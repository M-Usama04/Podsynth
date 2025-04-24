import streamlit as st
from dotenv import load_dotenv
import os
import re
import json
import numpy as np
import io
import base64
from youtube_transcript_api import YouTubeTranscriptApi
from deep_translator import GoogleTranslator
from gtts import gTTS
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors
import google.generativeai as genai
from textblob import TextBlob
import datetime
import pickle

# Load environment variables and configure API
load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")
if api_key:
    genai.configure(api_key=api_key)
else:
    st.error("Google API key not found. Please check your .env file.")

# Initialize session state for persistence
if "transcript_text" not in st.session_state:
    st.session_state.transcript_text = ""
if "transcript_segments" not in st.session_state:
    st.session_state.transcript_segments = []
if "final_transcript" not in st.session_state:
    st.session_state.final_transcript = ""
if "final_summary" not in st.session_state:
    st.session_state.final_summary = ""
if "transcript_pdf" not in st.session_state:
    st.session_state.transcript_pdf = None
if "summary_pdf" not in st.session_state:
    st.session_state.summary_pdf = None
if "transcript_audio" not in st.session_state:
    st.session_state.transcript_audio = None
if "summary_audio" not in st.session_state:
    st.session_state.summary_audio = None
if "video_id" not in st.session_state:
    st.session_state.video_id = ""
if "target_lang" not in st.session_state:
    st.session_state.target_lang = "English"
if "processing_complete" not in st.session_state:
    st.session_state.processing_complete = False
if "speaker_data" not in st.session_state:
    st.session_state.speaker_data = {}
if "speaker_summaries" not in st.session_state:
    st.session_state.speaker_summaries = {}
if "sentiment_data" not in st.session_state:
    st.session_state.sentiment_data = {}
if "speaker_sentiment" not in st.session_state:
    st.session_state.speaker_sentiment = {}
# New session state variables for insights
if "key_points" not in st.session_state:
    st.session_state.key_points = []
if "impactful_quotes" not in st.session_state:
    st.session_state.impactful_quotes = []
if "questions_answers" not in st.session_state:
    st.session_state.questions_answers = []
if "key_themes" not in st.session_state:
    st.session_state.key_themes = []
if "insights_pdf" not in st.session_state:
    st.session_state.insights_pdf = None
# New session state for history
if "video_history" not in st.session_state:
    st.session_state.video_history = {}

# Prompts
SUMMARY_PROMPT = """You are a YouTube video summarizer. You will take the transcript text
and summarize the entire video, providing the important points in under 250 words.
Please provide the summary of the text given here: """

DIARIZATION_PROMPT = """You're analyzing a transcript from a YouTube video to identify different speakers.
Please identify distinct speakers in this transcript and label each segment with the appropriate speaker.
Each speaker should be labeled as Speaker 1, Speaker 2, etc.
Return your response as a JSON object with this format:
{
    "speakers": [
        {
            "id": "Speaker 1",
            "segments": [
                {"text": "First segment text by Speaker 1"},
                {"text": "Another segment by Speaker 1"}
            ]
        },
        {
            "id": "Speaker 2",
            "segments": [
                {"text": "First segment text by Speaker 2"},
                {"text": "Another segment by Speaker 2"}
            ]
        }
    ]
}
Here's the transcript: """

SPEAKER_SUMMARY_PROMPT = """You're analyzing a transcript segment from a specific speaker in a YouTube video.
Please provide a concise summary (50-100 words) of this speaker's key points and contributions.
Focus on their main arguments, insights, or information they shared.
Here's the transcript segment for this speaker: """

# New prompts for insights extraction
KEY_POINTS_PROMPT = """Extract the 5-7 most valuable and important points from this YouTube video transcript.
Focus on insights, takeaways, or information that would be most useful to someone who hasn't watched the video.
Format your response as a JSON array with this structure:
{
    "key_points": [
        "First important point in a concise sentence",
        "Second important point in a concise sentence",
        "Etc."
    ]
}
Here's the transcript: """

QUOTES_PROMPT = """Extract 3-5 of the most impactful, insightful, or memorable quotes from this YouTube video transcript.
Choose quotes that represent significant ideas, are well-articulated, or capture key moments in the discussion.
Format your response as a JSON array with this structure:
{
    "quotes": [
        {
            "text": "The exact quote text",
            "speaker": "Speaker name/number if available, otherwise 'Unknown'"
        },
        {
            "text": "Another quote",
            "speaker": "Speaker name/number if available"
        }
    ]
}
Here's the transcript: """

QA_PROMPT = """Identify any questions and their corresponding answers from this YouTube video transcript.
Look for explicit questions asked and the responses given to them.
Format your response as a JSON array with this structure:
{
    "qa_pairs": [
        {
            "question": "The exact question text",
            "asker": "Speaker who asked the question (if known, otherwise 'Unknown')",
            "answer": "The answer provided in response to the question",
            "answerer": "Speaker who provided the answer (if known, otherwise 'Unknown')"
        }
    ]
}
Here's the transcript: """

THEMES_PROMPT = """Identify 3-5 main themes or topics discussed in this YouTube video transcript.
For each theme, provide a brief description and explanation of how it relates to the video content.
Format your response as a JSON array with this structure:
{
    "themes": [
        {
            "name": "Name of the theme/topic",
            "description": "Brief explanation of this theme and its importance in the video"
        },
        {
            "name": "Name of another theme/topic",
            "description": "Brief explanation of this theme"
        }
    ]
}
Here's the transcript: """

# Helper for audio player HTML
def get_audio_player_html(audio_data, label="Audio Player"):
    audio_base64 = base64.b64encode(audio_data.read()).decode()
    audio_html = f"""
        <div style="padding:10px; border-radius:5px; border:1px solid #ddd; margin-bottom:10px;">
            <p style="margin-bottom:5px; font-weight:bold;">{label}</p>
            <audio controls style="width:100%">
                <source src="data:audio/mp3;base64,{audio_base64}" type="audio/mp3">
                Your browser does not support the audio element.
            </audio>
        </div>
    """
    return audio_html

# Helper: Extract video ID from URL
def extract_video_id(youtube_url):
    # Handle different URL formats
    patterns = [
        r'(?:v=|\/)([0-9A-Za-z_-]{11}).*',  # Standard and short URL
        r'(?:embed\/)([0-9A-Za-z_-]{11})',  # Embed URL
        r'(?:youtu\.be\/)([0-9A-Za-z_-]{11})'  # Shortened URL
    ]
    
    for pattern in patterns:
        match = re.search(pattern, youtube_url)
        if match:
            return match.group(1)
    
    return None

# Helper: Get transcript from YouTube
def extract_transcript_details(youtube_video_url):
    try:
        video_id = extract_video_id(youtube_video_url)
        if not video_id:
            return "", video_id, None, "Invalid YouTube URL format"
        
        transcript_list = YouTubeTranscriptApi.get_transcript(video_id)
        # Directly extract text from transcript items
        transcript = " ".join([item['text'] for item in transcript_list])
        return transcript, video_id, transcript_list, None
    except Exception as e:
        error_message = str(e)
        return "", "", [], f"Failed to retrieve transcript: {error_message}"

# Helper: Get video title from video ID
def get_video_title(video_id):
    try:
        # This is a placeholder - in a production app, you would use the YouTube API
        # For now, let's return a generic title with the video ID
        return f"YouTube Video (ID: {video_id})"
    except Exception as e:
        return f"Unknown Video (ID: {video_id})"

# Helper: Perform speaker diarization using Gemini
def perform_speaker_diarization(transcript):
    try:
        model = genai.GenerativeModel("gemini-1.5-pro")
        response = model.generate_content(DIARIZATION_PROMPT + transcript)
        
        # Parse the JSON from the response
        content = response.text
        # Extract JSON if it's wrapped in markdown code blocks
        json_match = re.search(r'```(?:json)?\s*([\s\S]*?)\s*```', content)
        if json_match:
            content = json_match.group(1)
        
        # Clean any remaining text and parse JSON
        speaker_data = json.loads(content)
        return speaker_data, None
    except Exception as e:
        return {}, f"Speaker diarization failed: {str(e)}"

# Helper: Generate speaker summary
def generate_speaker_summary(speaker_text):
    try:
        model = genai.GenerativeModel("gemini-1.5-pro")
        response = model.generate_content(SPEAKER_SUMMARY_PROMPT + speaker_text)
        return response.text, None
    except Exception as e:
        return "", f"Speaker summary generation failed: {str(e)}"

# Helper: Extract insights using Gemini
def extract_insights(transcript, prompt_type):
    try:
        model = genai.GenerativeModel("gemini-1.5-pro")
        
        prompt = ""
        if prompt_type == "key_points":
            prompt = KEY_POINTS_PROMPT
        elif prompt_type == "quotes":
            prompt = QUOTES_PROMPT
        elif prompt_type == "qa":
            prompt = QA_PROMPT
        elif prompt_type == "themes":
            prompt = THEMES_PROMPT
        
        # Limit text length to avoid API limits
        max_length = 100000
        if len(transcript) > max_length:
            transcript = transcript[:max_length] + "... (truncated due to length)"
            
        response = model.generate_content(prompt + transcript)
        content = response.text
        
        # Extract JSON if it's wrapped in markdown code blocks
        json_match = re.search(r'```(?:json)?\s*([\s\S]*?)\s*```', content)
        if json_match:
            content = json_match.group(1)
            
        # Parse JSON response
        result = json.loads(content)
        return result, None
    except Exception as e:
        return {}, f"Insights extraction failed for {prompt_type}: {str(e)}"

# Helper: Perform sentiment analysis
def analyze_sentiment(text):
    try:
        # Use TextBlob for sentiment analysis
        blob = TextBlob(text)
        # Get polarity score (-1 to 1, where -1 is negative, 0 is neutral, 1 is positive)
        polarity = blob.sentiment.polarity
        # Get subjectivity score (0 to 1, where 0 is objective, 1 is subjective)
        subjectivity = blob.sentiment.subjectivity
        
        # Determine sentiment category
        if polarity > 0.1:
            sentiment = "Positive"
        elif polarity < -0.1:
            sentiment = "Negative"
        else:
            sentiment = "Neutral"
            
        return {
            "polarity": polarity,
            "subjectivity": subjectivity,
            "sentiment": sentiment
        }, None
    except Exception as e:
        return {}, f"Sentiment analysis failed: {str(e)}"

# Helper: Summarize with Gemini
def generate_summary(text, prompt):
    try:
        # Limit text length to avoid API limits
        max_length = 100000
        if len(text) > max_length:
            text = text[:max_length] + "... (truncated due to length)"
        
        model = genai.GenerativeModel("gemini-1.5-pro")
        response = model.generate_content(prompt + text)
        return response.text, None
    except Exception as e:
        return "", f"Summary generation failed: {str(e)}"

# Helper: Translate text
def translate_text(text, target_language):
    try:
        # Handle text in chunks if it's too long
        max_chunk_size = 4900  # Google Translator limit is 5000 characters
        
        if len(text) <= max_chunk_size:
            translator = GoogleTranslator(source='auto', target=target_language)
            return translator.translate(text), None
        
        # Split text into chunks for translation
        chunks = [text[i:i+max_chunk_size] for i in range(0, len(text), max_chunk_size)]
        translator = GoogleTranslator(source='auto', target=target_language)
        translated_chunks = [translator.translate(chunk) for chunk in chunks]
        
        return ''.join(translated_chunks), None
    except Exception as e:
        return "", f"Translation failed: {str(e)}"

# Helper: Create PDF
def create_pdf(text, title="Document", is_transcript_with_speakers=False, speaker_data=None, is_insights=False, insights_data=None):
    try:
        buffer = io.BytesIO()
        doc = SimpleDocTemplate(buffer, pagesize=letter)
        styles = getSampleStyleSheet()
        
        # Create custom styles
        speaker_style = ParagraphStyle(
            'SpeakerStyle',
            parent=styles['Heading2'],
            textColor=colors.blue,
            spaceBefore=12,
            spaceAfter=6
        )
        
        heading_style = ParagraphStyle(
            'HeadingStyle',
            parent=styles['Heading1'],
            textColor=colors.black,
            spaceBefore=12,
            spaceAfter=6
        )
        
        subheading_style = ParagraphStyle(
            'SubheadingStyle',
            parent=styles['Heading2'],
            textColor=colors.darkblue,
            spaceBefore=10,
            spaceAfter=5
        )
        
        highlight_style = ParagraphStyle(
            'HighlightStyle',
            parent=styles['Normal'],
            textColor=colors.black,
            backColor=colors.lightgrey,
            spaceBefore=6,
            spaceAfter=6,
            borderWidth=1,
            borderColor=colors.grey,
            borderPadding=5
        )
        
        story = []
        
        # Add title
        story.append(Paragraph(title, styles['Title']))
        story.append(Spacer(1, 12))
        
        if is_insights and insights_data:
            # Format insights document
            # Key Points
            story.append(Paragraph("Key Points", heading_style))
            if "key_points" in insights_data and insights_data["key_points"]:
                for i, point in enumerate(insights_data["key_points"], 1):
                    story.append(Paragraph(f"{i}. {point}", styles["Normal"]))
                    story.append(Spacer(1, 6))
            else:
                story.append(Paragraph("No key points extracted.", styles["Normal"]))
            story.append(Spacer(1, 12))
            
            # Impactful Quotes
            story.append(Paragraph("Impactful Quotes", heading_style))
            if "quotes" in insights_data and insights_data["quotes"]:
                for i, quote in enumerate(insights_data["quotes"], 1):
                    quote_text = quote.get("text", "")
                    speaker = quote.get("speaker", "Unknown")
                    story.append(Paragraph(f'"{quote_text}"', highlight_style))
                    story.append(Paragraph(f"- {speaker}", styles["Italic"]))
                    story.append(Spacer(1, 8))
            else:
                story.append(Paragraph("No impactful quotes extracted.", styles["Normal"]))
            story.append(Spacer(1, 12))
            
            # Questions and Answers
            story.append(Paragraph("Questions & Answers", heading_style))
            if "qa_pairs" in insights_data and insights_data["qa_pairs"]:
                for i, qa in enumerate(insights_data["qa_pairs"], 1):
                    question = qa.get("question", "")
                    asker = qa.get("asker", "Unknown")
                    answer = qa.get("answer", "")
                    answerer = qa.get("answerer", "Unknown")
                    
                    story.append(Paragraph(f"Q{i}: {question}", subheading_style))
                    story.append(Paragraph(f"Asked by: {asker}", styles["Italic"]))
                    story.append(Spacer(1, 4))
                    story.append(Paragraph(f"A: {answer}", styles["Normal"]))
                    story.append(Paragraph(f"Answered by: {answerer}", styles["Italic"]))
                    story.append(Spacer(1, 10))
            else:
                story.append(Paragraph("No question-answer pairs extracted.", styles["Normal"]))
            story.append(Spacer(1, 12))
            
            # Key Themes
            story.append(Paragraph("Key Themes", heading_style))
            if "themes" in insights_data and insights_data["themes"]:
                for i, theme in enumerate(insights_data["themes"], 1):
                    name = theme.get("name", "")
                    description = theme.get("description", "")
                    story.append(Paragraph(f"{i}. {name}", subheading_style))
                    story.append(Paragraph(description, styles["Normal"]))
                    story.append(Spacer(1, 8))
            else:
                story.append(Paragraph("No key themes extracted.", styles["Normal"]))
                
        elif is_transcript_with_speakers and speaker_data and 'speakers' in speaker_data:
            # Format document with speaker segments
            for speaker in speaker_data['speakers']:
                # Add speaker heading
                story.append(Paragraph(speaker['id'], speaker_style))
                
                # Add each segment for this speaker
                for segment in speaker['segments']:
                    if segment['text'].strip():
                        p = Paragraph(segment['text'], styles["Normal"])
                        story.append(p)
                        story.append(Spacer(1, 6))
                
                story.append(Spacer(1, 12))
        else:
            # Regular text document
            paragraphs = text.split('\n')
            for para in paragraphs:
                if para.strip():  # Skip empty paragraphs
                    p = Paragraph(para, styles["Normal"])
                    story.append(p)
                    story.append(Spacer(1, 6))
        
        doc.build(story)
        buffer.seek(0)
        return buffer, None
    except Exception as e:
        return None, f"PDF creation failed: {str(e)}"

# Helper: Convert to MP3
def create_audio(text, language='en'):
    try:
        audio_buffer = io.BytesIO()
        tts = gTTS(text=text, lang=language, slow=False)
        tts.write_to_fp(audio_buffer)
        audio_buffer.seek(0)
        return audio_buffer, None
    except Exception as e:
        return None, f"Audio creation failed: {str(e)}"

# Helper: Save data to history
def save_to_history(video_id, video_title, data_dict):
    try:
        # Create a timestamp
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # Create history entry
        history_entry = {
            "video_id": video_id,
            "video_title": video_title,
            "timestamp": timestamp,
            "data": data_dict
        }
        
        # Add to history dictionary using video_id as key
        st.session_state.video_history[video_id] = history_entry
        
        # Optionally save to disk for persistence across sessions
        try:
            with open('.streamlit/video_history.pkl', 'wb') as f:
                pickle.dump(st.session_state.video_history, f)
        except Exception as e:
            st.warning(f"Could not save history to disk: {e}")
            
        return True
    except Exception as e:
        st.warning(f"Error saving history: {e}")
        return False

# Helper: Load history from disk
def load_history_from_disk():
    try:
        with open('.streamlit/video_history.pkl', 'rb') as f:
            st.session_state.video_history = pickle.load(f)
        return True
    except Exception:
        # If file doesn't exist or can't be read, just use empty dict
        if "video_history" not in st.session_state:
            st.session_state.video_history = {}
        return False

# Process YouTube URL function
def process_youtube_url():
    with st.spinner("Extracting transcript from YouTube..."):
        transcript_text, video_id, transcript_segments, transcript_error = extract_transcript_details(st.session_state.youtube_link)
        
        if transcript_error:
            st.error(transcript_error)
            return False
        
        st.session_state.transcript_text = transcript_text
        st.session_state.transcript_segments = transcript_segments
        st.session_state.video_id = video_id
        video_title = get_video_title(video_id)
        
        # Generate summary
        with st.spinner("Generating summary..."):
            summary, summary_error = generate_summary(transcript_text, SUMMARY_PROMPT)
            if summary_error:
                st.error(summary_error)
                return False
        
        # Perform speaker diarization
        with st.spinner("Identifying speakers in the transcript..."):
            speaker_data, diarization_error = perform_speaker_diarization(transcript_text)
            if diarization_error:
                st.warning(diarization_error)
                # Create a fallback speaker data structure if diarization fails
                speaker_data = {
                    "speakers": [
                        {
                            "id": "Speaker (All)",
                            "segments": [{"text": transcript_text}]
                        }
                    ]
                }
            
            st.session_state.speaker_data = speaker_data
            
            # Generate summaries for each speaker
            speaker_summaries = {}
            for speaker in speaker_data.get("speakers", []):
                speaker_id = speaker["id"]
                speaker_text = " ".join([segment["text"] for segment in speaker["segments"]])
                
                if speaker_text:
                    speaker_summary, summary_error = generate_speaker_summary(speaker_text)
                    if not summary_error:
                        speaker_summaries[speaker_id] = speaker_summary
                    else:
                        speaker_summaries[speaker_id] = f"Could not generate summary: {summary_error}"
            
            st.session_state.speaker_summaries = speaker_summaries
        
        # Extract insights
        with st.spinner("Extracting key insights from the transcript..."):
            # Extract key points
            key_points_data, key_points_error = extract_insights(transcript_text, "key_points")
            if key_points_error:
                st.warning(key_points_error)
                st.session_state.key_points = []
            else:
                st.session_state.key_points = key_points_data.get("key_points", [])
            
            # Extract impactful quotes
            quotes_data, quotes_error = extract_insights(transcript_text, "quotes")
            if quotes_error:
                st.warning(quotes_error)
                st.session_state.impactful_quotes = []
            else:
                st.session_state.impactful_quotes = quotes_data.get("quotes", [])
            
            # Extract Q&A pairs
            qa_data, qa_error = extract_insights(transcript_text, "qa")
            if qa_error:
                st.warning(qa_error)
                st.session_state.questions_answers = []
            else:
                st.session_state.questions_answers = qa_data.get("qa_pairs", [])
            
            # Extract key themes
            themes_data, themes_error = extract_insights(transcript_text, "themes")
            if themes_error:
                st.warning(themes_error)
                st.session_state.key_themes = []
            else:
                st.session_state.key_themes = themes_data.get("themes", [])
            
            # Create insights PDF
            insights_data = {
                "key_points": st.session_state.key_points,
                "quotes": st.session_state.impactful_quotes,
                "qa_pairs": st.session_state.questions_answers,
                "themes": st.session_state.key_themes
            }
            
            insights_pdf, pdf_error = create_pdf(
                "", 
                title="Video Insights",
                is_insights=True,
                insights_data=insights_data
            )
            
            if pdf_error:
                st.warning(pdf_error)
            else:
                st.session_state.insights_pdf = insights_pdf
        
        # Perform sentiment analysis
        with st.spinner("Analyzing sentiment..."):
            # Overall sentiment
            sentiment_data, sentiment_error = analyze_sentiment(transcript_text)
            if sentiment_error:
                st.warning(sentiment_error)
            st.session_state.sentiment_data = sentiment_data
            
            # Per speaker sentiment
            speaker_sentiment = {}
            for speaker in speaker_data.get("speakers", []):
                speaker_id = speaker["id"]
                speaker_text = " ".join([segment["text"] for segment in speaker["segments"]])
                
                if speaker_text:
                    sentiment, _ = analyze_sentiment(speaker_text)
                    speaker_sentiment[speaker_id] = sentiment
            
            st.session_state.speaker_sentiment = speaker_sentiment
        
        lang_code = lang_codes[st.session_state.target_lang]
        
        # Translate if required
        if lang_code != 'en':
            with st.spinner(f"Translating to {st.session_state.target_lang}..."):
                final_transcript, transcript_trans_error = translate_text(transcript_text, lang_code)
                if transcript_trans_error:
                    st.warning(transcript_trans_error)
                    final_transcript = transcript_text
                
                final_summary, summary_trans_error = translate_text(summary, lang_code)
                if summary_trans_error:
                    st.warning(summary_trans_error)
                    final_summary = summary
        else:
            final_transcript = transcript_text
            final_summary = summary
        
        st.session_state.final_transcript = final_transcript
        st.session_state.final_summary = final_summary
        
        # Create PDF files
        with st.spinner("Creating PDF files..."):
            transcript_pdf, pdf_error = create_pdf(final_transcript, title="Transcript")
            summary_pdf, pdf_error_summary = create_pdf(final_summary, title="Summary")
            
            # Create speaker transcript PDF
            speaker_pdf, speaker_pdf_error = create_pdf(
                "", 
                title="Speaker Transcript",
                is_transcript_with_speakers=True,
                speaker_data=speaker_data
            )
            
            if pdf_error:
                st.warning(pdf_error)
            if pdf_error_summary:
                st.warning(pdf_error_summary)
            if speaker_pdf_error:
                st.warning(speaker_pdf_error)
                
            st.session_state.transcript_pdf = transcript_pdf
            st.session_state.summary_pdf = summary_pdf
            st.session_state.speaker_pdf = speaker_pdf
        
        # Create audio files - limit transcript audio to prevent timeouts
        with st.spinner("Creating audio files..."):
            # Limit transcript audio to first ~1-2 minutes for preview
            transcript_preview = final_transcript.split()[:500]  # About 500 words
            transcript_preview_text = " ".join(transcript_preview)
            
            transcript_audio, transcript_audio_error = create_audio(transcript_preview_text, lang_code)
            summary_audio, summary_audio_error = create_audio(final_summary, lang_code)
            
            if transcript_audio_error:
                st.warning(transcript_audio_error)
            if summary_audio_error:
                st.warning(summary_audio_error)
                
            st.session_state.transcript_audio = transcript_audio
            st.session_state.summary_audio = summary_audio
        
        # Save to history
        data_to_save = {
            "summary": final_summary,
            "transcript": final_transcript,
            "speaker_data": speaker_data,
            "speaker_summaries": speaker_summaries,
            "sentiment_data": sentiment_data,
            "speaker_sentiment": speaker_sentiment,
            "key_points": st.session_state.key_points,
            "impactful_quotes": st.session_state.impactful_quotes,
            "questions_answers": st.session_state.questions_answers,
            "key_themes": st.session_state.key_themes,
            "target_language": st.session_state.target_lang
        }
        save_to_history(video_id, video_title, data_to_save)
        
        st.session_state.processing_complete = True
        return True

# Helper: Display sentiment gauge and chart
def display_sentiment_gauge(polarity, title="Sentiment"):
    # Convert polarity (-1 to 1) to percentage (0 to 100)
    polarity_percentage = (polarity + 1) / 2 * 100
    
    # Determine color based on sentiment
    if polarity > 0.1:
        color = "green"
    elif polarity < -0.1:
        color = "red"
    else:
        color = "gray"
    
    # Create a simple gauge visualization
    gauge_html = f"""
    <div style="margin: 10px 0;">
        <p style="margin-bottom: 5px; font-weight: bold;">{title}</p>
        <div style="width: 100%; background-color: #eee; border-radius: 5px; height: 20px;">
            <div style="width: {polarity_percentage}%; background-color: {color}; height: 20px; border-radius: 5px;"></div>
        </div>
        <div style="display: flex; justify-content: space-between; margin-top: 5px;">
            <span>Neutral</span>
            <span>Positive</span>
        </div>
        <p style="text-align: center; margin-top: 10px;">
            Score: {polarity:.2f} ({sentiment_to_text(polarity)})
        </p>
    </div>
    """
    return gauge_html

# Helper: Convert polarity to text label
def sentiment_to_text(polarity):
    if polarity > 0.5:
        return "Very Positive"
    elif polarity > 0.1:
        return "Positive"
    elif polarity < -0.5:
        return "Very Negative"
    elif polarity < -0.1:
        return "Negative"
    else:
        return "Neutral"

# Helper: Load a specific video from history
def load_video_from_history(video_id):
    if video_id in st.session_state.video_history:
        history_entry = st.session_state.video_history[video_id]
        data = history_entry["data"]
        
        # Load data back into session state
        st.session_state.final_summary = data.get("summary", "")
        st.session_state.final_transcript = data.get("transcript", "")
        st.session_state.speaker_data = data.get("speaker_data", {})
        st.session_state.speaker_summaries = data.get("speaker_summaries", {})
        st.session_state.sentiment_data = data.get("sentiment_data", {})
        st.session_state.speaker_sentiment = data.get("speaker_sentiment", {})
        st.session_state.key_points = data.get("key_points", [])
        st.session_state.impactful_quotes = data.get("impactful_quotes", [])
        st.session_state.questions_answers = data.get("questions_answers", [])
        st.session_state.key_themes = data.get("key_themes", [])
        st.session_state.target_lang = data.get("target_language", "English")
        st.session_state.video_id = video_id
        
        # Regenerate PDFs and audio files on demand
        # We'll do this in the specific tab sections when needed
        
        st.session_state.processing_complete = True
        return True
    return False

# ---------------- UI Starts ------------------
st.set_page_config(page_title="YouTube AI Notes", layout="wide")
st.title("Podsynth: Smart podcast summarizer")

# Try to load history from disk
load_history_from_disk()

# Input form to prevent reloading on every interaction
with st.form("youtube_form"):
    youtube_link = st.text_input("🔗 Enter YouTube Video Link:", key="youtube_link")
    
    lang_codes = {
        "English": "en", "Hindi": "hi", "Spanish": "es", "French": "fr", "German": "de",
        "Tamil": "ta", "Telugu": "te", "Gujarati": "gu", "Bengali": "bn", "Japanese": "ja"
    }
    
    target_lang = st.selectbox("🌐 Select Target Language for Output:", 
        options=list(lang_codes.keys()), key="target_lang")
    
    # Submit button
    submit_button = st.form_submit_button("🚀 Generate Notes")
    
    if submit_button:
        process_youtube_url()

# Display results if processing is complete
if st.session_state.processing_complete:
    # Display YouTube video if we have a video ID
    if st.session_state.video_id:
        st.video(f"https://www.youtube.com/watch?v={st.session_state.video_id}")
    
    # Display tabs UI
    tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
        "📜 Transcript", 
        "📝 Summary", 
        "👥 Speaker Diarization",
        "🗣️ Speaker Summaries",
        "😊 Sentiment Analysis",
        "🔍 Insights",  # New tab
        "📚 History"    # New tab
    ])
    
    # Tab 1: Basic Transcript
    with tab1:
        st.markdown(f"### 🎧 Transcript ({st.session_state.target_lang})")
        st.text_area("", st.session_state.final_transcript, height=300)
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.session_state.transcript_pdf:
                st.download_button("⬇️ Download Transcript as PDF", 
                                  data=st.session_state.transcript_pdf, 
                                  file_name="Transcript.pdf",
                                  mime="application/pdf")
        
        with col2:
            if st.session_state.transcript_audio:
                st.download_button("⬇️ Download Transcript Preview as MP3", 
                                  data=st.session_state.transcript_audio, 
                                  file_name="Transcript_Preview.mp3",
                                  mime="audio/mp3")
        
        # Audio player for transcript
        if st.session_state.transcript_audio:
            st.session_state.transcript_audio.seek(0)  # Reset pointer
            st.markdown(get_audio_player_html(st.session_state.transcript_audio, 
                                             f"Transcript Preview ({st.session_state.target_lang})"), 
                       unsafe_allow_html=True)
            st.caption("Note: Audio preview is limited to approximately the first 500 words")
    
    # Tab 2: Summary
    with tab2:
        st.markdown(f"### 🧠 Summary ({st.session_state.target_lang})")
        st.text_area("", st.session_state.final_summary, height=300)
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.session_state.summary_pdf:
                st.download_button("⬇️ Download Summary as PDF", 
                                  data=st.session_state.summary_pdf, 
                                  file_name="Summary.pdf",
                                  mime="application/pdf")
        
        with col2:
            if st.session_state.summary_audio:
                st.download_button("⬇️ Download Summary as MP3", 
                                  data=st.session_state.summary_audio, 
                                  file_name="Summary.mp3",
                                  mime="audio/mp3")
        
        # Audio player for summary
        if st.session_state.summary_audio:
            st.session_state.summary_audio.seek(0)  # Reset pointer
            st.markdown(get_audio_player_html(st.session_state.summary_audio, 
                                             f"Summary Audio ({st.session_state.target_lang})"), 
                       unsafe_allow_html=True)
    
    # Tab 3: Speaker Diarization
    with tab3:
        st.markdown("### 👥 Speaker Diarization")
        st.write("This feature identifies and separates different speakers in the transcript.")
        
        speaker_data = st.session_state.speaker_data
        if speaker_data and "speakers" in speaker_data:
            # Download full speaker transcript
            if "speaker_pdf" in st.session_state and st.session_state.speaker_pdf:
                st.download_button(
                    "⬇️ Download Complete Speaker Transcript as PDF",
                    data=st.session_state.speaker_pdf,
                    file_name="Speaker_Transcript.pdf",
                    mime="application/pdf"
                )
            
            for speaker in speaker_data["speakers"]:
                with st.expander(f"{speaker['id']} - Click to view transcript"):
                    speaker_text = "\n\n".join([segment["text"] for segment in speaker["segments"]])
                    st.text_area("", speaker_text, height=200)
                    
                    # Create audio for this speaker's transcript
                    if st.button(f"🔊 Generate Audio for {speaker['id']}", key=f"audio_{speaker['id']}"):
                        lang_code = lang_codes[st.session_state.target_lang]
                        with st.spinner(f"Creating audio for {speaker['id']}..."):
                            # Limit to first 1000 words
                            preview_text = " ".join(speaker_text.split()[:1000])
                            speaker_audio, audio_error = create_audio(preview_text, lang_code)
                            
                            if audio_error:
                                st.warning(audio_error)
                            elif speaker_audio:
                                st.markdown(get_audio_player_html(speaker_audio, 
                                                                f"{speaker['id']} Audio"), 
                                          unsafe_allow_html=True)
                                st.download_button(
                                    f"⬇️ Download {speaker['id']} Audio",
                                    data=speaker_audio,
                                    file_name=f"{speaker['id'].replace(' ', '_')}_Audio.mp3",
                                    mime="audio/mp3"
                                )
        else:
            st.warning("Speaker diarization could not be performed for this transcript.")
    
    # Tab 4: Speaker Summaries
    with tab4:
        st.markdown("### 🗣️ Speaker Summaries")
        st.write("Key takeaways from each speaker's contribution to the conversation.")
        
        speaker_summaries = st.session_state.speaker_summaries
        if speaker_summaries:
            for speaker_id, summary in speaker_summaries.items():
                with st.expander(f"{speaker_id} - Summary"):
                    st.write(summary)
                    
                    # Generate PDF for this speaker's summary
                    if st.button(f"📄 Create PDF for {speaker_id}'s Summary", key=f"pdf_{speaker_id}"):
                        with st.spinner(f"Creating PDF for {speaker_id}..."):
                            summary_pdf, pdf_error = create_pdf(
                                summary, 
                                title=f"Summary of {speaker_id}'s Contribution"
                            )
                            
                            if pdf_error:
                                st.warning(pdf_error)
                            elif summary_pdf:
                                st.download_button(
                                    f"⬇️ Download {speaker_id}'s Summary PDF",
                                    data=summary_pdf,
                                    file_name=f"{speaker_id.replace(' ', '_')}_Summary.pdf",
                                    mime="application/pdf"
                                )
        else:
            st.warning("Speaker summaries could not be generated for this transcript.")
    
    # Tab 5: Sentiment Analysis
    with tab5:
        st.markdown("### 😊 Sentiment Analysis")
        st.write("Analysis of emotional tone throughout the conversation.")
        
        # Overall sentiment
        sentiment_data = st.session_state.sentiment_data
        speaker_sentiment = st.session_state.speaker_sentiment
        
        if sentiment_data:
            st.subheader("Overall Sentiment")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown(display_sentiment_gauge(
                    sentiment_data.get("polarity", 0), 
                    "Overall Emotional Tone"
                ), unsafe_allow_html=True)
            
            with col2:
                st.markdown(f"""
                **Overall Assessment:** {sentiment_data.get('sentiment', 'Neutral')}  
                **Polarity Score:** {sentiment_data.get('polarity', 0):.2f}  
                **Subjectivity:** {sentiment_data.get('subjectivity', 0):.2f}  
                """)
            
            # Per-speaker sentiment
            if speaker_sentiment:
                st.subheader("Sentiment by Speaker")
                
                for speaker_id, sentiment in speaker_sentiment.items():
                    with st.expander(f"{speaker_id} - Sentiment Analysis"):
                        scol1, scol2 = st.columns(2)
                        
                        with scol1:
                            st.markdown(display_sentiment_gauge(
                                sentiment.get("polarity", 0), 
                                f"{speaker_id}'s Emotional Tone"
                            ), unsafe_allow_html=True)
                        
                        with scol2:
                            st.markdown(f"""
                            **Assessment:** {sentiment.get('sentiment', 'Neutral')}  
                            **Polarity Score:** {sentiment.get('polarity', 0):.2f}  
                            **Subjectivity:** {sentiment.get('subjectivity', 0):.2f}  
                            """)
        else:
            st.warning("Sentiment analysis could not be performed for this transcript.")
    
    # Tab 6: Insights (NEW)
    with tab6:
        st.markdown("### 🔍 Key Insights Extraction")
        st.write("Automatically extracted valuable insights from the video content.")
        
        # Download full insights PDF
        if st.session_state.insights_pdf:
            st.download_button(
                "⬇️ Download Complete Insights as PDF",
                data=st.session_state.insights_pdf,
                file_name="Video_Insights.pdf",
                mime="application/pdf"
            )
        
        # Key points section
        st.subheader("📌 Key Points")
        key_points = st.session_state.key_points
        if key_points:
            for i, point in enumerate(key_points, 1):
                st.markdown(f"**{i}.** {point}")
        else:
            st.info("No key points were extracted from this video.")
        
        # Impactful quotes section
        st.subheader("💬 Impactful Quotes")
        quotes = st.session_state.impactful_quotes
        if quotes:
            for quote in quotes:
                quote_text = quote.get("text", "")
                speaker = quote.get("speaker", "Unknown")
                
                st.markdown(f"""
                <div style="padding: 10px; background-color: #f0f2f6; border-radius: 5px; margin-bottom: 10px;">
                    <div style="font-style: italic; margin-bottom: 5px;">"{quote_text}"</div>
                    <div style="text-align: right; font-weight: bold;">— {speaker}</div>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.info("No impactful quotes were extracted from this video.")
        
        # Questions and answers section
        st.subheader("❓ Questions & Answers")
        qa_pairs = st.session_state.questions_answers
        if qa_pairs:
            for i, qa in enumerate(qa_pairs, 1):
                question = qa.get("question", "")
                asker = qa.get("asker", "Unknown")
                answer = qa.get("answer", "")
                answerer = qa.get("answerer", "Unknown")
                
                with st.expander(f"Q{i}: {question}"):
                    st.markdown(f"**Asked by:** {asker}")
                    st.markdown(f"**Answer:** {answer}")
                    st.markdown(f"**Answered by:** {answerer}")
        else:
            st.info("No question-answer pairs were extracted from this video.")
        
        # Key themes section
        st.subheader("🏷️ Key Themes")
        themes = st.session_state.key_themes
        if themes:
            cols = st.columns(min(3, len(themes)))
            for i, theme in enumerate(themes):
                col_idx = i % len(cols)
                name = theme.get("name", "")
                description = theme.get("description", "")
                
                with cols[col_idx]:
                    st.markdown(f"""
                    <div style="padding: 15px; background-color: #e6f3ff; border-radius: 5px; margin-bottom: 10px;">
                        <div style="font-weight: bold; margin-bottom: 5px; font-size: 1.1em;">{name}</div>
                        <div>{description}</div>
                    </div>
                    """, unsafe_allow_html=True)
        else:
            st.info("No key themes were extracted from this video.")
    
    # Tab 7: History (NEW)
    with tab7:
        st.markdown("### 📚 Video Processing History")
        st.write("Access previously processed videos without reprocessing them.")
        
        # Check if we have any history
        if st.session_state.video_history:
            # Create a sorted list of history entries (most recent first)
            history_entries = list(st.session_state.video_history.values())
            history_entries.sort(key=lambda x: x["timestamp"], reverse=True)
            
            # Display history entries
            for entry in history_entries:
                video_id = entry["video_id"]
                video_title = entry["video_title"]
                timestamp = entry["timestamp"]
                
                # Create a collapsible section for each video
                with st.expander(f"{video_title} - {timestamp}"):
                    # Show a thumbnail and basic info
                    st.image(f"https://img.youtube.com/vi/{video_id}/mqdefault.jpg", 
                            width=240)
                    
                    st.markdown(f"**Video ID:** {video_id}")
                    st.markdown(f"**Processed on:** {timestamp}")
                    st.markdown(f"**Language:** {entry['data'].get('target_language', 'English')}")
                    
                    # Button to load this video's data
                    if st.button(f"📂 Load This Video's Data", key=f"load_{video_id}"):
                        load_video_from_history(video_id)
                        st.success("Video data loaded successfully! Navigate to other tabs to view the content.")
                        st.rerun()  # Force a rerun to update all tabs
                    
                    # Button to watch the video
                    if st.button(f"▶️ Watch Video", key=f"watch_{video_id}"):
                        st.video(f"https://www.youtube.com/watch?v={video_id}")
                    
                    # Option to delete this entry
                    if st.button(f"🗑️ Delete Entry", key=f"delete_{video_id}"):
                        if video_id in st.session_state.video_history:
                            del st.session_state.video_history[video_id]
                            # Update disk storage
                            try:
                                with open('.streamlit/video_history.pkl', 'wb') as f:
                                    pickle.dump(st.session_state.video_history, f)
                            except Exception as e:
                                st.warning(f"Could not update history file: {e}")
                            st.success("Entry deleted successfully!")
                            st.rerun()  # Force a rerun to update the history list
        else:
            st.info("No video processing history available. Process some videos to see them here!")
            
        # Button to clear all history
        if st.session_state.video_history and st.button("🧹 Clear All History"):
            st.session_state.video_history = {}
            # Update disk storage
            try:
                with open('.streamlit/video_history.pkl', 'wb') as f:
                    pickle.dump(st.session_state.video_history, f)
            except Exception as e:
                st.warning(f"Could not update history file: {e}")
            st.success("History cleared successfully!")
            st.rerun()  # Force a rerun to update the UI
    
    st.success("✅ Notes generation complete!")
    
    # Add a reset button to start over
    if st.button("🔄 Process Another Video"):
        # Reset all session state variables
        for key in list(st.session_state.keys()):
            if key != "youtube_link" and key != "target_lang" and key != "video_history":
                if key in st.session_state:
                    del st.session_state[key]
        
        st.session_state.processing_complete = False
        # Force a rerun to show the form again
        st.rerun()