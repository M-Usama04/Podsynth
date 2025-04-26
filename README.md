# 🎧 PodSynth - AI-Powered Podcast Summarizer

> Final Year Project | Machine Learning | Natural Language Processing | Full-Stack AI Web App

---

## 📋 Overview

**PodSynth** is a powerful AI-based podcast summarizer that simplifies long podcasts and videos into concise, intelligent summaries. It leverages state-of-the-art ML/NLP models for transcription, summarization, speaker diarization, sentiment analysis, and multilingual translation — all combined with a modern, responsive frontend.

This project includes a **Streamlit backend** (Python) and a **React.js frontend** for a seamless user experience.

---

## 🚀 Key Features

- 🎙️ **Transcription**: Convert audio/video to accurate text using OpenAI Whisper.
- 🧠 **Smart Summarization**: Generate concise summaries via Gemini 1.5 Pro / BART model.
- 👥 **Speaker Diarization**: Identify and separate multiple speakers (pyannote.audio).
- 😊 **Sentiment Analysis**: Detect emotional tone using TextBlob/VADER.
- 🌎 **Translation**: Translate summaries into Hindi and Urdu.
- 🔊 **Text-to-Speech**: Generate audio summaries with gTTS.
- 📥 **Download Options**: Export transcripts and summaries.
- 💻 **Beautiful UI/UX**: Built with React.js, Vite, and Streamlit.

---

## 🛠️ Tech Stack

| Layer     | Technology                                                                 |
|-----------|-----------------------------------------------------------------------------|
| Backend   | Python, Streamlit, OpenAI Whisper, pyannote.audio, Huggingface Transformers |
| Frontend  | React.js, Vite, React Bootstrap, Tailwind CSS, Axios                        |
| NLP Tools | TextBlob, VADER, Google Translate API, gTTS                                 |
| Utilities | FFmpeg, youtube-dl / yt-dlp (for video/audio download)                      |

---

## 📁 Project Structure

```
podsynth/
├── backend/
│   ├── ready2.py            # Streamlit app (main backend logic)
│   ├── utils/               # (optional) helper modules (speaker, summarizer, etc.)
│   ├── requirements.txt     # Python dependencies
├── frontend/
│   ├── src/
│   │   ├── components/       # React components
│   │   ├── App.jsx
│   │   ├── main.jsx
│   ├── public/
│   ├── package.json         # npm dependencies
├── README.md
```

---

## 🔥 Installation & Setup

### 1. Backend (Streamlit + Python)

```bash
cd backend
python -m venv venv
source venv/bin/activate    # For Windows: venv\Scripts\activate
pip install -r requirements.txt
streamlit run ready2.py
```

### 2. Frontend (React + Vite)

```bash
cd frontend
npm install
npm run dev
```

---

## 💡 How to Use

- Upload an audio/video file **or** paste a YouTube link.
- Navigate through tabs: **Transcription**, **Summary**, **Speaker Diarization**, **Sentiment Analysis**, **Translation**.
- Download the transcript, summaries, or listen to text-to-speech.
- Enjoy a clean, responsive user experience.

---

## 🧠 Machine Learning Models

| Feature                  | Model/Library                         | Algorithm         |
|---------------------------|---------------------------------------|-------------------|
| Transcription             | OpenAI Whisper (Base Model)           | Speech Recognition |
| Summarization             | Gemini 1.5 Pro or BART (Hugging Face) | Abstractive Summarization |
| Speaker Diarization       | pyannote.audio                        | Clustering + Embeddings |
| Sentiment Analysis        | TextBlob / VADER                      | Polarity Scoring  |
| Translation (Hindi/Urdu)  | Google Translate API                  | Neural Machine Translation |
| Text-to-Speech (TTS)      | gTTS                                  | TTS Synthesis |


## 📜 License

This project is licensed under the MIT License.

---

## 🙏 Acknowledgments

- OpenAI Whisper
- Huggingface Transformers
- pyannote.audio Team
- Streamlit Community
- React.js Community
- Google Cloud APIs
- GitHub Copilot & Cursor AI assistance

---

> ✨ **PodSynth** simplifies podcasts. Saves time. Boosts understanding. 🌟
