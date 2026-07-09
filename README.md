# 🎤 Presentation Confidence Coach

Record yourself responding to real-world scenarios and get scored by AI coaches on speaking skills, technical accuracy, and presentation quality.

## Features

### 🎯 12 Categories — 121 Scenarios
- General Interviews, Behavioral, Technical Interviews, Sales Pitches, Presentations, Difficult Conversations, Negotiation, Networking, Career Growth, Client Management, **Learning / CCNA (Teach it)**, and a Live Conversations placeholder.

### 🤖 4 Specialist AI Coaches
| Coach | Specialises In |
|---|---|
| 🫂 **The Comforter** | Interviews, Difficult Conversations, Networking, Client Management, Career Growth, Negotiation, Learning, Dress |
| 🔥 **The Closer** | Sales Pitches |
| 💥 **The Igniter** | Presentations |
| 🎓 **The Network Mentor** | Technical accuracy for Learning / CCNA scenarios |

### 🎓 Technical Scoring (Learning Section)
CCNA explanations scored on: **Accuracy, Clarity, Teaching Quality, Depth, Analogies, Structure** — with technically relevant feedback.

### 📸 Dress & Presentation Analysis
Snap a photo and get AI analysis of facial hair, clothing, grooming — with a top tip and verdict.

### 🎛️ Real Audio Analysis
- Pitch variance, volume stability, pause ratio, speaking pace
- Category-aware feedback (no "show more warmth" for technical topics)

### 🏆 Gamification
- Easy (≥7.0), Medium (≥8.0), Hard (≥8.5) pass thresholds
- Points per scenario, leaderboard, level-up every 200pts
- Score breakdown per metric

### ✏️ Custom Scenarios
Paste a job URL to auto-parse role details, or write your own scenario description.

## Quick Start

```bash
# Backend
cd backend
pip install -r ../requirements.txt
uvicorn main:app --host 0.0.0.0 --port 8001 --ssl-keyfile ../ssl/key.pem --ssl-certfile ../ssl/cert.pem

# Frontend
# Serve frontend/index.html via any static server or open directly (needs HTTPS for microphone access)
```

### Prerequisites
- Python 3.10+
- [faster-whisper](https://github.com/SYSTRAN/faster-whisper) (loaded at startup)
- [ollama](https://ollama.com/) with `hermes3:3b` and `gemma3:4b` models

## Tech Stack
- **Backend**: FastAPI + faster-whisper (transcription) + ollama (scoring)
- **Frontend**: Vanilla JS + CSS (glassmorphism, dark theme, spring animations)
- **Audio**: Web Audio API → PCM → Whisper transcription
- **Photo**: getUserMedia → base64 → gemma3:4b vision analysis

## Scoring System
Each metric is scored 0.0–10.0 then averaged:
- **Voice**: Pitch Variety, Volume Stability, Pause Ratio, Speaking Pace
- **Content**: Structure (intro-body-outro), Specificity (concrete examples), Empathy/Warmth, Confidence
- **Technical (Learning only)**: Accuracy, Clarity, Teaching Quality, Depth, Analogies, Structure
