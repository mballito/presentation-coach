"""
Presentation Confidence Game — Backend API
FastAPI server for video analysis and coaching scores.
"""

import json
import os
import subprocess
import tempfile
import re
import math
from pathlib import Path
from typing import Optional
from datetime import datetime

import httpx
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse

app = FastAPI(title="Presentation Coach API")

# ─── Catch-all error handler ─────────────────────────────────────────
class AnalysisFallback(Exception):
    pass

@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    if isinstance(exc, HTTPException):
        raise exc
    return JSONResponse(status_code=200, content={
        "error": "Analysis error, showing demo scores",
        "_is_demo": True,
        "scenario_id": "fallback",
        "scenario_title": "Analysis Error",
        "active_coach": "comforter",
        "coaches": {
            "comforter": {"name": "The Comforter", "emoji": "💙",
                "focus": "Did you connect?", "score": 5.0,
                "short": "General — warmth, clarity, confidence"},
            "closer": {"name": "The Closer", "emoji": "🔴",
                "focus": "Did you land the deal?", "score": 5.0,
                "short": "Sales — directness, confidence, closure"},
            "igniter": {"name": "The Igniter", "emoji": "🟡",
                "focus": "Did you bring energy?", "score": 5.0,
                "short": "Presentations — energy, storytelling, variety"}
        },
        "transcript": "[Analysis failed — server error]",
        "objective_metrics": {"speaking_pace": 0, "filler_word_count": 0,
            "filler_words_per_minute": 0, "pause_ratio": 0,
            "total_words": 0, "duration_seconds": 0},
        "connector": {"name": "The Connector", "emoji": "🔵",
            "focus": "Do people feel heard?", "score": 5.0,
            "key_metrics": {"warmth": 5.0, "empathy": 5.0, "eye_contact": 5.0, "speaking_pace": 5.0}},
        "closer": {"name": "The Closer", "emoji": "🔴",
            "focus": "Did you land it?", "score": 5.0,
            "key_metrics": {"directness": 5.0, "confidence": 5.0, "closure": 5.0, "filler_words": 5.0}},
        "igniter": {"name": "The Igniter", "emoji": "🟡",
            "focus": "Did you bring energy?", "score": 5.0,
            "key_metrics": {"positive_framing": 5.0, "storytelling": 5.0, "engagement": 5.0, "vocal_variety": 5.0}},
        "feedback": ["The AI analysis encountered an error. Try again in a moment."],
        "audio_analysis": {}
    })

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE_DIR = Path(__file__).parent
SCENARIOS_PATH = BASE_DIR / "scenarios.json"
RECORDINGS_DIR = BASE_DIR / "recordings"
RECORDINGS_DIR.mkdir(exist_ok=True)

OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "hermes3:3b"

# ─── Load Whisper model once at startup ──────────────────────────────
_whisper_model = None

def get_whisper_model():
    global _whisper_model
    if _whisper_model is None:
        from faster_whisper import WhisperModel
        _whisper_model = WhisperModel("base", device="cpu", compute_type="int8")
    return _whisper_model


# ─── Helpers ────────────────────────────────────────────────────────

@app.on_event("startup")
async def preload_models():
    """Preload Whisper model at startup so first request isn't slow."""
    try:
        get_whisper_model()
        print(f"[startup] Whisper model loaded")
    except Exception as e:
        print(f"[startup] Could not preload Whisper: {e}")


def load_scenarios():
    with open(SCENARIOS_PATH) as f:
        return json.load(f)


def extract_audio(video_path: str, audio_path: str) -> bool:
    try:
        subprocess.run(
            ["ffmpeg", "-i", video_path, "-vn", "-acodec", "pcm_s16le",
             "-ar", "16000", "-ac", "1", audio_path, "-y"],
            capture_output=True, timeout=60
        )
        return Path(audio_path).exists() and Path(audio_path).stat().st_size > 0
    except Exception:
        return False


def transcribe_ollama(audio_path: str) -> str:
    try:
        model = get_whisper_model()
        segments, info = model.transcribe(audio_path, beam_size=5)
        text_parts = []
        for seg in segments:
            text_parts.append(seg.text.strip())
        transcript = " ".join(text_parts)
        return transcript if transcript.strip() else "[No speech detected]"
    except Exception as e:
        return f"[Transcription failed: {str(e)}]"


def calculate_objective_metrics(transcript: str, duration_seconds: float):
    if not transcript or transcript.startswith("[") or duration_seconds <= 0:
        return {
            "speaking_pace": 0, "filler_word_count": 0,
            "filler_words_per_minute": 0, "pause_ratio": 0,
            "total_words": 0, "duration_seconds": duration_seconds
        }
    words = transcript.split()
    word_count = len(words)
    minutes = duration_seconds / 60
    pace = round(word_count / minutes) if minutes > 0 else 0
    filler_pattern = r'\b(um|uh|er|ah|like|you know|sort of|kind of|basically|actually|literally|right\?|okay\?|so yeah|i mean|you see|well\s)\b'
    filler_matches = re.findall(filler_pattern, transcript.lower())
    filler_count = len(filler_matches)
    filler_per_min = round(filler_count / minutes, 1) if minutes > 0 else 0
    return {
        "speaking_pace": pace, "filler_word_count": filler_count,
        "filler_words_per_minute": filler_per_min, "pause_ratio": 0,
        "total_words": word_count, "duration_seconds": round(duration_seconds, 1)
    }


def score_pace(pace: int) -> float:
    if pace == 0: return 5.0
    if 140 <= pace <= 180: return 10.0
    if 120 <= pace <= 200: return 8.0
    if 100 <= pace <= 220: return 6.0
    if 80 <= pace <= 240: return 4.0
    return 2.0


def score_filler_words(per_min: float) -> float:
    if per_min <= 0.5: return 10.0
    if per_min <= 1.5: return 9.0
    if per_min <= 3: return 7.0
    if per_min <= 5: return 5.0
    if per_min <= 8: return 3.0
    return 1.0


def call_ollama_llm(prompt: str) -> str:
    payload = {
        "model": OLLAMA_MODEL, "prompt": prompt,
        "stream": False, "temperature": 0.3, "max_tokens": 1024
    }
    try:
        resp = httpx.post(OLLAMA_URL, json=payload, timeout=60)
        if resp.status_code == 200:
            return resp.json().get("response", "")
        return ""
    except Exception:
        return ""


def score_ai_metrics(transcript: str, scenario_title: str, custom_context: str = ""):
    if not transcript or transcript.startswith("["):
        defaults = {
            "directness": 5.0, "structure": 5.0, "confidence_markers": 5.0,
            "empathy_markers": 5.0, "positive_framing": 5.0, "specificity": 5.0,
            "storytelling": 5.0, "engagement_hooks": 5.0, "closure": 5.0, "warmth": 5.0
        }
        return defaults
    esc_transcript = transcript.replace('"', '\\"')
    custom_section = ""
    if custom_context:
        custom_section = f'CUSTOM CONTEXT — The user provided these details about the scenario:\n{custom_context}\n\nUse this context to evaluate how well the response fits the specific situation.\n\n'
    prompt = (
        'You are a professional communication coach. Score the following '
        'interview/speech response on these 10 metrics from 0.0 to 10.0.\n\n'
        f'Scenario: "{scenario_title}"\n\n'
        f'{custom_section}'
        'Response transcript:\n'
        f'"{esc_transcript}"\n\n'
        'Return ONLY a JSON object with these keys (no markdown, no explanation):\n'
        '- directness: Did they answer head-on or circle around it?\n'
        '- structure: Did the answer have beginning/middle/end? STAR method?\n'
        '- confidence_markers: "I know/believe" vs "I think maybe/sort of"\n'
        '- empathy_markers: Acknowledging others perspectives\n'
        '- positive_framing: "Challenge" vs "Problem" language\n'
        '- specificity: Concrete examples with details vs vague statements\n'
        '- storytelling: Narrative arc, tension, resolution\n'
        '- engagement_hooks: Rhetorical questions, contrast, vivid language\n'
        '- closure: Did they land the ending or trail off?\n'
        '- warmth: Tone that makes listener feel comfortable vs cold\n\n'
        'Example: {"directness": 8.5, "structure": 7.0, ...}'
    )
    result = call_ollama_llm(prompt)
    defaults = {
        "directness": 5.0, "structure": 5.0, "confidence_markers": 5.0,
        "empathy_markers": 5.0, "positive_framing": 5.0, "specificity": 5.0,
        "storytelling": 5.0, "engagement_hooks": 5.0, "closure": 5.0, "warmth": 5.0
    }
    try:
        json_match = re.search(r'\{[\s\S]+\}', result)
        if json_match:
            scores = json.loads(json_match.group())
            for k in defaults:
                val = scores.get(k, defaults[k])
                try:
                    scores[k] = float(val) if val is not None else defaults[k]
                except (ValueError, TypeError):
                    scores[k] = defaults[k]
                scores[k] = min(max(scores[k], 0.0), 10.0)
            return scores
    except (json.JSONDecodeError, ValueError, TypeError):
        pass
    return defaults


# ─── Real Audio Analysis ────────────────────────────────────────────

def analyze_audio(audio_path: str) -> dict:
    """
    Analyze audio file for real vocal metrics using librosa.
    Measures: vocal variety (pitch), volume stability, pause management.
    """
    try:
        import librosa
        import numpy as np
    except ImportError:
        return {"vocal_variety_score": 5.0, "volume_stability_score": 5.0,
                "pause_management_score": 5.0, "speech_duration": 0,
                "pause_ratio": 0.5, "pitch_std": 0, "volume_cv": 0.5}

    if not audio_path or not Path(audio_path).exists():
        return {"vocal_variety_score": 5.0, "volume_stability_score": 5.0,
                "pause_management_score": 5.0, "speech_duration": 0,
                "pause_ratio": 0.5, "pitch_std": 0, "volume_cv": 0.5}

    try:
        y, sr = librosa.load(audio_path, sr=16000)
        total_recording = len(y) / sr

        # VAD: adaptive threshold with floor
        frame_len = int(0.025 * sr)
        hop_len = int(0.010 * sr)
        rms = librosa.feature.rms(y=y, frame_length=frame_len, hop_length=hop_len)[0]
        # Use 15th percentile + minimum floor to catch quiet speech
        threshold = max(float(np.percentile(rms, 15)), 0.001)
        speech_frames = rms > threshold

        # Find continuous speech segments (>0.3s for short utterances)
        speech_flag = np.concatenate([[False], speech_frames, [False]])
        starts = np.where(~speech_flag[:-1] & speech_flag[1:])[0] * hop_len / sr
        ends = np.where(speech_flag[:-1] & ~speech_flag[1:])[0] * hop_len / sr
        speech_regions = [(float(s), float(e)) for s, e in zip(starts, ends) if e - s > 0.3]

        total_speech = sum(e - s for s, e in speech_regions)
        pause_ratio = (total_recording - total_speech) / max(total_recording, 1)
        # Clamp pause_ratio: if we detected very little speech but recording has clear audio,
        # cap it so it doesn't report absurd silence numbers
        if total_speech < 1.0 and total_recording > 5.0:
            # Likely VAD missed the speech — use a default
            pause_ratio = 0.3

        # Speech mask for targeted analysis
        speech_mask = np.zeros(len(y), dtype=bool)
        for s, e in speech_regions:
            speech_mask[int(s*sr):int(e*sr)] = True

        # ── Vocal variety (pitch std during speech) ──
        pitch_std = 0
        if np.any(speech_mask):
            y_speech = y[speech_mask]
            f0, voiced, probs = librosa.pyin(y_speech, fmin=65, fmax=1047, sr=sr)
            pitches = f0[~np.isnan(f0)]
            if len(pitches) > 50:
                pitch_std = float(np.std(pitches))

        # ── Volume stability (during speech only) ──
        volume_cv = 0.5
        if np.any(speech_mask):
            chunk_s = int(0.3 * sr)
            speech_idx = np.where(speech_mask)[0]
            energies = []
            for i in range(0, len(speech_idx), chunk_s):
                chunk = y[speech_idx[i:i+chunk_s]]
                energies.append(float(np.sqrt(np.mean(chunk**2))))
            energies = np.array(energies)
            mean_e = float(np.mean(energies))
            if mean_e > 0:
                volume_cv = float(np.std(energies) / mean_e)

        # ── Score conversions ──
        def _vocal(s):
            if s > 40: return 9.0
            if s > 25: return 7.5
            if s > 15: return 6.0
            if s > 8: return 5.0
            if s > 3: return 4.0
            return 3.0

        def _vol(cv):
            if cv < 0.20: return 9.0
            if cv < 0.30: return 7.5
            if cv < 0.40: return 6.0
            if cv < 0.55: return 4.5
            if cv < 0.75: return 3.0
            return 1.5

        def _pause(r):
            if r < 0.15: return 9.0
            if r < 0.25: return 7.5
            if r < 0.35: return 6.0
            if r < 0.50: return 4.0
            if r < 0.65: return 2.5
            return 1.0

        return {
            "vocal_variety_score": _vocal(pitch_std),
            "volume_stability_score": _vol(volume_cv),
            "pause_management_score": _pause(pause_ratio),
            "speech_duration": round(total_speech, 1),
            "pause_ratio": round(pause_ratio, 3),
            "pitch_std": round(pitch_std, 1),
            "volume_cv": round(volume_cv, 2)
        }

    except Exception as e:
        return {"vocal_variety_score": 5.0, "volume_stability_score": 5.0,
                "pause_management_score": 5.0, "speech_duration": 0,
                "pause_ratio": 0.5, "pitch_std": 0, "volume_cv": 0.5,
                "error": str(e)}


# ─── Coach Scoring ──────────────────────────────────────────────────

# Category → coach mapping
CATEGORY_COACH_MAP = {
    "sales-calls": "closer",
    "presentations": "igniter",
}
# Everything else uses The Comforter


def score_technical_explanation(transcript: str, scenario_title: str, scenario_context: dict = None) -> dict:
    """
    Score a CCNA/technical explanation on accuracy, clarity, and teaching quality.
    Uses the LLM to evaluate the technical content itself.
    """
    if not transcript or transcript.startswith("["):
        return {
            "accuracy": 5.0, "clarity": 5.0, "teaching_quality": 5.0,
            "depth": 5.0, "use_of_analogies": 5.0, "structure": 5.0,
            "key_points_covered": 5.0, "feedback": []
        }

    topic_info = scenario_context.get("topic", scenario_title) if scenario_context else scenario_title
    tip_info = scenario_context.get("teaching_tip", "") if scenario_context else ""
    esc_transcript = transcript.replace('"', '\\"')

    prompt = (
        'You are a senior Cisco Networking instructor evaluating a student who is EXPLAINING a technical topic. '
        'Judge the QUALITY OF THEIR EXPLANATION, not their communication style.\n\n'
        f'Topic they were asked to teach: "{scenario_title}"\n'
        f'Suggested teaching approach: "{tip_info}"\n\n'
        'Their explanation transcript:\n'
        f'"{esc_transcript}"\n\n'
        'Return ONLY valid JSON (no markdown, no explanation) with these keys:\n'
        '- "accuracy": 0-10 — Were they factually correct? Did they get any concepts wrong?\n'
        '- "clarity": 0-10 — Could a beginner follow this explanation?\n'
        '- "teaching_quality": 0-10 — Did they use good teaching techniques (analogies, examples, build from basics)?\n'
        '- "depth": 0-10 — Did they go beyond surface-level definitions?\n'
        '- "use_of_analogies": 0-10 — How effective were their real-world comparisons?\n'
        '- "structure": 0-10 — Logical flow: context → concept → example → summary?\n'
        '- "key_points_covered": 0-10 — Did they hit the important aspects of the topic?\n'
        '- "overall": 0-10 — Overall quality of the explanation\n'
        '- "feedback": [array of 2-4 specific, actionable sentences about how to improve the technical explanation]\n\n'
        'Example: {"accuracy": 8.0, "clarity": 7.0, "teaching_quality": 6.5, "depth": 5.0, "use_of_analogies": 7.5, "structure": 6.0, "key_points_covered": 7.0, "overall": 6.7, "feedback": ["You explained OSPF neighbours clearly, but you missed the DR/BDR election process on multi-access networks.", "Try adding a concrete example — walk through what happens when Router A sends a packet to Router C."]}'
    )

    result = call_ollama_llm(prompt)
    defaults = {
        "accuracy": 5.0, "clarity": 5.0, "teaching_quality": 5.0,
        "depth": 5.0, "use_of_analogies": 5.0, "structure": 5.0,
        "key_points_covered": 5.0, "overall": 5.0, "feedback": []
    }
    try:
        json_match = re.search(r'\{[\s\S]+\}', result)
        if json_match:
            scores = json.loads(json_match.group())
            for k in defaults:
                val = scores.get(k, defaults[k])
                try:
                    scores[k] = float(val) if val is not None else defaults[k]
                except (ValueError, TypeError):
                    scores[k] = defaults[k]
                if isinstance(scores[k], float):
                    scores[k] = min(max(scores[k], 0.0), 10.0)
            if not isinstance(scores.get("feedback"), list):
                scores["feedback"] = defaults["feedback"]
            return scores
    except (json.JSONDecodeError, ValueError, TypeError):
        pass
    return defaults

def calculate_coach_scores(objective: dict, ai_scores: dict, audio: dict = None, category_id: str = "") -> dict:
    """
    Calculate final scores for each coach based on weighted metrics.
    Uses real audio analysis if available, otherwise falls back.
    """
    if audio is None:
        audio = {}

    pace_s = score_pace(objective["speaking_pace"])
    filler_s = score_filler_words(objective["filler_words_per_minute"])

    # Real audio metrics (or fallback)
    vocal_variety = audio.get("vocal_variety_score", 5.0)
    volume_stab = audio.get("volume_stability_score", 5.0)
    pause_mgmt = audio.get("pause_management_score", 5.0)

    # ── The Comforter (general — interviews, difficult convos, networking, client, career, learning) ──
    comforter_raw = (
        ai_scores.get("warmth", 5) * 0.20 +
        ai_scores.get("empathy_markers", 5) * 0.15 +
        ai_scores.get("structure", 5) * 0.10 +
        ai_scores.get("specificity", 5) * 0.10 +
        ai_scores.get("confidence_markers", 5) * 0.10 +
        pace_s * 0.10 +
        filler_s * 0.10 +
        vocal_variety * 0.05 +
        pause_mgmt * 0.05 +
        volume_stab * 0.05
    )
    comforter_score = round(min(max(comforter_raw * 1.0, 0), 10), 1)

    # ── The Closer (sales specialist) ──
    closer_raw = (
        ai_scores.get("directness", 5) * 0.25 +
        ai_scores.get("confidence_markers", 5) * 0.20 +
        ai_scores.get("closure", 5) * 0.20 +
        filler_s * 0.15 +
        ai_scores.get("specificity", 5) * 0.10 +
        volume_stab * 0.10
    )
    closer_score = round(min(max(closer_raw * 1.0, 0), 10), 1)

    # ── The Igniter (presentations specialist) ──
    igniter_raw = (
        pace_s * 0.10 +
        ai_scores.get("positive_framing", 5) * 0.15 +
        ai_scores.get("storytelling", 5) * 0.15 +
        ai_scores.get("engagement_hooks", 5) * 0.10 +
        ai_scores.get("structure", 5) * 0.10 +
        vocal_variety * 0.15 +
        pause_mgmt * 0.10 +
        volume_stab * 0.05 +
        filler_s * 0.10
    )
    igniter_score = round(min(max(igniter_raw * 1.0, 0), 10), 1)

    # ── Determine which coach is active for this category ──
    active_coach_key = CATEGORY_COACH_MAP.get(category_id, "comforter")

    # Build all coaches data
    all_coaches = {
        "comforter": {
            "name": "The Comforter", "emoji": "💙",
            "focus": "Did you connect?",
            "score": comforter_score,
            "short": "General — warmth, clarity, confidence"
        },
        "closer": {
            "name": "The Closer", "emoji": "🔴",
            "focus": "Did you land the deal?",
            "score": closer_score,
            "short": "Sales — directness, confidence, closure"
        },
        "igniter": {
            "name": "The Igniter", "emoji": "🟡",
            "focus": "Did you bring energy?",
            "score": igniter_score,
            "short": "Presentations — energy, storytelling, variety"
        }
    }

    # Return only the active coach + all coaches (frontend can choose)
    active_coach = all_coaches[active_coach_key]

    # ── Feedback (based on active coach's focus AND category) ──
    feedback = []
    # Categories where empathy/warmth feedback makes sense
    empathy_categories = {"job-interviews", "difficult-convos", "networking-relationships",
                          "client-management", "career-growth", "negotiation"}
    # Categories where storytelling/energy feedback makes sense
    story_categories = {"presentations", "sales-calls", "networking-relationships"}
    # Categories where directness feedback makes sense
    direct_categories = {"sales-calls", "negotiation", "difficult-convos", "client-management"}
    if pace_s < 6:
        if objective["speaking_pace"] > 180:
            feedback.append("Try slowing down — you're speaking faster than ideal.")
        elif objective["speaking_pace"] > 0:
            feedback.append("Pick up the pace a little — you've got room to add energy.")
    if filler_s < 6:
        feedback.append(f"Watch your filler words ({objective['filler_words_per_minute']}/min). Try replacing 'um' with a pause.")
    if vocal_variety < 5:
        feedback.append("Your voice stays flat. Vary your pitch — go up and down to keep people engaged.")
    if volume_stab < 4:
        feedback.append("Your volume is inconsistent. Try to stay at a steady, confident level throughout.")
    if pause_mgmt < 4:
        pause_r = audio.get("pause_ratio", 0)
        feedback.append(f"Too much dead air ({pause_r:.0%} silence). Plan what you'll say so you don't trail off.")
    if ai_scores.get("directness", 5) < 6 and category_id in direct_categories:
        feedback.append("Get to the point faster. In sales, hesitation costs the deal.")
    if ai_scores.get("warmth", 5) < 6 and category_id in empathy_categories:
        feedback.append("Add more warmth — smile, use their name, show you care.")
    if ai_scores.get("confidence_markers", 5) < 6:
        feedback.append("Drop the hedging. Replace 'I think maybe' with 'I believe' or 'I know'.")
    if ai_scores.get("storytelling", 5) < 6 and category_id in story_categories:
        feedback.append("Tell a story, not a list. Set the scene, describe the tension, then the resolution.")
    if ai_scores.get("closure", 5) < 6 and category_id in direct_categories:
        feedback.append("Land the ending. Don't trail off — close with a strong statement.")
    if category_id in empathy_categories and ai_scores.get("empathy_markers", 5) < 5:
        feedback.append("Show you understand their perspective. Acknowledge their situation before jumping to solutions.")
    if category_id == "presentations" and ai_scores.get("positive_framing", 5) < 5:
        feedback.append("Reframe challenges as opportunities. 'This is a chance to...' beats 'This is a problem because...'")
    if category_id == "learning" and ai_scores.get("structure", 5) < 5:
        feedback.append("Structure your explanation: start with the 'why', then the 'how', then an example.")
    if category_id == "learning" and ai_scores.get("specificity", 5) < 5:
        feedback.append("Use a concrete example or analogy. Abstract concepts stick better when you ground them.")
    if not feedback:
        feedback.append(f"Solid response! {active_coach['name']} is happy with that. Try another scenario to keep improving.")

    return {
        "active_coach": active_coach_key,
        "coaches": all_coaches,
        "feedback": feedback,
        "objective_metrics": objective,
        "audio_analysis": {
            "speech_duration": audio.get("speech_duration", 0),
            "pause_ratio": audio.get("pause_ratio", 0),
            "pitch_std": audio.get("pitch_std", 0),
            "volume_cv": audio.get("volume_cv", 0)
        }
    }


# ─── API Endpoints ──────────────────────────────────────────────────

@app.get("/api/scenarios")
def get_scenarios():
    return load_scenarios()


@app.post("/api/analyze-outfit")
async def analyze_outfit(
    image: UploadFile = File(...),
    scenario_id: str = Form(...),
    scenario_title: str = Form("Dress Check"),
    outfit_context: str = Form("")
):
    """
    Analyze an outfit photo using vision LLM.
    Accepts an image and returns style/grooming feedback.
    """
    if not image.filename:
        raise HTTPException(400, "No image provided")

    # Save image
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    ext = Path(image.filename).suffix or ".jpg"
    img_path = RECORDINGS_DIR / f"outfit_{timestamp}{ext}"
    with open(img_path, "wb") as f:
        content = await image.read()
        f.write(content)

    # Encode image to base64 for ollama vision
    import base64
    with open(img_path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("utf-8")

    # Build vision prompt
    prompt = (
        "You are a professional image consultant and grooming coach. "
        "Analyze this photo of a person's outfit and appearance for the following scenario:\n\n"
        f"Scenario: {scenario_title}\n"
        f"Context: {outfit_context}\n\n"
        "Give feedback in this exact JSON format (no markdown, no explanation):\n"
        "{\n"
        '  "overall_score": <0-10>,\n'
        '  "facial_hair": {\n'
        '    "score": <0-10>,\n'
        '    "feedback": "<1 sentence>"\n'
        "  },\n"
        '  "clothing": {\n'
        '    "score": <0-10>,\n'
        '    "formality_fit": "<does it match the scenario?>",\n'
        '    "feedback": "<1 sentence>"\n'
        "  },\n"
        '  "grooming": {\n'
        '    "score": <0-10>,\n'
        '    "feedback": "<1 sentence>"\n'
        "  },\n"
        '  "scenario_fit": <0-10>,\n'
        '  "top_tip": "<best quick improvement tip>",\n'
        '  "verdict": "<pass/review/change>"\n'
        "}"
    )

    try:
        resp = httpx.post("http://localhost:11434/api/generate", json={
            "model": "gemma3:4b",
            "prompt": prompt,
            "images": [b64],
            "stream": False,
            "temperature": 0.2,
            "max_tokens": 1024
        }, timeout=120)

        raw = resp.json().get("response", "{}")

        # Try to parse JSON from response
        import re as _re
        json_match = _re.search(r'\{[\s\S]+\}', raw)
        if json_match:
            result = json.loads(json_match.group())
        else:
            result = {
                "overall_score": 5.0, "facial_hair": {"score": 5.0, "feedback": "Could not analyze."},
                "clothing": {"score": 5.0, "formality_fit": "Unknown", "feedback": "Could not analyze."},
                "grooming": {"score": 5.0, "feedback": "Could not analyze."},
                "scenario_fit": 5.0, "top_tip": "Try again with better lighting.",
                "verdict": "review"
            }
    except Exception as e:
        result = {
            "overall_score": 5.0, "facial_hair": {"score": 5.0, "feedback": f"Analysis error: {str(e)[:50]}"},
            "clothing": {"score": 5.0, "formality_fit": "Unknown", "feedback": "Vision model unavailable."},
            "grooming": {"score": 5.0, "feedback": "Could not analyze."},
            "scenario_fit": 5.0, "top_tip": "Start the backend for real AI analysis.",
            "verdict": "review",
            "_is_demo": True
        }

    # Clean up the image after analysis
    try:
        img_path.unlink(missing_ok=True)
    except Exception:
        pass

    result["scenario_title"] = scenario_title
    result["scenario_id"] = scenario_id
    return result


@app.post("/api/parse-job-url")
async def parse_job_url(url: str = Form(...)):
    """Fetch a job listing URL and extract structured info using the LLM."""
    if not url or not url.startswith(("http://", "https://")):
        raise HTTPException(400, "Invalid URL provided")

    try:
        async with httpx.AsyncClient(timeout=15, follow_redirects=True) as client:
            resp = await client.get(url, headers={
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
            })
            if resp.status_code != 200:
                raise HTTPException(400, f"Failed to fetch URL (status {resp.status_code})")

        html = resp.text

        import html as _html
        # Strip all HTML tags to get page text (rough extraction)
        text = re.sub(r'<[^>]+>', ' ', html)
        text = _html.unescape(text)
        text = re.sub(r'\s+', ' ', text).strip()
        # Take first 4000 chars (enough for a job listing)
        text = text[:4000]

        prompt = (
            "Extract job listing information from the text below. "
            "Return ONLY valid JSON (no markdown, no explanation) with these keys:\n"
            '  "role": the job title,\n'
            '  "company": the company name,\n'
            '  "salary": salary range if found,\n'
            '  "location": location if found,\n'
            '  "summary": 1-2 sentence summary of the role,\n'
            '  "key_skills": comma-separated key skills mentioned,\n'
            '  "industry": industry if discernible\n\n'
            f"Page text:\n{text[:4000]}"
        )

        resp_llm = httpx.post(
            "http://localhost:11434/api/generate",
            json={"model": OLLAMA_MODEL, "prompt": prompt, "stream": False, "temperature": 0.1, "max_tokens": 512},
            timeout=60
        )

        raw = resp_llm.json().get("response", "{}")
        json_match = re.search(r'\{[\s\S]+\}', raw)
        if json_match:
            parsed = json.loads(json_match.group())
            return {
                "success": True,
                "parsed": {
                    "role": parsed.get("role", ""),
                    "company": parsed.get("company", ""),
                    "salary": parsed.get("salary", ""),
                    "location": parsed.get("location", ""),
                    "summary": parsed.get("summary", ""),
                    "key_skills": parsed.get("key_skills", ""),
                    "industry": parsed.get("industry", ""),
                }
            }

        return {"success": False, "error": "Could not parse job listing from URL"}
    except httpx.HTTPError as e:
        raise HTTPException(400, f"Failed to fetch URL: {str(e)[:100]}")
    except Exception as e:
        return {"success": False, "error": f"Parse error: {str(e)[:100]}"}


@app.post("/api/analyze")
async def analyze_recording(
    video: UploadFile = File(...),
    scenario_id: str = Form(...),
    scenario_title: str = Form("Unknown Scenario"),
    duration: float = Form(0.0),
    custom_context: str = Form(""),
    category_id: str = Form("")
):
    if not video.filename:
        raise HTTPException(400, "No video file provided")

    # Save video
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    video_path = RECORDINGS_DIR / f"recording_{timestamp}_{video.filename}"
    with open(video_path, "wb") as f:
        content = await video.read()
        f.write(content)

    # Extract + transcribe audio
    audio_path = video_path.with_suffix(".wav")
    audio_ok = extract_audio(str(video_path), str(audio_path))

    transcript = ""
    if audio_ok:
        transcript = transcribe_ollama(str(audio_path))
    else:
        transcript = "[Audio extraction failed]"

    # Run audio analysis (real tone/volume/pause metrics)
    audio_metrics = analyze_audio(str(audio_path)) if audio_ok else {}

    # Calculate all scores
    objective = calculate_objective_metrics(transcript, duration)

    if category_id == "learning":
        # Technical explanation scoring for CCNA/teaching scenarios
        # Load scenario context to get topic and teaching tip
        try:
            all_scenarios_raw = load_scenarios()
            ctx = {}
            for cat in all_scenarios_raw.get("categories", []):
                for s in cat.get("scenarios", []):
                    if s["id"] == scenario_id:
                        ctx = s.get("context", {})
                        break
        except Exception:
            ctx = {}
        tech_scores = score_technical_explanation(transcript, scenario_title, ctx)
        result = calculate_coach_scores(objective, {}, audio_metrics, category_id)
        result["tech_scores"] = tech_scores
        result["active_coach"] = "tech-coach"
        # Override with technical scores
        result["coaches"]["tech-coach"] = {
            "name": "The Network Mentor", "emoji": "🎓",
            "focus": "Did you teach it well?",
            "score": tech_scores.get("overall", 5.0),
            "short": "Technical accuracy, clarity, teaching quality"
        }
        # Replace feedback with technical feedback
        if tech_scores.get("feedback"):
            result["feedback"] = tech_scores["feedback"]
    else:
        ai_scores = score_ai_metrics(transcript, scenario_title, custom_context)
        result = calculate_coach_scores(objective, ai_scores, audio_metrics, category_id)

    result["transcript"] = transcript
    result["scenario_id"] = scenario_id
    result["scenario_title"] = scenario_title
    result["audio_metrics_raw"] = audio_metrics

    return result


@app.get("/api/health")
def health():
    return {
        "status": "ok",
        "ollama_model": OLLAMA_MODEL,
        "recordings_count": len(list(RECORDINGS_DIR.glob("*.webm")))
    }


# Serve frontend
frontend_dir = BASE_DIR.parent / "frontend"
if frontend_dir.exists():
    app.mount("/", StaticFiles(directory=str(frontend_dir), html=True), name="frontend")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
