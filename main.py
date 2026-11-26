import json
import logging
import os
import re
import urllib.request
from functools import lru_cache
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from groq import Groq
from pydantic import BaseModel
from youtube_transcript_api import YouTubeTranscriptApi

PREFERRED = [
    "llama-3.3-70b-versatile",
    "openai/gpt-oss-120b",
    "openai/gpt-oss-20b",
    "llama-3.1-8b-instant",
    "meta-llama/llama-4-scout-17b-16e-instruct",
    "qwen/qwen3-32b",
]
NOT_CHAT = ("whisper", "tts", "guard", "orpheus", "playai", "embed", "compound", "safeguard")
MAX_CHARS = 20000
STYLES = {
    "short": "Write a short summary (3-4 sentences), then 5 key takeaways as bullet points.",
    "detailed": "Write a detailed, well-structured summary with headings and bullet points "
    "covering every major section of the video.",
    "simple": "Explain the video in very simple language, like to a 15-year-old, "
    "using short sentences and one everyday example.",
}
LANGUAGES = {
    "English": "English",
    "Hindi": "Hindi (Devanagari script)",
    "Hinglish": "Hinglish (Hindi written in English letters)",
}

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("queryflow")
app = FastAPI(title="Queryflow")
BASE = Path(__file__).parent / "static"


@lru_cache(maxsize=1)
def client() -> Groq:
    # Strip spaces and stray quotes, a common mistake when setting the key in cmd.
    key = os.environ.get("GROQ_API_KEY", "").strip().strip("\"'").strip()
    if not key:
        raise HTTPException(500, "Server is missing GROQ_API_KEY. Set it in the same window that runs uvicorn, then restart.")
    return Groq(api_key=key)


_models_cache: list = []


def models() -> list:
    """Chat models this key can use, best first. Only successful lookups are cached."""
    global _models_cache
    if _models_cache:
        return _models_cache
    try:
        ids = [m.id for m in client().models.list().data]
    except HTTPException:
        raise
    except Exception:  # noqa: BLE001
        return PREFERRED
    chat = [i for i in ids if not any(b in i.lower() for b in NOT_CHAT)]
    _models_cache = ([m for m in PREFERRED if m in chat] + [m for m in chat if m not in PREFERRED]) or PREFERRED
    return _models_cache


def ask(messages: list) -> str:
    errors = []  # (model, status code, message)
    for model in models()[:4]:
        try:
            r = client().chat.completions.create(model=model, messages=messages, temperature=0.3)
            return r.choices[0].message.content
        except HTTPException:
            raise
        except Exception as e:  # noqa: BLE001
            log.warning("Groq model %s failed: %r", model, e)
            errors.append((model, getattr(e, "status_code", None), str(e)))
    codes = [c for _, c, _ in errors]
    if 401 in codes or 403 in codes:
        raise HTTPException(
            401,
            "Groq rejected your API key. Set GROQ_API_KEY again with no quotes or spaces, "
            "or create a new key at console.groq.com, then restart uvicorn.",
        )
    if codes and all(c == 429 for c in codes):
        raise HTTPException(429, "Rate limit hit on the free tier. Wait a minute and try again.")
    if 413 in codes:
        raise HTTPException(413, "The transcript is too long for the free tier. Try a shorter video.")
    model, code, msg = errors[-1] if errors else ("?", None, "no models available")
    raise HTTPException(502, f"The AI service failed (HTTP {code}, {model}): {msg[:220]}")


@app.get("/api/test")
def test_ai():
    """Open /api/test in the browser to check the Groq connection."""
    return {"models": models()[:4], "reply": ask([{"role": "user", "content": "Reply with the single word: ok"}])}


def system_prompt(transcript: str, language: str) -> str:
    lang = LANGUAGES.get(language, "English")
    return (
        "You answer questions about a YouTube video using ONLY its transcript below. "
        "If the answer is not in the transcript, say so. Be clear and concise. "
        f"Always reply in {lang}.\n\nTRANSCRIPT:\n{transcript[:MAX_CHARS]}"
    )


def video_id_from(url: str):
    m = re.search(r"(?:v=|youtu\.be/|shorts/|embed/)([A-Za-z0-9_-]{11})", url or "")
    return m.group(1) if m else None


def fetch_title(video_id: str) -> str:
    try:
        u = f"https://www.youtube.com/oembed?format=json&url=https://www.youtube.com/watch?v={video_id}"
        with urllib.request.urlopen(u, timeout=5) as r:
            return json.load(r).get("title", "")
    except Exception:  # noqa: BLE001
        return ""


class LoadReq(BaseModel):
    url: str = ""
    transcript: str = ""


class SummaryReq(BaseModel):
    transcript: str
    style: str = "short"
    language: str = "English"


class Msg(BaseModel):
    role: str
    content: str


class ChatReq(BaseModel):
    transcript: str
    messages: list[Msg]
    language: str = "English"


def get_transcript(vid: str) -> str:
    """Try English/Hindi first; if neither exists, fall back to any caption track."""
    yt = YouTubeTranscriptApi()
    try:
        snippets = yt.fetch(vid, languages=["en", "en-US", "en-GB", "hi"])
    except Exception as e:  # noqa: BLE001
        if type(e).__name__ != "NoTranscriptFound":
            raise
        snippets = next(iter(yt.list(vid))).fetch()
    return " ".join(s.text for s in snippets)


def explain_transcript_error(e: Exception) -> str:
    n = type(e).__name__
    tail = " Paste the transcript manually instead."
    if n in ("IpBlocked", "RequestBlocked"):
        return "YouTube is blocking this server's IP address (common on cloud hosts)." + tail
    if n == "TranscriptsDisabled":
        return "This video has captions turned off." + tail
    if n in ("VideoUnavailable", "VideoUnplayable", "InvalidVideoId"):
        return "YouTube says this video is unavailable, or the link is wrong."
    if n == "AgeRestricted":
        return "This video is age-restricted, so captions can't be fetched." + tail
    if n in ("NoTranscriptFound", "CouldNotRetrieveTranscript"):
        return "No captions were found for this video." + tail
    return f"Couldn't fetch the transcript ({n})." + tail


@app.get("/api/health")
def health():
    return {"ok": True, "has_key": bool(os.environ.get("GROQ_API_KEY", "").strip())}


@app.post("/api/load")
def load(req: LoadReq):
    vid = video_id_from(req.url)
    transcript = re.sub(r"\s+", " ", req.transcript).strip()
    if not transcript:
        if not vid:
            raise HTTPException(400, "That doesn't look like a YouTube link. Paste a full video URL.")
        try:
            transcript = get_transcript(vid)
        except Exception as e:  # noqa: BLE001
            log.warning("transcript fetch failed for %s: %r", vid, e)
            raise HTTPException(422, explain_transcript_error(e))
    try:
        title = fetch_title(vid) if vid else ""
    except Exception:  # noqa: BLE001
        title = ""
    return {
        "video_id": vid,
        "title": title,
        "transcript": transcript,
        "words": len(transcript.split()),
        "truncated": len(transcript) > MAX_CHARS,
        "max_chars": MAX_CHARS,
    }


@app.post("/api/summary")
def summary(req: SummaryReq):
    msgs = [
        {"role": "system", "content": system_prompt(req.transcript, req.language)},
        {"role": "user", "content": STYLES.get(req.style, STYLES["short"])},
    ]
    return {"text": ask(msgs)}


@app.post("/api/chat")
def chat(req: ChatReq):
    history = [m.model_dump() for m in req.messages if m.role in ("user", "assistant")][-20:]
    msgs = [{"role": "system", "content": system_prompt(req.transcript, req.language)}] + history
    return {"text": ask(msgs)}


@app.get("/")
def index():
    return FileResponse(BASE / "index.html")


app.mount("/static", StaticFiles(directory=BASE), name="static")
