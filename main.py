import json
import logging
import os
import re
import urllib.error
import urllib.parse
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
    cached_id: str = ""


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


CACHE_FILE = Path(__file__).parent / "cache" / "transcripts.json"


def load_cache() -> dict:
    try:
        return json.loads(CACHE_FILE.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        return {}


FEATURED = load_cache()  # ready-made videos shipped with the site (see prefetch.py)
SEEN: dict = {}          # transcripts fetched while the server runs (saves API credits)


def get_transcript(vid: str) -> str:
    """Direct fetch with youtube-transcript-api. English/Hindi first, then any track."""
    kwargs = {}
    proxy = os.environ.get("YT_PROXY_URL", "").strip()
    if proxy:
        try:
            from youtube_transcript_api.proxies import GenericProxyConfig

            kwargs["proxy_config"] = GenericProxyConfig(http_url=proxy, https_url=proxy)
        except Exception:  # noqa: BLE001
            log.warning("YT_PROXY_URL is set but proxy support could not be loaded")
    yt = YouTubeTranscriptApi(**kwargs)
    try:
        snippets = yt.fetch(vid, languages=["en", "en-US", "en-GB", "hi"])
    except Exception as e:  # noqa: BLE001
        if type(e).__name__ != "NoTranscriptFound":
            raise
        snippets = next(iter(yt.list(vid))).fetch()
    return " ".join(s.text for s in snippets)


def supadata_transcript(vid: str):
    """Fallback that works from cloud hosts. Free key: https://dash.supadata.ai (100/month)."""
    key = os.environ.get("SUPADATA_API_KEY", "").strip().strip("\"'").strip()
    if not key:
        return None
    qs = urllib.parse.urlencode(
        {"url": f"https://www.youtube.com/watch?v={vid}", "text": "true", "mode": "native"}
    )
    req = urllib.request.Request(
        "https://api.supadata.ai/v1/transcript?" + qs, headers={"x-api-key": key}
    )
    with urllib.request.urlopen(req, timeout=40) as r:
        data = json.load(r)
    content = data.get("content")
    if isinstance(content, list):
        content = " ".join(c.get("text", "") for c in content)
    return (content or "").strip() or None


def explain_transcript_error(e) -> str:
    n = type(e).__name__ if e else ""
    tail = " Pick a ready example, paste the transcript, or upload a caption file."
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
    return f"Couldn't fetch the transcript ({n or 'unknown error'})." + tail


def payload(vid, title, transcript):
    return {
        "video_id": vid,
        "title": title,
        "transcript": transcript,
        "words": len(transcript.split()),
        "truncated": len(transcript) > MAX_CHARS,
        "max_chars": MAX_CHARS,
    }


@app.get("/api/health")
def health():
    return {
        "ok": True,
        "has_key": bool(os.environ.get("GROQ_API_KEY", "").strip()),
        "has_supadata": bool(os.environ.get("SUPADATA_API_KEY", "").strip()),
        "featured": len(FEATURED),
    }


@app.get("/api/featured")
def featured():
    return [
        {
            "id": k,
            "title": v.get("title") or k,
            "words": len(v.get("transcript", "").split()),
            "video_id": v.get("video_id"),
        }
        for k, v in FEATURED.items()
    ]


@app.post("/api/load")
def load(req: LoadReq):
    # 1) a ready example chosen from the site
    if req.cached_id:
        item = FEATURED.get(req.cached_id)
        if not item:
            raise HTTPException(404, "That ready example isn't available.")
        return payload(item.get("video_id"), item.get("title", ""), item["transcript"])

    vid = video_id_from(req.url)
    transcript = re.sub(r"\s+", " ", req.transcript).strip()

    # 2) a transcript pasted or uploaded by the visitor
    if transcript:
        title = ""
        try:
            title = fetch_title(vid) if vid else ""
        except Exception:  # noqa: BLE001
            pass
        return payload(vid, title, transcript)

    if not vid:
        raise HTTPException(400, "That doesn't look like a YouTube link. Paste a full video URL.")

    # 3) shipped or previously fetched copies
    for store in (FEATURED, SEEN):
        if vid in store:
            item = store[vid]
            return payload(vid, item.get("title", ""), item["transcript"])

    # 4) live fetch, then the Supadata fallback
    direct_error = None
    try:
        transcript = get_transcript(vid)
    except Exception as e:  # noqa: BLE001
        direct_error = e
        log.warning("direct transcript fetch failed for %s: %r", vid, e)
    if not transcript:
        try:
            transcript = supadata_transcript(vid) or ""
        except urllib.error.HTTPError as e:
            log.warning("Supadata HTTP %s for %s", e.code, vid)
        except Exception as e:  # noqa: BLE001
            log.warning("Supadata failed for %s: %r", vid, e)
    if not transcript:
        raise HTTPException(422, explain_transcript_error(direct_error))

    title = fetch_title(vid)
    SEEN[vid] = {"video_id": vid, "title": title, "transcript": transcript}
    return payload(vid, title, transcript)


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


if __name__ == "__main__":
    # Lets you start the site with:  python main.py
    import threading
    import webbrowser

    import uvicorn

    threading.Timer(2.0, lambda: webbrowser.open("http://127.0.0.1:8000")).start()
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
