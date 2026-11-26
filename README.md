# Queryflow

Paste a YouTube link and watch it run as a workflow: get the transcript, summarize it, then chat with the video. Answers come only from what was said in the video.

**Live site:** https://queryflow-hbhu.onrender.com

## Features
- Fetches English/Hindi captions, or any caption track the video has
- Summaries in three styles (short, detailed, simple) and three languages (English, Hindi, Hinglish)
- Chat grounded in the transcript
- Paste a transcript manually if YouTube blocks the fetch
- Run log showing each step and how long it took

## Tech stack
Python, FastAPI, plain HTML/CSS/JS, Groq API (free tier), youtube-transcript-api

## Project layout
```
queryflow/
  main.py            backend (API + serves the site)
  static/index.html  the whole frontend
  requirements.txt
  render.yaml        optional Render config
```

## Run locally (Windows cmd)
```bat
pip install -r requirements.txt
set GROQ_API_KEY=your-key-here
python main.py
```
Opens http://127.0.0.1:8000. Do not put quotes around the key. Get a free key at https://console.groq.com.

On Mac or Linux use `export GROQ_API_KEY="your-key-here"` instead of `set`.

## Check the AI connection
Open http://127.0.0.1:8000/api/test. It should show `"reply":"ok"`.

## Deploy free on Render
1. Push this folder to GitHub (never commit your API key).
2. Render -> New -> Web Service -> select the repo.
3. Build command: `pip install -r requirements.txt`
4. Start command: `uvicorn main:app --host 0.0.0.0 --port $PORT`
5. Add the environment variable `GROQ_API_KEY`.

## Notes
- The AI reads the first 20,000 characters of a transcript to stay inside free-tier limits.
- YouTube may block transcript requests from cloud servers. Use the paste-transcript option if that happens.
- Nothing is stored on the server.
