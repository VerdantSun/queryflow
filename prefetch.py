"""Save transcripts into cache/transcripts.json so the deployed site can offer them
as "ready examples" even when YouTube blocks the server.

Run this on YOUR computer (home internet is not blocked):
    python prefetch.py "https://www.youtube.com/watch?v=AAAAAAAAAAA" "https://youtu.be/BBBBBBBBBBB"
Then upload cache/transcripts.json to GitHub.
"""
import json
import sys

from main import CACHE_FILE, fetch_title, get_transcript, load_cache, video_id_from


def run(urls):
    cache = load_cache()
    for url in urls:
        vid = video_id_from(url)
        if not vid:
            print("Skipped (not a YouTube link):", url)
            continue
        try:
            text = get_transcript(vid)
        except Exception as e:  # noqa: BLE001
            print(f"FAILED {url}: {type(e).__name__}")
            continue
        cache[vid] = {"video_id": vid, "title": fetch_title(vid), "transcript": text}
        print(f"Saved {vid}: {len(text.split()):,} words")
    CACHE_FILE.parent.mkdir(exist_ok=True)
    CACHE_FILE.write_text(json.dumps(cache, ensure_ascii=False, indent=2), encoding="utf-8")
    print("Done. Upload", CACHE_FILE, "to GitHub.")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__)
    else:
        run(sys.argv[1:])
