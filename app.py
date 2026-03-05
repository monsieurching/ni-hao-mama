"""
app.py — Flask server: /questions, /chat (JSON match), /clip (audio stream), /admin.
No Claude. Responses are Baba's actual recorded words.
"""

import json
import os
import shutil
import subprocess
import threading
import time
from flask import Flask, request, Response, render_template
from dotenv import load_dotenv
from rag import retrieve, get_all_questions
from logger import log_question
from config import AUDIO_DIR, AUDIO_FILES

load_dotenv()

app = Flask(__name__)

_ffmpeg_candidates = [
    shutil.which("ffmpeg"),
    "/opt/homebrew/bin/ffmpeg",  # macOS Homebrew
    "/usr/bin/ffmpeg",           # Linux
    "/usr/local/bin/ffmpeg",
]
FFMPEG = next((p for p in _ffmpeg_candidates if p and os.path.exists(p)), "ffmpeg")
AUDIO_PATH = os.path.join(AUDIO_DIR, AUDIO_FILES[0]["file"])

# ── Popularity cache ─────────────────────────────────────────────────
_pop_cache = {"counts": None, "ts": 0.0}
_POP_TTL   = 300  # seconds


def _get_popular_counts():
    """Return {chip_label: play_count} from Supabase, cached for 5 min."""
    now = time.time()
    if _pop_cache["counts"] is not None and now - _pop_cache["ts"] < _POP_TTL:
        return _pop_cache["counts"]
    try:
        from logger import _get_client
        client = _get_client()
        if client is None:
            return {}
        rows = client.table("questions").select("matched_question").execute()
        counts = {}
        for r in rows.data:
            q = r.get("matched_question", "").strip()
            if q:
                counts[q] = counts.get(q, 0) + 1
        _pop_cache["counts"] = counts
        _pop_cache["ts"] = now
        return counts
    except Exception:
        return {}


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/questions")
def questions():
    popular = _get_popular_counts()
    qs = get_all_questions(popular_order=popular)
    return Response(json.dumps(qs, ensure_ascii=False), mimetype="application/json")


@app.route("/chat", methods=["POST"])
def chat():
    data  = request.get_json(force=True)
    query = data.get("message", "").strip()

    if not query:
        return Response(json.dumps({"error": "empty"}), mimetype="application/json")

    pair = retrieve(query)
    if not pair:
        return Response(
            json.dumps({"error": "I don't think we talked about that."}),
            mimetype="application/json",
        )

    matched_label = pair.get("visitor_question", pair.get("label", pair.get("question", "")))

    threading.Thread(
        target=log_question,
        args=(query, pair.get("answer_text", "")[:300], matched_label),
        daemon=True,
    ).start()

    mama_segs = pair.get("mama_segments", [
        {"start": pair["answer_start"], "end": pair["answer_end"]}
    ])
    mama_dur = sum(s["end"] - s["start"] for s in mama_segs)

    return Response(
        json.dumps({
            "matched_question": pair["question"],
            "answer_text":      pair["answer_text"],
            "clip_start":       pair["answer_start"],
            "clip_end":         pair["answer_end"],
            "baba_segments":    [[s["start"], s["end"]] for s in mama_segs],
            "baba_duration":    round(mama_dur),
            "start_fmt":        pair["start_fmt"],
        }, ensure_ascii=False),
        mimetype="application/json",
    )


@app.route("/clip")
def clip():
    """
    Stream an audio clip. Accepts either:
      ?start=X&end=Y          — simple range (fallback)
      ?segments=[[s,e],[s,e]] — stitch specific segments (Baba-only)
    """
    segments_param = request.args.get("segments")

    if segments_param:
        try:
            segs = json.loads(segments_param)
        except Exception:
            return Response(status=400)
    else:
        try:
            start = float(request.args.get("start", 0))
            end   = float(request.args.get("end", start + 60))
            segs  = [[start, end]]
        except (ValueError, TypeError):
            return Response(status=400)

    if len(segs) == 1:
        cmd = [
            FFMPEG,
            "-ss", str(segs[0][0]),
            "-to", str(segs[0][1]),
            "-i",  AUDIO_PATH,
            "-ac", "1", "-b:a", "64k",
            "-f",  "mp3", "pipe:1",
        ]
    else:
        filter_parts = []
        for i, (s, e) in enumerate(segs):
            filter_parts.append(
                f"[0:a]atrim=start={s}:end={e},asetpts=PTS-STARTPTS[s{i}]"
            )
        inputs = "".join(f"[s{i}]" for i in range(len(segs)))
        filter_parts.append(f"{inputs}concat=n={len(segs)}:v=0:a=1[out]")
        filter_complex = "; ".join(filter_parts)

        cmd = [
            FFMPEG,
            "-i", AUDIO_PATH,
            "-filter_complex", filter_complex,
            "-map", "[out]",
            "-ac", "1", "-b:a", "64k",
            "-f", "mp3", "pipe:1",
        ]

    def generate():
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
        try:
            while True:
                chunk = proc.stdout.read(8192)
                if not chunk:
                    break
                yield chunk
        finally:
            proc.terminate()
            proc.wait()

    return Response(
        generate(),
        mimetype="audio/mpeg",
        headers={"Cache-Control": "no-cache"},
    )


@app.route("/admin")
def admin():
    token       = request.args.get("token", "")
    admin_token = os.environ.get("ADMIN_TOKEN", "")
    if not admin_token or token != admin_token:
        return Response("Unauthorized", status=401)
    return render_template("admin.html")


@app.route("/admin/data")
def admin_data():
    token       = request.args.get("token", "")
    admin_token = os.environ.get("ADMIN_TOKEN", "")
    if not admin_token or token != admin_token:
        return Response("Unauthorized", status=401)
    try:
        from logger import _get_client
        client = _get_client()
        if client is None:
            url_set = bool(os.environ.get("SUPABASE_URL"))
            key_set = bool(os.environ.get("SUPABASE_KEY"))
            return Response(json.dumps({"error": f"Supabase not configured (URL={'set' if url_set else 'MISSING'}, KEY={'set' if key_set else 'MISSING'})"}), mimetype="application/json")
        rows = (
            client.table("questions")
            .select("*")
            .order("created_at", desc=True)
            .limit(500)
            .execute()
        )
        return Response(
            json.dumps({"total": len(rows.data), "rows": rows.data}),
            mimetype="application/json",
        )
    except Exception as e:
        return Response(json.dumps({"error": str(e)}), mimetype="application/json")


@app.route("/admin/env")
def admin_env():
    token       = request.args.get("token", "")
    admin_token = os.environ.get("ADMIN_TOKEN", "")
    if not admin_token or token != admin_token:
        return Response("Unauthorized", status=401)
    # Show all env var NAMES that contain "SUPA" or "DB" (not values)
    keys = [k for k in os.environ.keys() if any(x in k.upper() for x in ["SUPA", "DB", "RAILWAY", "PORT"])]
    return Response(json.dumps(sorted(keys)), mimetype="application/json")


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    app.run(debug=False, threaded=True, host="0.0.0.0", port=port)
