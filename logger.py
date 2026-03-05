"""
logger.py — Log questions and responses to Supabase.
Silently no-ops if SUPABASE_URL / SUPABASE_KEY are not configured.
"""

import os
import re
from dotenv import load_dotenv

_client = None


def _get_client():
    global _client
    url = os.environ.get("SUPABASE_URL", "")
    key = os.environ.get("SUPABASE_KEY", "")
    if not url or not key:
        return None
    if _client is None:
        from supabase import create_client
        _client = create_client(url, key)
    return _client


def _extract_timestamps(response: str) -> str:
    """Pull out any cited timestamps from the response."""
    matches = re.findall(r'\(Baba,\s*(\d{2}:\d{2})\)', response)
    return ", ".join(sorted(set(matches))) if matches else ""


def log_question(question: str, response: str, matched_question: str = ""):
    """Insert one row into the questions table. Fails silently."""
    try:
        client = _get_client()
        if client is None:
            return
        client.table("questions").insert({
            "question":          question[:500],
            "response_preview":  response[:300],
            "timestamps_cited":  _extract_timestamps(response),
            "response_length":   len(response),
            "matched_question":  matched_question[:500],
        }).execute()
    except Exception:
        pass  # Never crash the app over logging
