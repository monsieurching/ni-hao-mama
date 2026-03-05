"""
ingest.py — One-time M4A → ChromaDB ingestion pipeline.

Uses OpenAI Whisper (large-v3) to transcribe the audio recording,
groups segments into ~300-word chunks, and stores them in ChromaDB.

Requires ffmpeg: brew install ffmpeg

Run once (or re-run safely; chunk IDs are deterministic):
    python ingest.py
"""

import os
import whisper
import chromadb
from sentence_transformers import SentenceTransformer
from config import AUDIO_DIR, AUDIO_FILES, EMBEDDING_MODEL, CHROMA_PATH, COLLECTION_NAME, CHUNK_WORDS


def fmt_timestamp(seconds: float) -> str:
    """Convert seconds to MM:SS string."""
    m = int(seconds) // 60
    s = int(seconds) % 60
    return f"{m:02d}:{s:02d}"


def chunk_segments(segments: list[dict], chunk_words: int = CHUNK_WORDS) -> list[dict]:
    """
    Group Whisper segments into ~chunk_words-word chunks.

    Each chunk keeps the start time of its first segment and
    the end time of its last segment.
    """
    chunks = []
    current_words = []
    current_start = None
    current_end = None

    for seg in segments:
        text = seg["text"].strip()
        if not text:
            continue

        words = text.split()
        if current_start is None:
            current_start = seg["start"]

        current_words.extend(words)
        current_end = seg["end"]

        if len(current_words) >= chunk_words:
            chunks.append({
                "text":  " ".join(current_words),
                "start": current_start,
                "end":   current_end,
            })
            current_words = []
            current_start = None
            current_end = None

    # Flush remaining words
    if current_words and current_start is not None:
        chunks.append({
            "text":  " ".join(current_words),
            "start": current_start,
            "end":   current_end,
        })

    return chunks


TRANSCRIPT_CACHE = "./transcript.json"


def load_or_transcribe(audio_path: str) -> list[dict]:
    """Load transcript from cache if available, otherwise run Whisper."""
    if os.path.exists(TRANSCRIPT_CACHE):
        print(f"  Loading cached transcript from {TRANSCRIPT_CACHE}…")
        with open(TRANSCRIPT_CACHE) as f:
            import json
            segments = json.load(f)
        print(f"  Loaded {len(segments)} segments from cache.")
        return segments

    print(f"  Transcribing with Whisper large-v3 (this takes 20–40 min)…")
    model = whisper.load_model("large-v3")
    result = model.transcribe(
        audio_path,
        language=None,
        verbose=True,
        word_timestamps=False,
    )
    segments = result.get("segments", [])
    with open(TRANSCRIPT_CACHE, "w") as f:
        import json
        json.dump(segments, f)
    print(f"  Transcription complete: {len(segments)} segments. Saved to {TRANSCRIPT_CACHE}.")
    return segments


def main():
    print(f"Loading embedding model '{EMBEDDING_MODEL}'…")
    embed_model = SentenceTransformer(EMBEDDING_MODEL)

    print(f"Connecting to ChromaDB at '{CHROMA_PATH}'…")
    client = chromadb.PersistentClient(path=CHROMA_PATH)
    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )

    total_added = 0
    total_skipped = 0

    for audio in AUDIO_FILES:
        audio_path = os.path.join(AUDIO_DIR, audio["file"])
        if not os.path.exists(audio_path):
            print(f"  [WARN] File not found: {audio_path} — skipping.")
            continue

        print(f"\nProcessing: {audio['name']} ({audio['file']})")
        segments = load_or_transcribe(audio_path)

        chunks = chunk_segments(segments)
        print(f"  Grouped into {len(chunks)} chunks (~{CHUNK_WORDS} words each).")

        for chunk in chunks:
            start_fmt = fmt_timestamp(chunk["start"])
            # Deterministic ID: Baba__t{MM}:{SS}
            chunk_id = f"{audio['name']}__t{start_fmt}"

            existing = collection.get(ids=[chunk_id])
            if existing["ids"]:
                total_skipped += 1
                continue

            embedding = embed_model.encode(chunk["text"]).tolist()
            collection.add(
                ids=[chunk_id],
                embeddings=[embedding],
                documents=[chunk["text"]],
                metadatas=[{
                    "speaker":    audio["name"],
                    "start":      chunk["start"],
                    "end":        chunk["end"],
                    "start_fmt":  start_fmt,
                    "file":       audio["file"],
                }],
            )
            total_added += 1

        print(f"  Done: {audio['name']}")

    print(f"\nIngestion complete.")
    print(f"  Added:   {total_added} chunks")
    print(f"  Skipped: {total_skipped} chunks (already in DB)")
    print(f"  Total in collection: {collection.count()}")


if __name__ == "__main__":
    main()
