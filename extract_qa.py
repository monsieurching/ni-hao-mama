"""
extract_qa.py — Parse transcript.json into Q&A pairs and rebuild ChromaDB.

Uses '?' as the question marker. Merges consecutive question segments.
Answer = all segments until the next question.

Run once (fast — no Whisper):
    python extract_qa.py
"""

import json
import os
import shutil
import chromadb
from sentence_transformers import SentenceTransformer
from config import EMBEDDING_MODEL, CHROMA_PATH, COLLECTION_NAME

QA_PAIRS_FILE = "./qa_pairs.json"


def fmt_timestamp(seconds: float) -> str:
    m = int(seconds) // 60
    s = int(seconds) % 60
    return f"{m:02d}:{s:02d}"


def extract_qa_pairs(segments: list[dict]) -> list[dict]:
    """
    Split transcript into Q&A pairs.
    A segment containing '?' starts a question block.
    Consecutive question segments (gap < 8s) are merged.
    Everything after until the next question is the answer.
    """
    tagged = []
    for seg in segments:
        text = seg["text"].strip()
        if text:
            tagged.append({
                "start": seg["start"],
                "end":   seg["end"],
                "text":  text,
                "is_q":  "?" in text,
            })

    pairs = []
    i = 0

    while i < len(tagged):
        if not tagged[i]["is_q"]:
            i += 1
            continue

        # Collect consecutive question segments
        q_segs = [tagged[i]]
        i += 1
        while (
            i < len(tagged)
            and tagged[i]["is_q"]
            and tagged[i]["start"] - q_segs[-1]["end"] < 8
        ):
            q_segs.append(tagged[i])
            i += 1

        q_text = " ".join(s["text"] for s in q_segs).strip()

        # Collect answer segments until next question
        a_segs = []
        while i < len(tagged) and not tagged[i]["is_q"]:
            a_segs.append(tagged[i])
            i += 1

        # Skip trivially short answers (< 3 segments)
        if len(a_segs) >= 3:
            pairs.append({
                "question":     q_text,
                "answer_text":  " ".join(s["text"] for s in a_segs).strip(),
                "answer_start": a_segs[0]["start"],
                "answer_end":   a_segs[-1]["end"],
                "start_fmt":    fmt_timestamp(a_segs[0]["start"]),
            })

    return pairs


def main():
    if not os.path.exists("transcript.json"):
        print("[ERROR] transcript.json not found. Run ingest.py first.")
        return

    print("Loading transcript.json…")
    with open("transcript.json") as f:
        segments = json.load(f)
    print(f"  {len(segments)} segments loaded.")

    print("Extracting Q&A pairs…")
    pairs = extract_qa_pairs(segments)
    print(f"  Found {len(pairs)} Q&A pairs.")

    with open(QA_PAIRS_FILE, "w") as f:
        json.dump(pairs, f, ensure_ascii=False, indent=2)
    print(f"  Saved to {QA_PAIRS_FILE}")

    # Print question list for review
    print("\nQuestions found:")
    for i, p in enumerate(pairs):
        duration = int(p["answer_end"] - p["answer_start"])
        print(f"  [{i:02d}] ({p['start_fmt']}, {duration}s) {p['question'][:80]}")

    # Rebuild ChromaDB
    print(f"\nLoading embedding model '{EMBEDDING_MODEL}'…")
    model = SentenceTransformer(EMBEDDING_MODEL)

    print(f"Rebuilding ChromaDB at '{CHROMA_PATH}'…")
    if os.path.exists(CHROMA_PATH):
        shutil.rmtree(CHROMA_PATH)

    client = chromadb.PersistentClient(path=CHROMA_PATH)
    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )

    for i, pair in enumerate(pairs):
        # Embed question text for semantic matching against user queries
        embedding = model.encode(pair["question"]).tolist()
        collection.add(
            ids=[f"qa_{i}"],
            embeddings=[embedding],
            documents=[pair["question"]],
            metadatas=[{
                "qa_id":        i,
                "question":     pair["question"][:500],
                "answer_start": pair["answer_start"],
                "answer_end":   pair["answer_end"],
                "start_fmt":    pair["start_fmt"],
            }],
        )

    print(f"ChromaDB rebuilt with {len(pairs)} entries.")
    print("\nDone. Run 'python app.py' to start.")


if __name__ == "__main__":
    main()
