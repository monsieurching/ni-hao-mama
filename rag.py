"""
rag.py — Retrieval module for Q&A pairs.
"""

import json
import chromadb
from sentence_transformers import SentenceTransformer
from config import EMBEDDING_MODEL, CHROMA_PATH, COLLECTION_NAME

QA_PAIRS_FILE = "./qa_pairs.json"

_model      = None
_collection = None
_qa_pairs   = None

# Custom chips that bypass semantic search and play specific audio segments directly.
# Add entries here once the transcript is reviewed and timestamps are known.
CUSTOM_CHIPS = []


def _get_model():
    global _model
    if _model is None:
        _model = SentenceTransformer(EMBEDDING_MODEL)
    return _model


def _get_collection():
    global _collection
    if _collection is None:
        client = chromadb.PersistentClient(path=CHROMA_PATH)
        _collection = client.get_collection(COLLECTION_NAME)
    return _collection


def _get_qa_pairs():
    global _qa_pairs
    if _qa_pairs is None:
        with open(QA_PAIRS_FILE) as f:
            _qa_pairs = json.load(f)
    return _qa_pairs


def get_all_questions(popular_order=None) -> list[dict]:
    """Return chip list: custom chips first, then featured, then the rest.

    popular_order: optional dict mapping question label → play count (descending).
    When provided, the 'rest' group is sorted by popularity.
    """
    pairs = _get_qa_pairs()

    custom = [
        {"id": f"custom_{i}", "question": c["question"], "start_fmt": c["start_fmt"],
         "custom_segments": c["custom_segments"]}
        for i, c in enumerate(CUSTOM_CHIPS)
    ]

    def _card(i, p):
        segs = p.get("mama_segments") or [{"start": p["answer_start"], "end": p["answer_end"]}]
        return {
            "id":          i,
            "question":    p.get("visitor_question", p.get("label", p["question"])),
            "label":       p.get("label", ""),
            "start_fmt":   p["start_fmt"],
            "segments":    [[s["start"], s["end"]] for s in segs],
            "answer_text": p.get("answer_text", ""),
        }

    featured = [_card(i, p) for i, p in enumerate(pairs) if p.get("featured")]
    rest     = [_card(i, p) for i, p in enumerate(pairs) if not p.get("featured")]

    if popular_order:
        rest.sort(key=lambda q: -popular_order.get(q["question"], 0))

    return custom + featured + rest


def retrieve(query: str):
    """
    Find the best matching Q&A pair for a user query.
    Returns the full pair dict or None.
    """
    model      = _get_model()
    collection = _get_collection()
    pairs      = _get_qa_pairs()

    query_embedding = model.encode(query).tolist()

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=1,
        include=["metadatas", "distances"],
    )

    if not results["ids"][0]:
        return None

    qa_id = results["metadatas"][0][0]["qa_id"]
    return pairs[qa_id]
