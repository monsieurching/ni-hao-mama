"""
generate_labels.py — One-time script to generate clean English question labels
using Claude. Updates qa_pairs.json with a 'label' field for each pair.

Run once:
    python generate_labels.py
"""

import json
import os
from anthropic import Anthropic
from dotenv import load_dotenv

load_dotenv()

QA_PAIRS_FILE = "./qa_pairs.json"


def generate_labels(pairs: list[dict]) -> list[str]:
    client = Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])

    questions_block = "\n".join(
        f"[{i}] {p['question'][:200]}" for i, p in enumerate(pairs)
    )

    prompt = f"""Below are {len(pairs)} questions from a recorded interview with a Chinese immigrant mother. The questions are in mixed Chinese/Cantonese/English from a Whisper auto-transcription, so some are garbled.

For each question, write a SHORT, CLEAN English label (5-10 words max) that captures what it's actually asking about. If the question is unclear or garbled, infer from context. Return ONLY a JSON array of {len(pairs)} strings, one label per question, in order.

Questions:
{questions_block}

Return only valid JSON like: ["label 0", "label 1", ...]"""

    message = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=2000,
        messages=[{"role": "user", "content": prompt}],
    )

    response_text = message.content[0].text.strip()

    # Extract JSON array from response
    start = response_text.find("[")
    end   = response_text.rfind("]") + 1
    labels = json.loads(response_text[start:end])

    return labels


def main():
    with open(QA_PAIRS_FILE) as f:
        pairs = json.load(f)

    print(f"Generating clean labels for {len(pairs)} questions…")
    labels = generate_labels(pairs)

    for i, (pair, label) in enumerate(zip(pairs, labels)):
        pair["label"] = label
        print(f"  [{i:02d}] {label}")

    with open(QA_PAIRS_FILE, "w") as f:
        json.dump(pairs, f, ensure_ascii=False, indent=2)

    print(f"\nDone. Labels saved to {QA_PAIRS_FILE}")


if __name__ == "__main__":
    main()
