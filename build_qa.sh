#!/bin/bash
# build_qa.sh — Run Q&A extraction pipeline after ingest.py has completed.
# Requires transcript.json and chroma_db/ to already exist (from ingest.py).

set -e
cd "$(dirname "$0")"
source venv/bin/activate

echo "=== Step 1: extract_qa.py ==="
python3 extract_qa.py

echo ""
echo "=== Step 2: generate_labels.py ==="
python3 generate_labels.py

echo ""
echo "=== Step 3: refine_qa.py ==="
python3 refine_qa.py

echo ""
echo "=== Pipeline complete! ==="
echo "Review qa_pairs.json, then git add chroma_db/ qa_pairs.json && git commit && git push"
