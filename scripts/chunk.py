# scripts/chunk.py
import json
from src.chunker import chunk_all_articles

if __name__ == "__main__":
    with open("data/structured_articles.json", "r", encoding="utf-8") as f:
        arts = json.load(f)

    # parameters (tune here or wire to configs)
    MAX_TOKENS = 900
    OVERLAP_TOKENS = 150

    chunks = chunk_all_articles(arts, max_tokens=MAX_TOKENS, overlap_tokens=OVERLAP_TOKENS)

    print(f"Created {len(chunks)} chunks from {len(arts)} articles (tokens_per_chunk~{MAX_TOKENS}, overlap_tokens={OVERLAP_TOKENS})")

    with open("data/chunks.json", "w", encoding="utf-8") as fo:
        json.dump(chunks, fo, ensure_ascii=False, indent=2)

    # Quick stats
    lens = [c["approx_tokens"] for c in chunks]
    if lens:
        avg = sum(lens)/len(lens)
        print(f"avg approx tokens per chunk: {avg:.1f}")
        print("Top 5 longest chunks (tokens):", sorted(lens, reverse=True)[:5])
