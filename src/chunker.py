# src/chunker.py
import json
import re
import uuid
from typing import List, Dict

CHAR_PER_TOKEN = 4  # approximate, used for char <-> token conversions


def estimate_tokens(text: str) -> int:
    """Rough token estimate (fast): 1 token ~ 4 characters (English)."""
    return max(1, len(text) // CHAR_PER_TOKEN)


def split_into_clauses(text: str) -> List[str]:
    """
    Split text by major clause boundaries: lines starting with (1), (2), ...
    Returns list of clause-block strings. If no clause markers, returns original text in single element.
    """
    # Normalize newlines
    lines = text.split("\n")
    blocks = []
    current = []
    clause_re = re.compile(r"^\s*\(\d+\)")
    for ln in lines:
        if clause_re.match(ln):
            if current:
                blocks.append("\n".join(current).strip())
            current = [ln]
        else:
            current.append(ln)
    if current:
        blocks.append("\n".join(current).strip())
    # Clean empty blocks
    blocks = [b for b in blocks if b and b.strip()]
    return blocks if blocks else [text.strip()]


def split_subclauses(block: str) -> List[str]:
    """
    For blocks that are still long, split by subclause markers (a), (b), etc.
    """
    lines = block.split("\n")
    parts = []
    cur = []
    sub_re = re.compile(r"^\s*\([a-z]\)")
    for ln in lines:
        if sub_re.match(ln):
            if cur:
                parts.append("\n".join(cur).strip())
            cur = [ln]
        else:
            cur.append(ln)
    if cur:
        parts.append("\n".join(cur).strip())
    parts = [p for p in parts if p and p.strip()]
    return parts if parts else [block.strip()]


def split_by_sentences_to_chunks(text: str, max_chars: int, overlap_chars: int) -> List[str]:
    """
    Final fallback: split by sentences into sliding-window style chunks.
    - max_chars: char budget per chunk (approx)
    - overlap_chars: char overlap between chunks
    """
    # naive sentence splitter by punctuation
    sentences = re.split(r'(?<=[\.\?\!])\s+', text)
    chunks = []
    current = ""
    for sent in sentences:
        if not sent.strip():
            continue
        candidate = (current + " " + sent).strip() if current else sent.strip()
        if len(candidate) <= max_chars or not current:
            current = candidate
        else:
            chunks.append(current.strip())
            # start new chunk with overlap from end of current
            if overlap_chars > 0:
                # get overlap as last N chars of current
                ov = current[-overlap_chars:]
                current = (ov + " " + sent).strip()
            else:
                current = sent.strip()
    if current:
        chunks.append(current.strip())
    return chunks


def chunk_article(article: Dict, max_tokens: int = 900, overlap_tokens: int = 150) -> List[Dict]:
    """
    Chunk a single structured article.
    Returns list of chunk dicts (with metadata).
    """
    text = article.get("text", "").strip()
    if not text:
        return []

    max_chars = max_tokens * CHAR_PER_TOKEN
    overlap_chars = overlap_tokens * CHAR_PER_TOKEN

    approx_tokens = estimate_tokens(text)
    if approx_tokens <= max_tokens:
        # single chunk
        return [{
            "chunk_id": str(uuid.uuid4()),
            "article_raw_number": article.get("article_raw_number"),
            "article_number": article.get("article_number"),
            "article_title": article.get("article_title"),
            "part": article.get("part"),
            "chapter": article.get("chapter"),
            "page_start": article.get("page_start"),
            "page_end": article.get("page_end"),
            "chunk_index": 0,
            "total_chunks_for_article": 1,
            "text": text,
            "char_count": len(text),
            "approx_tokens": approx_tokens
        }]

    # try clause split
    clauses = split_into_clauses(text)
    pieces = []
    for clause in clauses:
        if estimate_tokens(clause) <= max_tokens:
            pieces.append(clause)
        else:
            # try subclauses
            subs = split_subclauses(clause)
            for s in subs:
                if estimate_tokens(s) <= max_tokens:
                    pieces.append(s)
                else:
                    # final fallback -> split by sentences windows
                    sentence_chunks = split_by_sentences_to_chunks(s, max_chars, overlap_chars)
                    pieces.extend(sentence_chunks)

    # Now re-window pieces into chunks respecting max_chars and overlap
    final_chunks = []
    current = ""
    current_pages = []  # we don't have granular per-line page here; keep article page range
    for piece in pieces:
        candidate = (current + "\n\n" + piece).strip() if current else piece
        if len(candidate) <= max_chars or not current:
            current = candidate
        else:
            final_chunks.append(current.strip())
            # start new chunk with overlap
            if overlap_chars > 0:
                ov = current[-overlap_chars:]
                current = (ov + "\n\n" + piece).strip()
            else:
                current = piece
    if current:
        final_chunks.append(current.strip())

    # Build chunk dicts with metadata
    out = []
    for idx, ctext in enumerate(final_chunks):
        out.append({
            "chunk_id": str(uuid.uuid4()),
            "article_raw_number": article.get("article_raw_number"),
            "article_number": article.get("article_number"),
            "article_title": article.get("article_title"),
            "part": article.get("part"),
            "chapter": article.get("chapter"),
            "page_start": article.get("page_start"),
            "page_end": article.get("page_end"),
            "chunk_index": idx,
            "total_chunks_for_article": len(final_chunks),
            "text": ctext,
            "char_count": len(ctext),
            "approx_tokens": estimate_tokens(ctext)
        })
    return out


def chunk_all_articles(articles: List[Dict], max_tokens: int = 900, overlap_tokens: int = 150) -> List[Dict]:
    all_chunks = []
    for art in articles:
        chunks = chunk_article(art, max_tokens=max_tokens, overlap_tokens=overlap_tokens)
        # ensure non-empty
        if chunks:
            # set total_chunks_for_article consistently (some functions already set, but ensure)
            total = len(chunks)
            for i, c in enumerate(chunks):
                c["chunk_index"] = i
                c["total_chunks_for_article"] = total
            all_chunks.extend(chunks)
    return all_chunks
