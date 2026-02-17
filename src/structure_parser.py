import re
from typing import List, Dict, Tuple


def _normalize(text: str) -> str:
    """Normalize minor formatting artifacts so patterns are simpler."""
    text = text.replace("\u00A0", " ")
    text = text.replace("\u2014", "-").replace("\u2013", "-")
    # remove markdown bold/italic artifacts left by loader (e.g., **1.**, **14.**)
    text = re.sub(r"\*+", "", text)
    # collapse multiple spaces
    text = re.sub(r"[ \t]{2,}", " ", text)
    return text


def _lines_with_page(pages: List[Dict]) -> List[Tuple[str, int]]:
    out = []
    for p in pages:
        page_no = p.get("page_number")
        text = p.get("text", "")
        text = _normalize(text)
        lines = text.split("\n")
        for ln in lines:
            out.append((ln.rstrip(), page_no))
    return out


def _next_nonempty(lines_with_page: List[Tuple[str, int]], start_idx: int, max_ahead: int = 6):
    """Return (line, page, idx) of next non-empty line within max_ahead, or (None,None,None)."""
    N = len(lines_with_page)
    i = start_idx + 1
    steps = 0
    while i < N and steps < max_ahead:
        ln, pg = lines_with_page[i]
        if ln.strip():
            return ln, pg, i
        i += 1
        steps += 1
    return None, None, None


def parse_structure(pages):

    lines_with_page = _lines_with_page(pages)

    part_pattern = re.compile(r"^\s*PART\b", re.IGNORECASE)
    header_pattern = re.compile(r"^\s*(\d+[A-Za-z]*)\.\s*(.*)$")

    structured = []

    current_part = None
    current_article_key = None
    current_article_num = None
    current_article_title = None
    buffer_lines = []
    buffer_pages = []

    last_article_num = 0

    def flush_article():
        nonlocal current_article_key, current_article_num, current_article_title
        nonlocal buffer_lines, buffer_pages, last_article_num

        if current_article_key is None:
            return

        structured.append({
            "article_raw_number": current_article_key,
            "article_number": current_article_num,
            "article_title": current_article_title,
            "part": current_part,
            "page_start": buffer_pages[0] if buffer_pages else None,
            "page_end": buffer_pages[-1] if buffer_pages else None,
            "text": "\n".join(buffer_lines).strip()
        })

        last_article_num = current_article_num if current_article_num else last_article_num

        current_article_key = None
        current_article_num = None
        current_article_title = None
        buffer_lines = []
        buffer_pages = []

    for line, pg in lines_with_page:

        stripped = line.strip()

        if not stripped:
            continue

        if part_pattern.match(stripped):
            current_part = stripped
            continue

        m = header_pattern.match(stripped)
        if m and current_part is not None:

            raw_num = m.group(1)
            remainder = m.group(2).strip()

            try:
                num_int = int(re.match(r"^(\d+)", raw_num).group(1))
            except:
                num_int = None

            # Strictly increasing sequence rule
            if num_int and num_int > last_article_num:

                flush_article()

                current_article_key = raw_num
                current_article_num = num_int
                current_article_title = remainder if remainder else None

                buffer_pages.append(pg)

                continue

        if current_article_key is not None:
            buffer_lines.append(stripped)
            buffer_pages.append(pg)

    flush_article()

    return structured
