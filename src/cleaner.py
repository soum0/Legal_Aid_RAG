import re 
from collections import Counter

def detect_repeated_header(pages):

    first_line = []

    for page in pages:
        lines = page['text'].split('\n')
        if len(lines) >0:
            first_line.append(lines[0].strip())

        counter = Counter(first_line)

        repeated = {
            line for line, count in counter.items()
            if count > 0.7*len(pages)
        }

        return repeated
    
def clean_page_text(text, repeated_headers):
    
    # Remove repeated header lines
    lines = text.split("\n")
    lines = [line for line in lines if line.strip() not in repeated_headers]

    text = "\n".join(lines)

    # Remove standalone page numbers
    text = re.sub(r"^\s*\d+\s*$", "", text, flags=re.MULTILINE)

    # Fix hyphenated line breaks
    text = re.sub(r"-\n", "", text)

    # Normalize multiple spaces
    text = re.sub(r"[ ]{2,}", " ", text)

    # Normalize multiple newlines
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip()



def clean_all_pages(pages):

    repeated_headers = detect_repeated_header(pages)

    cleaned_pages = []

    for page in pages:
        cleaned_text = clean_page_text(page["text"], repeated_headers)

        page["text"] = cleaned_text
        page["char_count"] = len(cleaned_text)

        cleaned_pages.append(page)

    return cleaned_pages

