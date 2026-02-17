from langchain_pymupdf4llm import PyMuPDF4LLMLoader
from pathlib import Path
import json

def load_pages(file_path:str) -> list:
    file_path = Path(file_path)

    if not file_path.exists():
        return FileNotFoundError("file not found")
    
    loader = PyMuPDF4LLMLoader(str(file_path))

    docs = loader.load()

    print(f'Total Pages detected : {len(docs)}')

    pages = []

    for i, doc in enumerate(docs):
        text = doc.page_content.strip()

        text = text.replace("\r\n", "\n")

        if len(text)< 15:
            continue

        page_data = {
            "page_number": doc.metadata.get("page", i) + 1,
            "text": text,
            "source_file": file_path.name,
            "total_pages": len(docs),
            "char_count": len(text),
        }

        pages.append(page_data)

        print(f"Loaded page {page_data['page_number']} | chars: {len(text)}")

    print(f"\nValid pages stored: {len(pages)}")

    return pages



def save_pages_to_json(pages: list, output_path: str):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(pages, f, ensure_ascii=False, indent=2)

    print(f"Saved raw pages to: {output_path}")