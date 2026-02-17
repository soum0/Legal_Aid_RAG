import json
from src.structure_parser import parse_structure


if __name__ == "__main__":

    with open("data/cleaned_pages.json", "r", encoding="utf-8") as f:
        pages = json.load(f)

    structured_articles = parse_structure(pages)

    with open("data/structured_articles.json", "w", encoding="utf-8") as f:
        json.dump(structured_articles, f, ensure_ascii=False, indent=2)

    print(f"Parsed {len(structured_articles)} articles.")
