import json
from src.cleaner import clean_all_pages
from src.loaders import save_pages_to_json


if __name__ == "__main__":

    # Load already saved raw pages
    with open("data/raw_pages.json", "r", encoding="utf-8") as f:
        pages = json.load(f)

    # Clean pages
    cleaned_pages = clean_all_pages(pages)

    # Save cleaned output
    save_pages_to_json(cleaned_pages, "data/cleaned_pages.json")