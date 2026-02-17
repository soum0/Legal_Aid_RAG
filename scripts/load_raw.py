from src.loaders import load_pages, save_pages_to_json

if __name__ == "__main__":

    pdf_path = "/Users/soumsingh/Desktop/RAG_PROJEXT/data/constitution.pdf"
    output_path = "data/raw_pages.json"

    pages = load_pages(pdf_path)
    save_pages_to_json(pages, output_path)
