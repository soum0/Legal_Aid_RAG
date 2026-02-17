from langchain_community.document_loaders import PyPDFLoader
from langchain_pymupdf4llm import PyMuPDF4LLMLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# loader = PyPDFLoader('/Users/soumsingh/Desktop/RAG_PROJEXT/constitution.pdf')
# docs = loader.load()

loader = PyMuPDF4LLMLoader('/Users/soumsingh/Desktop/RAG_PROJEXT/constitution.pdf')

docs = loader.load()
# print(docs.content)

print(len(docs))

print(docs[0].page_content)
print(docs[1].metadata)

docs = loader.lazy_load()

# c=0
# for documents in docs:
    
#     print(documents.metadata,'\n')
#     c+=1
#     if c>10:
#         break


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
