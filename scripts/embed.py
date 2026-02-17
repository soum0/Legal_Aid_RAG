# scripts/embed.py

import json
from langchain.schema import Document
from src.vector_store import VectorStore


if __name__ == "__main__":

    # Load chunks
    with open("data/chunks.json", "r", encoding="utf-8") as f:
        chunks = json.load(f)

    print(f"Loaded {len(chunks)} chunks.")

    # Convert chunks → LangChain Documents
    documents = []

    for chunk in chunks:
        documents.append(
            Document(
                page_content=f"Article {chunk.get('article_number')}\n\n{chunk['text']}"
,
                metadata={
                    "article_number": chunk.get("article_number"),
                    "start_page": chunk.get("start_page"),
                    "end_page": chunk.get("end_page")
                }
            )
        )

    print("Converted chunks to LangChain Documents.")

    # Initialize Vector Store
    vector_store = VectorStore()

    # Add documents (embedding happens automatically)
    vector_store.add_documents(documents)

    print("Documents embedded and stored successfully.")
