# scripts/evaluate_retriever.py

import json
from src.vector_store import VectorStore


def extract_articles_from_docs(docs):
    """
    Extract article numbers from retrieved documents.
    """
    articles = []

    for doc in docs:
        article = doc.metadata.get("article_number")
        if article:
            articles.append(str(article))

    return articles


def precision_at_k(retrieved, relevant, k):
    retrieved_k = retrieved[:k]
    hits = sum(1 for r in retrieved_k if r in relevant)
    return hits / k if k > 0 else 0


def recall_at_k(retrieved, relevant, k):
    retrieved_k = retrieved[:k]
    hits = sum(1 for r in retrieved_k if r in relevant)
    return hits / len(relevant) if len(relevant) > 0 else 0


if __name__ == "__main__":

    # Load evaluation dataset
    with open("data/eval_set.json", "r", encoding="utf-8") as f:
        eval_data = json.load(f)

    # Load vector store (must match embedding script)
    vector_store = VectorStore(
        persist_directory="data/chroma_db",
        collection_name="constitution"
    )

    retriever = vector_store.get_retriever(k=8, fetch_k=24)

    K = 8
    total_precision = 0
    total_recall = 0

    print("\n===== RETRIEVAL EVALUATION STARTED =====\n")

    for sample in eval_data:
        question = sample["question"]
        relevant = [str(r) for r in sample["relevant_articles"]]

        docs = retriever.invoke(question)

        retrieved_articles = extract_articles_from_docs(docs)

        p = precision_at_k(retrieved_articles, relevant, K)
        r = recall_at_k(retrieved_articles, relevant, K)

        total_precision += p
        total_recall += r

        print(f"\nQuestion: {question}")
        print(f"Relevant: {relevant}")
        print(f"Retrieved: {retrieved_articles}")
        print(f"Precision@{K}: {p:.2f}")
        print(f"Recall@{K}: {r:.2f}")

    avg_precision = total_precision / len(eval_data)
    avg_recall = total_recall / len(eval_data)

    print("\n=========================================")
    print(f"Average Precision@{K}: {avg_precision:.2f}")
    print(f"Average Recall@{K}: {avg_recall:.2f}")
    print("=========================================\n")
