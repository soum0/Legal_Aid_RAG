# # src/vector_store.py
# import chromadb
# from chromadb.config import Settings
# from typing import List, Dict, Any
# import os


# def _sanitize_metadata_item(meta: Dict[str, Any]) -> Dict[str, Any]:
#     clean = {}
#     for k, v in (meta or {}).items():
#         if v is None:
#             clean[k] = ""
#         elif isinstance(v, (str, int, float, bool)):
#             clean[k] = v
#         elif isinstance(v, (list, tuple)):
#             new_list = []
#             for x in v:
#                 if x is None:
#                     new_list.append("")
#                 elif isinstance(x, (str, int, float, bool)):
#                     new_list.append(x)
#                 else:
#                     new_list.append(str(x))
#             clean[k] = new_list
#         else:
#             clean[k] = str(v)
#     return clean


# class ChromaVectorStore:
#     def __init__(self, persist_directory="data/chroma_db"):

#         os.makedirs(persist_directory, exist_ok=True)

#         # Use PersistentClient instead of Client
#         self.client = chromadb.PersistentClient(path=persist_directory)

#         self.collection = self.client.get_or_create_collection(
#             name="constitution_chunks"
#         )

#     def add_documents(self, ids: List[str], embeddings: List[List[float]],
#                       metadatas: List[Dict], documents: List[str]):
#         safe_metadatas = [_sanitize_metadata_item(m) for m in (metadatas or [])]

#         n = len(ids)
#         if not (len(embeddings) == n and len(safe_metadatas) == n and len(documents) == n):
#             raise ValueError("ids, embeddings, metadatas and documents must be same length")

#         self.collection.add(
#             ids=ids,
#             embeddings=embeddings,
#             metadatas=safe_metadatas,
#             documents=documents
#         )

#     def persist(self):
#         """
#         Persist DB if the client exposes a persist method. Some chromadb builds
#         persist automatically; older/newer builds may differ. This method is
#         safe to call in either case.
#         """
#         # Preferred: call client.persist() if it exists
#         if hasattr(self.client, "persist") and callable(getattr(self.client, "persist")):
#             try:
#                 self.client.persist()
#                 print("Chroma client.persist() called successfully.")
#                 return
#             except Exception as e:
#                 print("Warning: client.persist() raised an error:", e)

#         # Fallback: if running a filesystem-backed chroma, there may already be files
#         # in persist_directory. We just notify the user.
#         import os
#         if os.path.exists(self.persist_directory):
#             print(f"Chroma persist not available on this client, but directory exists: {self.persist_directory}")
#             print("If you used an in-memory client, consider switching to a persistent build or Chroma server.")
#         else:
#             print("Chroma persist() not available and persist directory not found. Data may be in-memory only.")


# src/vector_store.py

from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings


class VectorStore:

    def __init__(
        self,
        persist_directory="data/chroma_db",
        collection_name="constitution"
    ):

        self.embedding_model = HuggingFaceEmbeddings(
            model_name="all-MiniLM-L6-v2"
        )

        self.vectorstore = Chroma(
            persist_directory=persist_directory,
            embedding_function=self.embedding_model,
            collection_name=collection_name
        )

    def add_documents(self, documents):
        """
        documents: List of LangChain Document objects
        """
        self.vectorstore.add_documents(documents)

    def get_retriever(self, k=8, fetch_k=24):

        return self.vectorstore.as_retriever(
            search_type="mmr",
            search_kwargs={
                "k": k,
                "fetch_k": fetch_k
            }
        )

    def get_vectorstore(self):
        return self.vectorstore
