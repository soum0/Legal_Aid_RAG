# src/lc_rag_chain.py

import os
from dotenv import load_dotenv
import streamlit as st
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain.prompts import ChatPromptTemplate
from langchain.schema.runnable import RunnablePassthrough

load_dotenv()

def build_rag_chain():

    # 1️⃣ Embeddings
    embedding_model = HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2"
    )

    # 2️⃣ Load Chroma
    vectorstore = Chroma(
        persist_directory="data/chroma_db",
        embedding_function=embedding_model
    )

    # 3️⃣ Base Retriever with MMR
    retriever = vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={
            "k": 4,
            "fetch_k": 10
        }
    )

    # 4️⃣ Groq LLM
    llm = ChatGroq(
        groq_api_key=st.secrets.get("GROQ_API_KEY"),
        model_name="llama-3.1-8b-instant",
        temperature=0
    )

    # 6️⃣ Prompt
    prompt = ChatPromptTemplate.from_template("""
You are a legal assistant answering questions strictly based on the Constitution of India.
You are Created by SOUMYA SINGH. 
Use ONLY the provided context.

If the answer is not found in the context, say:
"I could not find this in the provided constitutional text."

Context:
{context}

Question:
{question}

Answer:
""")

    # 7️⃣ Format Docs
    def format_docs(docs):
        return "\n\n".join(
            f"[Article {doc.metadata.get('article_number','')}] {doc.page_content}"
            for doc in docs
        )

    # 8️⃣ Chain — use the base retriever (NOT MultiQueryRetriever)
    rag_chain = (
        {
            "context": retriever | format_docs,
            "question": RunnablePassthrough()
        }
        | prompt
        | llm
    )

    return rag_chain
