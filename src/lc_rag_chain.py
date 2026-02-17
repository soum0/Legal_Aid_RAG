# src/lc_rag_chain.py

import os
from dotenv import load_dotenv

from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_groq import ChatGroq



from langchain.retrievers.multi_query import MultiQueryRetriever
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
            "k": 8,
            "fetch_k": 24
        }
    )

    # 4️⃣ Groq LLM
    llm = ChatGroq(
        groq_api_key=os.environ.get("GROQ_API_KEY"),
        model_name="mixtral-8x7b-32768",
        temperature=0
    )


    # 5️⃣ MultiQuery Retriever
    multi_retriever = MultiQueryRetriever.from_llm(
        retriever=retriever,
        llm=llm
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
            f"[Article {doc.metadata.get('article_raw_number','')}] {doc.page_content}"
            for doc in docs
        )

    # 8️⃣ Chain
    rag_chain = (
        {
            "context": multi_retriever | format_docs,
            "question": RunnablePassthrough()
        }
        | prompt
        | llm
    )

    return rag_chain
