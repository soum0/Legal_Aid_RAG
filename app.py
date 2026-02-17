# app.py

import streamlit as st
from dotenv import load_dotenv
from src.lc_rag_chain import build_rag_chain

load_dotenv()

# -----------------------
# Page Config
# -----------------------
st.set_page_config(
    page_title="Constitution of India - AI Assistant",
    page_icon="⚖️",
    layout="wide"
)

st.title("⚖️ Constitution of India - AI Assistant")
st.markdown("Ask any question about the Constitution of India.")

# -----------------------
# Load RAG Chain (cached)
# -----------------------
@st.cache_resource
def load_chain():
    return build_rag_chain()

rag_chain = load_chain()

# -----------------------
# Session State
# -----------------------
if "messages" not in st.session_state:
    st.session_state.messages = []

# -----------------------
# Display Chat History
# -----------------------
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# -----------------------
# User Input
# -----------------------
if prompt := st.chat_input("Ask a constitutional question..."):

    # Show user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Generate response
    with st.chat_message("assistant"):
        with st.spinner("Analyzing constitutional text..."):

            response = rag_chain.invoke(prompt)

            # Extract only content if AIMessage
            if hasattr(response, "content"):
                response = response.content

            st.markdown(response)

    # Save assistant response
    st.session_state.messages.append(
        {"role": "assistant", "content": response}
    )
