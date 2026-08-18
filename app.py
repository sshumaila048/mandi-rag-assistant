import streamlit as st
import json
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from groq import Groq

# ---------------------------------------------------------
# PAGE CONFIG
# ---------------------------------------------------------
st.set_page_config(
    page_title="Mandi Price & Farmer Scheme Assistant",
    page_icon="🌾",
    layout="wide"
)

# ---------------------------------------------------------
# LOAD DATA + BUILD TF-IDF INDEX (cached, lightweight, no torch)
# ---------------------------------------------------------
@st.cache_resource
def load_everything():
    with open('embeddings/all_chunks.json', 'r') as f:
        all_chunks = json.load(f)
    with open('embeddings/chunk_sources.json', 'r') as f:
        chunk_sources = json.load(f)

    # Build a TF-IDF index over all chunks (fast, low memory, no model download)
    vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)
    tfidf_matrix = vectorizer.fit_transform(all_chunks)

    return all_chunks, chunk_sources, vectorizer, tfidf_matrix

all_chunks, chunk_sources, vectorizer, tfidf_matrix = load_everything()

# ---------------------------------------------------------
# GROQ CLIENT
# ---------------------------------------------------------
groq_api_key = st.secrets["GROQ_API_KEY"]
client_groq = Groq(api_key=groq_api_key)

# ---------------------------------------------------------
# RETRIEVAL FUNCTION (TF-IDF + cosine similarity, replaces ChromaDB)
# ---------------------------------------------------------
def retrieve_chunks(question, top_k=3):
    query_vec = vectorizer.transform([question])
    similarities = cosine_similarity(query_vec, tfidf_matrix).flatten()
    top_indices = similarities.argsort()[-top_k:][::-1]

    retrieved = [all_chunks[i] for i in top_indices]
    sources = [chunk_sources[i] for i in top_indices]
    return retrieved, sources

# ---------------------------------------------------------
# CORE RAG FUNCTION
# ---------------------------------------------------------
def ask_assistant(question):
    try:
        retrieved_chunks, sources = retrieve_chunks(question, top_k=3)

        context = "\n".join(retrieved_chunks)
        prompt = f"""Answer the question using ONLY the information below. If the information doesn't contain the answer, say so clearly.

Information:
{context}

Question: {question}

Answer:"""

        response = client_groq.chat.completions.create(
            model="openai/gpt-oss-20b",
            messages=[{"role": "user", "content": prompt}]
        )

        return response.choices[0].message.content, retrieved_chunks, sources

    except Exception as e:
        return f"⚠️ Something went wrong: {str(e)}", [], []

# ---------------------------------------------------------
# SIDEBAR
# ---------------------------------------------------------
with st.sidebar:
    st.title("🌾 About")
    st.write(
        "This assistant answers questions about **Mumbai APMC mandi prices** "
        "(July–August 2026) and the **PM-KISAN scheme**, using real government data."
    )
    st.write("**Data sources:**")
    st.write("- AGMARKNET (data.gov.in)")
    st.write("- PM-KISAN Operational Guidelines")

    st.divider()
    if st.button("🗑️ Clear conversation"):
        st.session_state.messages = []
        st.rerun()

# ---------------------------------------------------------
# MAIN CHAT INTERFACE
# ---------------------------------------------------------
st.title("🌾 Mandi Price & Farmer Scheme Assistant")
st.caption("Ask about vegetable/grain prices in Mumbai APMC or the PM-KISAN scheme")

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])
        if msg["role"] == "assistant" and "sources" in msg:
            with st.expander("📄 View source data used"):
                for i, src in enumerate(msg["sources"]):
                    st.write(f"**Source {i+1}** ({msg['source_types'][i]}):")
                    st.write(src)
                    st.divider()

user_question = st.chat_input("Ask about a commodity price or PM-KISAN scheme...")

if user_question:
    st.session_state.messages.append({"role": "user", "content": user_question})
    with st.chat_message("user"):
        st.write(user_question)

    with st.chat_message("assistant"):
        with st.spinner("Searching data and generating answer..."):
            answer, sources, source_types = ask_assistant(user_question)
            st.write(answer)

            if sources:
                with st.expander("📄 View source data used"):
                    for i, src in enumerate(sources):
                        st.write(f"**Source {i+1}** ({source_types[i]}):")
                        st.write(src)
                        st.divider()

    st.session_state.messages.append({
        "role": "assistant",
        "content": answer,
        "sources": sources,
        "source_types": source_types
    })

if len(st.session_state.messages) == 0:
    st.write("**Try asking:**")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.info("What is the price of onion today?")
    with col2:
        st.info("Compare tomato and potato prices")
    with col3:
        st.info("What is PM-KISAN scheme?")
