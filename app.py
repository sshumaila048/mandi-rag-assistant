
import streamlit as st
import json
import requests
import pandas as pd
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
# LIVE DATA FETCH FROM data.gov.in (AGMARKNET) API
# Cached for 1 hour so we don't hit the API on every single question
# ---------------------------------------------------------
DATA_GOV_RESOURCE_ID = "35985678-0d79-46b4-9ed6-6f13308a1d24"
 
@st.cache_data(ttl=3600, show_spinner="Fetching latest mandi prices...")
def fetch_live_mandi_data(state="Maharashtra", district="Mumbai", limit=1000):
    api_key = st.secrets["DATA_GOV_API_KEY"]
    url = f"https://api.data.gov.in/resource/{DATA_GOV_RESOURCE_ID}"
    params = {
        "api-key": api_key,
        "format": "json",
        "limit": limit,
        "filters[state]": state,
        "filters[district]": district,
    }
    last_error = None
    for attempt in range(3):
        try:
            response = requests.get(url, params=params, timeout=30)
            response.raise_for_status()
            data = response.json()
            records = data.get("records", [])
            return pd.DataFrame(records)
        except Exception as e:
            last_error = e
            continue
    # All 3 attempts failed — signal failure to caller, don't crash the app
    st.session_state["live_fetch_failed"] = str(last_error)
    return pd.DataFrame()
 
def clean_live_data(df):
    if df.empty:
        return df
    df = df.drop_duplicates()
    price_cols = [c for c in ["min_price", "max_price", "modal_price"] if c in df.columns]
    df = df.dropna(subset=price_cols)
    for c in price_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=price_cols)
    return df
 
def row_to_text(row):
    return (
        f"On {row.get('arrival_date', 'an unspecified date')}, in {row.get('market', 'Mumbai APMC')} market, "
        f"{row.get('district', 'Mumbai')}, {row.get('state', 'Maharashtra')}, the price of {row.get('commodity', 'this commodity')} "
        f"({row.get('variety', 'Other')}, {row.get('grade', 'Local')} grade) was: "
        f"Minimum ₹{row.get('min_price', 'N/A')}, Maximum ₹{row.get('max_price', 'N/A')}, "
        f"Modal (most common) price ₹{row.get('modal_price', 'N/A')} per quintal."
    )
 
# ---------------------------------------------------------
# LOAD ALL SAVED STATIC CHUNKS (used for PM-KISAN always, and as fallback for prices)
# ---------------------------------------------------------
@st.cache_resource
def load_static_chunks():
    with open('embeddings/all_chunks.json', 'r') as f:
        all_chunks = json.load(f)
    with open('embeddings/chunk_sources.json', 'r') as f:
        chunk_sources = json.load(f)
    policy_chunks = [c for c, s in zip(all_chunks, chunk_sources) if s == "pmkisan_pdf"]
    static_price_chunks = [c for c, s in zip(all_chunks, chunk_sources) if s == "mandi_prices_csv"]
    return policy_chunks, static_price_chunks
 
# ---------------------------------------------------------
# BUILD COMBINED, LIVE (with static fallback) TF-IDF INDEX
# ---------------------------------------------------------
@st.cache_resource(ttl=3600)
def build_index():
    policy_chunks, static_price_chunks = load_static_chunks()
 
    live_df = fetch_live_mandi_data()
    live_df = clean_live_data(live_df)
 
    if not live_df.empty:
        price_chunks = live_df.apply(row_to_text, axis=1).tolist()
        price_sources = ["mandi_prices_live_api"] * len(price_chunks)
        data_mode = "live"
    else:
        # Live API failed or returned nothing — fall back to last known good data
        price_chunks = static_price_chunks
        price_sources = ["mandi_prices_static_fallback"] * len(price_chunks)
        data_mode = "fallback"
 
    policy_sources = ["pmkisan_pdf"] * len(policy_chunks)
 
    all_chunks = price_chunks + policy_chunks
    chunk_sources = price_sources + policy_sources
 
    vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)
    tfidf_matrix = vectorizer.fit_transform(all_chunks) if all_chunks else None
 
    return all_chunks, chunk_sources, vectorizer, tfidf_matrix, len(price_chunks), data_mode
 
all_chunks, chunk_sources, vectorizer, tfidf_matrix, live_count, data_mode = build_index()
 
# ---------------------------------------------------------
# GROQ CLIENT
# ---------------------------------------------------------
groq_api_key = st.secrets["GROQ_API_KEY"]
client_groq = Groq(api_key=groq_api_key)
 
# ---------------------------------------------------------
# RETRIEVAL
# ---------------------------------------------------------
def retrieve_chunks(question, top_k=3):
    if tfidf_matrix is None:
        return [], []
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
        if not retrieved_chunks:
            return "⚠️ No data available right now — the live price feed may be temporarily unavailable. Please try again shortly.", [], []
 
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
        "(fetched live from the AGMARKNET / data.gov.in API, with a static fallback for reliability) "
        "and the **PM-KISAN scheme**."
    )
    if data_mode == "live":
        st.success(f"🟢 Live data active — {live_count} price records")
    else:
        st.warning(f"🟡 Live API unavailable — using last saved data ({live_count} records)")
    st.write("**Data sources:**")
    st.write("- AGMARKNET (data.gov.in) — live API, static fallback")
    st.write("- PM-KISAN Operational Guidelines (PDF)")
 
    st.divider()
    if st.button("🔄 Refresh live prices now"):
        st.cache_data.clear()
        st.cache_resource.clear()
        st.rerun()
    if st.button("🗑️ Clear conversation"):
        st.session_state.messages = []
        st.rerun()
 
# ---------------------------------------------------------
# MAIN CHAT INTERFACE
# ---------------------------------------------------------
st.title("🌾 Mandi Price & Farmer Scheme Assistant")
st.caption("Ask about vegetable/grain prices in Mumbai APMC (live data) or the PM-KISAN scheme")
 
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
        with st.spinner("Searching live data and generating answer..."):
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
