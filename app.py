
"""
Streamlit front-end for the PDF RAG pipeline (FAISS + SentenceTransformers + Gemini).
 
Run with:
    streamlit run app.py
 
Expects the following modules to exist in the same folder (from your earlier days):
    Day01_pdf_utlis.py   -> extract_text, save_file
    Day02_chunking.py    -> load_file, split_text, save_chunks_pickle
    Day03_embeddings.py  -> save_embeddings
    Day4_Vector_Store.py -> load_chunk, load_embeddings, build_faiss, save_index, load_index
"""
 
from multiprocessing import process

import os
import tempfile

import streamlit as st
import google.generativeai as genai

genai.configure(
    api_key=st.secrets["GOOGLE_API_KEY"]
)


import streamlit as st
from sentence_transformers import SentenceTransformer
import google.generativeai as genai
 
from Day01_pdf_utlis import extract_text, save_file
from Day02_chunking import load_file, split_text, save_chunks_pickle
from Day03_embeddings import save_embeddings
from Day4_Vector_Store import load_chunk, load_embeddings, build_faiss, save_index, load_index
 
# -----------------------------------------------------------------------
# Put your Gemini API key here before deploying. Every user of the
# deployed app will use this same key — they won't need to enter one.
# Get a key at: https://aistudio.google.com/app/apikey
# -----------------------------------------------------------------------
GEMINI_API_KEY = os.getenv("GOOGLE_API_KEY")
 
genai.configure(api_key=GEMINI_API_KEY)
 
 
FILES_NEEDED = [
    "extracted_text.txt",
    "Chunks.pkl",
    "Embeddings.npy",
    "faiss_index.pkl",
    "chunks_texts.pkl",
]
 
 
# ---------------------------------------------------------------------------
# Cached resources
# ---------------------------------------------------------------------------
@st.cache_resource
def load_embedding_model():
    return SentenceTransformer("all-MiniLM-L6-v2")
 
 
def load_llm():
    return genai.GenerativeModel("gemini-2.5-flash")
 
 
# ---------------------------------------------------------------------------
# Pipeline logic (same steps as your CLI script, wrapped for the UI)
# ---------------------------------------------------------------------------
def build_pipeline(pdf_path, embed_model, status=None):
    def log(msg):
        if status is not None:
            status.write(msg)
 
    all_exists = all(os.path.exists(f) for f in FILES_NEEDED)
    if all_exists:
        log("All cached files found — loading from disk.")
        return
 
    log("Some files are missing — building the index now.")
 
    # Step 1: Extract text from PDF
    if not os.path.exists("extracted_text.txt"):
        log("Extracting text from PDF...")
        text = extract_text(pdf_path)
        save_file(text)
    else:
        log("extracted_text.txt already exists — skipping extraction.")
 
    # Step 2: Chunking
    if not os.path.exists("Chunks.pkl"):
        log("Splitting text into chunks...")
        text = load_file()
        chunks = split_text(text)
        save_chunks_pickle(chunks)
    else:
        log("Chunks.pkl already exists — skipping chunking.")
 
    # Step 3: Embeddings
    if not os.path.exists("Embeddings.npy"):
        log("Generating embeddings...")
        chunks = load_chunk()
        embeddings = embed_model.encode(chunks, show_progress_bar=False, batch_size=32)
        save_embeddings(embeddings)
    else:
        log("Embeddings.npy already exists — skipping embeddings.")
 
    # Step 4: Vector store
    if not os.path.exists("faiss_index.pkl"):
        log("Building FAISS index...")
        chunks = load_chunk()
        embeddings = load_embeddings()
        index = build_faiss(embeddings)
        save_index(index, chunks)
    else:
        log("faiss_index.pkl already exists — skipping vector store creation.")
 
    log("Pipeline ready.")
 
 
def semantic_search_with_score(question, index, chunks, embed_model, k=3):
    question_vector = embed_model.encode([question])
    distance, indices = index.search(question_vector, k=k)
    results = [chunks[i] for i in indices[0]]
    score = distance[0].tolist()
    return results, score
 
 
def get_answer(llm, question, relevant_chunks):
    context = "\n\n".join(relevant_chunks)
    prompt = f"""<|system|>
You are a strict PDF assistant. You ONLY answer from the context below.
RULES you must follow:
1. Use ONLY the context provided. No outside knowledge allowed.
2. If the question cannot be answered from the context, you MUST respond with exactly: "This information is not available in the PDF."
3. Do NOT guess. Do NOT use your training knowledge. Do NOT make up facts.
</s>
<|user|>
CONTEXT FROM PDF:
{context}
 
QUESTION: {question}
 
Remember: Answer ONLY using the context above. If not found, say "This information is not available in the PDF."
</s>
<|assistant|>
"""
    response = llm.generate_content(prompt)
    return response.text
 
 
# ---------------------------------------------------------------------------
# Streamlit UI
# ---------------------------------------------------------------------------
st.set_page_config(page_title="PDF Q&A Assistant", page_icon="📄", layout="wide")
st.title("📄 PDF Q&A Assistant")
st.caption("Upload a PDF, build a semantic index (FAISS + MiniLM), and ask questions answered strictly from the document via Gemini.")
 
embed_model = load_embedding_model()
 
with st.sidebar:
    st.header("Settings")
    threshold = st.slider(
        "Distance threshold (lower = stricter match required)",
        min_value=0.0, max_value=3.0, value=1.5, step=0.1,
    )
    top_k = st.slider("Chunks to retrieve (k)", min_value=1, max_value=10, value=3)
 
    if st.button("🔄 Clear cached index files"):
        for f in FILES_NEEDED:
            if os.path.exists(f):
                os.remove(f)
        for key in ("index", "chunks", "pdf_name", "messages"):
            st.session_state.pop(key, None)
        st.success("Cache cleared. Upload a PDF again to rebuild the index.")
 
uploaded_file = st.file_uploader("Upload a PDF", type=["pdf"])
 
if uploaded_file is not None and st.session_state.get("pdf_name") != uploaded_file.name:
    pdf_path = os.path.join(tempfile.gettempdir(), uploaded_file.name)
    with open(pdf_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
 
    # A different PDF than last time was uploaded — wipe old cached
    # files so the pipeline rebuilds from this new PDF instead of
    # silently reusing the previous one's chunks/embeddings/index.
    for f in FILES_NEEDED:
        if os.path.exists(f):
            os.remove(f)
 
    with st.status("Processing PDF...", expanded=True) as status:
        build_pipeline(pdf_path, embed_model, status=status)
        index, chunks = load_index()
        st.session_state.index = index
        st.session_state.chunks = chunks
        st.session_state.pdf_name = uploaded_file.name
        st.session_state.llm = load_llm()
        status.update(label="Index ready!", state="complete")
 
if "messages" not in st.session_state:
    st.session_state.messages = []
 
if "index" in st.session_state:
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
 
    question = st.chat_input("Ask a question about the PDF...")
 
    if question:
        st.session_state.messages.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)
 
        with st.chat_message("assistant"):
            with st.spinner("Searching and generating answer..."):
                relevant_chunks, scores = semantic_search_with_score(
                    question, st.session_state.index, st.session_state.chunks, embed_model, k=top_k
                )
 
                if scores[0] > threshold:
                    answer = "This information is not available in the PDF."
                else:
                    answer = get_answer(st.session_state.llm, question, relevant_chunks)
 
                st.markdown(answer)
                with st.expander("Retrieved context chunks"):
                    for i, chunk in enumerate(relevant_chunks):
                        st.markdown(f"**Chunk {i + 1}** — distance: {scores[i]:.3f}")
                        st.text(chunk)
 
        st.session_state.messages.append({"role": "assistant", "content": answer})
else:
    st.info("Upload a PDF file above to build the index and start asking questions.")
 