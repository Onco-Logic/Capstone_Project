import streamlit as st
import pandas as pd
import faiss
import os
import pickle
import torch  # ✅ NEW

from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM  # ✅ NEW

# === File Paths ===
CSV_PATH = "/Users/harris/Documents/Capstone_Project/Capstone_Project/Data/TCGA_Reports.csv"
INDEX_PATH = "/Users/harris/Documents/Capstone_Project/Capstone_Project/Data/Index/faiss_index.bin"
DOCS_PATH = "/Users/harris/Documents/Capstone_Project/Capstone_Project/Data/Index/docs.pkl"

# === MODEL ID for smaller LLaMA variant ===
MODEL_ID = "mistralai/Mistral-7B-Instruct-v0.1"  # 🔄 CHANGED: switched to smaller, CPU-friendly model

# === Load Sentence-BERT Embedder ===
@st.cache_resource
def load_embedder():
    return SentenceTransformer("all-MiniLM-L6-v2")

# === Load Mistral 7B (CPU-optimized) ===
@st.cache_resource
def load_llama4():  # 🔄 CHANGED: still using same function name for compatibility
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.float32,  # 🔄 CHANGED: use float32 for CPU
        device_map={"": "cpu"}     # 🔄 CHANGED: force CPU usage
    )
    return tokenizer, model

embedder = load_embedder()
tokenizer, model = load_llama4()  # ✅ Unchanged usage

# === Build or Load FAISS Index ===
@st.cache_resource
def load_index():
    if os.path.exists(INDEX_PATH) and os.path.exists(DOCS_PATH):
        index = faiss.read_index(INDEX_PATH)
        with open(DOCS_PATH, "rb") as f:
            docs = pickle.load(f)
    else:
        df = pd.read_csv(CSV_PATH)
        docs = df["text"].tolist()
        embeddings = embedder.encode(docs, show_progress_bar=True)
        index = faiss.IndexFlatL2(embeddings[0].shape[0])
        index.add(embeddings)

        os.makedirs("index", exist_ok=True)
        faiss.write_index(index, INDEX_PATH)
        with open(DOCS_PATH, "wb") as f:
            pickle.dump(docs, f)
    return index, docs

index, docs = load_index()

# === Streamlit Front-End ===
st.title("🧠 TCGA Pathology Report Assistant (Mistral-7B RAG)")  # 🔄 CHANGED: updated title
query = st.text_input("Ask a question about the pathology reports:")

if query:
    query_vec = embedder.encode([query])
    D, I = index.search(query_vec, k=3)
    context = "\n\n".join([docs[i] for i in I[0]])

    # 🔄 CHANGED: Build prompt
    prompt = f"Context:\n{context}\n\nQuestion: {query}\n\nAnswer:"  

    # 🔄 CHANGED: Tokenize and generate using Mistral-7B
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048)
    outputs = model.generate(**inputs, max_new_tokens=400)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # Optional: extract only the answer part
    answer = response.split("Answer:")[-1].strip()

    st.subheader("Answer")
    st.write(answer)
