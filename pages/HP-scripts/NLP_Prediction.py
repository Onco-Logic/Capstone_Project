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

# === MODEL ID for LLaMA 4 Scout 17B ===
MODEL_ID = "meta-llama/Llama-4-Scout-17B-16E-Instruct"  # 🔄 CHANGED

# === Load Sentence-BERT Embedder ===
@st.cache_resource
def load_embedder():
    return SentenceTransformer("all-MiniLM-L6-v2")

# === Load LLaMA 4 Scout ===
@st.cache_resource
def load_llama4():  # ✅ NEW
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, use_auth_token=True)  # 🔄 CHANGED
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.float16,  # 🔄 CHANGED
        device_map="auto",          # 🔄 CHANGED
        use_auth_token=True
    )
    return tokenizer, model

embedder = load_embedder()
tokenizer, model = load_llama4()  # 🔄 CHANGED

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
st.title("🧠 TCGA Pathology Report Assistant (LLaMA 4 Scout RAG)")
query = st.text_input("Ask a question about the pathology reports:")

if query:
    query_vec = embedder.encode([query])
    D, I = index.search(query_vec, k=3)
    context = "\n\n".join([docs[i] for i in I[0]])

    # 🔄 CHANGED: Build prompt
    prompt = f"Context:\n{context}\n\nQuestion: {query}\n\nAnswer:"  

    # 🔄 CHANGED: Tokenize and generate using Scout
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048).to(model.device)
    outputs = model.generate(**inputs, max_new_tokens=400)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # Optional: extract only the answer part
    answer = response.split("Answer:")[-1].strip()

    st.subheader("Answer")
    st.write(answer)
