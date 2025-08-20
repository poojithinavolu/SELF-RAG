import os
import torch
import streamlit as st
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM
from huggingface_hub import login
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

st.set_page_config(page_title="Self RAG Chatbot", page_icon="💬", layout="centered")
st.title("💬 Self RAG Chatbot")

HF_TOKEN = st.secrets["HF_TOKEN"]
login(HF_TOKEN)

APP_DIR = Path(__file__).resolve().parent
FAISS_DIR = APP_DIR / "faiss_index"
INDEX_NAME = "index"

EMBED_MODEL = "sentence-transformers/sentence-t5-large"
LLM_MODEL = "meta-llama/Llama-3.2-3b"

@st.cache_resource
def load_embeddings():
    return HuggingFaceEmbeddings(model_name=EMBED_MODEL, model_kwargs={"device": "cpu"})

@st.cache_resource
def load_faiss(_emb):   # 👈 leading underscore
    return FAISS.load_local(
        folder_path=str(FAISS_DIR),
        embeddings=_emb,
        index_name=INDEX_NAME,
        allow_dangerous_deserialization=True,
    )

@st.cache_resource
def load_model_and_tokenizer():
    tok = AutoTokenizer.from_pretrained(LLM_MODEL, token=HF_TOKEN)
    if tok.pad_token_id is None and tok.eos_token_id is not None:
        tok.pad_token = tok.eos_token
    mdl = AutoModelForCausalLM.from_pretrained(
        LLM_MODEL,
        token=HF_TOKEN,
        torch_dtype=torch.float32,
        device_map="cpu",
        low_cpu_mem_usage=True,
    )
    mdl.eval()
    return tok, mdl

embeddings = load_embeddings()
db = load_faiss(embeddings)
retriever = db.as_retriever(search_kwargs={"k": 3})
tokenizer, model = load_model_and_tokenizer()

query = st.text_input("Ask me something:")
if query:
    docs = retriever.invoke(query)
    context = "\n\n".join(d.page_content for d in docs) if docs else ""
    prompt = f"""
Answer the question strictly using only the information from the context below.
If not enough info, reply: "The context does not provide this information."

Context:
{context}

Question: {query}
Answer:
"""
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=4096)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=256,
            do_sample=False,
            temperature=0.0,
            pad_token_id=tokenizer.eos_token_id,
        )
    text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    st.write(text.split("Answer:", 1)[-1].strip())
