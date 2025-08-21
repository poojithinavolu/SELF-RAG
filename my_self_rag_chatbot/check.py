import streamlit as st
from pathlib import Path
from huggingface_hub import InferenceClient
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# ==============================
# 🔧 Streamlit Page Setup
# ==============================
st.set_page_config(page_title="RAG Chatbot", page_icon="💬", layout="centered")
st.title("💬 RAG Chatbot")

# ==============================
# 🔑 Hugging Face Token
# ==============================
HF_TOKEN = st.secrets["HF_TOKEN"]

# ==============================
# 📂 Paths
# ==============================
APP_DIR = Path(__file__).resolve().parent
FAISS_DIR = APP_DIR / "faiss_index"
INDEX_NAME = "index"

# ==============================
# 🤗 Models
# ==============================
EMBED_MODEL = "sentence-transformers/sentence-t5-large"
LLM_MODEL = "google/gemma-2-9b"   # can swap with any supported model <10B

# ==============================
# 🧠 Cache Embeddings
# ==============================
@st.cache_resource
def load_embeddings():
    return HuggingFaceEmbeddings(model_name=EMBED_MODEL, model_kwargs={"device": "cpu"})

# ==============================
# 📊 Cache FAISS
# ==============================
@st.cache_resource
def load_faiss(_emb):
    return FAISS.load_local(
        folder_path=str(FAISS_DIR),
        embeddings=_emb,
        index_name=INDEX_NAME,
        allow_dangerous_deserialization=True,
    )

# ==============================
# 🚀 Hugging Face Client
# ==============================
@st.cache_resource
def get_hf_client():
    return InferenceClient(model=LLM_MODEL, token=HF_TOKEN)

# ==============================
# ⚡ Load Everything
# ==============================
embeddings = load_embeddings()
db = load_faiss(embeddings)
retriever = db.as_retriever(search_kwargs={"k": 3})
client = get_hf_client()

# ==============================
# 💬 Chat Input
# ==============================
query = st.text_input("Ask me something:")

if query:
    # 1️⃣ Retrieve context from FAISS
    docs = retriever.invoke(query)
    context = "\n\n".join(d.page_content for d in docs) if docs else ""

    # 2️⃣ Create Prompt
    prompt = f"""
Answer the question strictly using only the information from the context below.
If not enough info, reply exactly: "The context does not provide this information."

Context:
{context}

Question: {query}
Answer:
"""

    # 3️⃣ Call Hugging Face Inference
    response = client.text_generation(
        prompt,
        max_new_tokens=256,
        temperature=0.2,
        do_sample=False,
        stream=False
    )

    # 4️⃣ Extract Clean Answer
    raw_answer = ""

    if isinstance(response, list) and len(response) > 0 and "generated_text" in response[0]:
        raw_answer = response[0]["generated_text"]
    elif isinstance(response, dict) and "generated_text" in response:
        raw_answer = response["generated_text"]
    elif isinstance(response, str):
        raw_answer = response
    else:
        raw_answer = str(response)   # fallback for debugging

    # Just keep answer part after "Answer:"
    answer = raw_answer.split("Answer:", 1)[-1].strip() or raw_answer.strip()

    # 5️⃣ Display
    st.subheader("📌 Answer")
    st.write(answer)
