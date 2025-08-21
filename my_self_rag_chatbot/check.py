import streamlit as st
from pathlib import Path
from huggingface_hub import InferenceClient
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# Streamlit page setup
st.set_page_config(page_title="RAG Chatbot", page_icon="💬", layout="centered")
st.title("💬 RAG Chatbot")

# Hugging Face token (keep in Streamlit secrets, not in code!)
HF_TOKEN = st.secrets["HF_TOKEN"]

# Paths
APP_DIR = Path(__file__).resolve().parent
FAISS_DIR = APP_DIR / "faiss_index"
INDEX_NAME = "index"

# Models
EMBED_MODEL = "sentence-transformers/sentence-t5-large"
LLM_MODEL = "meta-llama/Llama-3.2-3B-Instruct"   # ✅ your gated-access model

# Cache embeddings
@st.cache_resource
def load_embeddings():
    return HuggingFaceEmbeddings(model_name=EMBED_MODEL, model_kwargs={"device": "cpu"})

# Cache FAISS
@st.cache_resource
def load_faiss(_emb):
    return FAISS.load_local(
        folder_path=str(FAISS_DIR),
        embeddings=_emb,
        index_name=INDEX_NAME,
        allow_dangerous_deserialization=True,
    )

# Hugging Face inference client
@st.cache_resource
def get_hf_client():
    return InferenceClient(LLM_MODEL, token=HF_TOKEN)

# Load everything
embeddings = load_embeddings()
db = load_faiss(embeddings)
retriever = db.as_retriever(search_kwargs={"k": 3})
client = get_hf_client()

# Chat input
query = st.text_input("Ask me something:")
if query:
    docs = retriever.invoke(query)
    context = "\n\n".join(d.page_content for d in docs) if docs else ""

    prompt = f"""Answer the question strictly using only the information from the context below.
If not enough info, reply: "The context does not provide this information."

Context:
{context}

Question: {query}
Answer:"""

    response = client.text_generation(
        prompt,
        max_new_tokens=256,
        temperature=0.2,
        do_sample=False
    )

    # Extract clean answer
    answer = response.split("Answer:", 1)[-1].strip() or response.strip()
    st.write(answer)



