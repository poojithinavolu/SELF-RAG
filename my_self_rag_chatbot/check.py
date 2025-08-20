import streamlit as st
from pathlib import Path
from huggingface_hub import InferenceClient
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

st.set_page_config(page_title="RAG Chatbot", page_icon="💬", layout="centered")
st.title("💬 Self RAG Chatbot")

HF_TOKEN = st.secrets["HF_TOKEN"]

APP_DIR = Path(__file__).resolve().parent
FAISS_DIR = APP_DIR / "faiss_index"
INDEX_NAME = "index"

EMBED_MODEL = "sentence-transformers/sentence-t5-large"
LLM_MODEL = "meta-llama/Llama-3.2-3b"

@st.cache_resource
def load_embeddings():
    return HuggingFaceEmbeddings(model_name=EMBED_MODEL, model_kwargs={"device": "cpu"})

@st.cache_resource
def load_faiss(_emb):
    return FAISS.load_local(
        folder_path=str(FAISS_DIR),
        embeddings=_emb,
        index_name=INDEX_NAME,
        allow_dangerous_deserialization=True,
    )

@st.cache_resource
def get_hf_client():
    return InferenceClient(LLM_MODEL, token=HF_TOKEN)

embeddings = load_embeddings()
db = load_faiss(embeddings)
retriever = db.as_retriever(search_kwargs={"k": 3})
client = get_hf_client()

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

    response = client.text_generation(prompt, max_new_tokens=256, temperature=0.2, do_sample=False)
    answer = response.split("Answer:", 1)[-1].strip() or response.strip()
    st.write(answer)
