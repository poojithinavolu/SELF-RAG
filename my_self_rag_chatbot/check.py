import streamlit as st
from pathlib import Path
from huggingface_hub import InferenceClient
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

st.set_page_config(page_title="RAG Chatbot", page_icon="💬", layout="centered")
st.title("💬 RAG Chatbot")

HF_TOKEN = st.secrets["HF_TOKEN"]

APP_DIR = Path(__file__).resolve().parent
FAISS_DIR = APP_DIR / "faiss_index"
INDEX_NAME = "index"

EMBED_MODEL = "sentence-transformers/sentence-t5-large"
LLM_MODEL = "google/gemma-2-9b"   # or zephyr-7b-beta, etc.

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
    return InferenceClient(model=LLM_MODEL, token=HF_TOKEN)

embeddings = load_embeddings()
db = load_faiss(embeddings)
retriever = db.as_retriever(search_kwargs={"k": 3})
client = get_hf_client()

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

query = st.text_input("Ask me something:")

if query:
    # 1️⃣ Retrieve context from FAISS
    docs = retriever.invoke(query)
    context = "\n\n".join(d.page_content for d in docs) if docs else ""

    # 2️⃣ Create Prompt
    prompt = f"""
You are a helpful assistant for a RAG chatbot. 
Answer the following question **only** using the given context. 
If the context does not have the answer, reply exactly:
"The context does not provide this information."

Context:
{context}

Question: {query}

Final Answer:
"""

    # 3️⃣ Call Hugging Face Inference
    response = client.text_generation(
        prompt,
        max_new_tokens=256,
        temperature=0.2,
        do_sample=False,
        stream=False
    )

    # 4️⃣ Extract Answer
    raw_answer = ""
    if isinstance(response, str):
        raw_answer = response
    elif isinstance(response, dict) and "generated_text" in response:
        raw_answer = response["generated_text"]
    elif isinstance(response, list) and "generated_text" in response[0]:
        raw_answer = response[0]["generated_text"]

    answer = raw_answer.split("Final Answer:", 1)[-1].strip()

    # 5️⃣ Save into session_state
    st.session_state.chat_history.append({"question": query, "answer": answer})

# 🖼️ Display Chat History
if st.session_state.chat_history:
    st.subheader("📜 Conversation History")
    for i, chat in enumerate(st.session_state.chat_history, 1):
        st.markdown(f"**Q{i}:** {chat['question']}")
        st.markdown(f"**A{i}:** {chat['answer']}")
        st.markdown("---")

