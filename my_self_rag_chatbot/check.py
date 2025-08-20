import torch
import streamlit as st
from transformers import AutoTokenizer, AutoModelForCausalLM
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from huggingface_hub import login

HF_TOKEN = st.secrets["HF_TOKEN"]

login(HF_TOKEN)

FAISS_FOLDER = "faiss_index"
EMBED_MODEL = "sentence-transformers/sentence-t5-large"
LLM_MODEL = "meta-llama/Llama-3.2-3b"

# Load FAISS
embedding = HuggingFaceEmbeddings(model_name=EMBED_MODEL, model_kwargs={"device": "cpu"})
db = FAISS.load_local(FAISS_FOLDER, embedding, allow_dangerous_deserialization=True)
retriever = db.as_retriever(search_kwargs={"k": 3})

# Load LLM
tokenizer = AutoTokenizer.from_pretrained(LLM_MODEL, token=HF_TOKEN)
model = AutoModelForCausalLM.from_pretrained(
    LLM_MODEL,
    device_map="cpu",
    torch_dtype=torch.float32,
    token=HF_TOKEN
)

st.title("RAG Chatbot")

query = st.text_input("Ask me something:")
if query:
    docs = retriever.invoke(query)
    context = "\n\n".join([d.page_content for d in docs])

    prompt = f"""
    Answer the question strictly using only the information from the context below.
    If not enough info, reply: "The context does not provide this information."

    Context:
    {context}

    Question: {query}
    Answer:
    """

    inputs = tokenizer(prompt, return_tensors="pt").to("cpu")
    outputs = model.generate(**inputs, max_new_tokens=150)
    answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
    st.write(answer.split("Answer:", 1)[-1].strip())


