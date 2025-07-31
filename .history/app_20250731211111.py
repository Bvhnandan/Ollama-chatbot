import os
import streamlit as st
from src.ingestion import load_text_from_file, chunk_text
from src.embeddings import Embeddings
from src.vectorstore import get_vectorstore
from src.retriever import get_qa_chain

from langchain.memory import ConversationBufferMemory
from langchain.chat_models import ChatOpenAI  # We'll build a simple local wrapper for Ollama below

# ==== Config ====
PERSIST_DIR = "db/chromadb"
UPLOAD_DIR = "tmp_uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(PERSIST_DIR, exist_ok=True)

# ==== Models available (fill in with your local Ollama model names or LLM wrappers) ====
AVAILABLE_MODELS = {
    "mistral": "mistral",
    "llama3": "llama3",
    "gemma3": "gemma3",
}

st.title("Local AI Q&A Bot with LangChain & Multi-Model Comparison")

# Uploaded files UI
uploaded_files = st.file_uploader("Upload PDF/TXT/MD documents", accept_multiple_files=True, type=["pdf", "txt", "md"])

if uploaded_files:
    if st.button("Process Uploaded Documents"):
        texts, metadatas, ids = [], [], []
        for file in uploaded_files:
            filepath = os.path.join(UPLOAD_DIR, file.name)
            with open(filepath, "wb") as f:
                f.write(file.read())
            try:
                text = load_text_from_file(filepath)
            except Exception as e:
                st.error(f"Error loading {file.name}: {e}")
                continue
            chunks = chunk_text(text)
            texts.extend(chunks)
            metadatas.extend([{"filename": file.name}] * len(chunks))
            ids.extend([f"{file.name}-{i}" for i in range(len(chunks))])

        if texts:
            # Initialize embedding model and vectorstore
            embedder = Embeddings()
            embeddings = embedder.embed_documents(texts)
            vectorstore = get_vectorstore(PERSIST_DIR, embedder)

            # Add documents + embeddings to vectorstore
            vectorstore.add_texts(texts, metadatas=metadatas, ids=ids, embeddings=embeddings)            
            vectorstore.persist()
            st.success(f"Processed and stored {len(texts)} text chunks into vectorstore.")
        else:
            st.warning("No valid text chunks found in uploaded files.")

# Initialize LangChain Embeddings
embedder = Embeddings()

# Initialize or load vectorstore (empty if you have not processed docs yet)
vectorstore = get_vectorstore(PERSIST_DIR, embedder)

# Chat History Memory (per session)
if "memory" not in st.session_state:
    st.session_state.memory = ConversationBufferMemory(return_messages=True)

memory = st.session_state.memory

# Model selection (single model)
selected_model = st.selectbox("Select a model to chat with:", list(AVAILABLE_MODELS.keys()))

# Comparison checkbox: enable user to pick 2 models for comparison
compare_mode = st.checkbox("Compare answers from two models")

if compare_mode:
    models_for_compare = st.multiselect("Select up to 2 models for comparison:", list(AVAILABLE_MODELS.keys()), default=[selected_model])
    if len(models_for_compare) > 2:
        st.warning("Please select up to two models only.")
    if len(models_for_compare) < 2:
        st.info("Select two models to compare answers.")

question = st.text_input("Ask your question:")

def create_ollama_llm(model_name: str):
    """
    Minimal LangChain-compatible LLM wrapper for Ollama CLI.
    Customize this if you have a better LangChain Ollama integration available.
    """
    from langchain.llms.base import LLM
    import subprocess

    class OllamaLLM(LLM):
        def _call(self, prompt: str, stop=None):
            full_prompt = prompt.strip()
            result = subprocess.run(["ollama", "run", model_name, full_prompt], stdout=subprocess.PIPE)
            return result.stdout.decode("utf-8").strip()

        @property
        def _identifying_params(self):
            return {"model": model_name}

        @property
        def _llm_type(self):
            return "ollama"

    return OllamaLLM()

def run_qa_chain(question: str, model_name: str):
    llm = create_ollama_llm(model_name)
    qa_chain = get_qa_chain(llm, vectorstore.as_retriever(search_type="similarity"), memory)
    return qa_chain.run(question)

if question:
    if compare_mode and len(models_for_compare) == 2:
        cols = st.columns(2)
        answers = []
        for i, model in enumerate(models_for_compare):
            with cols[i]:
                st.markdown(f"### Answer from `{model}`")
                try:
                    answer = run_qa_chain(question, AVAILABLE_MODELS[model])
                    answers.append(answer)
                    st.write(answer)
                except Exception as e:
                    st.error(f"Error generating answer from {model}: {e}")
    else:
        try:
            answer = run_qa_chain(question, AVAILABLE_MODELS[selected_model])
            st.markdown(f"### Answer from `{selected_model}`")
            st.write(answer)
        except Exception as e:
            st.error(f"Error generating answer from {selected_model}: {e}")
