# Main Streamlit app - UI, upload, ingest, chat
import streamlit as st
import os
import shutil
from src.ingestion import extract_text_from_file, chunk_text
from src.embeddings import get_ollama_embeddings
from src.vectorstore import get_chroma_collection
from src.retriever import retrieve_relevant_chunks

UPLOAD_DIR = "tmp_uploads"

st.title("Ollama Q&A Chatbot (Mistral 7B, ChromaDB, Local)")

os.makedirs(UPLOAD_DIR, exist_ok=True)

uploaded_files = st.file_uploader("Upload PDF/TXT/MD docs", type=["pdf", "txt", "md"], accept_multiple_files=True)

if uploaded_files:
    st.write("Files uploaded. Click 'Process Documents' to add to knowledge base.")
    if st.button("Process Documents"):
        doc_texts, metadatas, ids = [], [], []
        for file in uploaded_files:
            path = os.path.join(UPLOAD_DIR, file.name)
            with open(path, "wb") as f:
                f.write(file.read())
            text = extract_text_from_file(path)
            chunks = chunk_text(text)
            doc_texts.extend(chunks)
            metadatas.extend([{"filename": file.name}] * len(chunks))
            ids.extend([f"{file.name}-{i}" for i in range(len(chunks))])
        # Get embeddings from Ollama
        embeddings = get_ollama_embeddings(doc_texts)
        # Store in ChromaDB
        collection = get_chroma_collection()
        collection.add(
            documents=doc_texts,
            embeddings=embeddings,
            metadatas=metadatas,
            ids=ids
        )
        print(f"Number of document chunks: {len(doc_texts)}")
        print(f"Number of embeddings generated: {len(embeddings)}")

        st.success("Documents processed and added!")
        # Clean upload dir
        for file in os.listdir(UPLOAD_DIR):
            os.remove(os.path.join(UPLOAD_DIR, file))

st.header("Ask a Question")

question = st.text_input("Your question:")

if question and st.button("Get Answer"):
    collection = get_chroma_collection()
    # Retrieve top relevant context
    results = retrieve_relevant_chunks(question, get_ollama_embeddings, collection)
    context = " ".join(results["documents"][0])  # simple context concat
    # Build prompt for Mistral 7B
    prompt = f"Context:\n{context}\n\nQuestion: {question}\nAnswer:"
    # Call Ollama LLM (CLI)
    import subprocess
    answer = subprocess.run(["ollama", "run", "mistral", prompt], stdout=subprocess.PIPE).stdout.decode()
    st.write(f"**Answer:** {answer.strip()}")
    # Optionally show source context
    st.expander("Show sources").write("\n\n".join(results["documents"][0]))
