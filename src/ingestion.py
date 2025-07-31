import os
from typing import List
from pypdf import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter

def load_text_from_file(filepath: str) -> str:
    ext = os.path.splitext(filepath)[1].lower()
    text = ""
    if ext == ".pdf":
        with open(filepath, "rb") as f:
            reader = PdfReader(f)
            for page in reader.pages:
                text += page.extract_text() or ""
    elif ext in (".txt", ".md"):
        with open(filepath, "r", encoding="utf-8") as f:
            text = f.read()
    else:
        raise ValueError(f"Unsupported file type: {ext}")
    return text

def chunk_text(text: str, chunk_size=500, chunk_overlap=50) -> List[str]:
    # Use LangChain's RecursiveCharacterTextSplitter for better splitting
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunks = splitter.split_text(text)
    return [chunk for chunk in chunks if chunk.strip()]