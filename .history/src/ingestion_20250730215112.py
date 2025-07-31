# Document ingestion and chunking logic
import os
from typing import List
import PyPDF2

def extract_text_from_file(filepath: str) -> str:
    ext = os.path.splitext(filepath)[1].lower()
    text = ""
    if ext == ".pdf":
        with open(filepath, "rb") as f:
            reader = PyPDF2.PdfReader(f)
            for page in reader.pages:
                text += page.extract_text() or ""
    elif ext in (".txt", ".md"):
        with open(filepath, "r", encoding="utf-8") as f:
            text = f.read()
    return text

def chunk_text(text: str, chunk_size=500, chunk_overlap=50) -> List[str]:
    # Simple sliding window chunking
    chunks = []
    i = 0
    while i < len(text):
        chunks.append(text[i:i+chunk_size])
        i += chunk_size - chunk_overlap
    return chunks
