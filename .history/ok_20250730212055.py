import os

# Define the folder structure
folders = [
    "ai-qa-bot",
    "config",
    "tmp_uploads",
    "db/chromadb",
    "src",
    "tests"
]

# Define files and their initial content (empty or minimal placeholders)
files = {
    "app.py": "# Main Streamlit app - UI, upload, ingest, chat\n",
    "requirements.txt": (
        "streamlit\n"
        "chromadb\n"
        "langchain\n"
        "ollama\n"
        "pypdf\n"
        "sentence-transformers\n"
    ),
    "README.md": "# AI Q&A Bot Project\n\nSetup and usage instructions.\n",
    ".gitignore": "venv/\n__pycache__/\n*.pyc\n.db/\n",
    "config/settings.yaml": "# Optional configuration file\n",
    "src/__init__.py": "",
    "src/ingestion.py": "# Document ingestion and chunking logic\n",
    "src/embeddings.py": "# Embedding generation with Ollama\n",
    "src/vectorstore.py": "# ChromaDB vector store logic\n",
    "src/retriever.py": "# RAG pipeline orchestration\n",
    "src/utils.py": "# Utility functions\n",
    "tests/test_workflow.py": "# Workflow testing scripts\n",
}

def create_structure():
    for folder in folders:
        os.makedirs(folder, exist_ok=True)
        print(f"Created folder: {folder}")
    for file_path, content in files.items():
        with open(file_path, "w") as f:
            f.write(content)
            print(f"Created file: {file_path}")

if __name__ == "__main__":
    create_structure()
