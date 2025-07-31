# ChromaDB vector store logic
import chromadb
from chromadb.config import Settings

PERSIST_DIR = "db/chromadb"

def get_chroma_collection(collection_name="documents"):
    client = chromadb.Client(Settings(persist_directory=PERSIST_DIR))
    return client.get_or_create_collection(collection_name)
