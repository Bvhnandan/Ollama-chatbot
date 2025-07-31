from sentence_transformers import SentenceTransformer
from typing import List

class Embeddings:
    def __init__(self, model_name="all-MiniLM-L6-v2"):
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer(model_name)

    def embed_documents(self, texts):
        return self.model.encode(texts, show_progress_bar=False).tolist()

    def embed_query(self, text):
        # Accepts a single string (the query); returns a single vector
        return self.model.encode([text])[0].tolist()
