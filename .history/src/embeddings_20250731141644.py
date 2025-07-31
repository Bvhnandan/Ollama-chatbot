from sentence_transformers import SentenceTransformer
from typing import List

def get_sentence_transformer_embeddings(texts: List[str], model_name: str = "all-MiniLM-L6-v2") -> List[List[float]]:
    model = SentenceTransformer(model_name)
    embeddings = model.encode(texts, show_progress_bar=True).tolist()
    return embeddings