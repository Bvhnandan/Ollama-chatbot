# Embedding generation with Ollama
import subprocess
import json
from typing import List

def get_ollama_embeddings(texts: List[str], model: str = "embedding-mistral") -> List[List[float]]:
    embeddings = []
    for text in texts:
        # Call Ollama embeddings via CLI, returns JSON with "embedding"
        command = ["ollama", "embeddings", "-m", model, text]
        result = subprocess.run(command, stdout=subprocess.PIPE)
        output = json.loads(result.stdout)
        embeddings.append(output["embedding"])
    return embeddings
