import subprocess
import json
from typing import List

def get_ollama_embeddings(texts: List[str], model: str = "embedding-mistral") -> List[List[float]]:
    embeddings = []
    for i, text in enumerate(texts):
        try:
            if not text.strip():
                print(f"Skipping empty text chunk at index {i}")
                continue
            command = ["ollama", "embeddings", "-m", model, "--json"]
            result = subprocess.run(command, input=text.encode("utf-8"), stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            if result.returncode != 0:
                print(f"Ollama error on chunk {i}: {result.stderr.decode().strip()}")
                continue
            output_str = result.stdout.decode().strip()
            if not output_str:
                print(f"No output for chunk {i}")
                continue
            obj = json.loads(output_str)
            embedding = obj.get("embedding")
            if embedding:
                embeddings.append(embedding)
            else:
                print(f"No 'embedding' field in output for chunk {i}")
        except Exception as e:
            print(f"Exception during embedding chunk {i}: {e}")
    return embeddings
