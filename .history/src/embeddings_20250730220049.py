import subprocess
import json
from typing import List

def get_ollama_embeddings(texts: List[str], model: str = "embedding-mistral") -> List[List[float]]:
    embeddings = []
    for text in texts:
        try:
            # Use --json for structured output (if supported by your Ollama version)
            command = ["ollama", "embeddings", "-m", model, "--json"]
            result = subprocess.run(command, input=text.encode("utf-8"), stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            if result.returncode != 0:
                print("Ollama error:", result.stderr.decode())
                continue
            output = result.stdout.decode().strip()
            if not output:
                print("No output from Ollama embeddings for:", text[:50])
                continue
            obj = json.loads(output)
            embeddings.append(obj["embedding"])
        except Exception as e:
            print("Embedding failed:", e)
    return embeddings