import requests
from typing import List

def get_ollama_embeddings(texts: List[str], model: str = "embedding-mistral") -> List[List[float]]:
    """
    Generate embeddings for a list of texts using Ollama's local REST API.

    Args:
        texts: List of text chunks to embed.
        model: Name of the Ollama embedding model installed locally.

    Returns:
        List of embeddings (each is a list of floats).
    """
    url = "http://localhost:11434/api/embeddings"
    embeddings = []

    for i, text in enumerate(texts):
        if not text.strip():
            print(f"Skipping empty text chunk at index {i}")
            continue

        payload = {
            "model": model,
            "prompt": text  # Depending on Ollama version, might also be "input"
        }

        try:
            response = requests.post(url, json=payload)
            if response.status_code == 200:
                data = response.json()
                embedding = data.get("embedding")
                if embedding:
                    embeddings.append(embedding)
                else:
                    print(f"No 'embedding' field in response for chunk {i}")
            else:
                print(f"Failed to get embedding for chunk {i}: {response.status_code} {response.text}")
        except Exception as e:
            print(f"Exception during embedding chunk {i}: {e}")

    return embeddings
    st.write("**Source Context:**")
    for doc in results["documents"][0]: 
        st.write(doc)