# RAG pipeline orchestration
def retrieve_relevant_chunks(query, embedding_fn, collection, top_k=3):
    query_emb = embedding_fn([query])[0]
    results = collection.query(query_embeddings=[query_emb], n_results=top_k)
    # Results includes documents, ids, metadatas, etc.
    return results
