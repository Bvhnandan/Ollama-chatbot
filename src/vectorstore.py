# Depending on your langchain version, adjust import here:
from langchain.vectorstores import Chroma
from langchain.embeddings.base import Embeddings


def get_vectorstore(persist_directory: str, embedding_function: Embeddings, collection_name: str = "documents"):
    """
    Initialize or load a LangChain Chroma vectorstore.

    Args:
        persist_directory (str): Directory for persistent storage.
        embedding_function (Embeddings): An embeddings instance implementing embed_documents and embed_query.
        collection_name (str): Name of the Chroma collection.

    Returns:
        Chroma: Initialized vectorstore instance.
    """
    vectorstore = Chroma(
        collection_name=collection_name,
        persist_directory=persist_directory,
        embedding_function=embedding_function,
    )
    return vectorstore
