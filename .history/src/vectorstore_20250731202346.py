from langchain.vectorstores import Chroma
from langchain.embeddings.base import Embeddings

def get_vectorstore(persist_directory: str, embedding_function: Embeddings, collection_name:str = "documents"):
    """
    Initializes or loads a LangChain Chroma vectorstore.
    """
    vectorstore = Chroma(
        collection_name=collection_name,
        persist_directory=persist_directory,
        embedding_function=embedding_function,
    )
    return vectorstore
