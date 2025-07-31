from langchain.chains import RetrievalQA
from langchain.llms.base import LLM
from langchain.memory import ConversationBufferMemory

def get_qa_chain(llm: LLM, retriever, memory: ConversationBufferMemory):
    """
    Build and return the LangChain RetrievalQA chain with chat history memory.
    """
    qa = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        return_source_documents=False,
        memory=memory,
    )
    return qa
