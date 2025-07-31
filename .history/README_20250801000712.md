Local AI Q&A Bot with Ollama, LangChain & ChromaDB

Introduction

Large Language Models (LLMs) and retrieval-augmented generation (RAG) are revolutionizing many sectors in the tech industry. This project implements a local AI Q&A chatbot that combines powerful language models (e.g., Mistral, Gemma3) running locally via Ollama, with a scalable open-source vector database (ChromaDB) and LangChain to orchestrate conversation memory and retrieval.

Use cases and industry relevance:
- Enterprise knowledge management: Quickly query internal documents, policies, and reports without sending data to cloud APIs.
- Healthcare and research: Interactive Q&A over medical papers, clinical data, or research articles with high data privacy.
- Customer support automation: Build self-contained FAQ bots accessing company manuals and previous tickets.
- Educational tools: Facilitate interactive study aides that reference course materials and external content.
- Model evaluation and comparison: Enables side-by-side comparisons across multiple local LLMs, supporting model benchmarking and selection for specific domains.

By using local models and embedding generation, organizations gain full control over data, reduce dependency on third-party APIs, and reduce latency—critical factors in highly regulated or latency-sensitive environments.

Features
- Upload documents (PDF, TXT, Markdown) directly from the web UI.
- Automatic text extraction and chunking for improved retrieval.
- Embedding generation with sentence-transformers (all-MiniLM-L6-v2).
- Persistent vector store using ChromaDB managed via LangChain.
- Retrieval-based Q&A with chat history using LangChain’s ConversationBufferMemory.
- Multi-model selection and answer comparison from local Ollama LLMs.
- Clear UI for question input, model switching, and comparing multiple answers side by side.

Prerequisites
- Python 3.8 or higher
- Ollama installed and models pulled locally (e.g., Mistral, Gemma3)
- Required Python libraries (listed below)

Setup Instructions

1. Clone or copy the project directory
   git clone <your_repo_url>  # Or copy folder manually
   cd ollama-chatbot

2. Create and activate a virtual environment (recommended)
   Using venv:
   python -m venv venv
   source venv/bin/activate   # macOS/Linux
   venv\Scripts\activate      # Windows

3. Install required Python packages
   pip install -r requirements.txt

   The requirements.txt includes:
   streamlit
   chromadb
   langchain
   langchain-community
   sentence-transformers
   pypdf
   requests

4. Install and start Ollama service
   - Install Ollama from https://ollama.com
   - Pull desired local models:
     ollama pull mistral
     ollama pull gemma3
     # Add other models as needed
   - Start Ollama daemon if not already running (you might not need this on Windows as Ollama auto-runs):
     ollama serve

5. Run the Streamlit app
   streamlit run app.py

6. Using the app
   - Upload your PDF, TXT, or MD files for ingestion.
   - Select a single active model to chat with.
   - Optionally, enable comparison mode to view answers from two models side-by-side.
   - Enter your questions and see responses along with persistent chat history.

Notes & Tips
- Keep the Ollama daemon running for the app to communicate with local LLMs.
- The embedding model defaults to all-MiniLM-L6-v2 from sentence-transformers, easy to replace if desired.
- Uploaded documents are temporarily saved to tmp_uploads/ folder and indexed with ChromaDB under db/chromadb.
- Modify AVAILABLE_MODELS in app.py to add or remove Ollama models you have locally.
- For production deployment, consider adding user authentication, logging, and more robust persistence for chat history.

Troubleshooting
- Ollama CLI errors: Ensure Ollama service is running and models are pulled using ollama list.
- ModuleNotFoundError: Install missing dependencies like langchain-community with pip.
- Empty embeddings or retrieval issues: Verify embedding function is working and documents have been processed.
- Performance issues: Try smaller chunk sizes or increase system memory.

License
(Include your project license here or your organization's preferred licensing terms.)

Acknowledgements
- Based on open-source projects and frameworks: Ollama, LangChain, ChromaDB, sentence-transformers.
- Inspired by Cygnus company requirements for secure, local AI workflows.

If you encounter any issues or want to request new features, please raise an issue or contact the maintainer.

Happy querying! 🚀
