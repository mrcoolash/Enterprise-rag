# 📄 Enterprise RAG - PDF Intelligence System

A production-ready Retrieval Augmented Generation (RAG) system for querying PDF documents using LangChain, FAISS, and local LLMs (Ollama). Built with FastAPI backend and Streamlit frontend.

## 🌟 Features

- **PDF Upload & Processing**: Upload PDF documents and automatically index them for intelligent search
- **Local RAG Pipeline**: Complete RAG system running locally with FAISS vector store
- **Semantic Search**: HuggingFace embeddings (sentence-transformers/all-MiniLM-L6-v2) for accurate document retrieval
- **Local LLM Integration**: Uses Ollama (Mistral model) for private, offline question-answering
- **FastAPI Backend**: RESTful API with async file upload and chat endpoints
- **Streamlit Frontend**: User-friendly web interface for document upload and interactive Q&A
- **Modular Architecture**: Clean separation of concerns with ingestion, vector store, and backend modules

## 🏗️ Architecture

```
Enterprise-rag-main/
├── app/                    # Frontend application
│   ├── streamlit_app.py   # Streamlit UI
│   └── query.py           # Query utilities
├── backend/               # FastAPI backend
│   ├── main.py           # API endpoints
│   ├── rag.py            # RAG logic & LLM integration
│   ├── models.py         # Pydantic models
│   └── app.py
├── ingestion/            # Document processing
│   ├── loader.py        # PDF document loader
│   └── chunker.py       # Text chunking logic
├── vectorstore/         # Vector database
│   └── build_index.py   # FAISS index builder
└── requirements.txt     # Python dependencies
```

## 🔧 Tech Stack

- **LangChain**: Framework for LLM applications
- **FAISS**: Facebook AI Similarity Search for vector storage
- **Ollama**: Local LLM runtime (Mistral model)
- **FastAPI**: Modern Python web framework
- **Streamlit**: Interactive web UI
- **HuggingFace Transformers**: Sentence embeddings
- **PyPDF**: PDF document parsing

## 📋 Prerequisites

Before starting, ensure you have:

1. **Python 3.8+** installed
2. **Ollama** installed and running
   - Download from: https://ollama.ai/
   - Pull Mistral model: `ollama pull mistral:latest`

## 🚀 Setup & Installation

### 1. Clone the Repository
```bash
cd Enterprise-rag-main
```

### 2. Create Virtual Environment
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Linux/Mac
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Install Additional Dependencies
```bash
# For Streamlit frontend
pip install streamlit

# For FastAPI backend
pip install fastapi uvicorn

# For HuggingFace embeddings
pip install sentence-transformers

# For Ollama integration
pip install langchain-ollama
```

### 5. Verify Ollama Installation
```bash
# Check if Ollama is running
ollama list

# Pull Mistral model if not already installed
ollama pull mistral:latest
```

## 🎯 Running the Application

### Method 1: Run Both Services (Recommended)

**Terminal 1 - Start Backend:**
```bash
cd backend
uvicorn main:app --reload --port 8001
```

The FastAPI backend will start at: `http://127.0.0.1:8001`
- API docs available at: `http://127.0.0.1:8001/docs`

**Terminal 2 - Start Frontend:**
```bash
cd app
streamlit run streamlit_app.py
```

The Streamlit app will open automatically at: `http://localhost:8501`

### Method 2: Using Command Chaining (Windows)

```bash
start cmd /k "cd backend && uvicorn main:app --reload --port 8001" && timeout /t 3 && cd app && streamlit run streamlit_app.py
```

## 📖 Usage Guide

### 1. Upload a PDF Document
- Open the Streamlit interface at `http://localhost:8501`
- Click "Browse files" and select a PDF document
- Click "Upload & Index" button
- Wait for the indexing to complete

### 2. Ask Questions
- Once uploaded, the chat interface will activate
- Type your question in the text input
- Click "Ask" button
- The system will:
  - Search the document using semantic similarity
  - Retrieve relevant chunks
  - Generate an answer using Mistral LLM

### 3. Example Questions
```
- "What is the main topic of this document?"
- "Summarize the key findings"
- "What are the recommendations mentioned?"
- "Explain the methodology used"
```

## 🔌 API Endpoints

### POST `/upload`
Upload and index a PDF document
```bash
curl -X POST "http://127.0.0.1:8001/upload" \
  -F "file=@document.pdf"
```

### POST `/chat`
Ask questions about the indexed document
```bash
curl -X POST "http://127.0.0.1:8001/chat" \
  -H "Content-Type: application/json" \
  -d '{"question": "What is this document about?"}'
```

## ⚙️ Configuration

### Modify LLM Model
Edit `backend/rag.py`:
```python
llm = OllamaLLM(
    model="mistral:latest",  # Change to: llama2, codellama, etc.
    temperature=0
)
```

### Adjust Chunk Size
Edit `backend/rag.py`:
```python
splitter = RecursiveCharacterTextSplitter(
    chunk_size=600,      # Increase for longer context
    chunk_overlap=100    # Overlap between chunks
)
```

### Change Embedding Model
Edit `backend/rag.py`:
```python
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
    # Alternative: "all-mpnet-base-v2" for better accuracy
)
```

## 🔍 How It Works

1. **Document Upload**: PDF is uploaded via Streamlit interface
2. **Processing**: Backend receives PDF and extracts text using PyPDFLoader
3. **Chunking**: Text is split into semantic chunks (600 tokens, 100 overlap)
4. **Embedding**: Each chunk is converted to vector embeddings
5. **Indexing**: Vectors stored in FAISS index for fast similarity search
6. **Query**: User question is embedded and matched against document chunks
7. **Retrieval**: Top 4 most relevant chunks are retrieved
8. **Generation**: Context + question sent to Ollama Mistral for answer generation

## 🛠️ Troubleshooting

### Ollama Not Found
```bash
# Verify Ollama installation
ollama --version

# Restart Ollama service
ollama serve
```

### Port Already in Use
```bash
# Use different port for backend
uvicorn main:app --reload --port 8002

# Update BACKEND_URL in streamlit_app.py
BACKEND_URL = "http://127.0.0.1:8002"
```

### Module Import Errors
```bash
# Install missing dependencies
pip install -r requirements.txt --upgrade

# Verify Python path
python -c "import sys; print(sys.path)"
```

## 📊 Performance Tips

- **Small PDFs**: Default settings work well (< 50 pages)
- **Large PDFs**: Increase chunk_size to 1000-1500
- **Better Accuracy**: Use "all-mpnet-base-v2" embeddings (slower but more accurate)
- **Faster Response**: Reduce k=4 to k=2 in similarity_search

## 🤝 Contributing

Contributions are welcome! Areas for improvement:
- Add support for multiple file formats (DOCX, TXT)
- Implement conversation history/memory
- Add authentication and user management
- Deploy to cloud (AWS, GCP, Azure)
- Add batch processing for multiple documents

## 📝 License

This project is open source and available for educational and commercial use.

## 🙏 Acknowledgments

- LangChain for the RAG framework
- Ollama for local LLM runtime
- FAISS for efficient vector search
- HuggingFace for embeddings

---

**Built with ❤️ for Enterprise Document Intelligence**
