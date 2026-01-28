import tempfile
from typing import List

from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaLLM

from langchain_core.documents import Document


# ---------- MODELS ----------
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

llm = OllamaLLM(
    model="mistral:latest",
    temperature=0
)



# ---------- STATE ----------
VECTORSTORE = None
CHUNKS: List[Document] = []


# ---------- HELPERS ----------
def format_docs(docs: List[Document]) -> str:
    return "\n\n".join(d.page_content for d in docs)


def process_pdf(uploaded_bytes: bytes):
    global VECTORSTORE, CHUNKS

    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(uploaded_bytes)
        path = tmp.name

    loader = PyPDFLoader(path)
    pages = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=600,
        chunk_overlap=100
    )

    chunks = [c for c in splitter.split_documents(pages)
              if c.page_content.strip()]

    if not chunks:
        raise ValueError("No readable text found in PDF.")

    VECTORSTORE = FAISS.from_documents(chunks, embeddings)
    CHUNKS = chunks


def chat_with_pdf(question: str) -> str:
    if VECTORSTORE is None:
        raise ValueError("No PDF indexed.")

    docs = VECTORSTORE.similarity_search(question, k=4)

    if not docs:
        return "I could not find relevant content in the document."

    context = "\n\n".join(d.page_content for d in docs)

    prompt = f"""
You are a helpful assistant.
Answer the question using ONLY the context below.

Context:
{context}

Question:
{question}
"""

    return llm.invoke(prompt)
