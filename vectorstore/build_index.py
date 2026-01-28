from dotenv import load_dotenv
load_dotenv()


from ingestion.loader import load_documents
from ingestion.chunker import chunk_documents
from langchain_huggingface import HuggingFaceEmbeddings


from langchain_community.vectorstores import FAISS
import os

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data" / "raw_docs"
INDEX_DIR = PROJECT_ROOT / "vectorstore" / "index"



def main():
    print("Loading documents...")
    documents = load_documents(DATA_DIR)

    print("Chunking documents...")
    chunks = chunk_documents(documents)
    print(f"Total chunks created: {len(chunks)}")

    print("Creating local embeddings...")
    embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)


    print("Building FAISS index...")
    vectorstore = FAISS.from_documents(chunks, embeddings)

    os.makedirs(INDEX_DIR, exist_ok=True)
    vectorstore.save_local(INDEX_DIR)

    print("FAISS index built and saved successfully!")


if __name__ == "__main__":
    main()
