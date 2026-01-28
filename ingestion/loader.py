from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader


def load_documents(folder_path: str):
    documents = []
    folder = Path(folder_path)

    if not folder.exists():
        raise FileNotFoundError(f"Folder not found: {folder_path}")

    pdf_files = list(folder.glob("*.pdf"))

    if not pdf_files:
        raise ValueError("No PDF files found in the folder.")

    for file in pdf_files:
        loader = PyPDFLoader(str(file))
        pages = loader.load()

        for page in pages:
            page.metadata = {
                "source": file.name,
                "page": page.metadata.get("page"),
                "total_pages": page.metadata.get("total_pages"),
                "file_type": "pdf"
            }

        documents.extend(pages)

    return documents
