from pathlib import Path
from typing import List

from langchain_ollama import ChatOllama
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough


# ===============================
# PATH SETUP
# ===============================
PROJECT_ROOT = Path(__file__).resolve().parents[1]
INDEX_DIR = PROJECT_ROOT / "vectorstore" / "index"


# ===============================
# EMBEDDINGS (MUST MATCH INDEX)
# ===============================
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)


# ===============================
# LOAD FAISS VECTOR STORE
# ===============================
vectorstore = FAISS.load_local(
    str(INDEX_DIR),
    embeddings,
    allow_dangerous_deserialization=True
)

retriever = vectorstore.as_retriever(search_kwargs={"k": 5})


# ===============================
# LOCAL LLM (NO API)
# ===============================
llm = ChatOllama(
    model="mistral",
    temperature=0
)


# ===============================
# HELPER: FORMAT CONTEXT
# ===============================
def format_docs(docs: List[Document]) -> str:
    return "\n\n".join(
        f"[Source: {d.metadata['source']}, page {d.metadata['page'] + 1}]\n"
        f"{d.page_content}"
        for d in docs
    )


# ===============================
# PROMPT
# ===============================
prompt = ChatPromptTemplate.from_template(
    """
You are a document question answering system.
Answer the question using ONLY the context provided.
If the answer is not present, say:
"I don't know based on the provided documents."

Context:
{context}

Question:
{question}
"""
)


# ===============================
# RAG CHAIN
# ===============================
rag_chain = (
    {
        "context": retriever | format_docs,
        "question": RunnablePassthrough(),
    }
    | prompt
    | llm
)


# ===============================
# QUERY LOOP
# ===============================
print("\nEnterprise RAG System (Local LLM)")
print("Type 'exit' to quit\n")

while True:
    question = input("Ask a question: ")

    if question.lower() == "exit":
        break

    response = rag_chain.invoke(question)

    print("\nAnswer:\n")
    print(response.content)
    print("\n" + "-" * 80 + "\n")
