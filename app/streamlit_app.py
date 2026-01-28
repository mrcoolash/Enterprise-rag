import streamlit as st
import requests

BACKEND_URL = "http://127.0.0.1:8001"

st.set_page_config(
    page_title="Enterprise PDF RAG",
    page_icon="📄",
    layout="wide"
)

st.title("📄 Enterprise PDF Intelligence")
st.caption("FastAPI backend • Local RAG • Ollama")

# =========================
# SESSION STATE
# =========================
if "pdf_uploaded" not in st.session_state:
    st.session_state.pdf_uploaded = False

# =========================
# PDF UPLOAD
# =========================
st.header("📤 Upload PDF")

uploaded_file = st.file_uploader(
    "Choose a PDF file",
    type=["pdf"]
)

if uploaded_file and st.button("Upload & Index"):
    with st.spinner("Uploading and indexing PDF..."):
        files = {
            "file": (
                uploaded_file.name,
                uploaded_file.getvalue(),
                "application/pdf"
            )
        }

        response = requests.post(
            f"{BACKEND_URL}/upload",
            files=files
        )

        if response.status_code == 200:
            st.success("PDF indexed successfully!")
            st.session_state.pdf_uploaded = True
        else:
            st.error(response.json().get("detail", "Upload failed"))

# =========================
# CHAT SECTION
# =========================
st.divider()
st.header("💬 Chat with PDF")

if not st.session_state.pdf_uploaded:
    st.info("Upload a PDF first to enable chat.")
else:
    question = st.text_input("Ask a question about the document")

    if st.button("Ask"):
        if not question.strip():
            st.warning("Please enter a question.")
        else:
            with st.spinner("Thinking..."):
                response = requests.post(
                    f"{BACKEND_URL}/chat",
                    json={"question": question}
                )

                if response.status_code == 200:
                    answer = response.json()["answer"]
                    st.markdown("### ✅ Answer")
                    st.write(answer)
                else:
                    st.error(response.json().get("detail", "Error generating answer"))
