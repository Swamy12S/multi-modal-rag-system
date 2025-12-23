import streamlit as st
import requests

API_URL = "http://127.0.0.1:8000"

st.set_page_config(page_title="Multi-Modal RAG", layout="centered")

st.title("📄 Multi-Modal RAG Demo")

# -----------------------------
# Upload PDF
# -----------------------------
st.subheader("1️⃣ Upload PDF")

uploaded_file = st.file_uploader(
    "Upload a PDF file",
    type=["pdf"]
)

if uploaded_file:
    if st.button("Upload to Backend"):
        with st.spinner("Uploading and indexing PDF..."):
            files = {
                "files": (uploaded_file.name, uploaded_file.getvalue(), "application/pdf")
            }

            response = requests.post(f"{API_URL}/upload", files=files)

            if response.status_code == 200:
                st.success("✅ PDF uploaded and indexed successfully")
            else:
                st.error(response.json().get("detail", "Upload failed"))

# -----------------------------
# Ask Question
# -----------------------------
st.subheader("2️⃣ Ask a Question")

question = st.text_input("Enter your question")

if st.button("Ask"):
    if not question.strip():
        st.warning("Please enter a question")
    else:
        with st.spinner("Generating answer..."):
            payload = {"question": question}
            response = requests.post(f"{API_URL}/ask", json=payload)

            if response.status_code == 200:
                answer = response.json().get("answer", "")
                st.markdown("### 📌 Answer")
                st.write(answer)
            else:
                st.error(response.json().get("detail", "Error generating answer"))
