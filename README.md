# Multi-Modal RAG System

A **production-ready Multi-Modal Retrieval-Augmented Generation (RAG) system** for asking questions over PDF documents using **FastAPI (backend)** and **Streamlit (UI)**.

This system supports **text, tables, and image OCR**, enabling accurate, context-grounded answers using **LangChain + HuggingFace models**.

---
## Multi-Modal RAG System Demo
https://drive.google.com/file/d/1K8hAZq1FWnCfxESvR-c64BDH2BQf0EcM/view?usp=sharing
## ✨ Features

- 📄 Upload multiple PDF documents
- 🧠 Multi-modal ingestion
  - Text
  - Tables
  - Images with OCR
- ✂️ Smart chunking (recursive + semantic)
- 🔍 FAISS-based vector search
- 🎯 Context-grounded answers (no hallucination)
- 📚 Optional citation-aware responses
- ⚡ FastAPI backend with OpenAPI docs
- 💬 Streamlit chat UI
- 🧪 Evaluation & verification suite

---

## 🏗️ Architecture
<img width="585" height="605" alt="image" src="https://github.com/user-attachments/assets/5e4b1e90-29d1-426c-a760-1f7cc8d3b1f6" />

## 📂 Project Structure
<img width="471" height="709" alt="image" src="https://github.com/user-attachments/assets/36684788-902f-4038-aeab-a28c8ad186fa" />
<img width="556" height="315" alt="image" src="https://github.com/user-attachments/assets/145f3879-fc01-4a37-b74e-d0109cbc7749" />




---

## 🧠 Models & Tech Stack

### Language Model
- HuggingFace Seq2Seq (default: `google/flan-t5-base`)

### Embeddings
- `sentence-transformers/all-MiniLM-L6-v2`

### Vector Store
- FAISS (CPU)

### Multi-Modal Processing
- PyMuPDF (PDF + images)
- pdfplumber (tables)
- pytesseract (OCR)
- Pillow (image handling)

---

## 📦 Dependencies (Core)

```txt
fastapi==0.104.1
uvicorn==0.24.0
streamlit==1.36.0
python-dotenv==1.0.1
python-multipart==0.0.6

langchain==0.2.6
langchain-core>=0.2.10,<0.3.0
langchain-community==0.2.6

sentence-transformers==3.0.0
transformers==4.35.2
huggingface-hub>=0.20,<1.0

faiss-cpu
PyMuPDF==1.23.8
pdfplumber==0.10.3
pytesseract==0.3.10
Pillow>=10.3,<11.0
---
### Ingestor

Extracts text from PDF documents and creates chunks (using semantic and character splitter) that are stored in a vector databse

### Retriever

Given a query, searches for similar documents, reranks the result and applies LLM chain filter before returning the response.

### QA Chain

Combines the LLM with the retriever to answer a given user question

## Tech Stack

- [Ollama](https://ollama.com/) - run local LLM
- [Groq API](https://groq.com/) - fast inference for mutliple LLMs
- [LangChain](https://www.langchain.com/) - build LLM-powered apps
- [Qdrant](https://qdrant.tech/) - vector search/database
- [FlashRank](https://github.com/PrithivirajDamodaran/FlashRank) - fast reranking
- [FastEmbed](https://qdrant.github.io/fastembed/) - lightweight and fast embedding generation
- [Streamlit](https://streamlit.io/) - build UI for data apps
- [PDFium](https://pdfium.googlesource.com/pdfium/) - PDF processing and text extraction


## Setup
1️⃣ Create Virtual Environment
python -m venv venv
venv\Scripts\activate

2️⃣ Install Dependencies
pip install -r requirements.txt

3️⃣ Environment Variables
Create .env file:

HUGGINGFACEHUB_API_TOKEN=your_huggingface_token
HUGGING_FACE_MODEL=google/flan-t5-base

(Optional)

GROQ_API_KEY=your_groq_key

🚀 Run the Application
🔹 Start Backend (FastAPI)
python -m uvicorn main:app --host 127.0.0.1 --port 8000

API: http://127.0.0.1:8000
Docs: http://127.0.0.1:8000/docs

🔹 Start UI (Streamlit)
streamlit run app.py

UI: http://localhost:8501

## API Endpoints
Health Check
GET /
GET /health

Upload PDFs
POST /upload

Ask Question
POST /ask

{
  "question": "What are the key financial highlights?"
}

## Testing

Run verification:
python test_implementation.py


## Expected output:
  All tests passed!


