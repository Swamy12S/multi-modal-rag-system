from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List
import tempfile
import os

from dotenv import load_dotenv

# IMPORTANT: this must exist in ragbase.chain
from ragbase.chain import build_qa_chain

# ------------------------
# Load environment variables
# ------------------------
load_dotenv()

# ------------------------
# FastAPI App
# ------------------------
app = FastAPI(
    title="Multi-Modal RAG API",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ------------------------
# Global QA chain
# ------------------------
qa_chain = None


# ------------------------
# Request Models
# ------------------------
class QuestionRequest(BaseModel):
    question: str


# ------------------------
# Health Check
# ------------------------
@app.get("/")
@app.get("/health")
def health():
    return {
        "status": "ok",
        "message": "API is running"
    }


# ------------------------
# Upload PDFs
# ------------------------
@app.post("/upload")
async def upload_pdfs(files: List[UploadFile] = File(...)):
    global qa_chain

    if not files:
        raise HTTPException(status_code=400, detail="No files uploaded")

    pdf_paths = []

    for file in files:
        if not file.filename.lower().endswith(".pdf"):
            raise HTTPException(
                status_code=400,
                detail="Only PDF files are allowed"
            )

        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            content = await file.read()
            tmp.write(content)
            pdf_paths.append(tmp.name)

    try:
        # Build QA chain AFTER PDFs are uploaded
        qa_chain = build_qa_chain(
            pdf_paths=pdf_paths,
            use_multimodal=False,   # faster
            use_citations=False
        )

        return {
            "message": f"{len(pdf_paths)} PDF(s) uploaded and indexed successfully"
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ------------------------
# Ask Question
# ------------------------
@app.post("/ask")
async def ask_question(payload: QuestionRequest):
    global qa_chain

    if qa_chain is None:
        raise HTTPException(
            status_code=400,
            detail="Upload PDFs before asking questions"
        )

    question = payload.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="Question cannot be empty")

    try:
        result = qa_chain.invoke({"question": question})

        # ✅ FIX: handle both AIMessage and string outputs
        if hasattr(result, "content"):
            answer = result.content
        else:
            answer = str(result)

        return {
            "question": question,
            "answer": answer
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
