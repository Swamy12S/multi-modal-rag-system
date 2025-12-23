# ragbase/config.py
import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    # HuggingFace
    HUGGINGFACE_API_TOKEN = os.getenv("HUGGINGFACEHUB_API_TOKEN")

    # Models
    LLM_MODEL = os.getenv("HUGGING_FACE_MODEL", "google/flan-t5-base")
    EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

    # RAG
    CHUNK_SIZE = 1000
    CHUNK_OVERLAP = 200

    # Paths
    UPLOAD_DIR = "tmp"
