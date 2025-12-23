# ragbase/model.py

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import HuggingFacePipeline
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
from ragbase.config import Config


def create_embeddings():
    """
    Creates sentence-transformer embeddings
    """
    return HuggingFaceEmbeddings(
        model_name=Config.EMBEDDING_MODEL
    )


def create_llm():
    """
    Creates HuggingFace LLM pipeline
    """
    tokenizer = AutoTokenizer.from_pretrained(Config.LLM_MODEL)
    model = AutoModelForSeq2SeqLM.from_pretrained(Config.LLM_MODEL)

    hf_pipeline = pipeline(
        "text2text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=512,
        temperature=0.3
    )

    return HuggingFacePipeline(pipeline=hf_pipeline)
