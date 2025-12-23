import re
from typing import List

from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.language_models import BaseLanguageModel
from langchain_core.vectorstores import VectorStoreRetriever

from ragbase.model import create_llm   # ✅ USE EXISTING FUNCTION
from ragbase.retriever import build_retriever


# ------------------------
# SYSTEM PROMPT
# ------------------------
SYSTEM_PROMPT = """
You are a document-based assistant.

Rules:
1. Answer ONLY using the provided context.
2. If the answer is not present, say:
   "The document does not contain enough information to answer this question."
3. Keep the answer concise (max 3 sentences).
4. Do not use outside knowledge.

Context:
{context}
"""


# ------------------------
# Helpers
# ------------------------
def clean_text(text: str) -> str:
    return re.sub(r"https?://\S+|www\.\S+", "", text)


def format_documents(docs: List[Document], max_chars: int = 3000) -> str:
    context = ""

    for doc in docs:
        content = clean_text(doc.page_content)
        if len(context) + len(content) > max_chars:
            break
        context += content + "\n---\n"

    return context.strip()


# ------------------------
# Build QA Chain
# ------------------------
def build_qa_chain(
    pdf_paths: List[str],
    use_multimodal: bool = False,
    use_citations: bool = False,
):
    # ✅ Create LLM using your config logic (Ollama / HF / Groq)
    llm: BaseLanguageModel = create_llm()

    # ✅ Build retriever
    retriever: VectorStoreRetriever = build_retriever(
        pdf_paths=pdf_paths,
        top_k=3,
        use_multimodal=use_multimodal,
    )

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", SYSTEM_PROMPT),
            ("human", "{question}"),
        ]
    )

    # ✅ Deterministic, string-output chain
    chain = (
        RunnablePassthrough.assign(
            context=lambda x: format_documents(
                retriever.invoke(x["question"])
            )
        )
        | prompt
        | llm
        | StrOutputParser()   # 🔥 RETURNS STRING
    )

    return chain
