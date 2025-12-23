from typing import List

from langchain_core.documents import Document
from langchain_core.vectorstores import VectorStoreRetriever
from langchain_community.vectorstores import FAISS
from ragbase.ingestor import ingest_documents
from ragbase.model import create_embeddings


class MultiModalRetriever(VectorStoreRetriever):
    """
    Retriever that prioritizes modalities:
    text > table > image_ocr
    """

    def invoke(self, query, config=None):
        results = super().invoke(query, config)

        modality_groups = {}
        for doc in results:
            modality = doc.metadata.get("modality", "text")
            modality_groups.setdefault(modality, []).append(doc)

        ordered = []
        for m in ["text", "table", "image_ocr"]:
            ordered.extend(modality_groups.get(m, []))

        # any remaining modalities
        for m, docs in modality_groups.items():
            if m not in ["text", "table", "image_ocr"]:
                ordered.extend(docs)

        return ordered


def build_retriever(
    pdf_paths: List[str],
    top_k: int = 5,
    use_multimodal: bool = True,
) -> VectorStoreRetriever:
    """
    Build a fast FAISS-based retriever from PDFs.
    """

    # 1️⃣ Ingest documents
    documents: List[Document] = ingest_documents(
        pdf_paths,
        use_multimodal=use_multimodal
    )

    # 2️⃣ Embeddings
    embeddings = create_embeddings()

    # 3️⃣ Vector store
    vectorstore = FAISS.from_documents(documents, embeddings)

    # 4️⃣ Retriever
    if use_multimodal:
        return MultiModalRetriever(
            vectorstore=vectorstore,
            search_type="similarity",
            search_kwargs={"k": top_k},
        )

    return vectorstore.as_retriever(search_kwargs={"k": top_k})
