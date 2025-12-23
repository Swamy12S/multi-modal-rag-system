from typing import List
from langchain_core.documents import Document
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter


def ingest_documents(
    pdf_paths: List[str],
    use_multimodal: bool = False,
) -> List[Document]:
    """
    Load PDFs and split into text chunks.
    (Text-only for MVP, fast and stable)
    """

    documents: List[Document] = []

    for pdf_path in pdf_paths:
        loader = PyPDFLoader(pdf_path)
        pages = loader.load()

        for page in pages[:10]:  # ⚡ limit pages for speed
            page.metadata["modality"] = "text"
            documents.append(page)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=100,
    )

    return splitter.split_documents(documents)
