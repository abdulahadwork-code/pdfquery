"""
Core PDFQUERY RAG pipeline: PDF loading, hybrid retrieval + reranking, and
the QA chain. Pulled out of app.py so the Streamlit UI and the evaluation
harness (evaluate.py) both build the pipeline from one place — an eval
report is only meaningful if it's testing the same code path users hit.
"""

import os
import uuid

from pypdf import PdfReader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import FastEmbedEmbeddings
from langchain_community.retrievers import BM25Retriever
from langchain_chroma import Chroma
from langchain_groq import ChatGroq
# NOTE: LangChain 1.0 split these out of the main `langchain` package into
# `langchain-classic` (pip install langchain-classic). If you're on an older
# langchain (<1.0), use `from langchain.retrievers import ...` instead.
from langchain_classic.retrievers import EnsembleRetriever, ContextualCompressionRetriever
from langchain_classic.retrievers.document_compressors import FlashrankRerank
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableParallel, RunnableLambda
from langchain_core.output_parsers import StrOutputParser

# Pydantic forward-ref issue seen on some langchain/flashrank version pairs —
# harmless no-op when not needed, so it's safe to always call this.
FlashrankRerank.model_rebuild()

GROQ_KEY = os.environ.get("GROQ_API_KEY")

THINK_END = "</" + "think>"


def strip_think(text):
    """
    Qwen3 can still emit a stray <think>...</think> block even with
    reasoning_effort="none" on some model/version combos. Strip it
    defensively so it never leaks into the UI.
    """
    if THINK_END in text:
        return text.split(THINK_END)[-1].strip()
    return text


RAG_PROMPT = ChatPromptTemplate.from_template(
    """Answer using ONLY the provided context.

Answer ONLY what the question asks for. The context may contain extra topics
(qualifications, benefits, company info, etc.) — deliberately EXCLUDE anything
the question does not ask about, even if it appears in the context.

If the answer is not present in the context, respond exactly with:

"The provided document does not contain this information."

Context:
{context}

Question:
{question}
"""
)


def load_pdf(pdf_path):

    reader = PdfReader(pdf_path)

    docs = [
        Document(
            page_content=page.extract_text() or "",
            metadata={
                "source": os.path.basename(pdf_path),
                "page": i + 1
            },
        )
        for i, page in enumerate(reader.pages)
    ]

    # Scanned / image-only pages return "" from extract_text(). Indexing
    # empty documents pollutes the vector store and wastes embedding calls,
    # so drop them here.
    docs = [d for d in docs if d.page_content.strip()]

    return docs


def format_docs(docs):
    return "\n\n".join(
        doc.page_content
        for doc in docs
    )


def build_chain(pdf_path):
    """
    Build a completely new RAG pipeline for the given PDF.

    The vector store is intentionally NOT cached because every uploaded
    document should have its own fresh embeddings and collection.

    Returns (chain, vectorstore). chain.invoke(question) returns a dict:
        {"question": ..., "context": [Document, ...], "answer": "..."}
    """

    docs = load_pdf(pdf_path)

    if not docs:
        raise ValueError(
            "No extractable text found in this PDF. It may be scanned or "
            "image-only and needs OCR before it can be indexed."
        )

    chunks = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    ).split_documents(docs)

    # OllamaEmbeddings requires a locally running Ollama server, which won't
    # exist on most deployment targets (Streamlit Cloud, etc.) and will hang
    # or crash there. FastEmbed runs a local ONNX model with no external
    # service required, so it works the same in dev and prod.
    embeddings = FastEmbedEmbeddings()

    collection_name = f"pdfquery_{uuid.uuid4().hex}"

    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        collection_name=collection_name
    )

    # --- Hybrid retrieval ---------------------------------------------
    # Dense (embedding) search alone misses exact keyword/number/name matches
    # that a user's question quotes verbatim from the PDF (e.g. clause
    # numbers, product codes, proper nouns). BM25 is a sparse, term-frequency
    # retriever that's strong exactly where dense search is weak, so we run
    # both and merge results with EnsembleRetriever (Reciprocal Rank Fusion).
    # Each side pulls a wider candidate pool (k=8) than we'll actually use,
    # since the reranker below narrows it back down to the best 4.
    bm25_retriever = BM25Retriever.from_documents(chunks)
    bm25_retriever.k = 8

    dense_retriever = vectorstore.as_retriever(
        search_kwargs={"k": 8}
    )

    hybrid_retriever = EnsembleRetriever(
        retrievers=[bm25_retriever, dense_retriever],
        weights=[0.4, 0.6],  # favor semantic match slightly, keyword still counts
    )

    # --- Reranking -------------------------------------------------------
    # EnsembleRetriever's fusion score is a cheap heuristic (rank position,
    # not relevance). A cross-encoder reranker actually reads the query
    # against each candidate and scores true relevance. FlashRank runs a
    # small ONNX cross-encoder locally — no GPU, no API key, no torch — so
    # it stays deployment-friendly.
    compressor = FlashrankRerank(model="ms-marco-MiniLM-L-12-v2", top_n=4)

    retriever = ContextualCompressionRetriever(
        base_compressor=compressor,
        base_retriever=hybrid_retriever,
    )

    llm = ChatGroq(
        model="qwen/qwen3.6-27b",
        api_key=GROQ_KEY,
        temperature=0,
        # ChatGroq validates this as an explicit field now — passing it
        # inside model_kwargs (as before) raises a pydantic validation
        # error asking for exactly this.
        reasoning_effort="none",
    )

    # Single retrieval per question — returns both the answer and the
    # source docs together instead of invoking the retriever twice.
    chain = (
        RunnableParallel(
            question=RunnablePassthrough(),
            context=retriever,
        )
        | RunnablePassthrough.assign(
            answer=(
                RunnableLambda(
                    lambda x: {
                        "context": format_docs(x["context"]),
                        "question": x["question"],
                    }
                )
                | RAG_PROMPT
                | llm
                | StrOutputParser()
            )
        )
    )

    return chain, vectorstore