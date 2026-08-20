import os
import uuid

from pypdf import PdfReader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import FastEmbedEmbeddings
from langchain_community.retrievers import BM25Retriever
from langchain_chroma import Chroma
from langchain_groq import ChatGroq
from langchain_classic.retrievers import EnsembleRetriever, ContextualCompressionRetriever
from langchain_classic.retrievers.document_compressors import FlashrankRerank
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableParallel, RunnableLambda
from langchain_core.output_parsers import StrOutputParser

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

    embeddings = FastEmbedEmbeddings()

    collection_name = f"pdfquery_{uuid.uuid4().hex}"

    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        collection_name=collection_name
    )

    bm25_retriever = BM25Retriever.from_documents(chunks)
    bm25_retriever.k = 8

    dense_retriever = vectorstore.as_retriever(
        search_kwargs={"k": 8}
    )

    hybrid_retriever = EnsembleRetriever(
        retrievers=[bm25_retriever, dense_retriever],
        weights=[0.4, 0.6],  
    )
    compressor = FlashrankRerank(model="ms-marco-MiniLM-L-12-v2", top_n=4)

    retriever = ContextualCompressionRetriever(
        base_compressor=compressor,
        base_retriever=hybrid_retriever,
    )

    llm = ChatGroq(
        model="qwen/qwen3.6-27b",
        api_key=GROQ_KEY,
        temperature=0,

        reasoning_effort="none",
    )

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