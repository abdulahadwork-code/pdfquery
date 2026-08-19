import os
import sys
import json
import random
import argparse

from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_text_splitters import RecursiveCharacterTextSplitter

from rag_pipeline import load_pdf, GROQ_MODEL

load_dotenv()

GROQ_KEY = os.environ.get("GROQ_API_KEY")

GEN_PROMPT = ChatPromptTemplate.from_template(
    """You are building a test set to evaluate a document Q&A system.

Read the passage below and write ONE question that is fully answerable
using ONLY this passage, plus the correct answer.

Rules:
- The question must NOT be answerable without this passage (avoid generic
  questions like "what is the main topic?").
- The answer must be short, factual, and directly supported by the passage.
- Respond with ONLY valid JSON, no markdown fences, no preamble:
  {{"question": "...", "ground_truth": "..."}}

Passage:
{passage}
"""
)


def parse_json_response(raw):
    raw = raw.strip()

    if raw.startswith("```"):
        raw = raw.strip("`")
        if raw.lower().startswith("json"):
            raw = raw[4:].strip()

    return json.loads(raw)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("pdf_path")
    parser.add_argument(
        "--n", type=int, default=8,
        help="Number of QA pairs to generate (default: 8)"
    )
    parser.add_argument("--out", default="eval_dataset.json")
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for chunk sampling, for reproducible eval sets"
    )
    args = parser.parse_args()

    if not GROQ_KEY:
        sys.exit("GROQ_API_KEY is not set — add it to your .env file.")

    docs = load_pdf(args.pdf_path)
    if not docs:
        sys.exit("No extractable text found in this PDF.")
    chunks = RecursiveCharacterTextSplitter(
        chunk_size=1200, chunk_overlap=100
    ).split_documents(docs)

    random.seed(args.seed)
    n = min(args.n, len(chunks))
    sample = random.sample(chunks, n)

    llm = ChatGroq(
        model=GROQ_MODEL,
        api_key=GROQ_KEY,
        temperature=0,
        reasoning_effort="none",
    )

    gen_chain = GEN_PROMPT | llm | StrOutputParser()

    rows = []
    for i, chunk in enumerate(sample, 1):
        page = chunk.metadata.get("page")
        print(f"[{i}/{n}] generating QA pair from page {page} ...")

        raw = gen_chain.invoke({"passage": chunk.page_content})

        try:
            row = parse_json_response(raw)
            assert "question" in row and "ground_truth" in row
        except Exception:
            print(f"  skipped — model did not return valid JSON: {raw[:120]!r}")
            continue

        row["source_page"] = page
        rows.append(row)

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2, ensure_ascii=False)

    print(f"\nWrote {len(rows)} QA pairs to {args.out}")
    print("Review them before trusting eval scores — auto-generated ground truths can be wrong.")


if __name__ == "__main__":
    main()
