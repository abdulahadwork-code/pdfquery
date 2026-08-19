import os
import sys
import json
import argparse

from dotenv import load_dotenv

from langchain_groq import ChatGroq
from langchain_community.embeddings import FastEmbedEmbeddings

from ragas import evaluate, EvaluationDataset
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.metrics import (
    Faithfulness,
    ResponseRelevancy,
    LLMContextPrecisionWithReference,
    LLMContextRecall,
)

from rag_pipeline import build_chain, strip_think, GROQ_MODEL

load_dotenv()

GROQ_KEY = os.environ.get("GROQ_API_KEY")
EVAL_MODEL = os.environ.get("GROQ_EVAL_MODEL", GROQ_MODEL)
NON_METRIC_COLUMNS = {"user_input", "retrieved_contexts", "response", "reference"}

def load_eval_set(path):
    with open(path, "r", encoding="utf-8") as f:
        rows = json.load(f)

    if not rows:
        raise ValueError(f"{path} contains no questions.")

    for i, row in enumerate(rows):
        if "question" not in row or "ground_truth" not in row:
            raise ValueError(
                f"Row {i} in {path} must have 'question' and 'ground_truth' keys."
            )

    return rows

def run_chain_on_eval_set(chain, eval_rows):
    """Runs each question through the chain and assembles a ragas-ready dataset."""

    records = []

    for i, row in enumerate(eval_rows, 1):
        question = row["question"]
        reference = row["ground_truth"]

        print(f"[{i}/{len(eval_rows)}] {question}")

        result = chain.invoke(question)

        answer = strip_think(result["answer"])
        contexts = [doc.page_content for doc in result["context"]]

        records.append(
            {
                "user_input": question,
                "retrieved_contexts": contexts,
                "response": answer,
                "reference": reference,
            }
        )

    return records

def main():
    parser = argparse.ArgumentParser(description="RAGAS eval harness for PDFQUERY")
    parser.add_argument("pdf_path", help="Path to the PDF to index and evaluate against")
    parser.add_argument(
        "eval_set_path",
        nargs="?",
        default="eval_dataset.json",
        help="JSON file of {question, ground_truth} pairs (default: eval_dataset.json)",
    )
    parser.add_argument(
        "--out",
        default="eval_results.csv",
        help="Where to write the per-question results CSV (default: eval_results.csv)",
    )
    args = parser.parse_args()

    if not GROQ_KEY:
        sys.exit("GROQ_API_KEY is not set — add it to your .env file.")

    eval_rows = load_eval_set(args.eval_set_path)

    print(f"Building RAG chain for {args.pdf_path} ...")
    chain, vectorstore = build_chain(args.pdf_path)

    try:
        print(f"Running {len(eval_rows)} questions through the chain ...")
        records = run_chain_on_eval_set(chain, eval_rows)

        dataset = EvaluationDataset.from_list(records)

        evaluator_llm = LangchainLLMWrapper(
            ChatGroq(
                model=GROQ_MODEL,
                api_key=GROQ_KEY,
                temperature=0,
                reasoning_effort="none",
            )
        )
        evaluator_embeddings = LangchainEmbeddingsWrapper(FastEmbedEmbeddings())

        metrics = [
            Faithfulness(llm=evaluator_llm),
            ResponseRelevancy(llm=evaluator_llm, embeddings=evaluator_embeddings),
            LLMContextPrecisionWithReference(llm=evaluator_llm),
            LLMContextRecall(llm=evaluator_llm),
        ]

        print("Scoring with RAGAS (this calls the LLM once per metric per question)...")
        result = evaluate(dataset=dataset, metrics=metrics)
        df = result.to_pandas()
        df.to_csv(args.out, index=False)
        metric_cols = [c for c in df.columns if c not in NON_METRIC_COLUMNS]
        print("\n=== Per-question scores ===")
        with_pd_options(df, metric_cols)
        print("\n=== Averages ===")
    
        for col in metric_cols:
            print(f"{col:45s} {df[col].mean():.3f}")

        print(f"\nFull results (incl. contexts + answers) written to {args.out}")
    finally:
        try:
            vectorstore.delete_collection()
        except Exception:
            pass

def with_pd_options(df, metric_cols):
    import pandas as pd

    with pd.option_context("display.max_colwidth", 60, "display.width", 160):
        print(df[["user_input"] + metric_cols].to_string(index=False))

if __name__ == "__main__":
    main()
