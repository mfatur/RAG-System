import json
from pathlib import Path

from datasets import Dataset
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from ragas.run_config import RunConfig
from ragas import evaluate
from ragas.metrics import (
    answer_relevancy,
    context_precision,
    context_recall,
    faithfulness,
)


PROJECT_DIR = Path(__file__).resolve().parents[2]
EVALUATION_DIR = PROJECT_DIR / "ragas-evaluation"
RESULTS_FILE = EVALUATION_DIR / "results" / "rag_results.json"
SCORES_FILE = EVALUATION_DIR / "results" / "ragas_scores.csv"

load_dotenv(PROJECT_DIR / "rag-langchain" / ".env")

with RESULTS_FILE.open("r", encoding="utf-8") as file:
    rows = json.load(file)

dataset = Dataset.from_list(rows)

llm = ChatGroq(
    model="openai/gpt-oss-120b",
    temperature=0,
    max_tokens=2048,
)

embeddings = HuggingFaceEmbeddings(
    model_name="BAAI/bge-small-en-v1.5",
)
answer_relevancy.strictness = 1
result = evaluate(
    dataset=dataset,
    metrics=[
        faithfulness,
        answer_relevancy,
        context_precision,
        context_recall,
    ],
    llm=llm,
    embeddings=embeddings,
    column_map={
        "user_input": "question",
        "response": "answer",
        "retrieved_contexts": "contexts",
        "reference": "ground_truth",
    },
    run_config=RunConfig(
        max_workers=1,
        max_retries=1,
        max_wait=60,
        timeout=300,
    ),
    raise_exceptions=True,
)

scores = result.to_pandas()
scores.to_csv(SCORES_FILE, index=False)

print("\n=== RAGAS SCORES ===")
print(scores.mean(numeric_only=True).to_string())
print(f"\nDetail scores saved to: {SCORES_FILE}")