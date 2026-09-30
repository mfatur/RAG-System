import csv
import json
import sys
from pathlib import Path

# Tambahkan folder src dari project RAG
rag_project = Path("D:/RAG-llm/rag-langchain")
sys.path.append(str(rag_project / "src"))

from rag import run_rag


# Lokasi dataset
dataset_path = Path(
    "D:/RAG-llm/ragas-evaluation/dataset/evaluation_dataset.csv"
)

# Lokasi output
output_path = Path(
    "D:/RAG-llm/ragas-evaluation/results/rag_results.json"
)


results = []


with open(dataset_path, "r", encoding="utf-8") as file:

    reader = csv.DictReader(file)

    for i, row in enumerate(reader, start=1):

        question = row["question"]
        ground_truth = row["ground_truth"]

        print(f"\nProcessing {i}/15...")
        print(f"Question: {question}")

        rag_result = run_rag(question)

        results.append({
            "question": question,
            "ground_truth": ground_truth,
            "answer": rag_result["answer"],
            "contexts": rag_result["contexts"]
        })


with open(
    output_path,
    "w",
    encoding="utf-8"
) as file:

    json.dump(
        results,
        file,
        indent=2,
        ensure_ascii=False
    )


print("\n================================")
print("RAG evaluation dataset berhasil dibuat.")
print("================================")
print(f"Output: {output_path}")