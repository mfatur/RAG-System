"""
Build an instruction dataset (chat format) for LoRA fine-tuning a small model
to act as the generator of the ticket RAG system.

Run from anywhere:
    python lora-finetuning/build_sft_dataset.py

Expected location of this file: RAG-llm/lora-finetuning/build_sft_dataset.py
Output: lora-finetuning/data/train.jsonl and test.jsonl

Design notes:
- Questions are generated from ticket METADATA (status, priority, channel, ...),
  because the 'Resolution' column of this synthetic dataset is random text.
- Split is done by Ticket ID, so no ticket appears in both train and test.
- Tickets that appear in the RAGAS evaluation set are excluded from both,
  so that evaluation set stays untouched.
- Out-of-scope questions teach the model to use the fallback answer.
"""

import json
import random
import re
from pathlib import Path

import pandas as pd

# =========================
# Config
# =========================

ROOT = Path(__file__).resolve().parents[1]
CSV_PATH = ROOT / "rag-langchain" / "data" / "clean_tickets.csv"
EVAL_PATH = ROOT / "ragas-evaluation" / "dataset" / "evaluation_dataset.csv"
OUT_DIR = Path(__file__).resolve().parent / "data"

SEED = 42
N_TRAIN_TICKETS = 1500
N_TEST_TICKETS = 200
LOOKUPS_PER_TICKET = 2
P_OUT_OF_SCOPE = 0.3      # chance to add one out-of-scope question per ticket
CHUNK_CHARS = 400         # mimic the 400-character chunks used in the RAG

FALLBACK = "I don't have enough information to answer that."

SYSTEM_PROMPT = (
    "You are a customer support assistant.\n"
    "Answer the user's question based only on the provided context.\n"
    f'If the answer cannot be found in the context, say: "{FALLBACK}"\n'
    "Do not make up information.\n"
    "Answer in one concise, complete sentence.\n"
    "For ticket lookup questions, include the ticket ID and the requested field."
)

# field key -> (column name, question templates, answer template)
LOOKUP_FIELDS = {
    "status": (
        "Ticket Status",
        [
            "What is the status of ticket {tid}?",
            "What's the current status of ticket {tid}?",
            "Can you tell me the status of ticket {tid}?",
            "Ticket {tid}: what is its status?",
        ],
        "The status of ticket {tid} is {value}.",
    ),
    "priority": (
        "Ticket Priority",
        [
            "What is the priority of ticket {tid}?",
            "What priority level does ticket {tid} have?",
            "Can you tell me the priority of ticket {tid}?",
        ],
        "The priority of ticket {tid} is {value}.",
    ),
    "channel": (
        "Ticket Channel",
        [
            "What channel was used for ticket {tid}?",
            "Which channel was ticket {tid} submitted through?",
            "Through which channel did ticket {tid} come in?",
        ],
        "Ticket {tid} was submitted through {value}.",
    ),
    "type": (
        "Ticket Type",
        [
            "What type of issue is ticket {tid}?",
            "What is the ticket type of ticket {tid}?",
            "How is ticket {tid} classified by type?",
        ],
        "The ticket type of ticket {tid} is {value}.",
    ),
    "product": (
        "Product Purchased",
        [
            "What product is associated with ticket {tid}?",
            "Which product is ticket {tid} about?",
            "What product does ticket {tid} concern?",
        ],
        "The product associated with ticket {tid} is {value}.",
    ),
    "subject": (
        "Ticket Subject",
        [
            "What is the subject of ticket {tid}?",
            "What subject does ticket {tid} have?",
            "Can you tell me the subject of ticket {tid}?",
        ],
        "The subject of ticket {tid} is {value}.",
    ),
}

# Information that does NOT exist in the data -> fallback answer expected
OUT_OF_SCOPE_QUESTIONS = [
    "What is the warranty period for the {product} mentioned in ticket {tid}?",
    "What is the name of the customer service agent handling ticket {tid}?",
    "How do you perform a hard reset on the {product} in ticket {tid}?",
    "What is the customer's phone number in ticket {tid}?",
    "What is the refund amount for ticket {tid}?",
    "What is the shipping tracking number for ticket {tid}?",
    "When was the {product} in ticket {tid} purchased?",
    "What is the serial number of the {product} in ticket {tid}?",
    "How long will it take to resolve ticket {tid}?",
    "What is the home address of the customer in ticket {tid}?",
]


# =========================
# Helpers
# =========================

def build_context(row):
    """Mimic the context format produced by rag.py (header + ticket details)."""
    header = (
        f"Ticket ID: {row['Ticket ID']}\n"
        f"Product: {row['Product Purchased']}\n"
        f"Subject: {row['Ticket Subject']}\n"
        f"Status: {row['Ticket Status']}\n"
        f"Ticket Type: {row['Ticket Type']}\n"
        f"Priority: {row['Ticket Priority']}\n"
        f"Channel: {row['Ticket Channel']}\n"
        "Ticket details:\n"
    )
    details = (
        f"Ticket ID: {row['Ticket ID']}\n"
        f"Product: {row['Product Purchased']}\n"
        f"Ticket Type: {row['Ticket Type']}\n"
        f"Subject: {row['Ticket Subject']}\n"
        f"Description: {row['Ticket Description']}"
    )
    return header + details[:CHUNK_CHARS]


def make_example(row, question, answer, kind, field):
    user_msg = f"Context:\n{build_context(row)}\n\nQuestion:\n{question}\n\nAnswer:"
    return {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_msg},
            {"role": "assistant", "content": answer},
        ],
        "meta": {
            "ticket_id": str(row["Ticket ID"]),
            "kind": kind,
            "field": field,
            "expected": answer,
        },
    }


def examples_for_ticket(row, rng):
    tid = row["Ticket ID"]
    examples = []

    for field in rng.sample(list(LOOKUP_FIELDS), k=LOOKUPS_PER_TICKET):
        column, questions, answer_tpl = LOOKUP_FIELDS[field]
        question = rng.choice(questions).format(tid=tid)
        answer = answer_tpl.format(tid=tid, value=row[column])
        examples.append(make_example(row, question, answer, "lookup", field))

    if rng.random() < P_OUT_OF_SCOPE:
        question = rng.choice(OUT_OF_SCOPE_QUESTIONS).format(
            tid=tid, product=row["Product Purchased"]
        )
        examples.append(make_example(row, question, FALLBACK, "out_of_scope", "none"))

    return examples


def load_eval_ticket_ids():
    """Ticket IDs used in the RAGAS evaluation set (to keep them out of training)."""
    if not EVAL_PATH.exists():
        print(f"[warn] {EVAL_PATH} not found; no evaluation tickets excluded.")
        return set()
    eval_df = pd.read_csv(EVAL_PATH, dtype=str).fillna("")
    text = " ".join(eval_df.astype(str).agg(" ".join, axis=1))
    ids = set(re.findall(r"\bticket\s*#?\s*(\d+)\b", text, flags=re.IGNORECASE))
    print(f"Excluding {len(ids)} tickets used in the RAGAS evaluation set.")
    return ids


def write_jsonl(path, rows):
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


# =========================
# Main
# =========================

def main():
    rng = random.Random(SEED)

    df = pd.read_csv(CSV_PATH, dtype=str).fillna("")
    df["Ticket ID"] = df["Ticket ID"].str.strip()
    df = df.drop_duplicates(subset="Ticket ID")

    eval_ids = load_eval_ticket_ids()
    df = df[~df["Ticket ID"].isin(eval_ids)]

    ids = df["Ticket ID"].tolist()
    rng.shuffle(ids)
    test_ids = set(ids[:N_TEST_TICKETS])
    train_ids = set(ids[N_TEST_TICKETS:N_TEST_TICKETS + N_TRAIN_TICKETS])
    assert not (train_ids & test_ids), "train/test overlap!"

    train, test = [], []
    for _, row in df.iterrows():
        if row["Ticket ID"] in train_ids:
            train += examples_for_ticket(row, rng)
        elif row["Ticket ID"] in test_ids:
            test += examples_for_ticket(row, rng)

    rng.shuffle(train)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    write_jsonl(OUT_DIR / "train.jsonl", train)
    write_jsonl(OUT_DIR / "test.jsonl", test)

    for name, rows in [("train", train), ("test", test)]:
        oos = sum(r["meta"]["kind"] == "out_of_scope" for r in rows)
        print(f"{name}: {len(rows)} examples ({oos} out-of-scope, {len(rows) - oos} lookup)")

    print("\n=== SAMPLE ===")
    sample = train[0]["messages"]
    print(sample[1]["content"])
    print("\n-> ", sample[2]["content"])


if __name__ == "__main__":
    main()