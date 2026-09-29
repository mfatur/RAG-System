import pandas as pd
from pathlib import Path

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma


PROJECT_DIR = Path(__file__).resolve().parents[1]
DATA_FILE = PROJECT_DIR / "data" / "clean_tickets.csv"
DB_DIR = PROJECT_DIR / "chroma_db"
COLLECTION_NAME = "support_tickets_clean"


# ==========================================
# 1. Load cleaned data
# ==========================================

df = pd.read_csv(
    DATA_FILE,
    keep_default_na=False
)

df = df.replace(["nan", "NaN", "None"], "")


# ==========================================
# 2. Buat documents
# ==========================================

documents = []
metadatas = []

for _, row in df.iterrows():

    text = f"""
Ticket ID: {row['Ticket ID']}
Product: {row['Product Purchased']}
Ticket Type: {row['Ticket Type']}
Subject: {row['Ticket Subject']}
Description: {row['Ticket Description']}
Status: {row['Ticket Status']}
Resolution: {row['Resolution']}
Priority: {row['Ticket Priority']}
Channel: {row['Ticket Channel']}
"""

    documents.append(text.strip())
    metadatas.append({
        "ticket_id": str(row["Ticket ID"]),
        "product": str(row["Product Purchased"]),
        "subject": str(row["Ticket Subject"]),
        "status": str(row["Ticket Status"]),
        "ticket_type": str(row["Ticket Type"]),
        "priority": str(row["Ticket Priority"]),
        "channel": str(row["Ticket Channel"]),
    })

print("Jumlah documents:", len(documents))


# ==========================================
# 3. Chunking
# ==========================================

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=400,
    chunk_overlap=50
)

chunks = text_splitter.create_documents(
    documents,
    metadatas=metadatas
)

print("Jumlah chunks:", len(chunks))


# ==========================================
# 4. Load embedding model
# ==========================================

embeddings = HuggingFaceEmbeddings(
    model_name="BAAI/bge-small-en-v1.5"
)

existing_db = Chroma(
    collection_name=COLLECTION_NAME,
    persist_directory=str(DB_DIR),
    embedding_function=embeddings,
)
existing_db.delete_collection()


# ==========================================
# 5. Buat Chroma
# ==========================================

db = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    persist_directory=str(DB_DIR),
    collection_name=COLLECTION_NAME,
)


print("Chroma berhasil dibuat.")
print(f"Data disimpan di folder: {DB_DIR}")