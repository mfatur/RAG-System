import pandas as pd
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings


# 1. Load data
df = pd.read_csv(
    "data/clean_tickets.csv",
    keep_default_na=False
)

df = df.replace(["nan", "NaN", "None"], "")


# 2. Buat document
documents = []

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


# 3. Chunking
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=400,
    chunk_overlap=50
)

chunks = text_splitter.create_documents(documents)

print("Jumlah chunks:", len(chunks))


# 4. Load embedding model
embeddings = HuggingFaceEmbeddings(
    model_name="BAAI/bge-small-en-v1.5"
)


# 5. Ubah 1 chunk menjadi vector
vector = embeddings.embed_query(
    chunks[0].page_content
)


# 6. Lihat hasil
print("Panjang vector:", len(vector))
print("5 angka pertama:", vector[:5])