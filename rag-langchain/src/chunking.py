import pandas as pd
from langchain_text_splitters import RecursiveCharacterTextSplitter


# 1. Load clean dataset
df = pd.read_csv(
    "data/clean_tickets.csv",
    keep_default_na=False
)

# Pastikan nilai kosong tidak menjadi teks "nan"
df = df.replace(["nan", "NaN", "None"], "")


# 2. Ubah setiap ticket menjadi satu teks
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


print("Jumlah document:", len(documents))


# 3. Buat text splitter
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=400,
    chunk_overlap=50
)


# 4. Pecah document menjadi chunks
chunks = text_splitter.create_documents(documents)


print("Jumlah chunks:", len(chunks))


# 5. Tampilkan beberapa contoh chunk
print("\n=== CHUNK 1 ===")
print(chunks[0].page_content)

print("\n=== CHUNK 2 ===")
print(chunks[1].page_content)