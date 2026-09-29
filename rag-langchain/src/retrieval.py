from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma


# ==========================================
# 1. Load embedding model
# ==========================================

embeddings = HuggingFaceEmbeddings(
    model_name="BAAI/bge-small-en-v1.5"
)


# ==========================================
# 2. Load Chroma yang sudah dibuat
# ==========================================

db = Chroma(
    persist_directory="chroma_db",
    embedding_function=embeddings
)


# ==========================================
# 3. Pertanyaan user
# ==========================================

query = "I have an issue with product setup. What should I do?"


# ==========================================
# 4. Cari 5 document paling relevan
# ==========================================

results = db.similarity_search(
    query,
    k=5
)


# ==========================================
# 5. Tampilkan hasil
# ==========================================

print("\n=== RETRIEVAL RESULTS ===")

for i, document in enumerate(results, start=1):

    print(f"\n--- RESULT {i} ---")
    print(document.page_content)