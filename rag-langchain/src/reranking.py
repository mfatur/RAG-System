from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from sentence_transformers import CrossEncoder


# ==========================================
# 1. Load embedding model
# ==========================================

embeddings = HuggingFaceEmbeddings(
    model_name="BAAI/bge-small-en-v1.5"
)


# ==========================================
# 2. Load Chroma
# ==========================================

db = Chroma(
    persist_directory="chroma_db",
    embedding_function=embeddings
)


# ==========================================
# 3. Query
# ==========================================

query = "I have an issue with product setup. What should I do?"


# ==========================================
# 4. Ambil Top 5 dari Chroma
# ==========================================

results = db.similarity_search(
    query,
    k=5
)


# ==========================================
# 5. Load CrossEncoder
# ==========================================

reranker = CrossEncoder(
    "cross-encoder/ms-marco-MiniLM-L-6-v2"
)


# ==========================================
# 6. Buat pasangan Query + Document
# ==========================================

pairs = [
    [query, document.page_content]
    for document in results
]


# ==========================================
# 7. Hitung relevance score
# ==========================================

scores = reranker.predict(pairs)


# ==========================================
# 8. Gabungkan document + score
# ==========================================

ranked_results = list(zip(results, scores))


# ==========================================
# 9. Urutkan dari score tertinggi
# ==========================================

ranked_results.sort(
    key=lambda x: x[1],
    reverse=True
)


# ==========================================
# 10. Tampilkan hasil
# ==========================================

print("\n=== RERANKING RESULTS ===")

for i, (document, score) in enumerate(
    ranked_results,
    start=1
):

    print(f"\n--- RESULT {i} ---")
    print(f"Score: {score:.4f}")
    print(document.page_content)