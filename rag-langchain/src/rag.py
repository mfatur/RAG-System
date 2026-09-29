import re
from pathlib import Path

from dotenv import load_dotenv

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.prompts import PromptTemplate
from langchain_groq import ChatGroq

from sentence_transformers import CrossEncoder


PROJECT_DIR = Path(__file__).resolve().parents[1]
DB_DIR = PROJECT_DIR / "chroma_db"
COLLECTION_NAME = "support_tickets_clean"

load_dotenv(PROJECT_DIR / ".env")


# =========================
# 1. Load models
# =========================

embeddings = HuggingFaceEmbeddings(
    model_name="BAAI/bge-small-en-v1.5"
)

reranker = CrossEncoder(
    "cross-encoder/ms-marco-MiniLM-L-6-v2"
)

llm = ChatGroq(
    model="qwen/qwen3.8-27b",
    temperature=0,
    max_tokens=2048
)


# =========================
# 2. Load Chroma
# =========================

db = Chroma(
    collection_name=COLLECTION_NAME,
    persist_directory=str(DB_DIR),
    embedding_function=embeddings,
)
retriever = db.as_retriever(
    search_kwargs={"k": 5}
)

# =========================
# 3. Prompt
# =========================

prompt = PromptTemplate(
    input_variables=["context", "question"],
    template="""
You are a customer support assistant.

Answer the user's question based only on the provided context.

If the answer cannot be found in the context, say:
"I don't have enough information to answer that."

Do not make up information.
Answer in one concise, complete sentence.
For ticket lookup questions, include the ticket ID and the requested field.
Use only facts from the context. If the context does not answer the question, use the fallback response.
Context:
{context}

Question:
{question}

Answer:
"""
)


# =========================
# 4. RAG function
# =========================

def run_rag(query):

    # Use metadata filtering for exact ticket ID queries.
    ticket_match = re.search(
        r"\bticket(?:\s+id)?\s*#?\s*(\d+)\b",
        query,
        re.IGNORECASE,
    )
    if ticket_match:
        results = db.similarity_search(
            query,
            k=5,
            filter={"ticket_id": ticket_match.group(1)},
        )
    else:
        results = retriever.invoke(query)

    # Reranking
    pairs = [
        [query, document.page_content]
        for document in results
    ]

    scores = reranker.predict(pairs)

    ranked_results = list(zip(results, scores))

    ranked_results.sort(
        key=lambda x: x[1],
        reverse=True
    )

    # Take top 3
    top_documents = [
        document
        for document, score in ranked_results[:3]
    ]

    # Context
    context_blocks = [
    (
        f"Ticket ID: {document.metadata.get('ticket_id', 'unknown')}\n"
        f"Product: {document.metadata.get('product', 'unknown')}\n"
        f"Subject: {document.metadata.get('subject', 'unknown')}\n"
        f"Status: {document.metadata.get('status', 'unknown')}\n"
        f"Ticket Type: {document.metadata.get('ticket_type', 'unknown')}\n"
        f"Priority: {document.metadata.get('priority', 'unknown')}\n"
        f"Channel: {document.metadata.get('channel', 'unknown')}\n"
        f"Ticket details:\n{document.page_content}"
        )

        for document in top_documents
    ]

    context = "\n\n".join(context_blocks)

    # Prompt
    final_prompt = prompt.format(
        context=context,
        question=query
    )

    # LLM
    response = llm.invoke(final_prompt)

    return {
        "answer": response.content,
        "contexts": context_blocks
    }


# =========================
# 5. Test
# =========================

if __name__ == "__main__":

    query = "What is the priority of ticket 1?"

    result = run_rag(query)

    print("\n=== ANSWER ===")
    print(result["answer"])

    print("\n=== CONTEXTS ===")

    for context in result["contexts"]:
        print("\n---")
        print(context)