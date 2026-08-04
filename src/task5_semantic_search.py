"""
Task 5 — Semantic Search Module (dense retrieval trên ChromaDB).

Dùng lại đúng embedding model + collection của Task 4 để vector query và vector
index nằm cùng không gian.
"""

from .task4_chunking_indexing import get_collection, get_embedding_model


def hyde_query(query: str) -> str:
    """Create an offline HyDE-style expansion for multilingual dense retrieval."""
    return f"Hướng dẫn và chính sách Shopee liên quan đến: {query.strip()}"


def semantic_search(query: str, top_k: int = 10, customer_role: str | None = None,
                    use_hyde: bool = False) -> list[dict]:
    """
    Tìm kiếm ngữ nghĩa bằng cosine similarity.

    Args:
        query: Câu truy vấn
        top_k: Số lượng kết quả tối đa
        customer_role: "buyer" | "seller" → chỉ lấy chunk của role đó + "both"

    Returns:
        List of {'content', 'score', 'metadata'} sorted by score descending.
    """
    if not query or top_k <= 0:
        return []
    collection = get_collection()
    if collection.count() == 0:
        return []

    query_vector = get_embedding_model().encode(
        hyde_query(query) if use_hyde else query, normalize_embeddings=True
    ).tolist()

    where = None
    if customer_role in ("buyer", "seller"):
        where = {"customer_role": {"$in": [customer_role, "both"]}}

    results = collection.query(
        query_embeddings=[query_vector],
        n_results=min(top_k, collection.count()),
        where=where,
        include=["documents", "metadatas", "distances"],
    )

    output = [
        {"content": doc, "score": round(max(0.0, 1.0 - dist), 4), "metadata": meta}
        for doc, meta, dist in zip(
            results["documents"][0], results["metadatas"][0], results["distances"][0]
        )
    ]
    output.sort(key=lambda x: x["score"], reverse=True)
    return output[:top_k]


if __name__ == "__main__":
    for r in semantic_search("quy định trả hàng hoàn tiền shopee", top_k=5):
        print(f"[{r['score']:.3f}] ({r['metadata']['customer_role']}) "
              f"{r['metadata']['source']}: {r['content'][:100]}...")
