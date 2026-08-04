"""
Task 9 — Retrieval Pipeline Hoàn Chỉnh.

Kết hợp semantic search + lexical search + reranking + PageIndex fallback
thành một pipeline thống nhất.

Logic:
    1. Chạy semantic_search + lexical_search song song
    2. Merge kết quả (RRF hoặc weighted fusion)
    3. Rerank
    4. Nếu top result score < threshold → fallback sang PageIndex
    5. Return top_k results

⚠️ BẪY THƯỜNG GẶP — đọc kỹ trước khi code:
    Nếu bạn dùng điểm RRF đã fuse (Task 7) để so với score_threshold, bạn sẽ gặp bug
    thật: RRF max score luôn ≈ 1/(k+1) ≈ 0.0164 (k=60) BẤT KỂ nội dung có liên quan
    hay không. Nếu đặt threshold thấp (như 0.005) để "hợp" với thang điểm RRF, thực
    chất KHÔNG câu hỏi nào đủ thấp để trigger fallback nữa — kể cả query hoàn toàn vô
    nghĩa vẫn trả về kết quả "hybrid" (rác) thay vì fallback đúng như thiết kế.

    Cách sửa đúng: giữ điểm cosine similarity GỐC của semantic_search (trước khi qua
    RRF) làm căn cứ quyết định fallback, tách biệt khỏi điểm RRF dùng để sắp xếp kết
    quả cuối cùng. Calibrate threshold bằng cách tự đo: chạy vài câu hỏi chắc chắn
    liên quan và vài câu chắc chắn lạc đề/rác qua semantic_search, xem khoảng cách
    điểm số giữa hai nhóm rồi chọn ngưỡng nằm giữa.
"""

import os

from .task5_semantic_search import semantic_search
from .task6_lexical_search import lexical_search
from .task7_reranking import rerank, rerank_rrf
from .task8_pageindex_vectorless import pageindex_search


# =============================================================================
# CONFIGURATION
# =============================================================================

# Calibrate this value from semantic-search cosine scores after indexing a new
# corpus; the default is deliberately conservative for the bundled documents.
SCORE_THRESHOLD = 0.3   # Nếu best score (cosine gốc) < threshold → fallback PageIndex
DEFAULT_TOP_K = 5
# RRF fuses dense/BM25 rankings.  A second stage then actually reranks the
# fused passages by reading query-passage pairs.
RERANK_METHOD = os.getenv("RERANK_METHOD", "openai").lower()


def _chunk_key(item: dict) -> tuple:
    """Stable identity shared by dense, sparse and fused result objects."""
    metadata = item.get("metadata", {})
    return (metadata.get("source"), metadata.get("chunk_index"), item.get("content", ""))


def retrieve(
    query: str,
    top_k: int = DEFAULT_TOP_K,
    score_threshold: float = SCORE_THRESHOLD,
    use_reranking: bool = True,
) -> list[dict]:
    """
    Retrieval pipeline hoàn chỉnh với fallback logic.

    Pipeline:
        Query
          ├→ Semantic Search → dense_results (giữ điểm cosine gốc)
          ├→ Lexical Search  → sparse_results
          │
          ├→ Merge (RRF) → merged_results
          ├→ Rerank → reranked_results
          │
          └→ If dense_results[0]["score"] < threshold:
                └→ PageIndex Vectorless → fallback_results

    Args:
        query: Câu truy vấn
        top_k: Số lượng kết quả cuối cùng
        score_threshold: Ngưỡng điểm cosine gốc tối thiểu (KHÔNG phải điểm RRF)
        use_reranking: Có áp dụng reranking hay không

    Returns:
        List of {
            'content': str,
            'score': float,
            'metadata': dict,
            'source': str  # 'hybrid' hoặc 'pageindex'
        }
    """
    return retrieve_with_trace(query, top_k, score_threshold, use_reranking)["final_results"]


def retrieve_with_trace(
    query: str,
    top_k: int = DEFAULT_TOP_K,
    score_threshold: float = SCORE_THRESHOLD,
    use_reranking: bool = True,
) -> dict:
    """Run retrieval and return intermediate stages for UI debugging/evaluation."""
    if not query or top_k <= 0:
        return {
            "dense_results": [], "sparse_results": [], "fused_results": [],
            "final_results": [], "best_dense_score": 0.0,
            "score_threshold": score_threshold, "fallback_used": False,
        }
    # Keep the raw dense cosine score separate from the fused RRF score.
    dense_results = semantic_search(query, top_k=top_k * 2)
    sparse_results = lexical_search(query, top_k=top_k * 2)
    dense_scores = {_chunk_key(item): item["score"] for item in dense_results}
    sparse_scores = {_chunk_key(item): item["score"] for item in sparse_results}
    merged = rerank_rrf([dense_results, sparse_results], top_k=top_k * 2)
    for item in merged:
        item["source"] = "hybrid"
        key = _chunk_key(item)
        # RRF is a rank-fusion score (~0.01–0.03 with k=60), not a
        # semantic-confidence score. Preserve all values for transparent UI.
        item["retrieval_scores"] = {
            "rrf": item["score"],
            "dense_cosine": dense_scores.get(key),
            "bm25": sparse_scores.get(key),
        }
    final_results = rerank(query, merged, top_k=top_k, method=RERANK_METHOD) if use_reranking else merged[:top_k]
    best_dense_score = dense_results[0]["score"] if dense_results else 0.0
    fallback_used = False
    if best_dense_score < score_threshold:
        fallback = pageindex_search(query, top_k=top_k)
        if fallback:
            final_results = fallback
            fallback_used = True
    return {
        "dense_results": dense_results,
        "sparse_results": sparse_results,
        "fused_results": merged,
        "final_results": final_results[:top_k],
        "best_dense_score": best_dense_score,
        "score_threshold": score_threshold,
        "fallback_used": fallback_used,
    }


if __name__ == "__main__":
    test_queries = [
        "What payment methods does Shopee support?",
        "How do I request a return or refund?",
        "What evidence do I need for a refund request?",
        "xyzabc123nonsense",  # Query không có kết quả → test fallback
    ]

    for q in test_queries:
        print(f"\nQuery: {q}")
        print("-" * 60)
        results = retrieve(q, top_k=3)
        for i, r in enumerate(results, 1):
            print(f"  {i}. [{r['score']:.3f}] [{r['source']}] {r['content'][:80]}...")
