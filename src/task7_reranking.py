"""Task 7: reranking utilities (Jina cross-encoder, MMR and RRF)."""

import os

import numpy as np


def rerank_cross_encoder(query: str, candidates: list[dict], top_k: int = 5) -> list[dict]:
    """Rerank with Jina when configured; otherwise retain the existing ranking."""
    if not candidates or top_k <= 0:
        return []
    api_key = os.getenv("JINA_API_KEY")
    if not api_key:
        return sorted((item.copy() for item in candidates), key=lambda x: x.get("score", 0), reverse=True)[:top_k]
    import requests
    response = requests.post(
        "https://api.jina.ai/v1/rerank",
        headers={"Authorization": f"Bearer {api_key}"},
        json={"model": "jina-reranker-v2-base-multilingual", "query": query,
              "documents": [item["content"] for item in candidates],
              "top_n": min(top_k, len(candidates))},
        timeout=30,
    )
    response.raise_for_status()
    results = []
    for result in response.json().get("results", []):
        index = result.get("index")
        if isinstance(index, int) and 0 <= index < len(candidates):
            results.append({**candidates[index], "score": float(result.get("relevance_score", 0.0))})
    return results


def rerank_mmr(query_embedding: list[float], candidates: list[dict], top_k: int = 5,
               lambda_param: float = 0.7) -> list[dict]:
    """Select relevant but non-duplicative candidates by Maximal Marginal Relevance."""
    if not 0 <= lambda_param <= 1:
        raise ValueError("lambda_param must be between 0 and 1")
    if top_k <= 0 or not candidates:
        return []
    query = np.asarray(query_embedding, dtype=float)
    vectors = [np.asarray(item.get("embedding", []), dtype=float) for item in candidates]
    if not len(query) or any(vector.shape != query.shape for vector in vectors):
        raise ValueError("every candidate needs an embedding matching query_embedding")

    def cosine(left, right):
        denominator = np.linalg.norm(left) * np.linalg.norm(right)
        return float(np.dot(left, right) / denominator) if denominator else 0.0

    selected, remaining, selected_scores = [], list(range(len(candidates))), []
    while remaining and len(selected) < top_k:
        best_index, best_score = None, float("-inf")
        for index in remaining:
            redundancy = max((cosine(vectors[index], vectors[chosen]) for chosen in selected), default=0.0)
            mmr_score = lambda_param * cosine(query, vectors[index]) - (1 - lambda_param) * redundancy
            if mmr_score > best_score:
                best_index, best_score = index, mmr_score
        selected.append(best_index)
        selected_scores.append(best_score)
        remaining.remove(best_index)
    return [{**candidates[index], "mmr_score": score} for index, score in zip(selected, selected_scores)]


def rerank_rrf(ranked_lists: list[list[dict]], top_k: int = 5, k: int = 60) -> list[dict]:
    """Merge ranked lists with Reciprocal Rank Fusion (Cormack et al., 2009)."""
    if k < 0 or top_k <= 0:
        return []
    scores, candidates = {}, {}
    for ranked_list in ranked_lists:
        for rank, item in enumerate(ranked_list, 1):
            metadata = item.get("metadata", {})
            key = (metadata.get("source"), metadata.get("chunk_index"), item.get("content", ""))
            scores[key] = scores.get(key, 0.0) + 1.0 / (k + rank)
            candidates.setdefault(key, item)
    return [{**candidates[key], "score": score} for key, score in
            sorted(scores.items(), key=lambda pair: pair[1], reverse=True)[:top_k]]


def rerank(query: str, candidates: list[dict], top_k: int = 5, method: str = "rrf") -> list[dict]:
    """Unified reranking interface suitable for Task 9."""
    if not candidates or top_k <= 0:
        return []
    if method == "cross_encoder":
        return rerank_cross_encoder(query, candidates, top_k)
    if method == "mmr":
        from .task4_chunking_indexing import get_embedding_model
        embedding = get_embedding_model().encode(query, normalize_embeddings=True).tolist()
        return rerank_mmr(embedding, candidates, top_k)
    if method == "rrf":
        return sorted((item.copy() for item in candidates), key=lambda x: x.get("score", 0), reverse=True)[:top_k]
    raise ValueError(f"Unknown rerank method: {method}")
