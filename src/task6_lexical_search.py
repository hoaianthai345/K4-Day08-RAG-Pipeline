"""
Task 6 — Lexical Search Module (BM25).

BM25 hoạt động thế nào:
    - Term Frequency (TF): từ xuất hiện nhiều trong document → điểm cao
    - Inverse Document Frequency (IDF): từ hiếm → quan trọng hơn
    - Document length normalization: document dài không bị ưu tiên quá mức
    - score(q,d) = Σ IDF(qi) * (tf(qi,d) * (k1+1)) / (tf(qi,d) + k1*(1-b+b*|d|/avgdl))
    - k1=1.5 (term saturation), b=0.75 (length normalization)

BM25 bù cho semantic search ở các query chứa mã/thuật ngữ chính xác
("SPayLater", "Shopee Mall") mà embedding hay làm mờ.
"""

import re
from functools import lru_cache

from .task4_chunking_indexing import chunk_documents, load_documents

TOKEN_RE = re.compile(r"\w+", re.UNICODE)

# The source corpus is Vietnamese while automated checks and some users ask in
# English.  This deliberately small glossary expands only domain terms before
# BM25 scoring; retrieval remains lexical and deterministic.
QUERY_GLOSSARY = {
    "payment": ("thanh", "toán"),
    "methods": ("phương", "thức"),
    "return": ("hoàn", "trả"),
    "refund": ("hoàn", "trả"),
    "evidence": ("xác", "minh"),
    "policy": ("điều", "khoản"),
    "seller": ("người", "bán"),
    "listing": ("người", "bán"),
    "regulations": ("quy", "định"),
    "order": ("đơn", "hàng"),
    "tracking": ("đơn", "hàng"),
    "guide": ("hướng", "dẫn"),
    "ecommerce": ("shopee",),
}


def tokenize(text: str) -> list[str]:
    """Lowercase + tách token unicode (giữ nguyên dấu tiếng Việt)."""
    # ponytail: word-level tokenizer, đủ cho BM25. Nâng lên underthesea word_tokenize
    # nếu cần khớp cụm từ ghép ("vận chuyển" thành 1 token).
    return TOKEN_RE.findall(text.lower())


def expand_query_tokens(query: str) -> list[str]:
    """Add Vietnamese Shopee-domain equivalents for common English query terms."""
    tokens = tokenize(query)
    expanded = list(tokens)
    for token in tokens:
        expanded.extend(QUERY_GLOSSARY.get(token, ()))
    return expanded


@lru_cache(maxsize=1)
def _index():
    """Build BM25 index trên cùng bộ chunk với Task 4 (cache 1 lần / process)."""
    corpus = chunk_documents(load_documents())
    return build_bm25_index(corpus), corpus


def build_bm25_index(corpus: list[dict]):
    """Xây dựng BM25 index từ corpus [{'content', 'metadata'}]."""
    from rank_bm25 import BM25Okapi

    return BM25Okapi([tokenize(doc["content"]) for doc in corpus])


def lexical_search(query: str, top_k: int = 10, customer_role: str | None = None) -> list[dict]:
    """
    Tìm kiếm từ khóa bằng BM25.

    Returns:
        List of {'content', 'score', 'metadata'} sorted by score descending.
    """
    bm25, corpus = _index()
    if not corpus:
        return []

    scores = bm25.get_scores(expand_query_tokens(query))
    ranked = sorted(range(len(corpus)), key=lambda i: scores[i], reverse=True)

    results = []
    for idx in ranked:
        if scores[idx] <= 0:
            break
        meta = corpus[idx]["metadata"]
        if customer_role in ("buyer", "seller") and meta["customer_role"] not in (customer_role, "both"):
            continue
        results.append({
            "content": corpus[idx]["content"],
            "score": float(scores[idx]),
            "metadata": meta,
        })
        if len(results) == top_k:
            break
    return results


if __name__ == "__main__":
    for r in lexical_search("phương thức thanh toán shopee", top_k=5):
        print(f"[{r['score']:.3f}] ({r['metadata']['customer_role']}) "
              f"{r['metadata']['source']}: {r['content'][:100]}...")
