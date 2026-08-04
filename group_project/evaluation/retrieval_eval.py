"""
Chấm điểm RETRIEVAL bằng golden_dataset.json — không cần LLM, không cần API key.

RAGAS (eval_pipeline.py) đo cả generation nhưng phải gọi LLM để chấm, nên tốn tiền
và không chạy được khi thiếu key. Bộ này chỉ đo tầng retrieval bằng
`expected_sources` trong golden set, nên chạy offline và cho kết quả tất định —
dùng để so sánh semantic / BM25 / hybrid trước khi đụng tới RAGAS.

Chạy:
    python -m group_project.evaluation.retrieval_eval
"""

import re
import sys
import unicodedata
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from group_project.evaluation.eval_pipeline import load_golden_dataset  # noqa: E402

TOP_K = 5
MIN_FRAGMENT = 25  # mảnh evidence ngắn hơn thì dễ khớp ngẫu nhiên, không tính


def _norm(text: str) -> str:
    # NFC bắt buộc: corpus và golden set dùng khác dạng tổ hợp dấu tiếng Việt, hai
    # chuỗi hiện ra giống hệt nhau nhưng `in` trả về False nếu không chuẩn hoá.
    return re.sub(r"\s+", " ", unicodedata.normalize("NFC", text)).lower().strip()


def evidence_fragments(evidence: str | None) -> list[str]:
    """Tách evidence thành các mảnh độc lập (golden set nối nhiều ý bằng ' / ')."""
    if not evidence:
        return []
    parts = re.split(r"\s/\s|\.\.\.", evidence)
    return [f for f in (_norm(p) for p in parts) if len(f) >= MIN_FRAGMENT]


def score_one(retrieved: list[dict], expected: list[str], evidence: str | None) -> dict:
    """
    Hai mức chấm:
      - source-level: chunk đến từ đúng FILE (dễ — corpus chỉ 8 file)
      - evidence-level: chunk thực sự CHỨA câu văn mang đáp án (khó, mới là cái cần đo)
    """
    sources = [r.get("metadata", {}).get("source", "") for r in retrieved]
    hits = [i for i, s in enumerate(sources) if s in expected]

    frags = evidence_fragments(evidence)
    ev_hits = [
        i for i, r in enumerate(retrieved)
        if frags and any(f in _norm(r.get("content", "")) for f in frags)
    ]

    return {
        "hit": bool(hits),
        "rr": 1.0 / (hits[0] + 1) if hits else 0.0,
        "precision": len(hits) / len(sources) if sources else 0.0,
        "ev_hit": bool(ev_hits),
        "ev_rr": 1.0 / (ev_hits[0] + 1) if ev_hits else 0.0,
        "top_source": sources[0] if sources else "",
        "top_score": retrieved[0]["score"] if retrieved else 0.0,
    }


def evaluate_retrieval(search_fn, name: str, top_k: int = TOP_K) -> dict:
    """
    Chấm 1 hàm retrieval trên toàn bộ golden set.

    Câu `answerable: false` không có đáp án đúng nào — thay vì tính hit, ta ghi lại
    điểm top-1 để biết sàn nhiễu ở đâu mà đặt score_threshold cho Task 9.
    """
    golden = load_golden_dataset()
    rows, noise_scores = [], []

    for item in golden:
        results = search_fn(item["question"], top_k=top_k)
        if item["answerable"]:
            row = score_one(results, item["expected_sources"], item.get("evidence"))
            row["id"] = item["id"]
            row["difficulty"] = item["difficulty"]
            rows.append(row)
        else:
            noise_scores.append(results[0]["score"] if results else 0.0)

    n = len(rows) or 1
    return {
        "name": name,
        "n_answerable": len(rows),
        "hit_rate": sum(r["hit"] for r in rows) / n,
        "mrr": sum(r["rr"] for r in rows) / n,
        "precision": sum(r["precision"] for r in rows) / n,
        "ev_hit_rate": sum(r["ev_hit"] for r in rows) / n,
        "ev_mrr": sum(r["ev_rr"] for r in rows) / n,
        "max_noise_score": max(noise_scores) if noise_scores else 0.0,
        "misses": [r["id"] for r in rows if not r["ev_hit"]],
        "rows": rows,
    }


def print_report(summaries: list[dict], top_k: int = TOP_K):
    print(f"\n{'='*72}\nRETRIEVAL EVAL @ top_k={top_k}  (golden set, không dùng LLM)\n{'='*72}")
    print("            |---- đúng FILE (dễ) ----|--- đúng CHUNK evidence ---|")
    print(f"{'method':<12}{'hit':>8}{'MRR':>8}{'prec':>8}{'ev_hit':>10}{'ev_MRR':>9}{'nhiễu max':>12}")
    for s in summaries:
        print(f"{s['name']:<12}{s['hit_rate']:>8.3f}{s['mrr']:>8.3f}{s['precision']:>8.3f}"
              f"{s['ev_hit_rate']:>10.3f}{s['ev_mrr']:>9.3f}{s['max_noise_score']:>12.3f}")

    print(f"\n{'-'*72}\nev_hit_rate theo độ khó — trượt câu nào\n{'-'*72}")
    print(f"{'method':<12}{'easy':>8}{'medium':>8}{'hard':>8}   trượt")
    for s in summaries:
        by = {}
        for r in s["rows"]:
            by.setdefault(r["difficulty"], []).append(r["ev_hit"])
        cell = {d: (sum(v) / len(v) if v else 0.0) for d, v in by.items()}
        print(f"{s['name']:<12}{cell.get('easy',0):>8.3f}{cell.get('medium',0):>8.3f}"
              f"{cell.get('hard',0):>8.3f}   {','.join(s['misses']) or '-'}")

    print("\nGhi chú:")
    print("  hit/MRR/prec  = chunk đến đúng FILE. Corpus chỉ 8 file nên chỉ số này dễ đạt trần.")
    print("  ev_hit/ev_MRR = chunk thực sự CHỨA câu văn mang đáp án — đây mới là chỉ số để so sánh.")
    print("  nhiễu max     = điểm top-1 cao nhất ở các câu out_of_scope. score_threshold của")
    print("                  Task 9 phải nằm TRÊN mức này thì fallback mới có tác dụng.")
    print("                  Lưu ý: điểm BM25 không chặn trên [0,1] nên KHÔNG so trực tiếp với semantic.")


def validate_golden_set() -> list[str]:
    """
    Golden set chỉ dùng được nếu mọi `evidence` thật sự nằm trong corpus đã index.
    Không có bước này thì một evidence gõ sai sẽ bị đọc nhầm thành "retrieval kém".
    """
    from src.task4_chunking_indexing import chunk_documents, load_documents

    chunks = [_norm(c["content"]) for c in chunk_documents(load_documents())]
    sources = {c["metadata"]["source"] for c in chunk_documents(load_documents())}

    problems = []
    for item in load_golden_dataset():
        missing = [s for s in item["expected_sources"] if s not in sources]
        if missing:
            problems.append(f"{item['id']}: expected_sources không có trong corpus: {missing}")
        if not item["answerable"]:
            continue
        frags = evidence_fragments(item.get("evidence"))
        if not frags:
            problems.append(f"{item['id']}: thiếu evidence (câu answerable bắt buộc phải có)")
        elif not any(any(f in c for f in frags) for c in chunks):
            problems.append(f"{item['id']}: evidence KHÔNG tồn tại trong corpus — lỗi golden set")
    return problems


def main():
    from src.task5_semantic_search import semantic_search
    from src.task6_lexical_search import lexical_search

    problems = validate_golden_set()
    print("Kiểm tra golden set:", "OK — mọi evidence đều có trong corpus" if not problems
          else f"{len(problems)} VẤN ĐỀ")
    for p in problems:
        print("   ✗", p)

    summaries = [
        evaluate_retrieval(semantic_search, "semantic"),
        evaluate_retrieval(lexical_search, "bm25"),
    ]
    try:
        from src.task9_retrieval_pipeline import retrieve
        summaries.append(evaluate_retrieval(retrieve, "hybrid"))
    except (ImportError, NotImplementedError):
        print("(bỏ qua hybrid — task9 chưa implement)")

    print_report(summaries)


if __name__ == "__main__":
    main()
