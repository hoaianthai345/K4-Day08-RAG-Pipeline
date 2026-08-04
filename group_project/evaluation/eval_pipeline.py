"""RAGAS evaluation runner for the group RAG chatbot.

RAGAS is the selected evaluation framework because it is listed in the project
dependencies. DeepEval and TruLens remain optional and deliberately report a
clear installation requirement instead of failing at import time.
"""

import json
from collections.abc import Callable
from pathlib import Path

GOLDEN_DATASET_PATH = Path(__file__).parent / "golden_dataset.json"
RESULTS_PATH = Path(__file__).parent / "results.md"


def load_golden_dataset() -> list[dict]:
    return json.loads(GOLDEN_DATASET_PATH.read_text(encoding="utf-8"))


def _generate(pipeline, question: str, **kwargs) -> dict:
    """Accept either ``generate_with_citation`` or an object exposing it."""
    generator = getattr(pipeline, "generate_with_citation", pipeline)
    if not callable(generator):
        raise TypeError("rag_pipeline must be callable or expose generate_with_citation")
    return generator(question, **kwargs)


def evaluate_with_ragas(rag_pipeline, golden_dataset: list[dict]):
    """Run faithfulness, answer relevance, context recall and precision."""
    from datasets import Dataset
    from ragas import evaluate
    from ragas.metrics import answer_relevancy, context_precision, context_recall, faithfulness

    records = {"question": [], "answer": [], "contexts": [], "ground_truth": []}
    for item in golden_dataset:
        result = _generate(rag_pipeline, item["question"])
        records["question"].append(item["question"])
        records["answer"].append(result.get("answer", ""))
        records["contexts"].append([chunk.get("content", "") for chunk in result.get("sources", [])])
        records["ground_truth"].append(item["expected_answer"])
    return evaluate(Dataset.from_dict(records), metrics=[faithfulness, answer_relevancy, context_recall, context_precision]).to_pandas()


def evaluate_with_deepeval(rag_pipeline, golden_dataset: list[dict]):
    """Optional adapter; install ``deepeval`` before selecting this framework."""
    try:
        from deepeval import evaluate
        from deepeval.metrics import AnswerRelevancyMetric, ContextualPrecisionMetric, ContextualRecallMetric, FaithfulnessMetric
        from deepeval.test_case import LLMTestCase
    except ImportError as exc:
        raise RuntimeError("DeepEval is optional; run `pip install deepeval` to use it") from exc
    cases = []
    for item in golden_dataset:
        result = _generate(rag_pipeline, item["question"])
        cases.append(LLMTestCase(input=item["question"], actual_output=result.get("answer", ""),
                                 expected_output=item["expected_answer"],
                                 retrieval_context=[source.get("content", "") for source in result.get("sources", [])]))
    return evaluate(cases, [FaithfulnessMetric(threshold=0.7), AnswerRelevancyMetric(threshold=0.7),
                            ContextualRecallMetric(threshold=0.7), ContextualPrecisionMetric(threshold=0.7)])


def evaluate_with_trulens(rag_pipeline, golden_dataset: list[dict]):
    """Run a simple TruLens recording when the optional package is installed."""
    try:
        from trulens.apps.custom import TruCustomApp
    except ImportError as exc:
        raise RuntimeError("TruLens is optional; run `pip install trulens` to use it") from exc
    app = TruCustomApp(rag_pipeline, app_name="EcommerceSupport_RAG")
    with app as recording:
        for item in golden_dataset:
            _generate(rag_pipeline, item["question"])
    return recording


def _pipeline_with_reranking(use_reranking: bool):
    """
    Dựng 1 callable sinh câu trả lời với retrieval bật/tắt rerank.

    KHÔNG truyền use_reranking thẳng vào generate_with_citation() — hàm đó chỉ
    nhận (query, top_k, context_chunks). Truyền sai sẽ ném TypeError, bị nuốt bởi
    except, và A/B sẽ chạy CÙNG một config hai lần rồi cho ra hai bảng giống hệt.
    Cách đúng: gọi retrieve() với cấu hình mong muốn rồi bơm chunk vào generation.
    """
    from src.task9_retrieval_pipeline import retrieve
    from src.task10_generation import generate_with_citation

    def run(question: str, top_k: int = 5) -> dict:
        chunks = retrieve(question, top_k=top_k, use_reranking=use_reranking)
        return generate_with_citation(question, top_k=top_k, context_chunks=chunks)

    return run


def compare_configs(rag_pipeline, golden_dataset: list[dict]) -> dict:
    """So sánh baseline (RRF thuần) với cấu hình bật reranking."""
    return {
        "B_baseline_no_rerank": evaluate_with_ragas(_pipeline_with_reranking(False), golden_dataset),
        "A_hybrid_rerank": evaluate_with_ragas(_pipeline_with_reranking(True), golden_dataset),
    }


def export_results(results, comparison: dict) -> Path:
    """Write score tables for the selected evaluation and A/B comparison."""
    def table(value) -> str:
        if hasattr(value, "to_markdown"):
            return value.to_markdown(index=False)
        if isinstance(value, dict):
            return "\n".join(f"- **{key}**: {item}" for key, item in value.items())
        return str(value)

    content = "# RAG Evaluation Results\n\n## Overall Scores\n\n" + table(results)
    content += "\n\n## A/B Comparison\n"
    for name, value in comparison.items():
        content += f"\n### {name}\n\n{table(value)}\n"
    RESULTS_PATH.write_text(content + "\n", encoding="utf-8")
    return RESULTS_PATH


REFUSAL_MARKERS = ("không tìm thấy", "không có thông tin", "không đủ", "không thể trả lời")


def evaluate_refusal(pipeline, golden_dataset: list[dict]) -> dict:
    """
    Đo riêng nhóm out_of_scope: hệ thống có biết từ chối thay vì bịa không.

    RAGAS không đo được nhóm này — context_recall trên câu không có đáp án đúng
    là vô nghĩa. Nhưng đây mới là chỗ hallucination lộ ra, nên phải đo tách.
    """
    rows = []
    for item in golden_dataset:
        answer = pipeline(item["question"]).get("answer", "")
        rows.append({
            "id": item["id"],
            "refused": any(m in answer.lower() for m in REFUSAL_MARKERS),
            "answer": answer[:160],
        })
    n = len(rows) or 1
    return {"refusal_rate": sum(r["refused"] for r in rows) / n, "rows": rows}


def main():
    golden = load_golden_dataset()
    answerable = [g for g in golden if g.get("answerable", True)]
    out_of_scope = [g for g in golden if not g.get("answerable", True)]
    print(f"Golden set: {len(golden)} câu = {len(answerable)} trả lời được + {len(out_of_scope)} ngoài phạm vi")

    comparison = compare_configs(None, answerable)
    for name, df in comparison.items():
        print(f"\n=== {name} (n={len(answerable)}) ===")
        print(df[["faithfulness", "answer_relevancy", "context_recall", "context_precision"]].mean().round(4).to_string())

    print(f"\n=== Từ chối câu ngoài phạm vi (n={len(out_of_scope)}) ===")
    for name, use_rr in [("B_baseline_no_rerank", False), ("A_hybrid_rerank", True)]:
        r = evaluate_refusal(_pipeline_with_reranking(use_rr), out_of_scope)
        print(f"{name}: refusal_rate = {r['refusal_rate']:.3f}")
        for row in r["rows"]:
            print(f"   {row['id']} {'TU CHOI' if row['refused'] else 'TRA LOI '} | {row['answer'][:110]}")

    export_results(comparison["A_hybrid_rerank"], comparison)
    print(f"\nĐã ghi {RESULTS_PATH}")


if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    main()
