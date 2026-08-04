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


def compare_configs(rag_pipeline, golden_dataset: list[dict]) -> dict:
    """Compare baseline retrieval against the reranking-enabled configuration.

    The supplied pipeline may accept ``use_reranking``; older callables still
    work and are reported as the same generation configuration.
    """
    comparison = {}
    for name, kwargs in {"baseline": {"use_reranking": False}, "hybrid_rerank": {"use_reranking": True}}.items():
        try:
            comparison[name] = evaluate_with_ragas(lambda question, _kw=kwargs: _generate(rag_pipeline, question, **_kw), golden_dataset)
        except TypeError:
            comparison[name] = evaluate_with_ragas(rag_pipeline, golden_dataset)
    return comparison


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
