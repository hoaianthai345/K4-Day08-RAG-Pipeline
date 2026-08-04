"""Streamlit UI for the Shopee financial-services RAG chatbot."""

import json
import sys
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).parent
GOLDEN_DATASET_PATH = PROJECT_ROOT / "group_project" / "evaluation" / "golden_dataset.json"
load_dotenv(PROJECT_ROOT / ".env")
sys.path.insert(0, str(PROJECT_ROOT))

st.set_page_config(
    page_title="Shopee Financial Services RAG",
    page_icon="🛒",
    layout="wide",
    initial_sidebar_state="expanded",
)


@st.cache_data
def load_test_cases() -> list[dict]:
    """Read the golden dataset without reloading it on every Streamlit rerun."""
    if not GOLDEN_DATASET_PATH.exists():
        return []
    return json.loads(GOLDEN_DATASET_PATH.read_text(encoding="utf-8"))


def render_sources(sources: list[dict]) -> None:
    """Display retrieval evidence and distinguish ranking scores clearly."""
    if not sources:
        st.info("Không có source chunk được trả về.")
        return
    with st.expander(f"📚 Nguồn tham khảo ({len(sources)} chunks)", expanded=False):
        for index, source in enumerate(sources, 1):
            metadata = source.get("metadata", {})
            source_name = metadata.get("source", "Unknown")
            doc_type = metadata.get("type", "unknown")
            score_details = source.get("retrieval_scores", {})
            if score_details:
                details = [f"RRF: `{source.get('score', 0):.4f}`"]
                dense = score_details.get("dense_cosine")
                bm25 = score_details.get("bm25")
                if dense is not None:
                    details.append(f"Dense cosine: `{dense:.4f}`")
                if bm25 is not None:
                    details.append(f"BM25: `{bm25:.2f}`")
                rerank_relevance = score_details.get("rerank_relevance")
                if rerank_relevance is not None:
                    details.append(f"Rerank: `{rerank_relevance:.2f}`")
                score_display = " | ".join(details)
            else:
                score_display = f"score: `{source.get('score', 0):.4f}`"
            st.markdown(f"**[{index}] {source_name}** `{doc_type}` | {score_display}")
            st.text(source.get("content", "")[:500] + "...")
            if index < len(sources):
                st.divider()


def trace_table(results: list[dict]) -> list[dict]:
    """Convert one retrieval stage into rows suitable for Streamlit."""
    rows = []
    for rank, item in enumerate(results, 1):
        metadata = item.get("metadata", {})
        scores = item.get("retrieval_scores", {})
        rows.append({
            "Rank": rank,
            "Source": metadata.get("source", "Unknown"),
            "Chunk": metadata.get("chunk_index", ""),
            "Score": round(float(item.get("score", 0)), 4),
            "Dense cosine": None if scores.get("dense_cosine") is None else round(scores["dense_cosine"], 4),
            "BM25": None if scores.get("bm25") is None else round(scores["bm25"], 4),
            "Rerank relevance": None if scores.get("rerank_relevance") is None else round(scores["rerank_relevance"], 4),
            "Preview": item.get("content", "")[:180],
        })
    return rows


def render_trace(trace: dict) -> None:
    """Render every retrieval stage for a selected golden test case."""
    st.subheader("Trace retrieval")
    status = "đã dùng PageIndex fallback" if trace["fallback_used"] else "dùng hybrid retrieval"
    st.caption(
        f"Dense top-1: {trace['best_dense_score']:.4f} | "
        f"Threshold: {trace['score_threshold']:.2f} | {status}"
    )
    dense_tab, bm25_tab, fusion_tab, final_tab = st.tabs([
        "1. Dense / Chroma", "2. BM25", "3. RRF fusion", "4. Rerank + context"
    ])
    with dense_tab:
        st.dataframe(trace_table(trace["dense_results"]), use_container_width=True, hide_index=True)
    with bm25_tab:
        st.dataframe(trace_table(trace["sparse_results"]), use_container_width=True, hide_index=True)
    with fusion_tab:
        st.caption("RRF score chỉ dùng fusion; không phải confidence score.")
        st.dataframe(trace_table(trace["fused_results"]), use_container_width=True, hide_index=True)
    with final_tab:
        render_sources(trace["final_results"])


with st.sidebar:
    st.title("🛒 Shopee Financial RAG")
    st.caption("Trợ lý hỏi đáp về SPayLater, SEasy Vay Tiền Nhanh và SEasy Cho vay Người Bán.")
    st.divider()
    st.subheader("💡 Câu hỏi gợi ý")
    suggestions = [
        "SPayLater là gì và khoản tín dụng được cung cấp như thế nào?",
        "Điều kiện nào để được cấp khoản tín dụng qua SPayLater?",
        "Nếu thanh toán SPayLater trễ hạn thì có thể xảy ra điều gì?",
        "SEasy Vay Tiền Nhanh được dùng cho mục đích nào?",
        "Người Bán cần lưu ý gì khi sử dụng dịch vụ SEasy Cho vay Người Bán?",
    ]
    for index, suggestion in enumerate(suggestions):
        if st.button(suggestion, use_container_width=True, key=f"suggestion_{index}"):
            st.session_state["pending_query"] = suggestion
    st.divider()
    top_k = st.slider("Số chunks retrieval (top_k)", 3, 10, 5)
    st.caption("Dense retrieval + BM25 → RRF fusion → OpenAI generation có citation")

if "messages" not in st.session_state:
    st.session_state.messages = []
if "pending_query" not in st.session_state:
    st.session_state.pending_query = None
if "evaluation_runs" not in st.session_state:
    st.session_state.evaluation_runs = {}

chat_tab, evaluation_tab = st.tabs(["💬 Hỏi đáp", "🧪 Đánh giá test case"])

with chat_tab:
    st.title("🛒 Shopee Financial Services RAG")
    st.caption("Hỏi đáp dựa trên corpus SPayLater và SEasy đã index trong ChromaDB.")

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message["role"] == "assistant":
                render_sources(message.get("sources", []))

    user_input = st.chat_input("Nhập câu hỏi về SPayLater hoặc SEasy...")
    query = user_input or st.session_state.pending_query
    if query:
        st.session_state.pending_query = None
        st.session_state.messages.append({"role": "user", "content": query})
        with st.chat_message("user"):
            st.markdown(query)
        with st.chat_message("assistant"):
            with st.spinner("Đang tìm kiếm tài liệu và tổng hợp câu trả lời..."):
                try:
                    from src.task10_generation import generate_with_citation
                    response = generate_with_citation(query, top_k=top_k)
                    answer = response.get("answer", "Chưa thể trả lời.")
                    sources = response.get("sources", [])
                except Exception as exc:
                    answer = f"❌ Lỗi khi chạy RAG Pipeline: {exc}"
                    sources = []
            st.markdown(answer)
            render_sources(sources)
        st.session_state.messages.append({"role": "assistant", "content": answer, "sources": sources})

with evaluation_tab:
    st.title("🧪 Đánh giá theo Golden Dataset")
    st.caption("Chọn một test case, chạy pipeline và đối chiếu câu trả lời với expected answer. Mỗi lần chạy sẽ gọi API generation.")
    cases = load_test_cases()
    if not cases:
        st.warning("Không tìm thấy group_project/evaluation/golden_dataset.json.")
    else:
        st.metric("Số test case", len(cases))
        st.dataframe(
            [{"#": index + 1, "Question": item["question"], "Expected context": item.get("expected_context", "")}
             for index, item in enumerate(cases)],
            use_container_width=True,
            hide_index=True,
        )
        selected_index = st.selectbox(
            "Chọn test case để đánh giá",
            range(len(cases)),
            format_func=lambda index: f"Case {index + 1}: {cases[index]['question']}",
        )
        selected = cases[selected_index]
        left, right = st.columns(2)
        with left:
            st.subheader("Kỳ vọng")
            st.markdown(f"**Đáp án kỳ vọng:** {selected.get('expected_answer', 'N/A')}")
            st.markdown(f"**Context kỳ vọng:** {selected.get('expected_context', 'N/A')}")
        with right:
            st.subheader("Tự đánh giá")
            st.caption("So sánh factual accuracy, citation và mức phù hợp của source với phần kỳ vọng.")
            st.checkbox("Đáp án bám đúng evidence", key=f"faithful_{selected_index}")
            st.checkbox("Source truy xuất đúng chủ đề", key=f"context_{selected_index}")

        if st.button("Chạy test case này và lưu trace", type="primary"):
            with st.spinner("Đang chạy retrieval và generation..."):
                try:
                    from src.task9_retrieval_pipeline import retrieve_with_trace
                    from src.task10_generation import generate_with_citation
                    trace = retrieve_with_trace(selected["question"], top_k=top_k)
                    result = generate_with_citation(
                        selected["question"],
                        top_k=top_k,
                        context_chunks=trace["final_results"],
                    )
                    st.session_state.evaluation_runs[selected_index] = {
                        "trace": trace,
                        "result": result,
                    }
                except Exception as exc:
                    st.error(f"Không thể chạy test case: {exc}")

        run = st.session_state.evaluation_runs.get(selected_index)
        if run:
            st.subheader("Kết quả hệ thống")
            st.markdown(run["result"].get("answer", "Chưa có câu trả lời."))
            render_trace(run["trace"])
