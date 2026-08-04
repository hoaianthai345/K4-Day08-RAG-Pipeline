"""Task 8: optional PageIndex vectorless fallback.

The SDK is only imported when the feature is configured, so normal local RAG
use does not require a PageIndex account.
"""

import json
import os
import time
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()
PAGEINDEX_API_KEY = os.getenv("PAGEINDEX_API_KEY", "")
STANDARDIZED_DIR = Path(__file__).parent.parent / "data" / "standardized"
DOC_IDS_PATH = Path(__file__).parent.parent / "pageindex_doc_ids.json"


def _client():
    if not PAGEINDEX_API_KEY:
        return None
    try:
        from pageindex.client import PageIndexClient
    except ImportError as exc:
        raise RuntimeError("Install pageindex to enable PageIndex fallback") from exc
    return PageIndexClient(api_key=PAGEINDEX_API_KEY)


def upload_documents() -> dict[str, str]:
    """Upload PDF source documents and persist a source-to-document-id mapping.

    PageIndex accepts PDFs. The method deliberately does not pretend markdown
    files are PDFs; place original PDFs in ``data/landing/legal`` first.
    """
    client = _client()
    if client is None:
        return {}
    source_dir = Path(__file__).parent.parent / "data" / "landing" / "legal"
    mapping = {}
    for pdf_path in source_dir.glob("*.pdf"):
        response = client.submit_document(str(pdf_path))
        document_id = response.get("doc_id") or response.get("id")
        if document_id:
            mapping[pdf_path.name] = str(document_id)
    DOC_IDS_PATH.write_text(json.dumps(mapping, ensure_ascii=False, indent=2), encoding="utf-8")
    return mapping


def _flatten_contents(node: dict):
    for group in node.get("relevant_contents", []) or []:
        for item in group or []:
            content = item.get("relevant_content", "")
            if content:
                yield content, item.get("section_title", "")


def pageindex_search(query: str, top_k: int = 5) -> list[dict]:
    """Query uploaded PageIndex documents. Return ``[]`` when unconfigured."""
    if not query or top_k <= 0 or not PAGEINDEX_API_KEY or not DOC_IDS_PATH.exists():
        return []
    client = _client()
    document_ids = json.loads(DOC_IDS_PATH.read_text(encoding="utf-8"))
    results = []
    for name, document_id in document_ids.items():
        submitted = client.submit_query(doc_id=document_id, query=query)
        retrieval_id = submitted.get("retrieval_id") or submitted.get("id")
        if not retrieval_id:
            continue
        retrieval = client.get_retrieval(retrieval_id)
        for _ in range(30):
            if retrieval.get("status") in {"completed", "failed", "error"}:
                break
            time.sleep(1)
            retrieval = client.get_retrieval(retrieval_id)
        for node_rank, node in enumerate(retrieval.get("retrieved_nodes", []) or [], 1):
            for content, section in _flatten_contents(node):
                results.append({"content": content, "score": 1.0 / node_rank,
                                "metadata": {"source": name, "section": section},
                                "source": "pageindex"})
    return sorted(results, key=lambda item: item["score"], reverse=True)[:top_k]
