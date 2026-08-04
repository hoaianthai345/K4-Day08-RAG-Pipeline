"""
Task 4 — Chunking & Indexing vào Vector Store (ChromaDB).

Pipeline: load .md (data/standardized/) → chunk → embed (bge-m3) → index (ChromaDB).

Lưu ý: đổi corpus thì phải xóa chroma_db/ trước khi reindex, nếu không chunk cũ
và mới sẽ lẫn lộn trong cùng collection. run_pipeline() tự xóa collection cũ.
"""

import re
from functools import lru_cache
from pathlib import Path

STANDARDIZED_DIR = Path(__file__).parent.parent / "data" / "standardized"
CHROMA_DIR = Path(__file__).parent.parent / "chroma_db"


# =============================================================================
# CONFIGURATION
# =============================================================================

# 800 ký tự ~ 1 điều khoản/1 mục trong văn bản pháp lý Shopee: đủ ngữ cảnh để LLM
# trả lời mà không nhồi cả file 50 trang vào prompt (tốn tiền + loãng thông tin).
CHUNK_SIZE = 800
# 100 ký tự đệm ở ranh giới để câu quan trọng không bị cắt đôi giữa 2 chunk.
CHUNK_OVERLAP = 100
CHUNKING_METHOD = "recursive"  # recursive: an toàn cho markdown pha văn xuôi + list
MIN_CHUNK_CHARS = 50   # bỏ mảnh vụn ("A.", "1.") — không mang thông tin, chỉ làm nhiễu

# bge-m3: multilingual, mạnh với tiếng Việt, 1024 chiều, chạy local không cần API key.
EMBEDDING_MODEL = "BAAI/bge-m3"
EMBEDDING_DIM = 1024

VECTOR_STORE = "chromadb"  # local persistent, cosine search, không cần Docker
COLLECTION_NAME = "ecommerce_support_docs"

# Người mua và người bán chịu quy định khác nhau (phí sàn vs phí vận chuyển...),
# nên customer_role đi vào metadata để retrieval lọc đúng đối tượng.
CUSTOMER_ROLES = {"buyer", "seller", "both"}

FRONTMATTER_RE = re.compile(r"\A---\n(.*?)\n---\n", re.DOTALL)


# =============================================================================
# IMPLEMENTATION
# =============================================================================

def parse_frontmatter(text: str) -> tuple[dict, str]:
    """Tách YAML frontmatter đơn giản (key: value) khỏi body markdown."""
    m = FRONTMATTER_RE.match(text)
    if not m:
        return {}, text
    meta = {}
    for line in m.group(1).splitlines():
        # ponytail: frontmatter ở đây chỉ có key: value phẳng, không cần PyYAML
        key, sep, value = line.partition(":")
        if sep and not key.startswith(" "):
            meta[key.strip()] = value.strip().strip('"').strip("'")
    return meta, text[m.end():]


def infer_customer_role(meta: dict, content: str) -> str:
    """Lấy customer_role từ frontmatter, fallback suy luận từ nội dung."""
    role = meta.get("customer_role", "").lower()
    if role in CUSTOMER_ROLES:
        return role
    low = content.lower()
    has_seller = "người bán" in low or "seller" in low
    has_buyer = "người mua" in low or "buyer" in low
    if has_seller and has_buyer:
        return "both"
    if has_seller:
        return "seller"
    if has_buyer:
        return "buyer"
    return "both"


def load_documents() -> list[dict]:
    """
    Đọc toàn bộ markdown files từ data/standardized/.

    Returns:
        List of {'content': str, 'metadata': {'source', 'type', 'customer_role', ...}}
    """
    documents = []
    for md_file in sorted(STANDARDIZED_DIR.rglob("*.md")):
        raw = md_file.read_text(encoding="utf-8")
        meta, body = parse_frontmatter(raw)
        doc_type = "legal" if "legal" in md_file.parts else "news"
        documents.append({
            "content": body.strip() or raw,
            "metadata": {
                "source": md_file.name,
                "type": doc_type,
                "customer_role": infer_customer_role(meta, raw),
                "title": meta.get("title", md_file.stem),
                "category": meta.get("category", ""),
                "source_url": meta.get("source_url", ""),
            },
        })
    return documents


def chunk_documents(documents: list[dict]) -> list[dict]:
    """Chunk documents bằng RecursiveCharacterTextSplitter (giữ nguyên metadata)."""
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = []
    for doc in documents:
        for i, text in enumerate(splitter.split_text(doc["content"])):
            if len(text.strip()) < MIN_CHUNK_CHARS:
                continue  # mảnh vụn (tiêu đề lẻ, gạch đầu dòng rỗng) chỉ làm nhiễu retrieval
            chunks.append({
                "content": text,
                "metadata": {**doc["metadata"], "chunk_index": i},
            })
    return chunks


@lru_cache(maxsize=1)
def get_embedding_model():
    """SentenceTransformer dùng chung cho index (Task 4) và query (Task 5)."""
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(EMBEDDING_MODEL)


def embed_chunks(chunks: list[dict]) -> list[dict]:
    """Embed chunks, normalize để cosine similarity = dot product."""
    model = get_embedding_model()
    embeddings = model.encode(
        [c["content"] for c in chunks],
        batch_size=16,
        normalize_embeddings=True,
        show_progress_bar=True,
    )
    for chunk, emb in zip(chunks, embeddings):
        chunk["embedding"] = emb.tolist()
    return chunks


def get_collection():
    """Chroma collection persistent (cosine space)."""
    import chromadb

    CHROMA_DIR.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    return client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )


def index_to_vectorstore(chunks: list[dict]):
    """Upsert chunks vào ChromaDB (batch 200 để không vượt giới hạn 1 request)."""
    collection = get_collection()
    ids = [f"{c['metadata']['source']}_chunk_{c['metadata']['chunk_index']}" for c in chunks]
    for i in range(0, len(chunks), 200):
        batch = chunks[i:i + 200]
        collection.upsert(
            ids=ids[i:i + 200],
            documents=[c["content"] for c in batch],
            embeddings=[c["embedding"] for c in batch],
            metadatas=[c["metadata"] for c in batch],
        )


def reset_collection():
    """Xóa collection cũ để reindex sạch (chunk cũ không lẫn với chunk mới)."""
    import chromadb

    CHROMA_DIR.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    try:
        client.delete_collection(COLLECTION_NAME)
    except Exception:
        pass  # chưa từng index


def run_pipeline():
    """Chạy toàn bộ pipeline: load → chunk → embed → index."""
    print("=" * 50)
    print("Task 4: Chunking & Indexing")
    print(f"  Chunking: {CHUNKING_METHOD} (size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP})")
    print(f"  Embedding: {EMBEDDING_MODEL} (dim={EMBEDDING_DIM})")
    print(f"  Vector Store: {VECTOR_STORE}")
    print("=" * 50)

    docs = load_documents()
    print(f"\n✓ Loaded {len(docs)} documents")

    chunks = chunk_documents(docs)
    print(f"✓ Created {len(chunks)} chunks")

    roles: dict[str, int] = {}
    for c in chunks:
        role = c["metadata"]["customer_role"]
        roles[role] = roles.get(role, 0) + 1
    print(f"  customer_role: {roles}")

    chunks = embed_chunks(chunks)
    print(f"✓ Embedded {len(chunks)} chunks (dim={len(chunks[0]['embedding'])})")

    reset_collection()
    index_to_vectorstore(chunks)
    print(f"✓ Indexed to {VECTOR_STORE}: {get_collection().count()} chunks")


if __name__ == "__main__":
    run_pipeline()
