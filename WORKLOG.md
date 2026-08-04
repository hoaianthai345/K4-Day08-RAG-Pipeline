# WORKLOG — Nhật ký thay đổi & phản biện

Ghi lại **cái gì đã đổi**, **vì sao**, và **phản biện** (lý do có thể sai / khi nào phải sửa).
Mỗi checkpoint 1 mục. Đọc từ trên xuống để truy vết.

---

## CP2 — Task 4/5/6: Chunking, Indexing (ChromaDB), Semantic + BM25

**Ngày:** 2026-08-04

### 1. Dữ liệu đã nạp

| Việc | Chi tiết |
|---|---|
| Copy corpus | `folder lab 8/legal/*.md` (22 file) → `data/standardized/legal/` |
| | `folder lab 8/news/*.md` (5 file) → `data/standardized/news/` |
| | `sources.csv` → `data/standardized/legal/` (truy vết nguồn) |

Toàn bộ 27 file **đã có sẵn YAML frontmatter** với `customer_role`
(13 `both` / 8 `buyer` / 6 `seller`), `category`, `source_url`, `document_version`.

> **Phản biện:** copy thủ công, không qua `task3_convert_markdown.py`. Nghĩa là
> pipeline crawl → convert chưa được chạy end-to-end trong repo này. Nếu chấm điểm
> đòi hỏi Task 1–3 chạy được thì phải làm lại: Task 1 test yêu cầu **≥3 file
> PDF/DOCX** trong `data/landing/legal/` — hiện đang trống, sẽ **FAIL** ở CP4.
> Đây là món nợ đã biết, không phải bỏ sót.

### 2. `src/task4_chunking_indexing.py` — đã implement đầy đủ

| Config | Giá trị | Lý do |
|---|---|---|
| `CHUNK_SIZE` | 800 | Xấp xỉ 1 điều khoản/1 mục trong văn bản pháp lý Shopee. Đủ ngữ cảnh để LLM trả lời trọn ý, không nhồi cả file vào prompt (tốn token + loãng thông tin). |
| `CHUNK_OVERLAP` | 100 | 12.5% đệm ở ranh giới, để câu quan trọng không bị cắt đôi giữa 2 chunk. |
| `CHUNKING_METHOD` | `recursive` | `RecursiveCharacterTextSplitter`, separators `\n\n → \n → ". " → " "`: cắt ưu tiên ranh giới đoạn/câu, an toàn cho markdown pha văn xuôi + bullet list. |
| `EMBEDDING_MODEL` | `BAAI/bge-m3` | Multilingual, mạnh tiếng Việt, 1024 chiều, chạy local — không cần API key, không tốn tiền mỗi lần reindex. |
| `VECTOR_STORE` | ChromaDB, `hnsw:space=cosine` | Local persistent, không cần Docker. Truy vấn cosine vài ms thay vì quét tay từng file. |

Hàm mới thêm ngoài skeleton:

- `parse_frontmatter()` — tách YAML frontmatter khỏi body. **Không** dùng PyYAML:
  frontmatter ở đây chỉ là `key: value` phẳng, regex + `str.partition` là đủ.
- `infer_customer_role()` — ưu tiên `customer_role` trong frontmatter; nếu file
  thiếu nhãn thì suy luận từ nội dung ("người bán"/"người mua"), mặc định `both`.
- `get_embedding_model()` / `get_collection()` — `@lru_cache`, để Task 5 dùng
  **đúng model và đúng collection** đã index. Sai model = vector query lệch không
  gian với vector index → kết quả rác.
- `reset_collection()` — xóa collection trước khi reindex. Không có bước này,
  chunk cũ và chunk mới sống lẫn lộn, retrieval trả kết quả từ dữ liệu đã bỏ.

**Metadata mỗi chunk:** `source`, `type` (legal/news), `customer_role`, `title`,
`category`, `source_url`, `chunk_index`.

> **Phản biện `customer_role`:**
> 1. Nhãn ở **cấp file**, không phải cấp chunk. Một file `both` như *Chính sách
>    vận chuyển* có đoạn chỉ nói về nghĩa vụ Người Bán — chunk đó vẫn mang nhãn
>    `both`. Muốn chính xác hơn phải phân loại từng chunk (LLM hoặc rule) — chưa
>    làm vì 13/27 file là `both`, lọc theo file đã cắt được ~30% nhiễu, đủ dùng.
> 2. Filter đang là `role ∈ {role_hỏi, "both"}` — **không loại** `both`. Nếu loại
>    thì mất phần lớn corpus.
> 3. `infer_customer_role()` fallback bằng keyword là heuristic thô. Hiện **không
>    kích hoạt** vì cả 27 file đều có nhãn sẵn; nó chỉ là lưới an toàn cho file mới.
>
> **Phản biện 800/100:** con số chọn theo cảm tính về độ dài điều khoản, **chưa đo**.
> Cách kiểm chứng đúng là chạy RAGAS (CP5) với 800 và với 500/1000 rồi so
> context_precision. Nếu context_precision thấp mà recall cao → chunk quá to, giảm xuống.

### 3. `src/task5_semantic_search.py` — dense retrieval

- Embed query bằng **cùng** model Task 4, `normalize_embeddings=True`.
- `score = max(0, 1 - cosine_distance)` → về thang [0,1], sort giảm dần.
- Thêm tham số `customer_role` (tùy chọn) → Chroma `where={"customer_role": {"$in": [role, "both"]}}`.
- Trả `[]` khi collection rỗng, thay vì ném lỗi.

> **Phản biện:** `1 - distance` chỉ đúng khi collection dùng `hnsw:space=cosine`.
> Đổi sang `l2` thì công thức này vô nghĩa (distance không chặn trên bởi 2).
> Đã ghim `cosine` trong `get_collection()`, nhưng đây là ràng buộc ngầm giữa 2 file.

### 4. `src/task6_lexical_search.py` — BM25

- Corpus BM25 lấy từ **chính `chunk_documents(load_documents())`** — cùng bộ chunk
  với vector store, không phải đọc lại file thô. Nhờ vậy RRF ở Task 7 gộp được
  thứ hạng của **cùng một tập tài liệu**.
- Tokenize: `re.findall(r"\w+")` + lowercase, giữ nguyên dấu tiếng Việt.
- `@lru_cache` — build index 1 lần mỗi process.
- Cùng tham số `customer_role` như Task 5.

> **Phản biện tokenizer:** tách theo **âm tiết**, không theo **từ ghép**. "vận chuyển"
> thành 2 token `vận` + `chuyển`. BM25 vẫn chạy (IDF từng âm tiết), nhưng query
> "vận chuyển" cũng khớp nhầm document chỉ có "chuyển khoản". Nâng cấp:
> `underthesea.word_tokenize` — bỏ qua vì thêm 1 dependency nặng, và hybrid RRF
> (Task 7) đã có semantic search bù lại phần nhiễu này.
>
> **Phản biện chọn nguồn corpus:** BM25 đọc lại file + chunk lại thay vì đọc từ
> Chroma. Ưu điểm: chạy được cả khi chưa index. Rủi ro: nếu ai đó sửa file .md mà
> quên reindex Chroma, BM25 và vector store sẽ **lệch nhau**. Quy ước: sửa corpus →
> chạy lại `python -m src.task4_chunking_indexing`.

### 5. Môi trường

Cài vào `venv/` (trước đó chỉ có pip): `langchain-text-splitters`, `chromadb`,
`sentence-transformers`, `rank-bm25`, `numpy`, `python-dotenv`.

> **Phản biện:** `sentence-transformers` kéo theo `torch` (~2GB) và lần chạy đầu
> phải tải `bge-m3` (~2.2GB). Đây là chi phí một lần, nhưng nếu máy demo khác
> không có sẵn cache thì **live demo sẽ treo vài phút**. Phương án dự phòng:
> `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` (384 chiều) đã có
> trong HF cache của máy này — đổi `EMBEDDING_MODEL` + `EMBEDDING_DIM` rồi reindex.

### 6. Kết quả chạy

<!-- điền sau khi chạy -->

---
