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

> **Phản biện:** copy thủ công, không qua `task3_convert_markdown.py` → pipeline
> crawl → convert chưa chạy end-to-end. Hệ quả cụ thể đã đo được ở **mục 7**
> (2 test FAIL, chặn CP4). Đây là món nợ đã biết, không phải bỏ sót.

### 2. `src/task4_chunking_indexing.py` — đã implement đầy đủ

| Config | Giá trị | Lý do |
|---|---|---|
| `CHUNK_SIZE` | 800 | Xấp xỉ 1 điều khoản/1 mục trong văn bản pháp lý Shopee. Đủ ngữ cảnh để LLM trả lời trọn ý, không nhồi cả file vào prompt (tốn token + loãng thông tin). |
| `CHUNK_OVERLAP` | 100 | 12.5% đệm ở ranh giới, để câu quan trọng không bị cắt đôi giữa 2 chunk. |
| `CHUNKING_METHOD` | `recursive` | `RecursiveCharacterTextSplitter`, separators `\n\n → \n → ". " → " "`: cắt ưu tiên ranh giới đoạn/câu, an toàn cho markdown pha văn xuôi + bullet list. |
| `MIN_CHUNK_CHARS` | 50 | **Thêm sau khi đo:** lần chunk đầu có chunk ngắn nhất chỉ **3 ký tự** (mảnh vụn kiểu `"A."`, `"1."` bị splitter tách ra). Vô nghĩa về ngữ nghĩa, chỉ chiếm slot top-k và làm nhiễu BM25 → lọc bỏ. |
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

> **Phản biện — rủi ro này đã thành hiện thực ngay trong lúc làm:**
> `sentence-transformers` kéo theo `torch` (venv phình lên **1.2GB**), và lần chạy
> đầu phải tải `bge-m3` (~2.3GB). Download **thực tế bị chậm/khựng**: sau 3 phút
> vẫn 0 byte, rồi nhích lên ~260MB và đứng. Nghĩa là **live demo trên máy lạ có
> thể treo 10+ phút** — không phải rủi ro lý thuyết.
>
> Phương án dự phòng đã xác nhận có sẵn trong HF cache của máy này:
> - `AITeamVN/Vietnamese_Embedding` — chính là bge-m3 fine-tune cho tiếng Việt,
>   **cùng 1024 chiều** → đổi mỗi `EMBEDDING_MODEL`, giữ nguyên `EMBEDDING_DIM`.
> - `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` — 384 chiều,
>   nhẹ nhất, phải sửa cả `EMBEDDING_DIM`.
>
> Đổi model thì **bắt buộc reindex** (`python -m src.task4_chunking_indexing`),
> vì vector cũ nằm ở không gian khác.

### 6. Kết quả đo được

**Đã xác minh (chạy thật, không cần model):**

| Chỉ số | Giá trị |
|---|---|
| Documents load được | **27** (22 legal + 5 news) |
| `customer_role` cấp file | 13 both / 8 buyer / 6 seller — **khớp chính xác** với `grep` trên file gốc |
| Frontmatter | đã tách khỏi body, không lọt vào chunk |
| Chunks tạo ra | **740** (trước khi lọc `MIN_CHUNK_CHARS`) |
| Độ dài chunk (min/median/max) | 3 / 661 / 800 |
| Chunk vượt `CHUNK_SIZE × 1.1` | **0** |
| `customer_role` cấp chunk | 532 both / 108 seller / 100 buyer |

**Pytest:** `TestTask3` + `TestTask4` — **PASS toàn bộ** (12 passed).

**Chưa chạy được:** `embed_chunks()` + `index_to_vectorstore()` + Task 5 semantic
search — đang chờ tải xong bge-m3. Task 6 BM25 không phụ thuộc model nên chạy
được ngay sau khi có `rank-bm25`.

> **Phản biện con số 532 `both`:** 72% chunk mang nhãn `both`. Lọc theo
> `customer_role` chỉ loại được ~14% corpus cho mỗi role — **giá trị lọc thấp hơn
> nhiều so với kỳ vọng ban đầu**. Muốn filter thực sự có tác dụng thì phải gán
> nhãn ở **cấp chunk** (một đoạn trong *Chính sách vận chuyển* nói riêng về nghĩa
> vụ Người Bán thì phải là `seller`, không phải `both`). Chấp nhận ở CP2 vì đúng
> thời lượng, nhưng đừng quảng cáo quá lời khi demo.

### 7. Nợ kỹ thuật đã biết (chặn CP4 — 35/35 PASSED)

Chạy `pytest tests/test_individual.py -k "Task1 or Task2"` → **2 FAILED**:

| Test | Lỗi | Yêu cầu |
|---|---|---|
| `TestTask1::test_minimum_3_legal_files` | `0 not >= 3` | ≥3 file `.pdf/.docx` trong `data/landing/legal/`, mỗi file >1KB |
| `TestTask2::test_minimum_5_news_files` | `0 not >= 5` | ≥5 file `.json/.html/.md/.txt` trong `data/landing/news/`, mỗi file >500 bytes |

Nguyên nhân: corpus được copy thẳng vào `data/standardized/` (đã ở dạng markdown
chuẩn), **bỏ qua tầng `landing/`** — tức là Task 1–3 chưa chạy end-to-end.

> **Phản biện — 2 hướng sửa, phải chọn:**
> 1. **Crawl lại cho đúng:** chạy `task1_collect_legal_docs.py` /
>    `task2_crawl_news.py` để lưu file thô (HTML/JSON/PDF) vào `landing/`, rồi
>    `task3_convert_markdown.py` sinh ra `standardized/`. Đúng bản chất data
>    pipeline, nhưng tốn thời gian và phụ thuộc mạng.
> 2. **Chế file cho qua test:** ví dụ dùng `fpdf2` xuất .md thành PDF rồi nhét vào
>    `landing/legal/`. Test sẽ xanh, nhưng đó là **PDF do mình sinh ra, không phải
>    tài liệu gốc thu thập được** — `landing/` mất hết ý nghĩa truy vết nguồn, và
>    nếu bị hỏi trong demo thì không giải thích được.
>
> **Khuyến nghị: hướng 1.** Hướng 2 chỉ nên dùng nếu hết giờ, và nếu dùng thì
> phải nói thẳng trong demo là dữ liệu landing được tái tạo, không phải bản gốc.

---
