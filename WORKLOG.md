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

Cài vào `venv/` của repo (trước đó chỉ có pip): `langchain-text-splitters`,
`chromadb`, `sentence-transformers`, `rank-bm25`, `numpy`, `python-dotenv`.

> **Phản biện — "sao không dùng env đã cài sẵn?"** (câu hỏi đúng, tôi đã bỏ sót
> bước kiểm tra):
>
> Python hệ thống `/Library/Frameworks/Python.framework/Versions/3.13` **đã có
> sẵn cả 5 package**. Tôi chỉ kiểm tra `./venv/bin/pip list` thấy trống rồi cài
> luôn, không `find_spec` thử interpreter khác — mất 5 giây mà bỏ qua.
>
> Nhưng khi kiểm tra lại thì env đó **hỏng**:
> ```
> ImportError: cannot import name 'HybridCache' from 'transformers'
>   peft/peft_model.py → transformers  (xung đột phiên bản)
> ```
> `import sentence_transformers` chết ngay. Dùng env này sẽ phải đi gỡ xung đột
> `transformers`/`peft`, và sửa nó thì **đụng vào các project khác** đang dùng
> chung env hệ thống. → Cài vào `venv/` riêng vẫn là lựa chọn đúng, chỉ là tôi
> đến đúng kết luận bằng đường sai.
>
> **Không đổi env nữa**: 1.2GB đã cài xong, và model nằm ở `~/.cache/huggingface`
> — **dùng chung cho mọi interpreter**, nên đổi env cũng không làm download nhanh
> hơn một giây nào.
>
> **Bài học cho lần sau:** trước khi `pip install`, chạy
> `python -c "import importlib.util as u; print(u.find_spec('torch'))"` trên các
> interpreter có mặt. Nhưng "có package" ≠ "dùng được" — vẫn phải thử `import` thật.

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

### 8. Thêm chiến lược `markdown_header` (bổ sung sau)

`chunk_documents(documents, method=...)` giờ dispatch qua dict `SPLITTERS`:

| method | Cách hoạt động |
|---|---|
| `recursive` (mặc định) | `RecursiveCharacterTextSplitter` thuần |
| `markdown_header` | `MarkdownHeaderTextSplitter` (`#`→h1, `##`→h2, `###`→h3) cắt theo heading **trước**, rồi `RecursiveCharacterTextSplitter` cắt tiếp từng section cho vừa `CHUNK_SIZE`. Chunk mang thêm metadata `h1/h2/h3`. |

Hai quyết định thiết kế:
- **Bắt buộc cắt 2 tầng.** `MarkdownHeaderTextSplitter` **không** giới hạn độ dài —
  nó cắt theo cấu trúc, một section 5000 ký tự vẫn ra 1 chunk 5000 ký tự. Không
  có tầng recursive phía sau thì vi phạm `CHUNK_SIZE` và fail
  `test_chunks_respect_size_limit`.
- **`strip_headers=False`.** Mặc định splitter **xoá** dòng heading khỏi nội dung
  (chỉ giữ trong metadata). Giữ lại để BM25 còn khớp được từ khóa trong tiêu đề.

**Đo trên corpus thật:**

| method | chunks | median | max | vượt limit | chunk có h2/h3 |
|---|---|---|---|---|---|
| `recursive` | 724 | 664 | 800 | 0 | 0 |
| `markdown_header` | 724 | 664 | 800 | 0 | **28** (3.9%) |

623/724 chunk khác nội dung → hai chiến lược cắt **thật sự khác nhau**, chỉ trùng
tổng số một cách ngẫu nhiên.

> **Phản biện — trên corpus NÀY, `markdown_header` gần như vô dụng:**
> Đếm heading từng file: **22/27 file chỉ có đúng 1 dòng `# `**, không có `##`/`###`
> nào. Cấu trúc thật của văn bản pháp lý Shopee ("A. PHẠM VI VÀ ĐỐI TƯỢNG ÁP DỤNG",
> "1. Đối tượng áp dụng", "a. ...") nằm ở dạng **đoạn văn thường**, không phải
> heading markdown. Nên với 22 file đó, `markdown_header` = 1 section duy nhất =
> **thoái hoá về đúng `recursive`**.
>
> Chỉ 5 file có `##` (`dieu-khoan-dich-vu-shopee-mall.md` + 4 file news), cho ra
> 28 chunk có metadata phân cấp — 3.9% corpus.
>
> **Kết luận: giữ `recursive` làm mặc định.** `markdown_header` để đó cho 2 tình
> huống: (a) corpus tương lai có heading phân cấp thật, (b) cần demo so sánh
> chiến lược. Nếu muốn `markdown_header` thực sự có tác dụng ở đây thì việc phải
> làm là ở **Task 3** — sửa bước convert để "A.", "1.", "a." được xuất thành
> `##`/`###`, chứ không phải sửa Task 4.

### 9. Corpus bị thu hẹp — 27 file → 8 file

Người dùng đã xoá bớt file trong `data/standardized/legal/`. Corpus hiện tại:

| | Trước | Sau |
|---|---|---|
| legal | 22 | **3** — `dieu-khoan-dich-vu-spaylater`, `…-seasy-cho-vay-nguoi-ban`, `…-seasy-vay-tien-nhanh` |
| news | 5 | **5** — evoucher, shopeefood-dat-mon, shopeefood-lien-ket-tai-khoan, spaylater-thanh-toan-shopeefood, tai-khoan-ngan-hang |
| **tổng** | 27 | **8** |

Chủ đề thu về đúng một cụm: **SPayLater / SEasy / ShopeeFood / e-voucher**.

**Nguồn dữ liệu của ChromaDB:** `STANDARDIZED_DIR.rglob("*.md")` — **chỉ**
`data/standardized/`, và **chỉ** file `.md`. `sources.csv` nằm cùng thư mục nhưng
không bị nạp. Không có đường nào khác đi vào index.

**Xử lý:** tiến trình index đang chạy đã nạp 27 file cũ vào bộ nhớ từ lúc khởi
động → nếu để chạy tiếp sẽ ghi cả 19 file đã xoá vào ChromaDB. Đã `pkill` tiến
trình đó, `rm -rf chroma_db/`, chạy lại từ đầu.

> **Phản biện 1 — đây chính xác là cái bẫy ghi ở đầu file này.** "Đổi corpus phải
> xoá `chroma_db/` trước khi reindex." Lần này nó suýt xảy ra thật, và ở dạng khó
> thấy hơn: không phải collection cũ còn sót, mà là **tiến trình đang chạy giữ
> snapshot corpus cũ trong RAM**. `reset_collection()` không cứu được, vì nó chạy
> *bên trong* chính tiến trình đó và sẽ index lại 27 file cũ ngay sau khi xoá.
> Quy tắc rút ra: **sửa corpus → kill mọi tiến trình index đang chạy, rồi mới
> reindex.** Không có cách nào phát hiện tự động trong code hiện tại.
>
> **Phản biện 2 — corpus 8 file làm yếu vài kết luận trước đó.** Mọi con số ở mục
> 6 và 8 (740/724 chunks, 532 `both`, 28 chunk có h2/h3) đo trên corpus 27 file,
> **không còn đúng**. Phải đo lại. Riêng nhận định "22/27 file không có heading
> phân cấp" thì đảo chiều: trong 8 file còn lại, 4 file news **có** `##`, nên tỉ
> lệ chunk hưởng lợi từ `markdown_header` sẽ **cao hơn hẳn** — đáng đo lại trước
> khi kết luận giữ `recursive`.
>
> **Phản biện 3 — 8 file có đủ cho RAGAS ở CP5 không?** Corpus hẹp thì retrieval
> chính xác hơn (ít nhiễu), nhưng golden dataset phải hỏi **trong phạm vi** 8 file
> này. Câu hỏi về vận chuyển, trả hàng, sản phẩm cấm… giờ **không có tài liệu để
> trả lời** → context_recall sẽ tụt thảm. Kiểm tra
> `group_project/evaluation/golden_dataset.json` khớp với corpus mới trước CP5.

---
