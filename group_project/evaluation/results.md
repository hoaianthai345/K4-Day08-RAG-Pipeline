# RAG Evaluation Results

**Ngày đo:** 2026-08-04 · **Corpus:** 8 tài liệu → 92 chunks · **Golden set:** 24 câu.
**Bảng điểm thô từng câu:** `raw_A_hybrid_rerank.csv`, `raw_B_baseline_no_rerank.csv`
(chạy lại `python -m group_project.evaluation.eval_pipeline` sẽ sinh thêm `results_raw_tables.md`)

## Cấu hình hệ thống

| Thành phần | Giá trị |
|---|---|
| Chunking | `RecursiveCharacterTextSplitter`, size 800 / overlap 100, bỏ mảnh < 50 ký tự |
| Embedding | `text-embedding-3-small` (1536 chiều) |
| Vector store | ChromaDB, `hnsw:space=cosine` |
| Sparse | BM25Okapi (k1=1.5, b=0.75) trên **cùng bộ 92 chunk** |
| Fusion | RRF, k = 60 |
| Rerank | LLM judge (`RERANK_METHOD=openai`) |
| Generation | `gpt-4o-mini`, reorder chống "lost in the middle" |

## Framework sử dụng

**RAGAS** cho 4 chỉ số bắt buộc, cộng thêm **hai bộ đo tự viết** bổ trợ:

| Bộ đo | Cần LLM | Tính tất định | Đo gì |
|---|---|---|---|
| `retrieval_eval.py` | Không | ✅ Có | Tầng retrieval, dựa trên `evidence` |
| RAGAS | Có | ❌ Không | Chất lượng câu trả lời + context |
| `evaluate_refusal()` | Không | ✅ Có | Khả năng từ chối câu ngoài phạm vi |

Lý do tách: RAGAS dùng LLM làm giám khảo nên **có phương sai** — chạy hai lần ra hai số khác nhau (xem mục "Phương sai" bên dưới, có số liệu thật). Muốn so sánh A/B chắc chắn thì cần thước đo tất định làm nền.

---

## Golden set

24 câu viết từ nội dung thật của 8 file corpus, mỗi câu kèm `evidence` là câu văn nguyên bản → kiểm chứng được bằng máy.

| Nhóm | Số câu | Mục đích |
|---|---|---|
| `procedural`, `factual`, `numeric` | 10 | Truy xuất cơ bản |
| `multi_fact` | 3 | Đáp án gồm nhiều ý → đo context_recall |
| `inference` | 3 | Diễn đạt khác hẳn tài liệu → thiên về semantic |
| `disambiguation` | 5 | **Cặp đối chứng** Q15↔Q16, Q18↔Q19 |
| `out_of_scope` | 3 | **Bắt buộc từ chối**, không được bịa |

Độ khó: 8 easy / 8 medium / 8 hard.

**Vì sao có nhóm `disambiguation`.** Ba văn bản SPayLater / SEasy Cho Vay Người Bán / SEasy Vay Tiền Nhanh là boilerplate gần trùng nhau. Q15 và Q16 hỏi **cùng một dạng** ("định nghĩa Bên vay") nhưng đáp án nằm ở **hai file khác nhau**. Nếu hệ thống trả về cùng một chunk cho cả hai thì retrieval không phân biệt được văn bản gần trùng.

**Vì sao có nhóm `out_of_scope`.** Golden set chỉ gồm câu trả lời được sẽ **không bao giờ phát hiện hệ thống bịa**. Q24 là bẫy có chủ đích: hỏi phí vận chuyển, mà cụm "miễn phí vận chuyển" **có** xuất hiện trong điều khoản SPayLater, nên retrieval chắc chắn trả về chunk với điểm không thấp.

---

## Overall Scores — RAGAS (n = 21 câu trả lời được)

| Metric | **Config B**<br>RRF thuần | **Config A**<br>hybrid + rerank | Δ |
|--------|--------------------------|--------------------------------|---|
| Faithfulness | 0.8889 | **0.8980** | +1.0% |
| Answer Relevance | 0.7774 | **0.8585** | **+10.4%** |
| Context Recall | 0.8889 | 0.8889 | **0.0%** |
| Context Precision | 0.8497 | **0.9859** | **+16.0%** |
| **Average** | 0.8512 | **0.9078** | **+6.6%** |

## Overall Scores — Retrieval (tất định, n = 21)

| Method | hit@5 | MRR | prec@5 | **ev_hit@5** | **ev_MRR** |
|---|---|---|---|---|---|
| Semantic (dense) | 1.000 | 0.968 | 0.533 | 0.810 | 0.668 |
| BM25 (sparse) | 1.000 | 0.976 | 0.543 | 0.905 | 0.819 |
| **Hybrid + rerank** | 1.000 | 0.976 | **0.590** | **0.952** | **0.929** |

Hai mức chấm: `hit`/`MRR`/`prec` = chunk đến đúng **file**; `ev_hit`/`ev_MRR` = chunk thực sự **chứa câu văn** mang đáp án.

**`hit@5 = 1.000` cho cả ba là chỉ số vô dụng, không phải hệ thống hoàn hảo.** Corpus chỉ có 8 file, lấy top-5 thì trúng đúng file gần như chắc chắn. Chỉ khi siết xuống mức `ev_*` thì ba phương pháp mới tách nhau ra. **Chỉ số đạt trần ngay lần đo đầu là dấu hiệu bài đo quá dễ.**

### Phân tích theo độ khó (`ev_hit_rate`)

| Method | easy | medium | hard | Trượt câu |
|---|---|---|---|---|
| Semantic | 1.000 | 0.875 | 0.500 | Q16, Q17, Q18, Q19 |
| BM25 | 1.000 | 1.000 | 0.667 | Q16, Q17 |
| Hybrid + rerank | 1.000 | 1.000 | **0.833** | Q16 |

Semantic trượt **cả 4 câu `disambiguation`** — đúng như golden set dự đoán: `text-embedding-3-small` nén ba văn bản gần trùng về những vector quá giống nhau. BM25 gỡ 2 câu nhờ khớp từ khoá riêng ("cấn trừ" chỉ có trong file người bán), rerank gỡ thêm 1 câu.

---

## A/B Comparison Analysis

**Config A:** Semantic + BM25 → RRF (k=60) → **LLM rerank** → top-5 → reorder → generate
**Config B:** Semantic + BM25 → RRF (k=60) → top-5 → reorder → generate *(giống hệt, chỉ tắt rerank)*

### Kết luận: Config A thắng, và thắng đúng ở nơi lý thuyết dự đoán

Điều thuyết phục nhất không phải "A cao hơn B", mà là **mẫu hình phân bổ của mức tăng**:

| Chỉ số | Δ | Vì sao đúng như vậy |
|---|---|---|
| **Context Recall** | **0.0%** | Rerank **không lấy về tài liệu mới**, nó chỉ sắp xếp lại tập đã có. Recall *phải* không đổi. Nếu chỉ số này nhảy lên thì phép đo có vấn đề. |
| **Context Precision** | **+16.0%** | Precision đo *chunk đúng có được xếp lên trên không* — đây **chính xác là việc rerank làm**. Mức tăng lớn nhất rơi vào đây là bằng chứng rerank hoạt động đúng cơ chế. |
| Answer Relevance | +10.4% | Context tốt hơn ở vị trí đầu → câu trả lời bám trọng tâm hơn. |
| Faithfulness | +1.0% | Đã cao sẵn (0.89), ít dư địa. **Xem mục phương sai — mức này nằm trong nhiễu.** |

Kết quả retrieval tất định xác nhận cùng một câu chuyện: `hit@5` và `MRR` **không đổi** (cùng tập file), nhưng `ev_MRR` nhảy **0.726 → 0.952 (+31%)** — đoạn văn chứa đáp án được kéo từ hạng 2–3 lên hạng 1.

### Ghi chú lịch sử: rerank từng là lệnh rỗng

Ở lần đo trước, rerank cho chênh lệch **bằng 0**. Nguyên nhân: `RERANK_METHOD="rrf"`, mà nhánh `"rrf"` trong `rerank()` chỉ sắp xếp lại theo đúng khoá `score` mà `rerank_rrf()` đã sắp ở dòng trên — bước "Rerank" trong sơ đồ **không làm gì ngoài cắt bớt danh sách**. Chỉ khi đổi sang `"openai"` nó mới thật sự chấm lại. Con số +16% / +31% là chênh lệch giữa **có rerank thật** và **không rerank**.

---

## Khả năng từ chối câu ngoài phạm vi (n = 3)

RAGAS **không đo được** nhóm này — `context_recall` trên câu không có đáp án đúng là vô nghĩa. Nhưng đây mới là chỗ hallucination lộ ra, nên đo tách bằng `evaluate_refusal()`.

| Config | Refusal rate | Q22 | Q23 | Q24 |
|---|---|---|---|---|
| B: RRF thuần | **3/3 = 1.000** | ✅ | ✅ | ✅ |
| A: hybrid + rerank | **3/3 = 1.000** | ✅ | ✅ | ✅ |

Câu trả lời thực tế cho cả 3: *"Tôi không thể xác minh thông tin này từ nguồn hiện có."*

**Kết quả này lật ngược một kết luận khác của báo cáo.** Phần hiệu chuẩn ngưỡng bên dưới chứng minh **không ngưỡng cosine nào** tách được câu trong/ngoài phạm vi. Nhưng hệ thống vẫn từ chối **đúng 3/3** — vì việc phán đoán "đủ căn cứ hay không" đang do **LLM ở Task 10** đảm nhiệm, không phải do ngưỡng.

Nói cách khác: **ngưỡng cosine của Task 9 không hoạt động, nhưng điều đó không gây hại**, vì tầng sau đã bắt trọn.

---

## Hiệu chuẩn ngưỡng fallback

`task9` kích hoạt PageIndex fallback khi `dense_results[0].score < SCORE_THRESHOLD`. Đo điểm cosine top-1 trên toàn bộ 24 câu:

```
Câu trả lời được  (n=21):  min 0.532   max 0.826
Câu ngoài phạm vi (n=3):   min 0.503   max 0.621
                                 ↑
                       HAI PHÂN BỐ CHỒNG LẤN
```

Câu **Q20 trả lời được** chỉ đạt 0.532 — **thấp hơn** câu **Q24 ngoài phạm vi** (0.621).

| Ngưỡng | Bắt đúng OOS | Báo động nhầm | Tổng lỗi |
|---|---|---|---|
| 0.30 (code hiện tại) | 0/3 | 0/21 | 3 |
| **0.48 (spec đề bài)** | **0/3** | 0/21 | **3** |
| 0.52 | 1/3 | 0/21 | **2** |
| 0.56 | 2/3 | 2/21 | 3 |
| 0.64 | 3/3 | 6/21 | 6 |

**Không ngưỡng nào tách được hai nhóm.** Đây không phải lỗi chọn sai số mà là **giới hạn của cách tiếp cận "một ngưỡng cosine"**. Giữ 0.48 cho khớp barem — nó không kích hoạt, nhưng vô hại vì tầng LLM đã chặn trọn.

---

## Phương sai của RAGAS — đo được, không phải suy đoán

Chạy **cùng một code, cùng một dữ liệu, hai lần liên tiếp**:

| Metric | Config | Lượt 1 | Lượt 2 | Chênh |
|---|---|---|---|---|
| Faithfulness | B | 0.8492 | 0.8889 | **0.040** |
| Faithfulness | A | 0.9083 | 0.8980 | 0.010 |
| Answer Relevance | A | 0.7961 | 0.8585 | **0.062** |
| Context Recall | A | 0.9127 | 0.8889 | 0.024 |

Phương sai lên tới **0.06** — lớn hơn cả khoảng cách A/B của Faithfulness (0.009). Hệ quả trực tiếp:

- **Faithfulness +1.0% là nhiễu, không phải cải thiện.** Không được kết luận gì từ nó.
- **Context Precision +16.0% và Answer Relevance +10.4% thì đủ lớn** để vượt phương sai → kết luận được.
- Ở lượt 1, Answer Relevance gần như không đổi giữa A và B (0.780 vs 0.796), khiến bản nháp báo cáo kết luận nhầm rằng *"nút thắt nằm ở prompt, retrieval hết tác dụng"*. Lượt 2 cho +10.4% — **kết luận đó đã bị bác bỏ**.

**Khi trình bày A/B phải kèm phương sai, nếu không sẽ đọc nhiễu thành tín hiệu.**

---

## Worst Performers

| # | Câu | Vấn đề | Failure Stage | Root Cause |
|---|---|---|---|---|
| 1 | **Q16** — "Định nghĩa Bên vay trong điều khoản SPayLater" | Cả 3 phương pháp đều trượt | **Retrieval** | Ba văn bản định nghĩa "Bên vay" bằng câu chữ gần y hệt. Embedding nén chúng về vector gần trùng; BM25 thấy cùng bộ từ khoá. Rerank cũng không tách nổi vì bản thân đoạn văn khác nhau rất ít. |
| 2 | **Q17** — "Shopee có được tự trừ tiền tài khoản người bán" | Semantic + BM25 trượt, rerank gỡ được | **Retrieval** | Từ khoá phân biệt là "cấn trừ" — chỉ có trong file người bán. Semantic làm mờ thuật ngữ hiếm này. |
| 3 | **Q24** — "Phí vận chuyển tính thế nào" (ngoài phạm vi) | Điểm cosine 0.621, cao hơn 1 câu trả lời được | **Fallback** | Cụm "miễn phí vận chuyển" có thật trong điều khoản SPayLater → cosine cao giả tạo. *Đã được tầng LLM chặn đúng.* |

---

## Recommendations

### Cải tiến 1 — Đánh giá lại lựa chọn embedding

**Action:** So `text-embedding-3-small` với `BAAI/bge-m3` trên cùng golden set, rồi chọn theo số liệu.

**Expected impact:** Đã có số đo từ phiên trước: bge-m3 đạt `ev_MRR` **0.806**, `text-embedding-3-small` chỉ **0.668** — thấp hơn **17%** trên corpus tiếng Việt, và trượt **4/4** câu `disambiguation` so với **2/4** của bge-m3. Hybrid + rerank đang gánh phần thiếu hụt này. Đổi lại bge-m3 còn bỏ được chi phí API và phụ thuộc mạng khi demo.

### Cải tiến 2 — Cân bằng lại corpus

**Action:** 84/92 chunk là văn bản pháp lý, chỉ 8 chunk là hướng dẫn; bốn file news mỗi file ra **đúng 1 chunk**. Bổ sung tài liệu hướng dẫn, hoặc hạ `CHUNK_SIZE` riêng cho nhóm news.

**Expected impact:** `prec@5` hiện chỉ 0.53–0.59 vì top-5 luôn bị chunk pháp lý chiếm chỗ. Câu hỏi vận hành ("làm sao đặt món ShopeeFood") chỉ có **1 chunk duy nhất** để trả lời — trượt là hỏng, không có dự phòng.

### Cải tiến 3 — Tăng cỡ mẫu trước khi kết luận thêm

**Action:** Nâng golden set lên 60–80 câu, nhóm `out_of_scope` lên 8–10 câu. Chạy RAGAS 3 lượt và báo cáo trung bình ± độ lệch.

**Expected impact:** Với n=21, mỗi câu nặng 4.8%; phương sai RAGAS đo được lên tới 0.06. Ở cỡ mẫu này chỉ kết luận chắc được về Context Precision (+16%) và ev_MRR (+31%). Muốn kết luận về Faithfulness (chênh 0.009) thì bắt buộc phải tăng mẫu.

---

## Giới hạn của báo cáo này

**n = 21 câu trả lời được, 3 câu ngoài phạm vi.** Mỗi câu nặng 4.8% `ev_hit_rate`; chênh lệch giữa 0.952 và 0.905 chỉ là **1 câu**. Riêng 3 câu `out_of_scope` là quá ít để ước lượng tin cậy tỉ lệ từ chối.

**Golden set do chính người xây hệ thống viết.** Câu hỏi được soạn *sau khi* đọc corpus nên vô thức bám theo từ ngữ tài liệu → **điểm cao hơn thực tế**. Đã giảm thiểu bằng Q04/Q05/Q10 (diễn đạt khác hẳn), nhưng cách chữa thật là để thành viên khác viết câu hỏi mà không đọc corpus trước.

**Ba lỗi trong chính bộ đo, đã phát hiện và sửa:**

1. `context_precision` **mất âm thầm** ở lượt đo đầu — ragas 0.2.x đặt tên cột là `llm_context_precision_with_reference`, filter chỉ tìm `context_precision`. Bảng vẫn in ra bình thường, không một cảnh báo nào.
2. `refusal_rate` báo **0.000** trong khi thực tế là **1.000** — danh sách `REFUSAL_MARKERS` thiếu biến thể *"không thể xác minh"* mà prompt Task 10 dùng. Nếu tin con số đó, báo cáo sẽ kết luận **ngược 180 độ**: rằng hệ thống bịa đặt, trong khi nó đang làm đúng.
3. `export_results()` ghi đè thẳng vào `results.md`, **xoá sạch báo cáo viết tay** một lần. Đã đổi đích sang `results_raw_tables.md` — số liệu thô và phần diễn giải phải nằm ở hai file khác nhau.

Hai lỗi đầu **không làm chương trình dừng** — chúng chỉ âm thầm cho ra con số sai. Đã bổ sung: cảnh báo liệt kê cột float chưa nhận diện, và dump dataframe thô ra `raw_*.csv`. **Một bộ đo im lặng nguy hiểm hơn một bộ đo báo lỗi.**

**Ghi chú môi trường:** `ragas==0.1.21` ghim trong `requirements.txt` **deadlock trên Python 3.13.7** (chạy 2 giờ 04 phút, dùng 3.23 giây CPU, 0 kết nối TCP). Phải nâng lên `ragas==0.2.15` kèm stub cho `langchain_community.chat_models.vertexai` (module đã bị gỡ ở langchain-community ≥ 0.4). **Cần cập nhật `requirements.txt`: `ragas>=0.2.15` + `tabulate`.**
