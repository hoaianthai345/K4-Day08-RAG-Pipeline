"""
Task 1 — Thu thập văn bản chính sách thương mại điện tử / hỗ trợ khách hàng.

Hướng dẫn:
    1. Tìm tối thiểu 3 văn bản chính sách (PDF/DOCX) từ trang chính thức của một sàn TMĐT.
    2. Tải về và lưu vào data/landing/legal/
    3. Đặt tên file rõ ràng, không dấu, mô tả đúng nội dung.

Gợi ý nguồn (ví dụ trang công khai Shopee Vietnam — help.shopee.vn):
    - https://help.shopee.vn/portal/4/article/77251 (Chính sách trả hàng và hoàn tiền)
    - https://help.shopee.vn/portal/4/article/79198 (Phương thức thanh toán)
    - https://help.shopee.vn/portal/4/article/77244 (Chính sách bảo mật)

Gợi ý văn bản (chủ đề chính sách thương mại điện tử):
    - Chính sách đổi trả/hoàn tiền (Returns/Refund Policy)
    - Phương thức thanh toán (Payment Methods)
    - Chính sách bảo mật (Privacy Policy)
    - Quy định đăng bán sản phẩm cho người bán (Seller Listing Regulations)

Nhớ gắn metadata `customer_role` (`buyer`/`seller`/`both`) cho từng tài liệu — yêu cầu riêng
của K4 Variant (kế thừa từ Lab 07), cần thiết để viết benchmark query dùng metadata_filter.

Lưu ý: một số trang help center dùng JavaScript render nội dung (SPA) — crawl về chỉ thấy
tiêu đề mà không có nội dung thật. Đổi sang bài viết khác cùng domain thay vì cố xử lý,
và chỉ dùng nguồn công khai/được phép chia sẻ.
"""

import re
import unicodedata
from pathlib import Path

import requests

DATA_DIR = Path(__file__).parent.parent / "data" / "landing" / "legal"
STANDARDIZED_DIR = Path(__file__).parent.parent / "data" / "standardized" / "legal"


def setup_directory():
    """Tạo thư mục data/landing/legal/ nếu chưa có."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Directory ready: {DATA_DIR}")


def download_file(url: str, filename: str, timeout: int = 30) -> Path:
    """Download one original PDF/DOCX file into the landing zone.

    The helper intentionally rejects HTML responses: a help-centre page is not
    an original legal document and must not be relabelled as PDF/DOCX.
    """
    setup_directory()
    path = DATA_DIR / filename
    if path.suffix.lower() not in {".pdf", ".doc", ".docx"}:
        raise ValueError("filename must end in .pdf, .doc, or .docx")
    response = requests.get(url, timeout=timeout, stream=True)
    response.raise_for_status()
    if "text/html" in response.headers.get("content-type", "").lower():
        raise ValueError(f"{url} returned HTML, not an original document")
    payload = response.content
    if len(payload) <= 1024:
        raise ValueError(f"Downloaded file is too small ({len(payload)} bytes)")
    path.write_bytes(payload)
    print(f"✓ Saved: {path}")
    return path


def download_documents(documents: dict[str, str]) -> list[Path]:
    """Download a mapping of ``filename -> direct URL``."""
    return [download_file(url, filename) for filename, url in documents.items()]


def create_demo_legal_documents(limit: int = 3) -> list[Path]:
    """Create PDF fixtures from pre-standardized Markdown for an offline demo.

    These are explicitly reproducible fixtures, not replacements for original
    legal files. They allow the Task 1 -> Task 3 pipeline and its tests to run
    when the team has only the curated Markdown corpus.
    """
    from fpdf import FPDF

    setup_directory()
    sources = sorted(path for path in STANDARDIZED_DIR.glob("*.md") if path.name != ".gitkeep")[:limit]
    if len(sources) < limit:
        raise RuntimeError(f"Need at least {limit} Markdown files in {STANDARDIZED_DIR}")
    outputs = []
    for source in sources:
        # Built-in FPDF fonts are Latin-1. Normalising preserves a readable
        # offline fixture while the canonical Vietnamese corpus stays untouched.
        text = unicodedata.normalize("NFKD", source.read_text(encoding="utf-8"))
        text = text.encode("latin-1", "replace").decode("latin-1")
        text = re.sub(r"\n{3,}", "\n\n", text)
        output = DATA_DIR / f"{source.stem}.pdf"
        pdf = FPDF()
        pdf.set_auto_page_break(auto=True, margin=15)
        pdf.add_page()
        pdf.set_font("Helvetica", size=10)
        for paragraph in text.splitlines():
            # Reset x to the left margin after every cell; otherwise FPDF keeps
            # the cursor at the right edge and eventually has no usable width.
            pdf.multi_cell(0, 5, paragraph or " ", new_x="LMARGIN", new_y="NEXT")
        pdf.output(str(output))
        outputs.append(output)
    return outputs


# Download PDF/DOCX trực tiếp bằng ``download_file`` hoặc ``download_documents``.
#
# Ví dụ nếu có direct link:
#
# import requests
#
# def download_file(url: str, filename: str):
#     response = requests.get(url)
#     filepath = DATA_DIR / filename
#     filepath.write_bytes(response.content)
#     print(f"✓ Đã tải: {filepath}")
#
# Nếu trang là HTML thuần (không phải PDF sẵn), có thể convert nội dung text
# thành PDF đơn giản bằng thư viện fpdf2 (đã có trong requirements.txt).


if __name__ == "__main__":
    setup_directory()
