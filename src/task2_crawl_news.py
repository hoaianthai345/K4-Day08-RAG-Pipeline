"""
Task 2 — Crawl bài viết/hướng dẫn hỗ trợ khách hàng về thương mại điện tử.

Hướng dẫn:
    1. Crawl tối thiểu 5 bài viết từ trung tâm trợ giúp công khai của một sàn TMĐT.
    2. Sử dụng Crawl4AI hoặc thư viện crawling tương tự.
    3. Lưu output vào data/landing/news/
    4. Mỗi bài lưu 1 file JSON với metadata (url, title, date_crawled, content).

Cài đặt:
    pip install crawl4ai
    playwright install chromium   # bắt buộc — pip install crawl4ai KHÔNG tự tải browser binary,
                                   # thiếu bước này sẽ báo lỗi
                                   # "BrowserType.launch: Executable doesn't exist"

Gợi ý chủ đề: theo dõi đơn hàng, đổi phương thức thanh toán, bằng chứng hoàn tiền,
mua hàng xuyên biên giới.

Lưu ý: một số trang help center dùng JavaScript render (SPA) — nếu crawl về chỉ thấy
tiêu đề mà không có nội dung, đổi sang bài viết khác cùng domain thay vì cố xử lý.
"""

import asyncio
import json
import re
from datetime import datetime
from pathlib import Path

DATA_DIR = Path(__file__).parent.parent / "data" / "landing" / "news"
STANDARDIZED_DIR = Path(__file__).parent.parent / "data" / "standardized" / "news"


def setup_directory():
    """Tạo thư mục data/landing/news/ nếu chưa có."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)


ARTICLE_URLS = [
    "https://help.shopee.vn/portal/4/article/79583",
    "https://help.shopee.vn/portal/4/article/79563",
    "https://help.shopee.vn/portal/4/article/79521",
    "https://help.shopee.vn/portal/4/article/79076",
    "https://help.shopee.vn/portal/4/article/77265",
]


async def crawl_article(url: str) -> dict:
    """
    Crawl một bài viết và trả về dict chứa metadata + content.

    Returns:
        {
            "url": str,
            "title": str,
            "date_crawled": str (ISO format),
            "content_markdown": str
        }
    """
    from crawl4ai import AsyncWebCrawler

    async with AsyncWebCrawler() as crawler:
        result = await crawler.arun(url=url)
    if not getattr(result, "success", True):
        raise RuntimeError(f"Crawl failed for {url}: {getattr(result, 'error_message', '')}")
    metadata = getattr(result, "metadata", {}) or {}
    markdown = getattr(result, "markdown", "") or ""
    if len(markdown.strip()) < 200:
        raise RuntimeError(f"Crawled content is unexpectedly short for {url}")
    return {
        "url": url,
        "title": metadata.get("title") or "Unknown",
        "date_crawled": datetime.now().isoformat(timespec="seconds"),
        "content_markdown": markdown,
    }


async def crawl_all():
    """Crawl toàn bộ bài viết trong ARTICLE_URLS."""
    setup_directory()

    for i, url in enumerate(ARTICLE_URLS, 1):
        print(f"[{i}/{len(ARTICLE_URLS)}] Crawling: {url}")
        try:
            article = await crawl_article(url)
        except Exception as exc:
            print(f"  ✗ Failed: {exc}")
            continue

        # Lưu file JSON
        filename = f"article_{i:02d}.json"
        filepath = DATA_DIR / filename
        filepath.write_text(json.dumps(article, ensure_ascii=False, indent=2))
        print(f"  ✓ Saved: {filepath}")


def create_demo_articles_from_standardized(limit: int = 5) -> list[Path]:
    """Materialize JSON crawl fixtures from the curated Markdown corpus.

    This reproducible offline path is for the lab environment only; use
    ``crawl_all`` when collecting fresh web pages.
    """
    setup_directory()
    markdown_files = sorted(path for path in STANDARDIZED_DIR.glob("*.md") if path.name != ".gitkeep")[:limit]
    if len(markdown_files) < limit:
        raise RuntimeError(f"Need at least {limit} Markdown files in {STANDARDIZED_DIR}")
    output_paths = []
    for index, markdown_file in enumerate(markdown_files, 1):
        raw = markdown_file.read_text(encoding="utf-8")
        title_match = re.search(r"^title:\s*[\"']?(.*?)[\"']?\s*$", raw, re.MULTILINE)
        url_match = re.search(r"^source_url:\s*(.+)$", raw, re.MULTILINE)
        article = {
            "url": url_match.group(1).strip() if url_match else "not-stated",
            "title": title_match.group(1).strip() if title_match else markdown_file.stem,
            "date_crawled": datetime.now().isoformat(timespec="seconds"),
            "content_markdown": raw,
        }
        output = DATA_DIR / f"article_{index:02d}.json"
        output.write_text(json.dumps(article, ensure_ascii=False, indent=2), encoding="utf-8")
        output_paths.append(output)
    return output_paths


if __name__ == "__main__":
    if not ARTICLE_URLS:
        print("⚠ Hãy điền ARTICLE_URLS trước khi chạy!")
        print("Gợi ý: tìm trang hướng dẫn/hỗ trợ khách hàng trên help center của sàn TMĐT")
    else:
        asyncio.run(crawl_all())
