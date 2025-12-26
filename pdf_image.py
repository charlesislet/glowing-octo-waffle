import os
import fitz  # pymupdf
from pathlib import Path
import config

# ──────────────────────────────────────────────────────────────
# 👉 直接在此處設定要轉換的 PDF 檔案路徑、輸出資料夾與解析度倍率
PDF_PATH   = r"I:\KM\manual.pdf"   # <-- 只要改成自己的路徑
OUT_DIR    = config.STATIC_DIR / "pages"       # 輸出資料夾（程式會自動建立）
ZOOM_RATIO = 5.0                                # 1.0 = 72 DPI，數值越大解析度越高
# ──────────────────────────────────────────────────────────────

def _normalize_doc_name(pdf_path: Path) -> str:
    """Convert PDF filename to a safe doc name used in image paths."""
    return pdf_path.stem.replace(" ", "_")


def pdf_to_images(pdf_path: str, out_dir: str | os.PathLike = OUT_DIR, zoom: float = 2.0, page_range: tuple[int, int] | None = None,):
    """把 PDF 每一頁存成 PNG 圖片。"""
    pdf_path = Path(pdf_path)
    if not pdf_path.is_file():
        raise FileNotFoundError(f"找不到檔案: {pdf_path}")

    out_dir = Path(out_dir)
    doc_name = _normalize_doc_name(pdf_path)
    doc_dir = out_dir / doc_name
    doc_dir.mkdir(parents=True, exist_ok=True)

    doc = fitz.open(str(pdf_path))
    mat = fitz.Matrix(zoom, zoom)  # 設定放大倍率，影響解析度


    # 若未提供 page_range，預設處理全部頁面
    if page_range is None:
        page_iter = range(doc.page_count)                     # 0‑based
    else:
        start, end = page_range
        if not (1 <= start <= end <= doc.page_count):
            raise ValueError(
                f"page_range 必須在 1 與 {doc.page_count} 之間，且 start ≤ end"
            )
        # 轉換成 0‑based range，結束頁含在內
        page_iter = range(start - 1, end)

    image_paths = []
    for page_num in page_iter:
        page = doc.load_page(page_num)
        pix = page.get_pixmap(matrix=mat)

        img_path = doc_dir / f"{doc_name}_page_{page_num + 1}.png"
        pix.save(str(img_path))
        image_paths.append(str(img_path))
        print(f"✅ 第 {page_num + 1} 頁 → {img_path}")

    doc.close()
    print("\n全部完成！圖片已儲存在:", out_dir)
    return image_paths


def get_page_info(image_path: str) -> dict:
    """
    從圖像檔案路徑解析頁面資訊

    Args:
        image_path: 圖像檔案路徑

    Returns:
        包含 doc_name 和 page_num 的字典
    """
    filename = Path(image_path).stem
    # 檔名格式: {pdf_name}_page_{page_num}
    parts = filename.rsplit('_page_', 1)

    if len(parts) == 2:
        doc_name = parts[0]
        page_num = int(parts[1])
    else:
        doc_name = filename
        page_num = 1

    return {
        'doc_name': doc_name,
        'page_num': page_num
    }



if __name__ == "__main__":
    # 直接使用上面設定好的常數執行
    pdf_to_images(PDF_PATH, OUT_DIR, ZOOM_RATIO)