import os
import pdfplumber
import pandas as pd
from openpyxl import Workbook
from PIL import Image, ImageEnhance, ImageFilter
import numpy as np

try:
    from rapidocr import EngineType, LangDet, ModelType, OCRVersion, RapidOCR

    engine = RapidOCR(
        params={
            "Det.engine_type": EngineType.TORCH,
            "Det.lang_type": LangDet.CH,
            "Det.model_type": ModelType.MOBILE,
            "Det.ocr_version": OCRVersion.PPOCRV5,
        }
    )
except ImportError:
    from rapidocr_onnxruntime import RapidOCR

    engine = RapidOCR(use_angle_cls=True, text_score=0.08)


def normalize_ocr_output(ocr_output):
    """Support both old tuple output and new RapidOCROutput objects."""
    if isinstance(ocr_output, tuple):
        return ocr_output[0]

    boxes = getattr(ocr_output, "boxes", None)
    txts = getattr(ocr_output, "txts", None)
    scores = getattr(ocr_output, "scores", None)

    if boxes is None or txts is None:
        return []

    result = []
    for idx, text in enumerate(txts):
        box = boxes[idx]
        score = scores[idx] if scores is not None and idx < len(scores) else None
        result.append([box, text, score])
    return result


def run_ocr(img_ndarray):
    return normalize_ocr_output(engine(img_ndarray))


def enhance_for_ocr(img):
    """提升對比度與銳化，改善低品質掃描頁的 OCR 辨識率。"""
    img = ImageEnhance.Contrast(img).enhance(1.5)
    img = ImageEnhance.Sharpness(img).enhance(2.0)
    return img


def ocr_cropped_image(pixmap_img, scale=2):
    """Run RapidOCR on a cropped or full-page PIL image and return text."""
    if pixmap_img is None:
        return ""

    w, h = pixmap_img.size
    resized_img = pixmap_img.resize((w * scale, h * scale), Image.Resampling.LANCZOS)
    resized_img = enhance_for_ocr(resized_img)
    img_ndarray = np.array(resized_img)

    result = run_ocr(img_ndarray)

    if result:
        texts = [line[1] for line in result]
        return " ".join(texts)
    return ""


def crop_with_padding(page, x0, top, x1, bottom, padding=4):
    """裁切頁面區塊時四邊加 padding，避免字元被切邊。"""
    pw, ph = float(page.width), float(page.height)
    bbox = (
        max(0, x0 - padding),
        max(0, top - padding),
        min(pw, x1 + padding),
        min(ph, bottom + padding),
    )
    return page.within_bbox(bbox).to_image(resolution=400).original


def is_word_inside_bbox(word, bbox, margin=1):
    x0, top, x1, bottom = bbox
    cx = (word["x0"] + word["x1"]) / 2
    cy = (word["top"] + word["bottom"]) / 2
    return (
        x0 - margin <= cx <= x1 + margin
        and top - margin <= cy <= bottom + margin
    )


def words_to_line_blocks(words, row_threshold=3):
    if not words:
        return []

    df_words = pd.DataFrame(words)
    df_words = df_words.sort_values(["top", "x0"]).reset_index(drop=True)
    df_words["row_group"] = (df_words["top"].diff() > row_threshold).cumsum()

    blocks = []
    for _, group in df_words.groupby("row_group"):
        group = group.sort_values("x0")
        text = " ".join(str(word["text"]).strip() for _, word in group.iterrows() if str(word["text"]).strip())
        if text:
            blocks.append({
                "top": float(group["top"].min()),
                "kind": "text",
                "rows": [[text]],
            })
    return blocks


def normalize_table_rows(table_rows):
    normalized_rows = []
    for row in table_rows:
        normalized_row = []
        for cell in row:
            if cell is None or str(cell).strip() == "":
                normalized_row.append("")
            else:
                normalized_row.append(cell)
        normalized_rows.append(normalized_row)
    return normalized_rows


def append_tables_and_outside_text(ws, page):
    found_tables = page.find_tables()
    table_bboxes = [table.bbox for table in found_tables]

    words = page.extract_words()
    outside_words = [
        word for word in words
        if not any(is_word_inside_bbox(word, bbox) for bbox in table_bboxes)
    ]

    blocks = words_to_line_blocks(outside_words)
    for table in found_tables:
        table_rows = normalize_table_rows(table.extract())
        if table_rows:
            blocks.append({
                "top": float(table.bbox[1]),
                "kind": "table",
                "rows": table_rows,
            })

    for block in sorted(blocks, key=lambda item: item["top"]):
        for row in block["rows"]:
            ws.append(row)
        if block["kind"] == "table":
            ws.append([])


def process_pdf_to_excel(pdf_path, excel_path):
    wb = Workbook()
    default_sheet = wb.active
    wb.remove(default_sheet)

    print(f"Processing PDF: {os.path.basename(pdf_path)}")

    with pdfplumber.open(pdf_path) as pdf:
        for i, page in enumerate(pdf.pages):
            page_num = i + 1
            sheet_name = f"Page_{page_num}"
            ws = wb.create_sheet(title=sheet_name)
            print(f"  Processing page {page_num}...")

            # Normal pages: extract tables and keep text outside table areas.
            found_tables = page.find_tables()

            if found_tables:
                append_tables_and_outside_text(ws, page)

            else:
                text_objects = page.extract_words()

                if not text_objects:
                    print(f"    [RapidOCR] Page {page_num} has no extractable text; using OCR...")
                    pil_img = page.to_image(resolution=450).original
                    w, h = pil_img.size
                    resized_img = pil_img.resize((w * 3, h * 3), Image.Resampling.LANCZOS)
                    resized_img = enhance_for_ocr(resized_img)
                    result = run_ocr(np.array(resized_img))

                    if not result:
                        print(f"    Page {page_num}: no OCR text detected.")
                        continue

                    data = [
                        {
                            "cx": (line[0][0][0] + line[0][1][0]) / 2,
                            "cy": (line[0][0][1] + line[0][2][1]) / 2,
                            "text": line[1],
                        }
                        for line in result
                    ]

                    df = pd.DataFrame(data).sort_values("cy").reset_index(drop=True)
                    row_threshold = 14 * 3  # ~14pt font at 3x scale
                    df["row_id"] = (df["cy"].diff() > row_threshold).cumsum()

                    for _, group in df.groupby("row_id"):
                        ws.append(group.sort_values("cx")["text"].tolist())

                else:
                    df_words = pd.DataFrame(text_objects)
                    df_words["row_group"] = (df_words["top"].diff() > 3).cumsum()

                    for _, group in df_words.groupby("row_group"):
                        group = group.sort_values("x0")
                        row_content = []

                        for _, word in group.iterrows():
                            text = word["text"].strip()

                            if not text:
                                cropped = crop_with_padding(
                                    page,
                                    word["x0"], word["top"],
                                    word["x1"], word["bottom"],
                                )
                                text = ocr_cropped_image(cropped, scale=3)

                            row_content.append(text)

                        ws.append([" ".join(row_content)])

    wb.save(excel_path)
    print(f"Excel saved: {excel_path}")


if __name__ == "__main__":
    input_pdf = r"D:\work2\Pdftoexcel\sample-tables.pdf (已受保護).pdf"
    output_excel = r"D:\work2\Pdftoexcel\output_result98%.xlsx"

    if os.path.exists(input_pdf):
        process_pdf_to_excel(input_pdf, output_excel)
    else:
        print(f"PDF not found: {input_pdf}")
