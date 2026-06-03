import os
import pdfplumber
import pandas as pd
from openpyxl import Workbook
from PIL import Image
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


def ocr_cropped_image(pixmap_img, scale=2):
    """Run RapidOCR on a cropped or full-page PIL image and return text."""
    if pixmap_img is None:
        return ""

    w, h = pixmap_img.size
    resized_img = pixmap_img.resize((w * scale, h * scale), Image.Resampling.LANCZOS)
    img_ndarray = np.array(resized_img)

    result = run_ocr(img_ndarray)

    if result:
        texts = [line[1] for line in result]
        return " ".join(texts)
    return ""


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


def normalize_page14_text(value):
    text = "" if value is None else str(value).strip()
    return (
        text.replace("指数", "指數")
        .replace("镇", "鎮")
        .replace("壮图鄉", "壯圍鄉")
        .replace("莫山鄉", "員山鄉")
        .replace("涨跌", "漲跌")
        .replace("别", "別")
        .replace("燮動", "變動")
        .replace(" ", "")
        .replace(" ", "")
    )


def normalize_page14_number(value):
    text = normalize_page14_text(value)
    text = text.replace("。", ".").replace("．", ".")
    return text


def is_page14_data_row(row):
    if not row:
        return False
    first = normalize_page14_text(row[0])
    return "年" in first and "月" in first


def is_page14_month_row(row):
    if not row:
        return False
    first = normalize_page14_text(row[0])
    return "年" in first and "月" in first and "日" not in first


def write_page14_upper_rows(ws, upper_rows):
    """Rewrite the upper Page 14 monthly table with a fixed header."""
    ws.append(["蘇澳鎮(總指數)"])
    ws.append(["月份", "反算（原", "價格日期", "113.3.31", "114.3.31"])

    for row in upper_rows:
        if not is_page14_month_row(row):
            continue

        cleaned = [normalize_page14_number(cell) for cell in row if normalize_page14_number(cell)]
        if len(cleaned) < 2:
            continue

        ws.append(cleaned)


def write_page14_rows(ws, ocr_rows):
    """Rewrite the lower Page 14 OCR table into a stable report layout."""
    city_row_idx = None
    for idx, row in enumerate(ocr_rows):
        row_text = "".join(normalize_page14_text(cell) for cell in row)
        if "高雄市" in row_text and "宜蘭" in row_text:
            city_row_idx = idx
            break

    if city_row_idx is None:
        for row in ocr_rows:
            ws.append(row)
        return

    write_page14_upper_rows(ws, ocr_rows[:city_row_idx])

    ws.append(["", "", "", "", "高雄市", "宜蘭縣"])
    ws.append(["", "期", "別"])
    ws.append(["", "", "", "甲仙區", "總指數", "宜蘭市", "羅東鎮", "蘇澳鎮", "頭城鎮", "礁溪鄉", "壯圍鄉", "員山鄉"])

    data_rows = []
    for row in ocr_rows[city_row_idx + 1:]:
        if is_page14_data_row(row):
            cleaned = [normalize_page14_number(cell) for cell in row if normalize_page14_number(cell)]
            data_rows.append(cleaned)

    index_labels = {1: "定", 4: "基", 7: "指", 10: "數"}
    rate_label_chars = ["對", "上", "期", "漲", "跌", "率", "（%）"]

    for idx, row in enumerate(data_rows):
        if idx < 11:
            label = index_labels.get(idx, "")
        else:
            label_idx = idx - 11
            label = rate_label_chars[label_idx] if label_idx < len(rate_label_chars) else ""
        ws.append([label, "", row[0]] + row[1:])


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

            # Page 14: full-page OCR, then rebuild rows and columns by OCR boxes.
            if page_num == 14:
                print("    [RapidOCR] Page 14 uses OCR box data...")

                pil_img = page.to_image(resolution=450).original
                w, h = pil_img.size
                resized_img = pil_img.resize((w * 3, h * 3), Image.Resampling.LANCZOS)

                img_ndarray = np.array(resized_img)
                result = run_ocr(img_ndarray)

                if not result:
                    print("    Page 14: no OCR text detected.")
                    continue

                data = []
                for line in result:
                    box = line[0]  # [[x0,y0], [x1,y1], [x2,y2], [x3,y3]]
                    text = line[1]
                    cx = (box[0][0] + box[1][0]) / 2
                    cy = (box[0][1] + box[2][1]) / 2
                    data.append({"cx": cx, "cy": cy, "text": text})

                df_page14 = pd.DataFrame(data)
                df_page14 = df_page14.sort_values(by="cy").reset_index(drop=True)
                df_page14["row_id"] = 0

                current_row = 0
                current_y = df_page14.loc[0, "cy"]
                row_threshold = 14 * 3

                for idx, row in df_page14.iterrows():
                    if row["cy"] - current_y > row_threshold:
                        current_row += 1
                        current_y = row["cy"]
                    df_page14.at[idx, "row_id"] = current_row

                page14_rows = []
                for _, group in df_page14.groupby("row_id"):
                    group = group.sort_values(by="cx")
                    row_cells = [item["text"] for _, item in group.iterrows()]
                    page14_rows.append(row_cells)

                write_page14_rows(ws, page14_rows)

                continue

            # Normal pages: extract tables and keep text outside table areas.
            found_tables = page.find_tables()

            if found_tables:
                append_tables_and_outside_text(ws, page)

            else:
                text_objects = page.extract_words()

                if not text_objects:
                    print(f"    [RapidOCR] Page {page_num} has no extractable text; using OCR...")
                    pil_img = page.to_image(resolution=200).original
                    page_text = ocr_cropped_image(pil_img, scale=2)

                    for row_text in page_text.split("\n"):
                        ws.append([row_text])
                else:
                    df_words = pd.DataFrame(text_objects)
                    df_words["row_group"] = (df_words["top"].diff() > 3).cumsum()

                    for _, group in df_words.groupby("row_group"):
                        group = group.sort_values("x0")
                        row_content = []

                        for _, word in group.iterrows():
                            text = word["text"].strip()

                            if not text:
                                bbox = (word["x0"], word["top"], word["x1"], word["bottom"])
                                cropped = page.within_bbox(bbox).to_image(resolution=300).original
                                text = ocr_cropped_image(cropped, scale=3)

                            row_content.append(text)

                        ws.append([" ".join(row_content)])

    wb.save(excel_path)
    print(f"Excel saved: {excel_path}")


if __name__ == "__main__":
    input_pdf = r"D:\work\0601Yilan\1150213.pdf"
    output_excel = r"D:\work\0601Yilan\output_result98%.xlsx"

    if os.path.exists(input_pdf):
        process_pdf_to_excel(input_pdf, output_excel)
    else:
        print(f"PDF not found: {input_pdf}")
