import os
import re
import pdfplumber
import pandas as pd
from openpyxl import Workbook
from PIL import Image, ImageEnhance, ImageFilter
import numpy as np
from opencc import OpenCC

cc = OpenCC('s2twp')

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

    engine = RapidOCR(use_angle_cls=True, text_score=0.45)


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
    img = ImageEnhance.Contrast(img).enhance(8.0)
    img = ImageEnhance.Sharpness(img).enhance(8.0)
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
        texts = [cc.convert(line[1]) for line in result]
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


def get_ocr_blocks_for_page(page, table_bboxes, extracted_words, col_threshold=10, row_threshold=6):
    """
    針對第 14 頁進行全頁 OCR 解析，具備『動態軌道均值分行』與『全域防碰撞網格』機制。
    """
    pil_img = page.to_image(resolution=450).original
    w, h = pil_img.size
    scale = (w * 3) / float(page.width) 
    
    resized_img = pil_img.resize((w * 3, h * 3), Image.Resampling.LANCZOS)
    resized_img = enhance_for_ocr(resized_img)
    result = run_ocr(np.array(resized_img))

    if not result:
        return []

    raw_ocr_elements = []
    for line in result:
        poly = np.array(line[0])
        x0, top = np.min(poly, axis=0) / scale
        x1, bottom = np.max(poly, axis=0) / scale
        text = cc.convert(line[1])

        # 進行潛在的密合文字前置拆分（加上防呆以防外部未定義 advanced_split_text）
        try:
            parts = advanced_split_text(text)
        except NameError:
            parts = [p.strip() for p in re.split(r'\s{2,}', text) if p.strip()]

        if len(parts) > 1:
            total_chars = sum(len(p) for p in parts)
            full_w = x1 - x0
            curr_x = x0
            for p in parts:
                part_w = full_w * (len(p) / total_chars)
                raw_ocr_elements.append({
                    "x0": curr_x, "x1": curr_x + part_w,
                    "top": top, "bottom": bottom, "text": p
                })
                curr_x += part_w
        else:
            if text.strip():
                raw_ocr_elements.append({
                    "x0": x0, "x1": x1, "top": top, "bottom": bottom, "text": text.strip()
                })

    # 排除重複
    ocr_blocks = []
    for elem in raw_ocr_elements:
        cx, cy = (elem["x0"] + elem["x1"]) / 2, (elem["top"] + elem["bottom"]) / 2
        in_table = any(b[0] <= cx <= b[2] and b[1] <= cy <= b[3] for b in table_bboxes)
        
        in_word = False
        for w_obj in extracted_words:
             if w_obj["x0"] - 2 <= cx <= w_obj["x1"] + 2 and w_obj["top"] - 2 <= cy <= w_obj["bottom"] + 2:
                 in_word = True
                 break

        if not in_table and not in_word:
            ocr_blocks.append(elem)

    if not ocr_blocks:
        return []

    df = pd.DataFrame(ocr_blocks)
    df["cy"] = (df["top"] + df["bottom"]) / 2
    
    # -----------------------------------------------------------------
    # 【核心優化一】動態均值錨點縱向軌道分行演算法
    # 徹底防止不同行的上下方文字被錯誤歸類在同一行
    # -----------------------------------------------------------------
    sorted_elems = df.sort_values("cy").to_dict(orient="records")
    distinct_rows = []
    
    for elem in sorted_elems:
        cy = elem["cy"]
        placed = False
        for row in distinct_rows:
            # 以該行當前所有元素的平均中心 Y 軸作為基準，避免階梯式擴展誤判
            row_cy_avg = sum(e["cy"] for e in row) / len(row)
            if abs(cy - row_cy_avg) <= row_threshold:
                row.append(elem)
                placed = True
                break
        if not placed:
            distinct_rows.append([elem])
            
    # 將分好行的資料重新按真實 Y 軸高度由上至下排序
    distinct_rows = sorted(distinct_rows, key=lambda r: sum(e["cy"] for e in r) / len(r))
    # -----------------------------------------------------------------

    # =========================================================================
    # 【智慧修正軌道】建立全頁 X 軸欄位起點基準線
    # 僅採用元件數量 >= 4 的密集資料行，精準剔除置中大標題（如 宜蘭縣）產生的幽靈錨點
    # =========================================================================
    dense_x0_list = []
    for row_elems in distinct_rows:
        if len(row_elems) >= 4:  # 標準多欄表格行的元件數通常較多（>=4）
            for e in row_elems:
                dense_x0_list.append(e["x0"])
                
    # 防呆機制：若全頁都沒有大於等於 4 個元件的行，則退回使用全頁所有的 x0
    if not dense_x0_list:
        dense_x0_list = df["x0"].tolist()

    x0_list = sorted(dense_x0_list)
    columns = []
    for x in x0_list:
        if not columns:
            columns.append(x)
        else:
            if min(abs(x - c) for c in columns) > col_threshold:
                columns.append(x)
    columns = sorted(columns)
    # =========================================================================
    
    new_blocks = []

    # -----------------------------------------------------------------
    # 【核心優化二】純全域網格對齊與防碰撞（同一行內嚴格由左至右填入對應的 Excel 欄位）
    # -----------------------------------------------------------------
    for row_elems in distinct_rows:
        row_data = []
        last_col_idx = -1
        
        for item in sorted(row_elems, key=lambda e: e["x0"]):
            col_idx = np.argmin([abs(item["x0"] - c) for c in columns])
            
            # 欄位防碰撞：如果此格已被佔用或發生水平重疊，強制往右推一格
            if col_idx <= last_col_idx:
                col_idx = last_col_idx + 1
                
            while len(row_data) <= col_idx:
                row_data.append("")
                
            row_data[col_idx] = item["text"]
            last_col_idx = col_idx
        
        if any(row_data):
            min_top = min(e["top"] for e in row_elems)
            new_blocks.append({
                "top": float(min_top),
                "kind": "ocr_table_row",
                "rows": [row_data]
            })
            
    return new_blocks


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
            
            # ---------------------------------------------------------
            # 特殊混合模式：精準處理第 14 頁隱藏表格
            # ---------------------------------------------------------
            if page_num == 14:
                print(f"  Processing page {page_num} (Special Mode: Split & Anti-Collision Merge)...")
                found_tables = page.find_tables()
                table_bboxes = [table.bbox for table in found_tables]
                words = page.extract_words()

                # 1. 抓原生文字
                outside_words = [
                    w for w in words
                    if not any(is_word_inside_bbox(w, bbox) for bbox in table_bboxes)
                ]
                blocks = words_to_line_blocks(outside_words)

                # 2. 抓原生表格
                for table in found_tables:
                    table_rows = normalize_table_rows(table.extract())
                    if table_rows:
                        blocks.append({
                            "top": float(table.bbox[1]),
                            "kind": "table",
                            "rows": table_rows,
                        })

                # 3. 執行防碰撞與智慧切分 OCR 補丁（參數與自訂 signatures 同步優化為 10 與 6）
                ocr_blocks = get_ocr_blocks_for_page(page, table_bboxes, words, col_threshold=10, row_threshold=6)
                blocks.extend(ocr_blocks)

                # 4. 依 Y 軸由上至下寫入，維持原始排版
                for block in sorted(blocks, key=lambda item: item["top"]):
                    for row in block["rows"]:
                        ws.append(row)
                    if block["kind"] == "table":
                        ws.append([])
                        
                continue 
            # ---------------------------------------------------------

            print(f"  Processing page {page_num}...")
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
                    row_threshold = 14 * 3  
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
    input_pdf = r"D:\work\0601Yilan\1150213.pdf"
    output_excel = r"D:\work\0601Yilan\output_result0101%.xlsx"

    if os.path.exists(input_pdf):
        process_pdf_to_excel(input_pdf, output_excel)
    else:
        print(f"PDF not found: {input_pdf}")
