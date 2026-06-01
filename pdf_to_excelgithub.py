import os
import pdfplumber
import pandas as pd
from openpyxl import Workbook
from PIL import Image
import numpy as np
from rapidocr_onnxruntime import RapidOCR

# 初始化 OCR 引擎
engine = RapidOCR()

# OCR 信心分數門檻，低於此值的辨識結果會被捨棄
OCR_CONFIDENCE_THRESHOLD = 0.6

def ocr_cropped_image(pixmap_img, scale=2):
    """
    將傳入的圖片放大並進行 OCR 辨識，過濾低信心結果
    """
    if pixmap_img is None:
        return ""

    w, h = pixmap_img.size
    resized_img = pixmap_img.resize((w * scale, h * scale), Image.Resampling.LANCZOS)

    img_ndarray = np.array(resized_img)

    result, _ = engine(img_ndarray)

    if result:
        # 過濾低信心分數的辨識結果
        texts = [line[1] for line in result if len(line) < 3 or line[2] >= OCR_CONFIDENCE_THRESHOLD]
        return " ".join(texts)
    return ""

def crop_with_padding(page, x0, top, x1, bottom, padding=4):
    """裁切頁面區塊時加上 padding，避免字元被切邊"""
    pw = float(page.width)
    ph = float(page.height)
    bbox = (
        max(0, x0 - padding),
        max(0, top - padding),
        min(pw, x1 + padding),
        min(ph, bottom + padding),
    )
    return page.within_bbox(bbox).to_image(resolution=400).original

def process_pdf_to_excel(pdf_path, excel_path):
    wb = Workbook()
    default_sheet = wb.active
    wb.remove(default_sheet)

    print(f"📄 開始處理 PDF: {os.path.basename(pdf_path)}")

    with pdfplumber.open(pdf_path) as pdf:
        for i, page in enumerate(pdf.pages):
            page_num = i + 1
            sheet_name = f"Page_{page_num}"
            ws = wb.create_sheet(title=sheet_name)
            print(f"  ➜ 正在處理第 {page_num} 頁...")

            # 【特判通道】針對第 14 頁無法提取的表格，強制執行高精度網格 OCR 搶救
            if page_num == 14:
                print(f"    [重要提示] 偵測到第 14 頁，啟動強效網格座標對齊 OCR 搶救表格...")

                pil_img = page.to_image(resolution=450).original
                w, h = pil_img.size
                resized_img = pil_img.resize((w * 3, h * 3), Image.Resampling.LANCZOS)

                img_ndarray = np.array(resized_img)
                result, _ = engine(img_ndarray)

                if not result:
                    print("    ❌ 第 14 頁搶救失敗：未偵測到任何文字。")
                    continue

                data = []
                for line in result:
                    # 過濾低信心結果
                    if len(line) >= 3 and line[2] < OCR_CONFIDENCE_THRESHOLD:
                        continue
                    box = line[0]
                    text = line[1]
                    cx = (box[0][0] + box[1][0]) / 2
                    cy = (box[0][1] + box[2][1]) / 2
                    data.append({'cx': cx, 'cy': cy, 'text': text})

                df_page14 = pd.DataFrame(data)
                df_page14 = df_page14.sort_values(by='cy').reset_index(drop=True)
                df_page14['row_id'] = 0

                current_row = 0
                current_y = df_page14.loc[0, 'cy']
                # 動態計算行高閾值：取所有 cy 差值中位數的 0.6 倍，更能適應不同字型大小
                cy_diffs = df_page14['cy'].diff().dropna()
                positive_diffs = cy_diffs[cy_diffs > 2]
                row_threshold = float(positive_diffs.median() * 0.6) if not positive_diffs.empty else 14 * 3

                for idx, row in df_page14.iterrows():
                    if row['cy'] - current_y > row_threshold:
                        current_row += 1
                        current_y = row['cy']
                    df_page14.at[idx, 'row_id'] = current_row

                for r_id, group in df_page14.groupby('row_id'):
                    group = group.sort_values(by='cx')
                    row_cells = [item['text'] for _, item in group.iterrows()]
                    ws.append(row_cells)

                continue

            # --- 其餘頁數的常規處理流程 ---

            tables = page.extract_tables()

            if tables:
                for table in tables:
                    for row in table:
                        for col_idx, cell in enumerate(row):
                            if cell is None or str(cell).strip() == "":
                                row[col_idx] = ""
                        ws.append(row)

            else:
                text_objects = page.extract_words()

                if not text_objects:
                    # 純圖片頁面，用 300 DPI + scale=2 提升解析度
                    print(f"    [提示] 第 {page_num} 頁未偵測到文字，啟動全頁 OCR...")
                    pil_img = page.to_image(resolution=300).original
                    page_text = ocr_cropped_image(pil_img, scale=2)

                    for row_text in page_text.split('\n'):
                        ws.append([row_text])
                else:
                    df_words = pd.DataFrame(text_objects)
                    # 動態計算行高閾值，比固定 3px 更能適應不同字型
                    top_diffs = df_words['top'].diff().dropna()
                    positive_top_diffs = top_diffs[top_diffs > 0]
                    line_threshold = float(positive_top_diffs.median() * 0.5) if not positive_top_diffs.empty else 5
                    df_words['row_group'] = (df_words['top'].diff() > line_threshold).cumsum()

                    for _, group in df_words.groupby('row_group'):
                        group = group.sort_values('x0')
                        row_content = []

                        for _, word in group.iterrows():
                            text = word['text'].strip()

                            if not text:
                                # 裁切時加 padding，提高 resolution 至 400
                                cropped = crop_with_padding(
                                    page,
                                    word['x0'], word['top'],
                                    word['x1'], word['bottom'],
                                )
                                text = ocr_cropped_image(cropped, scale=3)

                            row_content.append(text)

                        ws.append([" ".join(row_content)])

    wb.save(excel_path)
    print(f"🎉 轉換完成！Excel 檔案已儲存至: {excel_path}")

# 🚀 執行轉換
if __name__ == "__main__":
    input_pdf = r"D:\work\0601Yilan\1150213.pdf"
    output_excel = r"D:\work\0601Yilan\output_result1.xlsx"

    if os.path.exists(input_pdf):
        process_pdf_to_excel(input_pdf, output_excel)
    else:
        print(f"❌ 找不到檔案 {input_pdf}，請確認路徑是否正確。")
