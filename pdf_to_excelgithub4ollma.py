import os
import pdfplumber
import pandas as pd
from openpyxl import Workbook
from PIL import Image
import numpy as np
from rapidocr_onnxruntime import RapidOCR
import io
import base64
import requests
import json

# 初始化 OCR 引擎
engine = RapidOCR()

# OCR 信心分數門檻
OCR_CONFIDENCE_THRESHOLD = 0.6

# Llama 3.2-Vision API 設定 (以 Ollama 本地端為例，若用雲端 API 請替換 URL 與 Headers)
LLAMA_API_URL = "http://localhost:11434/api/chat" 
LLAMA_MODEL_NAME = "llama3.2-vision" # 或者使用 "llama3.2-vision:11b"

def ocr_cropped_image(pixmap_img, scale=2):
    if pixmap_img is None:
        return ""
    w, h = pixmap_img.size
    resized_img = pixmap_img.resize((w * scale, h * scale), Image.Resampling.LANCZOS)
    img_ndarray = np.array(resized_img)
    result, _ = engine(img_ndarray)
    if result:
        texts = [line[1] for line in result if len(line) < 3 or line[2] >= OCR_CONFIDENCE_THRESHOLD]
        return " ".join(texts)
    return ""

def crop_with_padding(page, x0, top, x1, bottom, padding=4):
    pw = float(page.width)
    ph = float(page.height)
    bbox = (
        max(0, x0 - padding),
        max(0, top - padding),
        min(pw, x1 + padding),
        min(ph, bottom + padding),
    )
    return page.within_bbox(bbox).to_image(resolution=400).original

def call_llama_vision_table(pil_image):
    """
    將 PIL 圖片轉成 Base64 並調用 Llama 3.2-Vision 提取表格
    回傳標準的 JSON 格式二維陣列 (表格)
    """
    # 1. 將 PIL Image 轉換為 Base64 字串
    buffered = io.BytesIO()
    # 為了節省 Token 與傳輸時間，轉成 JPEG
    pil_image.convert("RGB").save(buffered, format="JPEG", quality=90)
    img_str = base64.b64encode(buffered.getvalue()).decode('utf-8')
    
    # 2. 設定 Prompt，嚴格要求模型輸出純 JSON 的二維陣列
    prompt = (
        "你是一個高精度的表格辨識專家。請仔細觀看這張圖片中的表格，"
        "並將其完整的內容轉換為一個 JSON 的二維陣列 (List of Lists)。\n"
        "規定：\n"
        "1. 必須完整保留欄位與對齊，空值請填空字串 \"\"。\n"
        "2. 請只輸出 JSON 陣列本身，絕對不要包含任何解釋、Markdown 標記 (如 ```json) 或前後言。\n"
        "範例輸出：[[\"欄位1\", \"欄位2\"], [\"資料1\", \"資料2\"]]"
    )
    
    payload = {
        "model": LLAMA_MODEL_NAME,
        "messages": [
            {
                "role": "user",
                "content": prompt,
                "images": [img_str]
            }
        ],
        "stream": False,
        "options": {
            "temperature": 0.1 # 低隨機性，確保結構穩定
        }
    }
    
    try:
        response = requests.post(LLAMA_API_URL, json=payload, timeout=60)
        response.raise_for_status()
        res_json = response.json()
        content = res_json['message']['content'].strip()
        
        # 清理可能不小心夾帶的 markdown 語法
        if content.startswith("```"):
            content = content.split("\n", 1)[1]
        if content.endswith("```"):
            content = content.rsplit("\n", 1)[0]
        content = content.strip()
        
        # 解析成 Python List
        table_data = json.loads(content)
        return table_data
    except Exception as e:
        print(f"    ❌ Llama 3.2-Vision 調用或解析失敗: {e}")
        # 如果模型回傳了非 JSON 的純文字，印出來 debug
        if 'content' in locals():
            print(f"    [模型原始回傳]: {content}")
        return None

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

            # === 【特判通道】第 14 頁：調用 Llama 3.2-Vision 搶救表格 ===
            if page_num == 14:
                print(f"    [重要提示] 偵測到第 14 頁，啟動 Llama 3.2-Vision AI 視覺表格辨識...")
                
                # 轉成圖片 (大模型通常不需要過高的解析度，DPI 150~200 即可，避免 Token 爆炸)
                pil_img = page.to_image(resolution=200).original
                
                # 呼叫 Llama Vision
                table_data = call_llama_vision_table(pil_img)
                
                if table_data and isinstance(table_data, list):
                    print(f"    解析成功，成功讀取到 {len(table_data)} 行表格資料。")
                    for row_cells in table_data:
                        if isinstance(row_cells, list):
                            ws.append(row_cells)
                        else:
                            # 預防萬一模型返回一維陣列
                            ws.append([str(row_cells)])
                else:
                    print("    ❌ 第 14 頁 AI 搶救失敗，退回原有的 OCR 網格對齊機制...")
                    # --- 這裡保留你原本的 RapidOCR 備用機制，確保 AI 壞掉時程式不會斷掉 ---
                    pil_img_backup = page.to_image(resolution=450).original
                    w, h = pil_img_backup.size
                    resized_img = pil_img_backup.resize((w * 3, h * 3), Image.Resampling.LANCZOS)
                    img_ndarray = np.array(resized_img)
                    result, _ = engine(img_ndarray)
                    
                    if not result:
                        print("    ❌ 備用 OCR 也未偵測到任何文字。")
                        continue
                        
                    data = []
                    for line in result:
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
                    current_y = df_page14.loc[0, 'cy'] if not df_page14.empty else 0
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

            # --- 其餘頁數的常規處理流程 (保持不變) ---
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
                    print(f"    [提示] 第 {page_num} 頁未偵測到文字，啟動全頁 OCR...")
                    pil_img = page.to_image(resolution=300).original
                    page_text = ocr_cropped_image(pil_img, scale=2)
                    for row_text in page_text.split('\n'):
                        ws.append([row_text])
                else:
                    df_words = pd.DataFrame(text_objects)
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

if __name__ == "__main__":
    input_pdf = r"D:\work\0601Yilan\1150213.pdf"
    output_excel = r"D:\work\0601Yilan\output_result12.xlsx"

    if os.path.exists(input_pdf):
        process_pdf_to_excel(input_pdf, output_excel)
    else:
        print(f"❌ 找不到檔案 {input_pdf}，請確認路徑是否正確。")
