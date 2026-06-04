"""
修正 Page_14 的 OCR 行政區錯字。
用字形像素比對取代人工維護的混淆表——把每個字渲染成點陣圖，
計算餘弦相似度，讓視覺上像的字自然得高分。
"""
import sys
import numpy as np
import openpyxl
from difflib import SequenceMatcher
from functools import lru_cache
from PIL import Image, ImageDraw, ImageFont

sys.stdout.reconfigure(encoding="utf-8")

FONT_PATH = "C:/Windows/Fonts/kaiu.ttf"   # 標楷體，Traditional Chinese
CHAR_SIZE = 48

_font = ImageFont.truetype(FONT_PATH, CHAR_SIZE - 6)


@lru_cache(maxsize=4096)
def char_vec(ch: str) -> np.ndarray:
    """把單一漢字渲染成灰階向量（已正規化）。"""
    img = Image.new("L", (CHAR_SIZE, CHAR_SIZE), 0)
    draw = ImageDraw.Draw(img)
    draw.text((4, 2), ch, fill=255, font=_font)
    arr = np.array(img, dtype=np.float32).ravel()
    norm = np.linalg.norm(arr)
    return arr / norm if norm > 0 else arr


def char_sim(a: str, b: str) -> float:
    """單字視覺相似度（餘弦，0~1）。"""
    if a == b:
        return 1.0
    return float(np.dot(char_vec(a), char_vec(b)))


def string_visual_sim(s: str, t: str) -> float:
    """
    字串視覺相似度：長度相同時逐字位置比對取均值；
    長度不同時退回 SequenceMatcher。
    """
    if len(s) != len(t):
        return SequenceMatcher(None, s, t).ratio()
    return sum(char_sim(s[i], t[i]) for i in range(len(s))) / len(s)


# ── 台灣行政區資料庫 ────────────────────────────────────────────────

TAIWAN_CITIES = [
    "基隆市", "台北市", "新北市", "桃園市", "新竹市", "新竹縣", "苗栗縣",
    "台中市", "彰化縣", "南投縣", "雲林縣", "嘉義市", "嘉義縣", "台南市",
    "高雄市", "屏東縣", "宜蘭縣", "花蓮縣", "台東縣", "澎湖縣", "金門縣", "連江縣",
]

TAIWAN_DISTRICTS = [
    # 基隆市
    "仁愛區", "信義區", "中正區", "中山區", "安樂區", "暖暖區", "七堵區",
    # 台北市
    "大同區", "松山區", "大安區", "萬華區", "士林區", "北投區", "內湖區", "南港區", "文山區",
    # 新北市
    "板橋區", "三重區", "中和區", "永和區", "新莊區", "新店區", "樹林區", "鶯歌區", "三峽區",
    "淡水區", "汐止區", "瑞芳區", "土城區", "蘆洲區", "五股區", "泰山區", "林口區", "深坑區",
    "石碇區", "坪林區", "三芝區", "石門區", "八里區", "平溪區", "雙溪區", "貢寮區", "金山區",
    "萬里區", "烏來區",
    # 桃園市
    "桃園區", "中壢區", "大溪區", "楊梅區", "蘆竹區", "大園區", "龜山區", "八德區", "龍潭區",
    "平鎮區", "新屋區", "觀音區", "復興區",
    # 新竹市
    "香山區",
    # 新竹縣
    "竹北市", "竹東鎮", "新埔鎮", "關西鎮", "湖口鄉", "新豐鄉", "芎林鄉", "橫山鄉", "北埔鄉",
    "寶山鄉", "峨眉鄉", "尖石鄉", "五峰鄉",
    # 苗栗縣
    "苗栗市", "頭份市", "竹南鎮", "後龍鎮", "通霄鎮", "苑裡鎮", "造橋鄉", "西湖鄉", "頭屋鄉",
    "公館鄉", "銅鑼鄉", "三義鄉", "大湖鄉", "獅潭鄉", "三灣鄉", "南庄鄉", "卓蘭鎮", "泰安鄉",
    # 台中市
    "北屯區", "西屯區", "南屯區", "太平區", "大里區", "霧峰區", "烏日區", "豐原區", "後里區",
    "石岡區", "東勢區", "和平區", "新社區", "潭子區", "大雅區", "神岡區", "大肚區", "沙鹿區",
    "龍井區", "梧棲區", "清水區", "大甲區", "外埔區",
    # 彰化縣
    "彰化市", "鹿港鎮", "和美鎮", "線西鄉", "伸港鄉", "福興鄉", "秀水鄉", "花壇鄉", "芬園鄉",
    "員林市", "溪湖鎮", "田中鎮", "大村鄉", "埔鹽鄉", "埔心鄉", "永靖鄉", "社頭鄉", "二水鄉",
    "北斗鎮", "二林鎮", "田尾鄉", "埤頭鄉", "芳苑鄉", "大城鄉", "竹塘鄉", "溪州鄉",
    # 南投縣
    "南投市", "埔里鎮", "草屯鎮", "竹山鎮", "集集鎮", "名間鄉", "鹿谷鄉", "中寮鄉", "魚池鄉",
    "國姓鄉", "水里鄉", "信義鄉",
    # 雲林縣
    "斗六市", "斗南鎮", "虎尾鎮", "西螺鎮", "土庫鎮", "北港鎮", "古坑鄉", "大埤鄉", "莿桐鄉",
    "林內鄉", "二崙鄉", "崙背鄉", "麥寮鄉", "褒忠鄉", "台西鄉", "元長鄉", "四湖鄉", "口湖鄉",
    "水林鄉",
    # 嘉義縣
    "太保市", "朴子市", "布袋鎮", "大林鎮", "民雄鄉", "溪口鄉", "新港鄉", "六腳鄉", "東石鄉",
    "義竹鄉", "鹿草鄉", "水上鄉", "中埔鄉", "竹崎鄉", "梅山鄉", "番路鄉", "大埔鄉", "阿里山鄉",
    # 台南市
    "中西區", "安平區", "安南區", "永康區", "歸仁區", "新化區", "左鎮區", "玉井區", "楠西區",
    "南化區", "仁德區", "關廟區", "龍崎區", "官田區", "麻豆區", "佳里區", "西港區", "七股區",
    "將軍區", "學甲區", "北門區", "新營區", "後壁區", "白河區", "東山區", "六甲區", "下營區",
    "柳營區", "鹽水區", "善化區", "大內區", "山上區", "新市區", "安定區",
    # 高雄市
    "新興區", "前金區", "苓雅區", "鹽埕區", "鼓山區", "旗津區", "前鎮區", "三民區", "楠梓區",
    "小港區", "左營區", "仁武區", "大社區", "岡山區", "路竹區", "阿蓮區", "田寮區", "燕巢區",
    "橋頭區", "梓官區", "彌陀區", "永安區", "湖內區", "鳳山區", "大寮區", "林園區", "鳥松區",
    "大樹區", "旗山區", "美濃區", "六龜區", "內門區", "杉林區", "甲仙區", "桃源區", "那瑪夏區",
    "茂林區", "茄萣區",
    # 屏東縣
    "屏東市", "三地門鄉", "霧臺鄉", "瑪家鄉", "九如鄉", "里港鄉", "高樹鄉", "鹽埔鄉", "長治鄉",
    "麟洛鄉", "竹田鄉", "內埔鄉", "萬丹鄉", "潮州鎮", "泰武鄉", "來義鄉", "萬巒鄉", "新埤鄉",
    "南州鄉", "林邊鄉", "東港鎮", "琉球鄉", "佳冬鄉", "新園鄉", "枋寮鄉", "枋山鄉", "春日鄉",
    "獅子鄉", "車城鄉", "牡丹鄉", "恆春鎮", "滿州鄉",
    # 宜蘭縣
    "宜蘭市", "羅東鎮", "蘇澳鎮", "頭城鎮", "礁溪鄉", "壯圍鄉", "員山鄉", "冬山鄉", "五結鄉",
    "三星鄉", "大同鄉", "南澳鄉",
    # 花蓮縣
    "花蓮市", "新城鄉", "秀林鄉", "吉安鄉", "壽豐鄉", "鳳林鎮", "光復鄉", "豐濱鄉", "瑞穗鄉",
    "萬榮鄉", "玉里鎮", "卓溪鄉", "富里鄉",
    # 台東縣
    "台東市", "綠島鄉", "蘭嶼鄉", "延平鄉", "卑南鄉", "鹿野鄉", "關山鎮", "海端鄉", "池上鄉",
    "東河鄉", "成功鎮", "長濱鄉", "太麻里鄉", "金峰鄉", "大武鄉", "達仁鄉",
    # 澎湖縣
    "馬公市", "湖西鄉", "白沙鄉", "西嶼鄉", "望安鄉", "七美鄉",
    # 金門縣
    "金門鎮", "金沙鎮", "金湖鎮", "金寧鄉", "烈嶼鄉", "烏坵鄉",
    # 連江縣
    "南竿鄉", "北竿鄉", "莒光鄉", "東引鄉",
]

ALL_ADMIN = TAIWAN_CITIES + TAIWAN_DISTRICTS

# ── 比對與修正 ───────────────────────────────────────────────────────

THRESHOLD = 0.72        # 一般門檻
ANCHOR_THRESHOLD = 0.65 # 有錨點字（完全相同字）時放寬的門檻

# 常用統計詞彙庫（非行政區，但同樣需要 OCR 修正）
STATS_TERMS = [
    "總指數", "基期", "定基指數", "月份", "年度", "季度",
    "漲跌率", "價格指數", "物價指數", "消費者物價",
]

ALL_CANDIDATES = ALL_ADMIN + STATS_TERMS


def _has_anchor(s: str, t: str) -> bool:
    """是否有至少一個字位置完全相同（視覺相似度=1.0）。"""
    return len(s) == len(t) and any(s[i] == t[i] for i in range(len(s)))


def best_match(text: str):
    best_score = 0.0
    best_candidate = None
    for candidate in ALL_CANDIDATES:
        if abs(len(text) - len(candidate)) > 1:
            continue
        score = string_visual_sim(text, candidate)
        if score > best_score:
            best_score = score
            best_candidate = candidate
    # 決定門檻：有錨點字可以放寬
    threshold = ANCHOR_THRESHOLD if _has_anchor(text, best_candidate or "") else THRESHOLD
    if best_score >= threshold:
        return best_candidate, best_score
    return None, best_score


def try_correct(value: str) -> str:
    if not isinstance(value, str) or len(value) < 2:
        return value
    match, score = best_match(value)
    if match and match != value:
        print(f"  [{score:.3f}] '{value}' → '{match}'")
        return match
    return value


def main():
    src = r"D:\work2\Pdftoexcel\output_result0102%.xlsx"
    dst = r"D:\work2\Pdftoexcel\output_result0102%_fixed.xlsx"

    wb = openpyxl.load_workbook(src)
    ws = wb["Page_14"]

    print("=== 掃描 Page_14（字形像素比對）===\n")
    changed = 0
    for row in ws.iter_rows():
        for cell in row:
            if not isinstance(cell.value, str):
                continue
            original = cell.value
            corrected = try_correct(original)
            if corrected != original:
                cell.value = corrected
                changed += 1

    print(f"\n共修正 {changed} 個儲存格")
    wb.save(dst)
    print(f"已儲存至: {dst}")


if __name__ == "__main__":
    main()
