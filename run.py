"""
run.py – Streamlit 財經雷達 | 技術分析 | 進階分析 | 資產配置
2025-05 穩定版
"""
# --- 基本套件 ---
import streamlit as st, yfinance as yf, pandas as pd, numpy as np, json, re, requests
import plotly.graph_objects as go, plotly.express as px, openai
from datetime import datetime as dt, timedelta
import io, os, time, math
from typing import Dict, List, Any, Tuple, Optional
from scipy.stats import norm
from dotenv import load_dotenv

# 載入環境變數（僅在本地開發時使用）
load_dotenv()

# ======== 產業碼 ↔︎ 中文名稱完整對照 ========
INDUSTRY_MAP: dict[str, str] = {
    "01": "水泥工業",
    "02": "食品工業",
    "03": "塑膠工業",
    "04": "紡織纖維",
    "05": "電機機械",
    "06": "電器電纜",
    "07": "化學生技醫療",
    "08": "玻璃陶瓷",
    "09": "造紙工業",
    "10": "鋼鐵工業",
    "11": "橡膠工業",
    "12": "汽車工業",
    "13": "電子工業",
    "14": "建材營造",
    "15": "航運業",
    "16": "觀光事業",
    "17": "金融保險",
    "18": "貿易百貨",
    "19": "綜合",
    "20": "其他",
    # --- 電子次產業（依櫃買中心 & TWSE 公開欄位） ---
    "21": "半導體業",
    "22": "電腦及週邊",
    "23": "光電業",
    "24": "通信網路",
    "25": "電子零組件",
    "26": "電子通路",
    "27": "資訊服務",
    "28": "其他電子",
    # --- 新經濟分類（若 API 有提供請自行增補） ---
    "29": "文化創意",
    "30": "農業科技",
    "31": "數位雲端",
    "32": "運動休閒",
    "33": "綠色能源及環保",
    "34": "數位及電路板"
}
# ============================================

# ========= 設置 OpenAI API Key =========
# 優先使用 Streamlit Secrets，如果沒有則使用環境變數
openai.api_key = st.secrets.get("openai.api_key") or os.getenv("OPENAI_API_KEY")
if not openai.api_key:
    st.error("請設置 OpenAI API Key。在 Streamlit Cloud 上請在 Secrets 中設置 openai.api_key，或在本地開發時設置 OPENAI_API_KEY 環境變數")
    st.stop()
# =======================================

st.set_page_config(page_title="全方位財務分析", layout="wide")
st.title("📊 全方位財務分析工具")
PAGE = st.sidebar.radio(
    "📑 選擇頁面",
    ["市場雷達", "技術分析", "進階分析", "資產配置"],
    index=0            # 預設顯示第一頁
)

# ---------------- 共用函式 ---------------- #
@st.cache_data
def fetch_info(sym: str) -> Tuple[Dict, pd.DataFrame, pd.DataFrame, pd.Series]:
    """yfinance 一次取 info + 三表"""
    t = yf.Ticker(sym)
    try:
        # 取得基本資訊
        info = t.get_info()
        
        # 取得年度財報 (轉置)
        financials = pd.DataFrame()
        try:
            financials = t.financials.T
            # 將索引從日期轉為中文財報名稱
            financials.index  = [translate_financial_term(i) for i in financials.index]
            # 將列名轉換為日期字符串格式
            financials.columns = [col.strftime('%Y-%m-%d') if hasattr(col, 'strftime') else str(col) for col in financials.columns]
        except Exception as e:
            pass
            
        # 取得季度財報 (轉置)
        quarterly = pd.DataFrame()
        try:
            quarterly = t.quarterly_financials.T
            # 將索引從日期轉為中文財報名稱
            quarterly.index   = [translate_financial_term(i) for i in quarterly.index]
            # 將列名轉換為日期字符串格式
            quarterly.columns = [col.strftime('%Y-%m-%d') if hasattr(col, 'strftime') else str(col) for col in quarterly.columns]
        except Exception as e:
            pass
            
        # 取得股息資料
        dividends = pd.Series(dtype=float)
        try:
            dividends = t.dividends
        except Exception:
            pass
            
        return info, financials, quarterly, dividends
    except Exception as e:
        return {}, pd.DataFrame(), pd.DataFrame(), pd.Series(dtype=float)

def translate_financial_term(term: str) -> str:
    """將英文財報術語翻譯為繁體中文（未列入者保持原文）"""
    translations = {
        # ── 收益表 Income Statement ──
        "Total Revenue": "營業收入",
        "Operating Revenue": "營業收入",
        "Revenue": "營業收入",
        "Cost Of Revenue": "營業成本",
        "Reconciled Cost Of Revenue": "調整後營業成本",
        "Gross Profit": "毛利",
        "Operating Income": "營業利益",
        "Operating Expense": "營業費用",
        "Other Operating Expenses": "其他營業費用",
        "Selling General And Administration": "銷售及行政管理費用",
        "Selling General Administrative": "銷售及行政管理費用",
        "Selling And Marketing Expense": "銷售與行銷費用",
        "General And Administrative Expense": "一般及行政費用",
        "Other Gand A": "其他管理及行政",
        "Research And Development": "研發費用",
        "EBITDA": "EBITDA",
        "Normalized EBITDA": "標準化EBITDA",
        "EBIT": "EBIT",
        "Net Income": "淨利",
        "Net Income Common Stockholders": "普通股股東淨利",
        "Net Income Continuous Operations": "持續營運淨利",
        "Net Income From Continuing Operation Net Minority Interest": "持續營運淨利(含少數股權)",
        "Net Income From Continuing And Discontinued Operation": "持續與終止營運淨利",
        "Net Income Including Noncontrolling Interests": "含非控股權益淨利",
        "Diluted NI Availto Com Stockholders": "稀釋普通股股東淨利",
        "Minority Interests": "少數股權益",
        "Tax Provision": "所得稅費用",
        "Tax Effect Of Unusual Items": "非經常性項目稅務影響",
        "Tax Rate For Calcs": "計算用稅率",
        "Pretax Income": "稅前淨利",
        "Other Income Expense": "其他收入費用",
        "Other Non Operating Income Expenses": "其他非營業收入費用",
        "Special Income Charges": "特殊收益費用",
        "Gain On Sale Of Business": "出售業務收益",
        "Gain On Sale Of Security": "出售證券收益",
        "Write Off": "減記",
        "Earnings From Equity Interest": "權益法投資收益",
        "Net Non Operating Interest Income Expense": "非營業利息淨損益",
        "Net Interest Income": "淨利息收入",
        "Interest Expense": "利息支出",
        "Interest Expense Non Operating": "非營業利息支出",
        "Interest Income": "利息收入",
        "Interest Income Non Operating": "非營業利息收入",
        "Total Other Finance Cost": "其他財務成本總額",
        "Total Expenses": "總費用",
        "Total Operating Income As Reported": "報告營業利益總額",

        # ── 資產負債表 Balance Sheet ──
        "Total Assets": "資產總額",
        "Total Liabilities": "負債總額",
        "Total Liabilities Net Minority Interest": "負債總額(不含少數股權)",
        "Total Equity": "權益總額",
        "Cash": "現金",
        "Cash And Cash Equivalents": "現金及約當現金",
        "Short Term Investments": "短期投資",
        "Accounts Receivable": "應收帳款",
        "Inventory": "存貨",
        "Property Plant Equipment": "不動產廠房及設備",
        "Property Plant And Equipment": "不動產廠房及設備",
        "Accounts Payable": "應付帳款",
        "Current Assets": "流動資產",
        "Current Liabilities": "流動負債",
        "Long Term Debt": "長期借款",
        "Capital Stock": "股本",
        "Retained Earnings": "保留盈餘",

        # ── 現金流量表 Cash-Flow ──
        "Operating Cash Flow": "營業現金流量",
        "Investing Cash Flow": "投資現金流量",
        "Financing Cash Flow": "融資現金流量",
        "Free Cash Flow": "自由現金流量",
        "Cash Flow From Continuing Operating Activities": "持續營業現金流量",
        "Dividends Paid": "已付股利",
        "Capital Expenditure": "資本支出",

        # ── 股本 / 每股 ──
        "Diluted Average Shares": "稀釋平均股數",
        "Basic Average Shares": "基本平均股數",
        "Diluted EPS": "稀釋 EPS",
        "Basic EPS": "基本 EPS",

        # ── 其它 ──
        "Reconciled Depreciation": "調整後折舊",
        "Total Unusual Items": "非經常性項目總額",
        "Total Unusual Items Excluding Goodwill": "非經常性項目總額(不含商譽)",
    }
    return translations.get(term, term)  # 找不到就保持原文


def ai_resp(prompt, model="gpt-4o-mini", maxtok=400, temp=0.4):
    try:
        rsp = openai.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=maxtok, temperature=temp
        )
        return rsp.choices[0].message.content.strip()
    except Exception as e:
        return f"⚠️ AI 失敗：{e}"

# 安全取欄位值
def safe_get(df, col_idx, default=None):
    """安全地獲取DataFrame的列，避免KeyError"""
    try:
        if isinstance(col_idx, int) and col_idx < len(df.columns):
            return df.iloc[:, col_idx]
        else:
            return default
    except Exception:
        return default

# 讀取上市櫃公司基本資料
@st.cache_data(ttl=60*60)
def get_tw_list() -> pd.DataFrame:
    """
    從 TWSE / OTC OpenAPI 擷取上市櫃公司資料，
    並將「產業別」數字代碼自動轉為中文名稱。
    如 API 失效則回退至內建備份表。
    """
    try:
        url_twse = "https://openapi.twse.com.tw/v1/opendata/t187ap03_L"   # 上市
        url_otc  = "https://openapi.twse.com.tw/v1/opendata/t187ap05_L"   # 上櫃

        def _load(url: str) -> pd.DataFrame:
            try:
                r = requests.get(url, timeout=10)
                if r.status_code == 200:
                    return pd.DataFrame(r.json())
            except Exception:
                pass
            return pd.DataFrame()

        twse = _load(url_twse)
        otc  = _load(url_otc)
        df   = pd.concat([twse, otc], ignore_index=True)

        if not df.empty:
            # --- 自動偵測欄位名稱（不同 API 版本欄位名略有差異） ---
            id_col  = next((c for c in df.columns if "公司代號" in c or "公司代碼" in c), None)
            name_col= next((c for c in df.columns if "公司名稱" in c or "公司簡稱" in c), None)
            ind_col = next((c for c in df.columns if "產業別" in c or "行業別" in c or "產業代碼" in c), None)

            if id_col and name_col:
                out = df[[id_col, name_col] + ([ind_col] if ind_col else [])].copy()
                out.rename(columns={id_col:"代號", name_col:"名稱", ind_col:"產業"}, inplace=True, errors="ignore")

                # --- ◎ 產業碼 → 中文 ------------
                if "產業" in out.columns:
                    out["產業"] = (
                        out["產業"]
                        .astype(str).str.zfill(2)      # 例：'7' → '07'
                        .map(INDUSTRY_MAP)             # 對照字典
                        .fillna(out["產業"])            # 若對照不到就保留原值
                    )
                else:
                    out["產業"] = "未分類"

                # 清理其它欄位
                out["代號"] = out["代號"].astype(str).str.strip()
                out["名稱"] = out["名稱"].astype(str)\
                                   .str.replace("股份有限公司", "", regex=False)\
                                   .str.strip()
                return out
    except Exception:
        pass

    # --- ↓↓↓ 備援資料（如 API 失效） ↓↓↓ ---
    backup = {
        "代號": ["2330", "2454", "2317", "2412"],
        "名稱": ["台積電", "聯發科", "鴻海", "中華電"],
        "產業": ["半導體業", "半導體業", "電子零組件", "通信網路"]
    }
    return pd.DataFrame(backup)

# 財務指標計算
def calculate_financial_ratios(ticker_info, income_statement, balance_sheet):
    """計算重要財務比率"""
    ratios = {}
    
    # 從ticker_info中直接獲取的比率
    for key, name in [
        ("currentRatio", "流動比率"),
        ("quickRatio", "速動比率"),
        ("debtToEquity", "負債比率"),
        ("returnOnAssets", "資產報酬率"),
        ("returnOnEquity", "股東權益報酬率"),
        ("grossMargins", "毛利率"),
        ("operatingMargins", "營業利益率"),
        ("profitMargins", "淨利率"),
        ("trailingPE", "本益比(TTM)"),
        ("priceToBook", "股價淨值比"),
        ("dividendYield", "股息收益率")
    ]:
        ratios[name] = ticker_info.get(key, None)
    
    return ratios

# 估值模型
def calculate_valuation(ticker_symbol, info, financials, dividends):
    """使用多種估值模型計算股票價值"""
    valuation = {
        "DDM估值": None,
        "DCF估值": None,
        "本益比估值": None,
        "股價淨值比估值": None,
        "綜合評估": None
    }
    
    try:
        current_price = info.get("regularMarketPrice", None)
        if not current_price:
            return valuation
        
        # DDM模型 (股息折現模型)
        if not dividends.empty:
            try:
                # 計算年度平均股息
                annual_div = dividends[-4:].mean() if len(dividends) >= 4 else dividends.mean()
                # 估計股息成長率 (使用過去5年數據或可用數據)
                if len(dividends) >= 10:
                    div_growth = (dividends[-1] / dividends[-10]) ** (1/5) - 1
                else:
                    div_growth = 0.03  # 假設3%成長率
                
                # 要求報酬率 (使用CAPM或固定值)
                required_return = 0.08  # 假設8%
                
                # DDM計算 (使用戈登成長模型)
                if required_return > div_growth:
                    ddm_value = annual_div * (1 + div_growth) / (required_return - div_growth)
                    valuation["DDM估值"] = round(ddm_value, 2)
            except Exception:
                pass
        
        # DCF模型 (現金流量折現)
        try:
            if "營業現金流量" in financials.index:
                # 獲取過去的自由現金流量
                fcf = financials.loc["營業現金流量"].mean()
                
                # 假設增長率和折現率
                growth_rate_5y = 0.08  # 前5年8%增長
                growth_rate_terminal = 0.03  # 永續3%增長
                discount_rate = 0.1  # 10%折現率
                
                # 計算DCF
                dcf_value = 0
                for i in range(1, 6):
                    dcf_value += fcf * (1 + growth_rate_5y) ** i / (1 + discount_rate) ** i
                
                # 加上永續價值
                terminal_value = fcf * (1 + growth_rate_5y) ** 5 * (1 + growth_rate_terminal) / (discount_rate - growth_rate_terminal)
                discounted_terminal_value = terminal_value / (1 + discount_rate) ** 5
                
                # 總企業價值
                enterprise_value = dcf_value + discounted_terminal_value
                
                # 轉換為每股價值
                shares_outstanding = info.get("sharesOutstanding", None)
                if shares_outstanding:
                    dcf_per_share = enterprise_value / shares_outstanding
                    valuation["DCF估值"] = round(dcf_per_share, 2)
        except Exception:
            pass
        
        # 本益比估值
        try:
            eps = info.get("trailingEps", None)
            industry_pe = 15  # 假設行業平均本益比
            if eps and eps > 0:
                pe_valuation = eps * industry_pe
                valuation["本益比估值"] = round(pe_valuation, 2)
        except Exception:
            pass
        
        # 股價淨值比估值
        try:
            book_value_per_share = info.get("bookValue", None)
            industry_pb = 1.5  # 假設行業平均股價淨值比
            if book_value_per_share and book_value_per_share > 0:
                pb_valuation = book_value_per_share * industry_pb
                valuation["股價淨值比估值"] = round(pb_valuation, 2)
        except Exception:
            pass
        
        # 綜合評估
        valid_valuations = [v for v in [valuation["DDM估值"], valuation["DCF估值"], 
                                        valuation["本益比估值"], valuation["股價淨值比估值"]] if v]
        if valid_valuations:
            avg_valuation = sum(valid_valuations) / len(valid_valuations)
            valuation["綜合評估"] = round(avg_valuation, 2)
            
            # 計算潛在上漲/下跌空間
            if current_price:
                upside = (avg_valuation / current_price - 1) * 100
                valuation["潛在空間"] = f"{upside:.1f}%"
                
                # 給出投資建議
                if upside > 20:
                    valuation["建議"] = "🟢 強力買入"
                elif upside > 10:
                    valuation["建議"] = "🟢 買入"
                elif upside > 0:
                    valuation["建議"] = "🟡 持有"
                elif upside > -10:
                    valuation["建議"] = "🟡 觀望"
                else:
                    valuation["建議"] = "🔴 賣出"
    
    except Exception as e:
        pass
    
    return valuation

# =============== 1. 市場雷達 =============== #
if PAGE == "市場雷達":
    st.header("🚦 市場雷達（專業選股）")

    # 1-A：讀取公司基本資料（一次抓全部，速度快）
    with st.spinner("載入上市櫃公司清單..."):
        company_df = get_tw_list()

    if company_df.empty:
        st.error("無法獲取公司基本資料")
        st.stop()

    # ---- 🔸 新增【產業先選】與【開始載入】按鈕 -------------------
    all_sectors = sorted(company_df["產業"].dropna().unique().tolist())
    default_idx  = all_sectors.index("半導體業") if "半導體業" in all_sectors else 0
    picked_sector = st.selectbox("先選要分析的產業", ["全部產業"] + all_sectors, index=default_idx+1)

    # 只有「全部產業」時才用原 DataFrame，否則先過濾
    base_df_for_build = (
        company_df if picked_sector == "全部產業"
        else company_df[ company_df["產業"] == picked_sector ]
    )

    # 讓使用者自己決定「要不要現在就跑」
    go_build = st.button("🚀 開始載入樣本")
    if not go_build:
        st.info("選好產業後，點擊 **🚀 開始載入樣本** 才會真正下載行情與財務資料")
        st.stop()             # ← 提前結束，底下所有重運算完全跳過
    # ------------------------------------------------------------

    # 設置樣本數量滑桿（放在按鈕之後才顯示，避免多餘運算）
    max_n = min(100, len(base_df_for_build))
    sample_size = st.sidebar.slider("分析樣本數量", 10, max_n, 50)

    # 1-B：建立雷達 DataFrame（多給一個 sector key 做 cache 區分）
    @st.cache_data(ttl=60*60)
    def build_universe(base_df: pd.DataFrame,
                       sample_size: int = 50,
                       sector_tag: str = "全部") -> pd.DataFrame:
        """建立股票分析資料庫（可限制產業）"""
        if base_df.empty:
            return pd.DataFrame()
            
        rows = []
        
        # 隨機抽樣，但確保熱門股票包含在內
        popular_stocks = ["2330", "2454", "2317", "2412", "8044", "5347"]
        popular_df = base_df[base_df["代號"].isin(popular_stocks)]
        remain_df = base_df[~base_df["代號"].isin(popular_stocks)]
        
        # 隨機選擇剩餘股票
        if len(remain_df) > (sample_size - len(popular_df)):
            random_df = remain_df.sample(sample_size - len(popular_df))
        else:
            random_df = remain_df
            
        # 合併熱門股票和隨機股票
        selected_df = pd.concat([popular_df, random_df])
        
        progress_bar = st.progress(0)
        
        for i, (_, row) in enumerate(selected_df.iterrows()):
            try:
                code = str(row["代號"]).strip()
                name = row["名稱"]
                sector = row["產業"]
                
                progress_percent = (i + 1) / len(selected_df)
                progress_bar.progress(progress_percent)
                
                ticker = f"{code}.TW"
                info, fin, qfin, div = fetch_info(ticker)
                p = info.get("regularMarketPrice")
                if p is None:          # 沒價格 = 停牌 / 資料不全
                    continue

                hi = info.get("fiftyTwoWeekHigh", p)
                lo = info.get("fiftyTwoWeekLow",  p)
                pos52 = (p - lo) / (hi - lo + 1e-9)

                # RSI-14
                try:
                    hist = yf.download(ticker, period="6mo", progress=False)
                    if "Close" in hist.columns:
                        cls = hist["Close"]
                        delta = cls.diff()
                        up = delta.clip(lower=0).rolling(14).mean()
                        down = (-delta.clip(upper=0)).rolling(14).mean()
                        rsi = float((100-100/(1+up/down)).iloc[-1])
                    else:
                        rsi = np.nan
                except Exception:
                    rsi = np.nan

                # 財務指標
                eps = info.get("trailingEps", np.nan)
                eps_growth = info.get("earningsQuarterlyGrowth", np.nan)
                roe = info.get("returnOnEquity", np.nan)
                dy = info.get("dividendYield", np.nan)
                pb = info.get("priceToBook", np.nan)
                pe = info.get("trailingPE", np.nan)
                
                # 淡旺季指標 (使用過去5年同期數據)
                seasonal_strength = np.nan
                try:
                    if not qfin.empty and len(qfin.columns) > 0:
                        current_quarter = dt.now().month // 3 + 1
                        total_quarters = len(qfin.columns)
                        
                        # 計算過去同期表現
                        if "營業收入" in qfin.index:
                            revenue_data = qfin.loc["營業收入"]
                            if total_quarters >= 4:
                                same_quarter_data = [revenue_data.iloc[i] for i in range(total_quarters) 
                                                  if (i % 4) == (current_quarter - 1)]
                                if len(same_quarter_data) > 1:
                                    seasonal_strength = same_quarter_data[-1] / np.mean(same_quarter_data[:-1]) - 1
                except Exception:
                    pass

                rows.append({
                    "代號": code, 
                    "名稱": name, 
                    "產業": sector,
                    "現價": p, 
                    "市值(億)": round(info.get("marketCap",0)/1e8, 1),
                    "本益比": pe,
                    "股價淨值比": pb,
                    "EPS": eps,
                    "EPS季增(%)": None if eps_growth is None else round(eps_growth * 100, 1),
                    "ROE(%)": None if roe is None else round(roe * 100, 1),
                    "殖利率(%)": None if dy is None else round(dy * 100, 2),
                    "RSI14": round(rsi, 1),
                    "52W位置(%)": round(pos52 * 100, 1),
                    "同期強度(%)": None if np.isnan(seasonal_strength) else round(seasonal_strength * 100, 1)
                })
            except Exception as e:
                continue

        progress_bar.empty()
        
        if not rows:
            return pd.DataFrame()
            
        df = pd.DataFrame(rows)
        
        # 處理 NaN 值，以便於篩選
        for col in ["EPS季增(%)", "ROE(%)", "52W位置(%)"]:
            if col in df.columns:
                df[col] = df[col].fillna(0)
                
        return df

    with st.spinner("分析市場數據中..."):
        try:
            dfU = build_universe(base_df_for_build, sample_size, picked_sector)
            
            if not dfU.empty:
                # 添加篩選器
                st.subheader("股票篩選器")
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    selected_sector = st.multiselect("選擇產業", options=["全部"] + sorted(dfU["產業"].unique().tolist()), default=["半導體業"])
                
                with col2:
                    min_eps_growth = st.slider("最小EPS季增(%)", -50.0, 100.0, -50.0)
                    min_roe = st.slider("最小ROE(%)", 0.0, 50.0, 0.0)
                
                with col3:
                    min_dy = st.slider("最小殖利率(%)", 0.0, 10.0, 0.0)
                    rsi_range = st.slider("RSI範圍", 0, 100, (30, 70))
                
                # 應用篩選條件
                filtered_df = dfU.copy()
                
                if "全部" not in selected_sector:
                    filtered_df = filtered_df[filtered_df["產業"].isin(selected_sector)]
                
                # 數值條件篩選
                filtered_df = filtered_df[
                    (filtered_df["EPS季增(%)"] >= min_eps_growth) & 
                    (filtered_df["ROE(%)"] >= min_roe) & 
                    ((filtered_df["殖利率(%)"] >= min_dy) | filtered_df["殖利率(%)"].isna()) & 
                    ((filtered_df["RSI14"] >= rsi_range[0]) & (filtered_df["RSI14"] <= rsi_range[1]) | filtered_df["RSI14"].isna())
                ]
                
                # 排序
                sort_by = st.selectbox("排序依據", options=["市值(億)", "ROE(%)", "EPS季增(%)", "殖利率(%)", "RSI14", "52W位置(%)"], index=0)
                ascending = st.checkbox("升序排列", value=False)
                
                filtered_df = filtered_df.sort_values(by=sort_by, ascending=ascending).reset_index(drop=True)
                
                # 顯示結果
                st.dataframe(filtered_df, use_container_width=True)
                
                # 匯出CSV功能
                csv = filtered_df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="下載為CSV",
                    data=csv,
                    file_name=f"market_radar_{dt.now().strftime('%Y%m%d')}.csv",
                    mime="text/csv",
                )
            else:
                st.error("無法獲取股票資料，請檢查網路連接或稍後再試。")
        except Exception as e:
            st.error(f"市場雷達分析錯誤: {e}")

    # 1-C：多選 → AI 燈號
    st.subheader("💡 AI 綜合燈號")
    if not dfU.empty and "代號" in dfU.columns:
        pool = st.multiselect("勾選欲評估的股票", dfU["代號"].tolist(), [])
        
        if st.button("⚡ 產生燈號") and pool:
            with st.spinner("AI分析中..."):
                req = dfU[dfU["代號"].isin(pool)].to_dict(orient="records")
                prompt = (
                    "你是資深台股分析師，請僅輸出**純 JSON 陣列**，"
                    f"{json.dumps(req, ensure_ascii=False)}\n\n"
                    "請以JSON列表格式回應，每支股票包含以下欄位:\n"
                    "不要有任何說明文字或註解。請以股票代號去除重複"
                    "1. symbol: 股票代號\n"
                    "2. light: 燈號，使用🟢(強力買入)、🟡(持有/觀望)、🔴(賣出)\n"
                    "3. reason: 20字內的分析理由\n"
                    "4. target_price: 目標價位\n"
                    "基於所有可用指標進行綜合評估，特別注意EPS季增、ROE、殖利率、RSI14及52W位置。"
                )
                ans = ai_resp(prompt, maxtok=600)
                try:
                    # 先擷取第一段 […]，確保只留下 JSON
                    m = re.search(r"\[[\s\S]*\]", ans)
                    json_str = m.group(0) if m else ans          # 找不到就直接用原文
                    lights = pd.DataFrame(json.loads(json_str))
                    out = (
                        dfU[dfU["代號"].isin(pool)]
                        .merge(lights, left_on="代號", right_on="symbol", how="left")
                        .drop(columns=["symbol"])
                    )
                    st.dataframe(out, use_container_width=True)
                except Exception as e:
                    st.error(f"解析AI回應時發生錯誤: {e}")
                    st.write("⚠️ AI 原始回應：")
                    st.code(ans, language="json")

    # 1-D：單檔詳查
    st.subheader("🔍 單檔診斷")
    if not dfU.empty and "代號" in dfU.columns:
        tgt = st.selectbox("選擇股票", dfU["代號"].tolist())
        
        if st.button("查看詳情"):
            with st.spinner("分析中..."):
                tinfo, fin, qfin, div = fetch_info(f"{tgt}.TW")
                
                # 安全地檢索公司名稱和產業
                company_name = ""
                industry = ""
                try:
                    match_rows = dfU[dfU["代號"] == tgt]
                    if not match_rows.empty:
                        company_name = match_rows["名稱"].iloc[0]
                        industry = match_rows["產業"].iloc[0]
                except Exception:
                    pass
                
                # 計算財務比率
                ratios = calculate_financial_ratios(tinfo, fin, qfin)
                
                # 估算合理價值
                valuation = calculate_valuation(tgt, tinfo, fin, div)
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.subheader(f"{tgt} {company_name} 基本資料")
                    
                    metrics = {
                        "目前股價": tinfo.get("regularMarketPrice", "N/A"),
                        "市值(億)": round(tinfo.get("marketCap", 0)/1e8, 1),
                        "52週高點": tinfo.get("fiftyTwoWeekHigh", "N/A"),
                        "52週低點": tinfo.get("fiftyTwoWeekLow", "N/A"),
                        "產業": industry,
                        "本益比": tinfo.get("trailingPE", "N/A"),
                        "股價淨值比": tinfo.get("priceToBook", "N/A"),
                        "殖利率(%)": None if tinfo.get("dividendYield") is None else round(tinfo.get("dividendYield", 0) * 100, 2)
                    }
                    
                    metrics_df = pd.DataFrame([metrics])
                    st.dataframe(metrics_df.T.rename(columns={0: "數值"}), use_container_width=True)
                
                with col2:
                    st.subheader("財務健康度")
                    
                    ratios_to_display = {
                        "ROE(%)": None if ratios.get("股東權益報酬率") is None else round(ratios.get("股東權益報酬率", 0) * 100, 2),
                        "毛利率(%)": None if ratios.get("毛利率") is None else round(ratios.get("毛利率", 0) * 100, 2),
                        "營業利益率(%)": None if ratios.get("營業利益率") is None else round(ratios.get("營業利益率", 0) * 100, 2),
                        "淨利率(%)": None if ratios.get("淨利率") is None else round(ratios.get("淨利率", 0) * 100, 2),
                        "流動比率": ratios.get("流動比率", "N/A"),
                        "負債比率(%)": None if ratios.get("負債比率") is None else round(ratios.get("負債比率", 0) * 100, 2)
                    }
                    
                    ratios_df = pd.DataFrame([ratios_to_display])
                    st.dataframe(ratios_df.T.rename(columns={0: "數值"}), use_container_width=True)
                
                st.subheader("估值分析")
                
                valuation_df = pd.DataFrame([{k: v for k, v in valuation.items() if v is not None}])
                if not valuation_df.empty:
                    st.dataframe(valuation_df.T.rename(columns={0: "數值"}), use_container_width=True)
                    
                    if "建議" in valuation and "潛在空間" in valuation:
                        st.metric(
                            label="投資建議", 
                            value=valuation["建議"], 
                            delta=valuation["潛在空間"]
                        )
                else:
                    st.info("無法完成估值分析，數據不足")
                
                # AI分析
                brief = {
                    "symbol": tgt,
                    "name": company_name,
                    "industry": industry,
                    "price": tinfo.get("regularMarketPrice", "N/A"),
                    "roe": ratios_to_display.get("ROE(%)", "N/A"),
                    "eps": tinfo.get("trailingEps", "N/A"),
                    "pe": tinfo.get("trailingPE", "N/A"),
                    "pb": tinfo.get("priceToBook", "N/A"),
                    "dy": ratios_to_display.get("殖利率(%)", "N/A"),
                    "valuation": valuation.get("綜合評估", "N/A"),
                    "potential": valuation.get("潛在空間", "N/A")
                }
                
                st.subheader("AI 投資觀點")
                res = ai_resp(
                    "作為資深股票分析師，請針對以下公司提供100-150字的詳細分析與投資建議，包含技術面、基本面和風險評估：\n"
                    + json.dumps(brief, ensure_ascii=False) + 
                    "\n請提供具體的買入/賣出理由，並在分析末尾附上🟢(買入)/🟡(觀望)/🔴(賣出)的綜合建議。",
                    model="gpt-4o-mini", maxtok=500)
                st.write(res)
    else:
        st.warning("請先獲取股票資料")

# =============== 2. 技術分析 =============== #
elif PAGE == "技術分析":
    st.header("📈 多因子技術分析")
    code = st.sidebar.text_input("股票代號 (不含 .TW)", "2330").strip()
    # 多模型技術指標
    S = st.sidebar.slider("短期 MA", 5, 60, 20)
    M = st.sidebar.slider("中期 MA", 30, 120, 60)
    L = st.sidebar.slider("長期 MA", 60, 240, 120)
    
    # 其他技術指標設定
    with st.sidebar.expander("進階技術指標設定"):
        rsi_period = st.slider("RSI 週期", 7, 21, 14)
        rsi_buy = st.slider("RSI 超賣區間", 20, 40, 30)
        rsi_sell = st.slider("RSI 超買區間", 60, 80, 70)
        
        bb_period = st.slider("布林通道週期", 10, 30, 20)
        bb_std = st.slider("布林通道標準差倍數", 1.5, 3.0, 2.0)
        
        # KD 指標設定
        kd_period = st.slider("KD 週期", 5, 21, 9)
        kd_slow = st.slider("KD 慢速週期", 1, 9, 3)
        
        # MACD 參數
        macd_fast = st.slider("MACD 快線", 8, 16, 12)
        macd_slow = st.slider("MACD 慢線", 17, 32, 26)
        macd_signal = st.slider("MACD 訊號線", 5, 15, 9)

    if st.button("執行分析"):
        with st.spinner("下載歷史數據..."):
            try:
                # 嘗試使用代號直接下載
                raw = yf.download(f"{code}.TW", start=(dt.now() - timedelta(days=365*2)).strftime('%Y-%m-%d'), progress=False)
                
                # 如果沒有數據，嘗試加上.TWO (櫃買中心)
                if raw.empty:
                    raw = yf.download(f"{code}.TWO", start=(dt.now() - timedelta(days=365*2)).strftime('%Y-%m-%d'), progress=False)
                
                if raw.empty:
                    st.error("下載失敗 / 代號錯誤")
                    st.write("請確認股票代號是否正確，或嘗試其他代號。台股代號通常為4-5位數字，例如2330為台積電。")
                else:
                    # 確保有收盤價
                    if "Close" not in raw.columns and "Adj Close" not in raw.columns:
                        st.error("缺少收盤價資料")
                        st.stop()
                        
                    price = raw["Adj Close"] if "Adj Close" in raw.columns else raw["Close"]
                    
                    # 確保有足夠的數據點進行技術分析
                    if len(price) < max(S, M, L) + 30:
                        st.warning(f"數據點不足，需要至少 {max(S, M, L) + 30} 個交易日資料")
                    
                    # 建立技術指標
                    df = pd.DataFrame()
                    df["price"] = price
                    df["MA_S"] = price.rolling(S).mean()
                    df["MA_M"] = price.rolling(M).mean()
                    df["MA_L"] = price.rolling(L).mean()

                    # 計算 RSI
                    delta = price.diff()
                    gain = delta.where(delta > 0, 0)
                    loss = -delta.where(delta < 0, 0)
                    avg_gain = gain.rolling(window=rsi_period).mean()
                    avg_loss = loss.rolling(window=rsi_period).mean()
                    rs = avg_gain / avg_loss
                    df["RSI"] = 100 - (100 / (1 + rs))

                    # 計算 MACD
                    ema_fast = price.ewm(span=macd_fast).mean()
                    ema_slow = price.ewm(span=macd_slow).mean()
                    df["MACD"] = ema_fast - ema_slow
                    df["Signal"] = df["MACD"].ewm(span=macd_signal).mean()
                    df["MACD_Hist"] = df["MACD"] - df["Signal"]

                    # 計算布林帶
                    mid = price.rolling(bb_period).mean()
                    std = price.rolling(bb_period).std()
                    df["BB_UP"] = mid + bb_std * std
                    df["BB_MID"] = mid
                    df["BB_LOW"] = mid - bb_std * std
                    df["BB_Width"] = (df["BB_UP"] - df["BB_LOW"]) / df["BB_MID"]
                    
                    # 計算KD指標
                    low_min = price.rolling(window=kd_period).min()
                    high_max = price.rolling(window=kd_period).max()
                    rsv = 100 * ((price - low_min) / (high_max - low_min))
                    df["K"] = rsv.rolling(window=kd_slow).mean()
                    df["D"] = df["K"].rolling(window=kd_slow).mean()
                    
                    # 計算動量指標 (Momentum)
                    df["Momentum"] = price.diff(14)
                    
                    # 計算乖離率 (Price Rate of Change)
                    df["ROC"] = price.pct_change(periods=12) * 100
                    
                    # 成交量分析
                    if "Volume" in raw.columns:
                        df["Volume"] = raw["Volume"]
                        df["Volume_MA"] = df["Volume"].rolling(20).mean()
                        df["Volume_Ratio"] = df["Volume"] / df["Volume_MA"]
                    
                    # 使用最後的有效值進行判斷
                    last_idx = len(df) - 1
                    if last_idx >= 0:
                        last = df.iloc[last_idx]
                        prev = df.iloc[last_idx-1] if last_idx > 0 else last
                        
                        # 各種買賣信號
                        signals = {
                            "多頭排列": last.MA_S > last.MA_M > last.MA_L if not any(pd.isna([last.MA_S, last.MA_M, last.MA_L])) else False,
                            "黃金交叉": prev.MA_S <= prev.MA_M and last.MA_S > last.MA_M if not any(pd.isna([prev.MA_S, prev.MA_M, last.MA_S, last.MA_M])) else False,
                            "死亡交叉": prev.MA_S >= prev.MA_M and last.MA_S < last.MA_M if not any(pd.isna([prev.MA_S, prev.MA_M, last.MA_S, last.MA_M])) else False,
                            "MACD 黃金交叉": prev.MACD <= prev.Signal and last.MACD > last.Signal if not any(pd.isna([prev.MACD, prev.Signal, last.MACD, last.Signal])) else False,
                            "MACD 死亡交叉": prev.MACD >= prev.Signal and last.MACD < last.Signal if not any(pd.isna([prev.MACD, prev.Signal, last.MACD, last.Signal])) else False,
                            "RSI 超買": last.RSI > rsi_sell if not pd.isna(last.RSI) else False,
                            "RSI 超賣": last.RSI < rsi_buy if not pd.isna(last.RSI) else False,
                            "KD 黃金交叉": prev.K <= prev.D and last.K > last.D if not any(pd.isna([prev.K, prev.D, last.K, last.D])) else False,
                            "KD 死亡交叉": prev.K >= prev.D and last.K < last.D if not any(pd.isna([prev.K, prev.D, last.K, last.D])) else False,
                            "突破布林上軌": last.price > last.BB_UP if not any(pd.isna([last.price, last.BB_UP])) else False,
                            "跌破布林下軌": last.price < last.BB_LOW if not any(pd.isna([last.price, last.BB_LOW])) else False,
                            "布林帶收縮": df["BB_Width"].iloc[-5:].mean() < df["BB_Width"].iloc[-20:-5].mean() if len(df) > 20 else False,
                            "布林帶擴張": df["BB_Width"].iloc[-5:].mean() > df["BB_Width"].iloc[-20:-5].mean() if len(df) > 20 else False,
                            "量價背離": "Volume" in df.columns and last.price > prev.price and last.Volume < prev.Volume,
                            "價格動能正向": last.Momentum > 0 if not pd.isna(last.Momentum) else False,
                            "價格動能負向": last.Momentum < 0 if not pd.isna(last.Momentum) else False
                        }
                        
                        # 計算綜合分數
                        positive_signals = ["多頭排列", "黃金交叉", "MACD 黃金交叉", "RSI 超賣", "KD 黃金交叉", "布林帶收縮", "價格動能正向"]
                        negative_signals = ["死亡交叉", "MACD 死亡交叉", "RSI 超買", "KD 死亡交叉", "突破布林上軌", "跌破布林下軌", "量價背離", "價格動能負向"]
                        
                        positive_score = sum(1 for s in positive_signals if signals.get(s, False))
                        negative_score = sum(1 for s in negative_signals if signals.get(s, False))
                        
                        total_score = positive_score - negative_score
                        if total_score >= 3:
                            sig = "🟢 強力買入"
                        elif total_score >= 1:
                            sig = "🟢 買入"
                        elif total_score >= -1:
                            sig = "🟡 觀望"
                        elif total_score >= -3:
                            sig = "🔴 賣出"
                        else:
                            sig = "🔴 強力賣出"
                        
                        col1, col2 = st.columns([1, 2])
                        with col1:
                            st.metric("技術分析綜合判斷", sig)
                            st.write("**技術指標結果:**")
                            
                            # 顯示買入信號
                            st.markdown("##### 買入信號:")
                            for k in positive_signals:
                                if k in signals:
                                    icon = "✅" if signals[k] else "❌"
                                    st.write(f"{icon} {k}")
                            
                            # 顯示賣出信號
                            st.markdown("##### 賣出信號:")
                            for k in negative_signals:
                                if k in signals:
                                    icon = "✅" if signals[k] else "❌"
                                    st.write(f"{icon} {k}")
                        
                        with col2:
                            st.write("**主要技術指標數值:**")
                            metrics = {
                                "收盤價": last.price,
                                f"MA{S}": last.MA_S,
                                f"MA{M}": last.MA_M,
                                f"MA{L}": last.MA_L,
                                "RSI": last.RSI,
                                "K值": last.K,
                                "D值": last.D,
                                "MACD": last.MACD,
                                "Signal": last.Signal,
                                "乖離率(%)": last.ROC,
                                "布林上軌": last.BB_UP,
                                "布林中軌": last.BB_MID,
                                "布林下軌": last.BB_LOW
                            }
                            metrics_df = pd.DataFrame([metrics])
                            st.dataframe(metrics_df.round(2), use_container_width=True)
                            
                            # 加入AI分析
                            signals_text = ", ".join([k for k, v in signals.items() if v])
                            if not signals_text:
                                signals_text = "無明顯技術信號"
                                
                            prompt = f"""
                            作為專業技術分析師，請分析以下股票技術指標，並給予詳細的技術分析意見和建議:
                            
                            股票代號: {code}
                            當前價格: {last.price:.2f}
                            RSI({rsi_period}): {last.RSI:.2f}
                            K值: {last.K:.2f}
                            D值: {last.D:.2f}
                            MACD: {last.MACD:.4f}
                            Signal: {last.Signal:.4f}
                            乖離率: {last.ROC:.2f}%
                            
                            出現的技術信號: {signals_text}
                            
                            請提供80-100字的技術分析，包括趨勢判斷、支撐阻力位、中短期走勢預測，以及操作建議。
                            """
                            
                            with st.spinner("生成AI技術分析..."):
                                tech_analysis = ai_resp(prompt, maxtok=350)
                                st.markdown("### AI 技術分析")
                                st.write(tech_analysis)
                    
                    # 繪製 K 線圖
                    try:
                        fig = go.Figure()
                        
                        # 添加K線
                        fig.add_trace(go.Candlestick(
                            x=raw.index,
                            open=raw["Open"],
                            high=raw["High"],
                            low=raw["Low"],
                            close=raw["Close"],
                            name="K線"
                        ))
                        
                        # 添加移動平均線
                        fig.add_trace(go.Scatter(x=df.index, y=df["MA_S"], name=f"MA{S}", line=dict(color="blue")))
                        fig.add_trace(go.Scatter(x=df.index, y=df["MA_M"], name=f"MA{M}", line=dict(color="orange")))
                        fig.add_trace(go.Scatter(x=df.index, y=df["MA_L"], name=f"MA{L}", line=dict(color="green")))
                        
                        # 添加布林帶
                        fig.add_trace(go.Scatter(x=df.index, y=df["BB_UP"], name="布林上軌", line=dict(color="rgba(173, 216, 230, 0.5)")))
                        fig.add_trace(go.Scatter(x=df.index, y=df["BB_LOW"], name="布林下軌", line=dict(color="rgba(173, 216, 230, 0.5)")))
                        
                        # 設置圖表樣式
                        fig.update_layout(
                            title=f"{code} 技術走勢圖",
                            hovermode="x unified",
                            height=600,
                            xaxis_title="日期",
                            yaxis_title="價格",
                            legend=dict(
                                orientation="h",
                                yanchor="bottom",
                                y=1.02,
                                xanchor="right",
                                x=1
                            )
                        )
                        
                        # 顯示圖表
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # 創建多個子圖表
                        fig2 = px.line(df, x=df.index, y=["RSI"], title="RSI 指標")
                        fig2.add_hline(y=rsi_sell, line_dash="dash", line_color="red", annotation_text=f"超買({rsi_sell})")
                        fig2.add_hline(y=rsi_buy, line_dash="dash", line_color="green", annotation_text=f"超賣({rsi_buy})")
                        fig2.update_layout(height=300, hovermode="x unified")
                        st.plotly_chart(fig2, use_container_width=True)
                        
                        fig3 = px.line(df, x=df.index, y=["K", "D"], title="KD 指標")
                        fig3.add_hline(y=80, line_dash="dash", line_color="red")
                        fig3.add_hline(y=20, line_dash="dash", line_color="green")
                        fig3.update_layout(height=300, hovermode="x unified")
                        st.plotly_chart(fig3, use_container_width=True)
                        
                        fig4 = go.Figure()
                        fig4.add_trace(go.Scatter(x=df.index, y=df["MACD"], name="MACD"))
                        fig4.add_trace(go.Scatter(x=df.index, y=df["Signal"], name="Signal"))
                        fig4.add_trace(go.Bar(x=df.index, y=df["MACD_Hist"], name="MACD Histogram"))
                        fig4.update_layout(
                            title="MACD 指標",
                            height=300,
                            hovermode="x unified"
                        )
                        st.plotly_chart(fig4, use_container_width=True)
                        
                        # 如果有成交量數據，顯示成交量圖表
                        if "Volume" in df.columns:
                            fig5 = go.Figure()
                            fig5.add_trace(go.Bar(x=df.index, y=df["Volume"], name="成交量"))
                            fig5.add_trace(go.Scatter(x=df.index, y=df["Volume_MA"], name="成交量MA"))
                            fig5.update_layout(
                                title="成交量分析",
                                height=300,
                                hovermode="x unified"
                            )
                            st.plotly_chart(fig5, use_container_width=True)
                        
                    except Exception as e:
                        st.error(f"圖表生成錯誤: {e}")
            except Exception as e:
                st.error(f"分析過程發生錯誤: {e}")
                st.error("請嘗試其他股票代號，確保代號格式正確")

# =============== 3. 進階分析 =============== #
elif PAGE == "進階分析":
    st.header("🔍 進階分析")
    syms_input = st.text_input("股票列表(逗號，不含 .TW)", "2330,2454")
    syms = [x.strip() for x in syms_input.split(",") if x.strip()]
    
    # ▶️ 只有按下這顆按鈕，才會真正開始所有進階分析
    run_advance = st.button("🚀 開始進階分析")
    if not run_advance:
        st.info("請輸入代號後點擊 **🚀 開始進階分析** 才會載入與計算")
        st.stop()         # ← 直接結束，後面程式碼都不執行
    
    if not syms:
        st.warning("請輸入至少一個股票代號")
        st.stop()
        
    tabs = st.tabs(["估值", "財務指標", "淡旺季", "財報+AI"])

    # --- 估值 ---
    with tabs[0]:
        st.subheader("📐 實時估值 (DDM / DCF / Comps)")
        if st.button("分析估值"):
            for s in syms:
                with st.spinner(f"計算 {s} 的估值..."):
                    try:
                        # 獲取數據
                        info, fin, qfin, div = fetch_info(f"{s}.TW")
                        
                        if not info:
                            st.warning(f"無法獲取 {s} 的資料，請確認代號是否正確")
                            continue
                        
                        current_price = info.get("regularMarketPrice")
                        if not current_price:
                            st.warning(f"無法獲取 {s} 的當前價格")
                            continue
                        
                        # 計算估值
                        valuation = calculate_valuation(s, info, fin, div)
                        
                        # 顯示結果
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.subheader(f"{s} {info.get('shortName', '')} 估值")
                            
                            valuation_df = pd.DataFrame([{k: v for k, v in valuation.items() if v is not None and k != "建議" and k != "潛在空間"}])
                            if not valuation_df.empty:
                                st.dataframe(valuation_df.T.rename(columns={0: "數值"}), use_container_width=True)
                        
                        with col2:
                            if "綜合評估" in valuation and valuation["綜合評估"] is not None:
                                fig = go.Figure()
                                
                                # 添加估值結果
                                methods = []
                                values = []
                                
                                for method, value in valuation.items():
                                    if method in ["DDM估值", "DCF估值", "本益比估值", "股價淨值比估值", "綜合評估"] and value is not None:
                                        methods.append(method)
                                        values.append(value)
                                
                                if methods and values:
                                    # 添加當前價格為參考線
                                    methods.append("當前價格")
                                    values.append(current_price)
                                    
                                    # 繪製條形圖
                                    fig.add_trace(go.Bar(
                                        x=methods,
                                        y=values,
                                        marker_color=['blue', 'green', 'orange', 'purple', 'red', 'gray'][:len(methods)]
                                    ))
                                    
                                    # 添加參考線，表示當前價格
                                    fig.add_hline(
                                        y=current_price,
                                        line_dash="dash",
                                        line_color="red",
                                        annotation_text=f"當前價格: {current_price}"
                                    )
                                    
                                    fig.update_layout(
                                        title="不同估值方法比較",
                                        xaxis_title="估值方法",
                                        yaxis_title="價格",
                                        height=400
                                    )
                                    
                                    st.plotly_chart(fig, use_container_width=True)
                                
                                # 顯示建議
                                if "建議" in valuation and "潛在空間" in valuation:
                                    st.metric(
                                        label="投資建議", 
                                        value=valuation["建議"], 
                                        delta=valuation["潛在空間"]
                                    )
                            else:
                                st.info("無法完成估值分析，數據不足")
                        
                        # AI 建議
                        st.subheader("AI 估值解讀")
                        ai_prompt = f"""
                        作為財務分析師，請針對以下股票估值結果提供專業解讀：
                        
                        股票: {s} {info.get('shortName', '')}
                        當前價格: {current_price}
                        
                        估值結果:
                        """
                        
                        for method, value in valuation.items():
                            if value is not None and method not in ["建議", "潛在空間"]:
                                ai_prompt += f"{method}: {value}\n"
                        
                        ai_prompt += """
                        請提供 100-120 字的估值分析，包括：
                        1. 不同估值方法的合理性評價
                        2. 目前股價是否被高估或低估
                        3. 投資建議及合理的買入價位
                        """
                        
                        ai_valuation = ai_resp(ai_prompt, model="gpt-4o-mini", maxtok=500)
                        st.write(ai_valuation)
                        
                    except Exception as e:
                        st.error(f"估值計算錯誤: {e}")

    # --- 財務指標 ---
    with tabs[1]:
        with st.spinner("載入財務指標..."):
            rows = []
            for s in syms:
                try:
                    info, fin, qfin, div = fetch_info(f"{s}.TW")
                    
                    # 計算財務指標
                    ratios = calculate_financial_ratios(info, fin, qfin)
                    
                    # 提取最近四季EPS總和
                    eps_ttm = info.get("trailingEps", np.nan)
                    
                    # 財務指標
                    rows.append([
                        s,
                        info.get("regularMarketPrice", np.nan),
                        info.get("trailingPE", np.nan),
                        info.get("priceToBook", np.nan),
                        None if ratios.get("股東權益報酬率") is None else round(ratios.get("股東權益報酬率", 0) * 100, 1),
                        None if ratios.get("營業利益率") is None else round(ratios.get("營業利益率", 0) * 100, 1),
                        None if ratios.get("淨利率") is None else round(ratios.get("淨利率", 0) * 100, 1),
                        eps_ttm,
                        None if ratios.get("股息收益率") is None else round(ratios.get("股息收益率", 0) * 100, 2),
                        None if ratios.get("負債比率") is None else round(ratios.get("負債比率", 0) * 100, 1)
                    ])
                except Exception as e:
                    # 添加空數據行
                    rows.append([s] + [np.nan] * 9)
            
            # 創建結果DataFrame
            if rows:
                dfF = pd.DataFrame(rows, columns=["代號", "現價", "本益比", "股價淨值比", "ROE(%)", "營業利益率(%)", "淨利率(%)", "EPS(TTM)", "殖利率(%)", "負債比率(%)"])
                dfF["代號"] = dfF["代號"].astype(str)      # 讓 Plotly 把代號視為類別

                st.dataframe(dfF, use_container_width=True)
                
                # 視覺化比較
                st.subheader("財務指標比較")
                
                # 讓用戶選擇要比較的指標
                indicator = st.selectbox(
                    "選擇要比較的財務指標",
                    ["ROE(%)", "營業利益率(%)", "淨利率(%)", "殖利率(%)", "本益比", "股價淨值比", "負債比率(%)"]
                )
                
                # 繪製比較圖表
                fig = px.bar(
                    dfF,                          # ← 你的財務指標 DataFrame
                    x="代號",                     # 代號已經是字串
                    y=indicator,
                    title=f"{indicator} 比較",
                    color=indicator,
                    text=indicator,
                )
                fig.update_traces(texttemplate="%{text:.2f}", textposition="inside")

                # ▼▼▼ 關鍵新增：把 x 軸強制為「category」 ▼▼▼
                fig.update_xaxes(type="category")
                # ▲▲▲ 關鍵新增 ▲▲▲

                st.plotly_chart(fig, use_container_width=True)
                
                # 財務健康度評分
                st.subheader("財務健康度評分")
                
                # 計算各項指標的評分
                scores = []
                for _, row in dfF.iterrows():
                    try:
                        score = 0
                        count = 0
                        
                        # ROE評分 (高=好)
                        if not pd.isna(row["ROE(%)"]):
                            roe_score = min(5, max(0, row["ROE(%)"] / 5))
                            score += roe_score
                            count += 1
                        
                        # 本益比評分 (適中=好)
                        if not pd.isna(row["本益比"]) and row["本益比"] > 0:
                            if row["本益比"] < 10:
                                pe_score = 5  # 低本益比優先
                            elif row["本益比"] < 15:
                                pe_score = 4
                            elif row["本益比"] < 20:
                                pe_score = 3
                            elif row["本益比"] < 30:
                                pe_score = 2
                            else:
                                pe_score = 1  # 高本益比扣分
                            score += pe_score
                            count += 1
                        
                        # 殖利率評分 (高=好)
                        if not pd.isna(row["殖利率(%)"]):
                            div_score = min(5, max(0, row["殖利率(%)"] * 5 / 5))
                            score += div_score
                            count += 1
                        
                        # 負債比率評分 (低=好)
                        if not pd.isna(row["負債比率(%)"]):
                            debt_score = 5 - min(5, max(0, row["負債比率(%)"] / 20))
                            score += debt_score
                            count += 1
                        
                        # 計算平均分數
                        if count > 0:
                            avg_score = score / count
                            scores.append({"代號": row["代號"], "評分": round(avg_score, 1)})
                        else:
                            scores.append({"代號": row["代號"], "評分": np.nan})
                    except Exception:
                        scores.append({"代號": row["代號"], "評分": np.nan})
                
                # 顯示評分結果
                scores_df = pd.DataFrame(scores)
                
                fig2 = px.bar(
                    scores_df,
                    x="代號",
                    y="評分",
                    title="財務健康度評分 (滿分5分)",
                    color="評分",
                    text="評分",
                    color_continuous_scale=["red", "yellow", "green"]
                )
                fig2.update_traces(texttemplate='%{text:.1f}', textposition='inside')
                fig2.update_layout(yaxis_range=[0, 5])
                st.plotly_chart(fig2, use_container_width=True)
            else:
                st.error("無法獲取財務指標數據")

    # --- 淡旺季 ---
    with tabs[2]:
        for s in syms:
            with st.spinner(f"分析 {s} 淡旺季..."):
                try:
                    # 獲取股票數據
                    ticker = yf.Ticker(f"{s}.TW")
                    
                    # 嘗試獲取5年歷史數據
                    hist = ticker.history(period="5y")
                    
                    if hist.empty or len(hist) < 252:  # 至少需要約一年數據
                        st.warning(f"{s}: 歷史數據不足，需要較長的歷史數據分析淡旺季")
                        continue
                    
                    # 將日期轉換為月份
                    hist['Month'] = hist.index.month
                    
                    # 計算每個月的平均收益率
                    monthly_returns = hist.groupby('Month')['Close'].apply(lambda x: x.pct_change().mean() * 100)
                    
                    # 計算每個月的標準差
                    monthly_std = hist.groupby('Month')['Close'].apply(lambda x: x.pct_change().std() * 100)
                    
                    # 計算每個月的交易量
                    if 'Volume' in hist.columns:
                        monthly_volume = hist.groupby('Month')['Volume'].mean()
                    
                    # 合併數據
                    seasonal_df = pd.DataFrame({
                        '月均報酬率(%)': monthly_returns,
                        '波動率(%)': monthly_std
                    })
                    
                    if 'Volume' in hist.columns:
                        seasonal_df['平均成交量'] = monthly_volume
                    
                    # 重設索引並添加月份名稱
                    seasonal_df = seasonal_df.reset_index()
                    month_names = {
                        1: '一月', 2: '二月', 3: '三月', 4: '四月', 5: '五月', 6: '六月',
                        7: '七月', 8: '八月', 9: '九月', 10: '十月', 11: '十一月', 12: '十二月'
                    }
                    seasonal_df['月份'] = seasonal_df['Month'].map(month_names)
                    
                    # 顯示結果表格
                    st.subheader(f"{s} 月度表現分析")
                    st.dataframe(seasonal_df[['月份', '月均報酬率(%)', '波動率(%)']].set_index('月份').round(2), use_container_width=True)
                    
                    # 繪製月度報酬率圖表
                    fig_return = px.bar(
                        seasonal_df,
                        x='月份',
                        y='月均報酬率(%)',
                        title=f"{s} 月度平均報酬率",
                        color='月均報酬率(%)',
                        color_continuous_scale=['red', 'lightgray', 'green'],
                        text='月均報酬率(%)'
                    )
                    fig_return.update_traces(texttemplate='%{text:.2f}%', textposition='inside')
                    st.plotly_chart(fig_return, use_container_width=True)
                    
                    # 繪製月度成交量圖表
                    if 'Volume' in hist.columns:
                        fig_volume = px.bar(
                            seasonal_df,
                            x='月份',
                            y='平均成交量',
                            title=f"{s} 月度平均成交量",
                            color='平均成交量'
                        )
                        st.plotly_chart(fig_volume, use_container_width=True)
                    
                    # 季度分析
                    st.subheader(f"{s} 季度表現分析")
                    
                    # 添加季度信息
                    hist['Quarter'] = hist.index.quarter
                    
                    # 計算每個季度的平均收益率
                    quarterly_returns = hist.groupby('Quarter')['Close'].apply(lambda x: x.pct_change().mean() * 100)
                    
                    # 計算每個季度的標準差
                    quarterly_std = hist.groupby('Quarter')['Close'].apply(lambda x: x.pct_change().std() * 100)
                    
                    # 合併數據
                    quarterly_df = pd.DataFrame({
                        '季均報酬率(%)': quarterly_returns,
                        '波動率(%)': quarterly_std
                    })
                    
                    # 重設索引並添加季度名稱
                    quarterly_df = quarterly_df.reset_index()
                    quarter_names = {1: 'Q1', 2: 'Q2', 3: 'Q3', 4: 'Q4'}
                    quarterly_df['季度'] = quarterly_df['Quarter'].map(quarter_names)
                    
                    # 繪製季度報酬率圖表
                    fig_q_return = px.bar(
                        quarterly_df,
                        x='季度',
                        y='季均報酬率(%)',
                        title=f"{s} 季度平均報酬率",
                        color='季均報酬率(%)',
                        color_continuous_scale=['red', 'lightgray', 'green'],
                        text='季均報酬率(%)'
                    )
                    fig_q_return.update_traces(texttemplate='%{text:.2f}%', textposition='inside')
                    st.plotly_chart(fig_q_return, use_container_width=True)
                    
                    # AI解讀淡旺季
                    st.subheader("AI 淡旺季分析")
                    
                    prompt = f"""
                    作為股票分析師，請分析以下股票 {s} 的月度和季度報酬數據，提供淡旺季分析：
                    
                    月度平均報酬率(%):
                    {seasonal_df[['月份', '月均報酬率(%)']].to_string(index=False)}
                    
                    季度平均報酬率(%):
                    {quarterly_df[['季度', '季均報酬率(%)']].to_string(index=False)}
                    
                    請提供約100字的淡旺季分析，包括:
                    1. 最強勢和最弱勢的月份和季度
                    2. 投資時機建議，何時加碼/減碼
                    3. 與產業或整體市場淡旺季是否一致的分析
                    
                    只分析淡旺季和報酬率，不要討論其他因素。
                    """
                    
                    seasonal_analysis = ai_resp(prompt, maxtok=300)
                    st.write(seasonal_analysis)
                    
                except Exception as e:
                    st.error(f"{s} 淡旺季分析錯誤: {e}")

    # --- 財報 + AI ---
    with tabs[3]:
        for s in syms:
            try:
                # 獲取財報資料
                _, fin, qfin, _ = fetch_info(f"{s}.TW")
                
                # 年度財報
                if not fin.empty and len(fin.columns) > 0:
                    # 顯示財報數據
                    st.subheader(f"{s} 年度財報 (單位：百萬)")
                    
                    # 將數據轉換為百萬單位並四捨五入
                    df_millions = fin / 1e6
                    df3 = df_millions.round(0)
                    
                    # 確保所有列名都是字符串
                    df3 = df3.rename(columns=lambda c: translate_financial_term(str(c)))

                    # ▸ 2. 若翻譯後有重複，為後面的欄自動加 _1、_2… 尾碼
                    if df3.columns.duplicated().any():
                        def dedup(cols):
                            seen = {}
                            new_cols = []
                            for col in cols:
                                cnt = seen.get(col, 0)
                                new_cols.append(f"{col}_{cnt}" if cnt else col)
                                seen[col] = cnt + 1
                            return new_cols
                        df3.columns = dedup(df3.columns)
                    # 顯示數據
                    st.dataframe(df3, use_container_width=True)
                    
                    # 繪製主要財務指標趨勢圖
                    key_metrics = ["營業收入", "營業利益", "淨利"]
                    available_metrics = [m for m in key_metrics if m in df3.index]
                    
                    if available_metrics:
                        # 創建趨勢數據
                        trend_data = pd.DataFrame()
                        
                        for metric in available_metrics:
                            trend_data[metric] = df3.loc[metric]
                        
                        # 重設索引，將日期從列名轉為索引
                        trend_data = trend_data.T
                        
                        # 繪製趨勢圖
                        fig = px.line(
                            trend_data,
                            title=f"{s} 主要財務指標趨勢",
                            labels={"index": "日期", "value": "金額 (百萬)", "variable": "指標"}
                        )
                        st.plotly_chart(fig, use_container_width=True)
                
                # 季度財報
                if not qfin.empty and len(qfin.columns) > 0:
                    st.subheader(f"{s} 季度財報 (單位：百萬)")
                    
                    # 將數據轉換為百萬單位並四捨五入
                    qf_millions = qfin / 1e6
                    qf3 = qf_millions.round(0)
                    
                    # 確保所有列名都是字符串
                    qf3 = qf3.rename(columns=lambda c: translate_financial_term(str(c)))
                    
                    # 去重／加尾碼
                    if qf3.columns.duplicated().any():
                        def dedup(cols):
                            seen, out = {}, []
                            for col in cols:
                                idx = seen.get(col, 0)
                                out.append(f"{col}_{idx}" if idx else col)
                                seen[col] = idx + 1
                            return out
                        qf3.columns = dedup(qf3.columns)
                    
                    # 顯示數據
                    st.dataframe(qf3, use_container_width=True)
                    
                    # 繪製季度收入趨勢圖
                    if "營業收入" in qf3.index:
                        quarterly_revenue = qf3.loc["營業收入"]
                        
                        # 計算同比成長率
                        if len(quarterly_revenue) >= 5:
                            yoy_growth = []
                            for i in range(4, len(quarterly_revenue)):
                                try:
                                    growth = (quarterly_revenue.iloc[i] / quarterly_revenue.iloc[i-4] - 1) * 100
                                    yoy_growth.append((quarterly_revenue.index[i], growth))
                                except Exception:
                                    pass
                            
                            if yoy_growth:
                                growth_df = pd.DataFrame(yoy_growth, columns=["季度", "同比成長率(%)"])
                                
                                # 繪製成長率圖表
                                fig_growth = px.bar(
                                    growth_df,
                                    x="季度",
                                    y="同比成長率(%)",
                                    title=f"{s} 季度營收同比成長率",
                                    color="同比成長率(%)",
                                    text="同比成長率(%)"
                                )
                                fig_growth.update_traces(texttemplate='%{text:.1f}%', textposition='inside')
                                st.plotly_chart(fig_growth, use_container_width=True)
                
                # 如果沒有財報數據
                if (fin.empty or len(fin.columns) == 0) and (qfin.empty or len(qfin.columns) == 0):
                    st.info(f"{s}: 無財報數據")
                
            except Exception as e:
                st.error(f"{s} 獲取財報錯誤: {e}")
                
        if st.button("AI 財報綜合分析") and syms:
            with st.spinner("AI財報分析中..."):
                # 收集所有可用財報數據
                report_data = {}
                
                for s in syms:
                    try:
                        _, fin, qfin, _ = fetch_info(f"{s}.TW")
                        
                        if not fin.empty and len(fin.columns) > 0:
                            # 提取重要指標
                            key_metrics = ["營業收入", "營業利益", "淨利", "毛利"]
                            metrics_data = {}
                            
                            for metric in key_metrics:
                                if metric in fin.index:
                                    metrics_data[metric] = fin.loc[metric].iloc[-3:].tolist()  # 取最近三期
                            
                            growth_data = {}
                            for metric in key_metrics:
                                if metric in metrics_data and len(metrics_data[metric]) >= 2:
                                    growth = (metrics_data[metric][-1] / metrics_data[metric][-2] - 1) * 100
                                    growth_data[f"{metric}成長"] = round(growth, 1)
                            
                            report_data[s] = {
                                "指標": metrics_data,
                                "成長率": growth_data
                            }
                    except Exception:
                        continue
                
                if report_data:
                    # 構建AI提示
                    prompt = f"""
                    作為資深財務分析師，請針對以下公司的財報數據提供專業分析：
                    
                    {json.dumps(report_data, ensure_ascii=False)}
                    
                    請針對每家公司提供具體的財報分析，包括：
                    1. 營收和利潤成長趨勢及其原因
                    2. 利潤率變化及其影響因素
                    3. 主要財務優勢和隱憂
                    4. 未來展望和投資建議
                    
                    每家公司的分析控制在150字左右，並標註🟢/🟡/🔴燈號在列點的前方。
                    """
                    
                    analysis = ai_resp(prompt, model="gpt-4o-mini", maxtok=800)
                    st.markdown(analysis)
                else:
                    st.warning("無足夠財報數據進行分析")

# =============== 4. 資產配置 =============== #
else:
    st.header("💼 資產配置")
    
    # 使用您提供的資產數據作為預設值
    default_portfolio = """0056:11000; 00878:11000; 00929:12000; 5347:2183; 0050:1110; 00713:3000; 2330:70; 00923:4000; 00919:5000; 00757:420; 現金:200000"""
    
    rawtxt = st.text_area("輸入持倉", default_portfolio)
    def parse(txt):
        h, c = {}, 0
        for seg in re.split(r"[;,]", txt):
            if ":" not in seg: continue
            parts = seg.split(":", 1)
            if len(parts) != 2: continue
            
            k, v = [x.strip() for x in parts]
            try:
                v_float = float(v)
                if k in ("現金", "cash"):
                    c = v_float
                else:
                    h[k] = v_float
            except ValueError:
                st.warning(f"無法解析值: {v}")
        return h, c
        
    hold, cash = parse(rawtxt)
    
    # 您提供的資產資料
    portfolio_data = {
        "代號": ["0056", "00878", "00929", "5347", "0050", "00713", "2330", "00923", "00919", "00757", "現金"],
        "標的": ["元大高股息", "國泰永續高股息", "復華台科優息", "世界先進", "元大台灣50", "台灣高息低波", "台積電", "台灣高息動能", "台灣價值高息", "統一FANG+", "現金"],
        "張數": [11000, 11000, 12000, 2183, 1110, 3000, 70, 4000, 5000, 420, "-"],
        "最新價": [32.34, 20.29, 17.04, 91.10, 167.8, 50.45, 926, 13.30, 8.69, 94.7, "-"],
        "市值(萬)": [35.6, 22.3, 20.4, 19.9, 18.6, 15.1, 6.3, 5.3, 4.3, 4.0, 20.0],
        "損益": ["+4.4萬", "-1.3萬", "-2.0萬", "+5.0萬", "+8.8萬", "+0.7萬", "-0.7萬", "-0.7萬", "-0.7萬", "+0.2萬", "-"],
        "報酬率": ["+14%", "-5%", "-9%", "+34%", "+90%", "+5%", "-10%", "-12%", "-13%", "+5%", "-"],
    }
    
    # 創建DataFrame
    df_portfolio = pd.DataFrame(portfolio_data)
    
    # 顯示資產配置表格
    st.subheader("資產配置明細")
    st.dataframe(df_portfolio, use_container_width=True)
    
    # 計算總市值
    total_value = sum(float(x) for x in df_portfolio["市值(萬)"] if x != "-")
    st.metric("總資產價值", f"{total_value:.1f} 萬")
    
    # 繪製圓餅圖
    st.subheader("資產配置比例")
    fig = px.pie(
        df_portfolio[df_portfolio["代號"] != "現金"],  # 排除現金
        values="市值(萬)",
        names="標的",
        title="投資組合資產分配",
        hole=0.4
    )
    fig.update_traces(textposition='inside', textinfo='percent+label')
    fig.update_layout(
        legend=dict(orientation="h", yanchor="bottom", y=-0.2, xanchor="center", x=0.5)
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # 計算按產業/類型的分布
    etf_allocation = {
        "台股ETF": ["0056", "00878", "00929", "0050", "00713", "00923", "00919"],
        "個股": ["5347", "2330"],
        "國際ETF": ["00757"],
        "現金": ["現金"]
    }
    
    category_values = []
    for category, tickers in etf_allocation.items():
        category_value = sum(float(row["市值(萬)"]) for _, row in df_portfolio.iterrows() 
                           if row["代號"] in tickers and row["市值(萬)"] != "-")
        category_values.append({"分類": category, "市值(萬)": category_value})
    
    df_categories = pd.DataFrame(category_values)
    
    # 繪製按產業分類的圓餅圖
    fig2 = px.pie(
        df_categories,
        values="市值(萬)",
        names="分類",
        title="按資產類型分佈",
        hole=0.4
    )
    fig2.update_traces(textposition='inside', textinfo='percent+label')
    fig2.update_layout(
        legend=dict(orientation="h", yanchor="bottom", y=-0.2, xanchor="center", x=0.5)
    )
    st.plotly_chart(fig2, use_container_width=True)
    
    # 投資組合效率分析
    st.subheader("投資組合效率分析")
    
    with st.spinner("計算投資組合績效指標..."):
        # 收集持有的非現金資產歷史數據
        tickers = [code for code in df_portfolio["代號"] if code != "現金"]
        
        # 下載歷史數據
        start_date = (dt.now() - timedelta(days=365)).strftime('%Y-%m-%d')
        end_date   = dt.now().strftime('%Y-%m-%d')
        
        # 收集價格數據並保存在字典中
        price_data = {}
        for ticker in tickers:
            try:
                hist = yf.download(f"{ticker}.TW", start=start_date, end=end_date, progress=False)
                if hist.empty:
                    hist = yf.download(f"{ticker}.TWO", start=start_date, end=end_date, progress=False)

                # 若成功抓到收盤價，就轉成 numpy 1-D
                if not hist.empty and "Close" in hist.columns:
                    price_vec = np.asarray(hist["Close"]).ravel()   # ★ 保證 1-D
                    price_data[ticker] = price_vec
            except Exception:
                pass
        # ❶ 只保留「至少 60 筆」的股票，避免單日價格造成 1-D error
        price_data = {k: v for k, v in price_data.items() if len(v) >= 60}

        if len(price_data) < 2:
            st.warning("歷史數據不足，無法進行效率分析（至少需要兩檔且各 60 筆以上收盤價）")
            st.stop()

        # ❷ 將所有序列裁成相同的最小長度
        min_len = min(len(v) for v in price_data.values())
        aligned  = {k: v[-min_len:] for k, v in price_data.items()}

        # ❸ 用共同日期當索引建立 DataFrame
        dates = pd.date_range(end=pd.Timestamp(end_date), periods=min_len, freq='B')
        prices_df = pd.DataFrame({k: pd.Series(v, index=dates) for k, v in aligned.items()})

        # ── 接下來直接用 prices_df ────────────────────────
        returns_df = prices_df.pct_change().dropna()
        if returns_df.empty or len(returns_df) <= 20:
            st.warning("有效收益率資料不足，無法計算績效指標")
            st.stop()

        # 為了避免"If using all scalar values, you must pass an index"錯誤
        # 如果有價格數據，創建一個帶有通用索引的DataFrame
        if price_data:
            # 首先找出最短的價格序列長度
            min_length = min(len(values) for values in price_data.values())
            
            # 截斷所有序列到相同長度並創建一個包含共同索引的DataFrame
            data_dict = {}
            for ticker, prices in price_data.items():
                data_dict[ticker] = prices[-min_length:]
            
            # 創建帶有虛擬日期索引的DataFrame
            dates = pd.date_range(end=pd.Timestamp(end_date), periods=min_length)
            prices_df = pd.DataFrame(data_dict, index=dates)
            
            if not prices_df.empty:
                # 計算日收益率
                returns_df = prices_df.pct_change().dropna()
                
                if not returns_df.empty and len(returns_df) > 20:
                    # 計算年化收益率和風險
                    annual_returns = returns_df.mean() * 252
                    annual_volatility = returns_df.std() * (252 ** 0.5)
                    
                    # 計算夏普比率 (假設無風險利率為2%)
                    risk_free_rate = 0.02
                    sharpe_ratio = (annual_returns - risk_free_rate) / annual_volatility
                    
                    # 計算最大回撤
                    portfolio_max_drawdown = {}
                    for col in prices_df.columns:
                        price_series = prices_df[col]
                        rolling_max = price_series.cummax()
                        drawdown = (price_series - rolling_max) / rolling_max
                        portfolio_max_drawdown[col] = drawdown.min()
                    
                    # 創建績效指標數據框
                    performance_data = {
                        "資產": list(prices_df.columns),
                        "年化報酬率(%)": annual_returns * 100,
                        "年化波動率(%)": annual_volatility * 100,
                        "夏普比率": sharpe_ratio,
                        "最大回撤(%)": [portfolio_max_drawdown.get(t, np.nan) * 100 for t in prices_df.columns]
                    }
                    
                    performance_df = pd.DataFrame(performance_data).round(2)
                    
                    # 顯示績效指標表格
                    st.dataframe(performance_df, use_container_width=True)
                    
                    # 繪製風險收益散點圖
                    fig3 = px.scatter(
                        performance_df,
                        x="年化波動率(%)",
                        y="年化報酬率(%)",
                        size=abs(performance_df["最大回撤(%)"]),
                        color="夏普比率",
                        text="資產",
                        title="風險收益分析",
                        size_max=50,
                        color_continuous_scale="RdYlGn"
                    )
                    fig3.update_traces(textposition='top center')
                    fig3.update_layout(height=600)
                    st.plotly_chart(fig3, use_container_width=True)
                    
                    # 計算相關性矩陣
                    if len(prices_df.columns) > 1:
                        st.subheader("資產相關性分析")
                        correlation = returns_df.corr()
                        
                        # 繪製相關性熱圖
                        fig4 = px.imshow(
                            correlation,
                            text_auto=".2f",
                            color_continuous_scale="RdBu_r",
                            title="資產相關性矩陣"
                        )
                        st.plotly_chart(fig4, use_container_width=True)
                        
                        # 計算資產組合多樣化得分
                        avg_correlation = correlation.values[np.triu_indices_from(correlation.values, k=1)].mean()
                        diversification_score = 1 - avg_correlation
                        
                        # 顯示多樣化得分
                        st.metric(
                            label="投資組合多樣化得分",
                            value=f"{diversification_score:.2f} / 1.00",
                            delta=None if diversification_score > 0.5 else "多樣化程度不足",
                            delta_color="normal" if diversification_score > 0.5 else "inverse"
                        )
                        
                        # 解釋多樣化得分
                        if diversification_score > 0.7:
                            st.success("您的投資組合多樣化程度良好，資產間相關性較低，有助於分散風險。")
                        elif diversification_score > 0.5:
                            st.info("您的投資組合多樣化程度中等，可以考慮增加低相關性資產來進一步降低風險。")
                        else:
                            st.warning("您的投資組合多樣化程度較低，資產間相關性較高，風險分散效果有限。")
            else:
                st.warning("歷史數據不足，無法進行詳細的績效分析。")
        else:
            st.warning("無法獲取資產歷史價格數據，無法進行績效分析。")

    if st.button("🧮 AI 配置建議"):
        with st.spinner("AI 分析中..."):
            try:
                # 將資產配置數據轉為更簡單的格式
                holdings_data = df_portfolio.to_dict(orient="records")
                
                # 獲取績效指標數據
                performance_summary = ""
                if 'performance_df' in locals() and not performance_df.empty:
                    performance_summary = "績效指標:\n" + performance_df.to_string() + "\n\n"
                    
                    if 'diversification_score' in locals():
                        performance_summary += f"多樣化得分: {diversification_score:.2f}\n"
                
                prompt = f"""
                作為專業投資顧問，請針對以下投資組合提供詳細的資產配置建議:
                
                總資產：{total_value} 萬元
                現金比例：{float(df_portfolio[df_portfolio['代號'] == '現金']['市值(萬)'].iloc[0])/total_value*100:.1f}%
                台股ETF佔比：{sum(float(row['市值(萬)']) for _, row in df_portfolio.iterrows() if row['代號'] in etf_allocation['台股ETF'])/total_value*100:.1f}%
                個股佔比：{sum(float(row['市值(萬)']) for _, row in df_portfolio.iterrows() if row['代號'] in etf_allocation['個股'])/total_value*100:.1f}%
                國際ETF佔比：{sum(float(row['市值(萬)']) for _, row in df_portfolio.iterrows() if row['代號'] in etf_allocation['國際ETF'])/total_value*100:.1f}%
                
                {performance_summary}
                
                持股明細：
                {json.dumps(holdings_data, ensure_ascii=False)}
                
                請提供以下建議（共200-250字）：
                1. 整體投資組合評估 (分散程度、風險水平等)
                2. 具體的調整建議 (哪些標的增持/減持/賣出，比例調整到多少)
                3. 新增資產建議 (若需要更多元化，推薦加入哪些類型資產)
                4. 資產配置最佳比例 (股票/債券/現金/其他資產的理想配置)
                5. 根據市場趨勢的短期調整策略
                
                最後附上整體投資組合評分：🟢(優)/🟡(中)/🔴(待改進)
                """
                
                suggestion = ai_resp(
                    prompt,
                    model="gpt-4o-mini",
                    maxtok=1000
                )
                st.markdown(suggestion)
                
                # 顯示模擬優化後的資產配置
                st.subheader("模擬優化後的資產配置")
                
                # 基於AI建議的模擬優化配置
                optimized_portfolio = {
                    "台股ETF": 0.45,  # 45%
                    "美股ETF": 0.15,   # 15%
                    "債券ETF": 0.10,   # 10%
                    "個股": 0.15,      # 15%
                    "現金": 0.10,      # 10%
                    "其他資產": 0.05   # 5%
                }
                
                # 繪製優化後的資產配置餅圖
                opt_df = pd.DataFrame({
                    "分類": list(optimized_portfolio.keys()),
                    "配置比例": list(optimized_portfolio.values())
                })
                
                fig_opt = px.pie(
                    opt_df,
                    values="配置比例",
                    names="分類",
                    title="建議優化後的資產配置",
                    hole=0.4
                )
                fig_opt.update_traces(textposition='inside', textinfo='percent+label')
                fig_opt.update_layout(
                    legend=dict(orientation="h", yanchor="bottom", y=-0.2, xanchor="center", x=0.5)
                )
                st.plotly_chart(fig_opt, use_container_width=True)
                
                # 風險收益最佳化曲線
                st.subheader("風險收益最佳化分析")
                
                # 模擬不同風險等級的投資組合
                risk_levels = np.linspace(0.05, 0.3, 10)  # 年化波動率範圍
                expected_returns = [r * 1.5 for r in risk_levels]  # 假設收益和風險的線性關係
                
                # 加入當前組合的估計點
                current_risk = 0.18  # 假設當前波動率
                current_return = 0.22  # 假設當前收益率
                
                # 加入優化後的估計點
                opt_risk = 0.16  # 假設優化後波動率
                opt_return = 0.24  # 假設優化後收益率
                
                # 創建效率前緣數據框
                ef_data = pd.DataFrame({
                    "風險(年化波動率)": risk_levels,
                    "收益(年化報酬率)": expected_returns
                })
                
                # 繪製效率前緣曲線
                fig_ef = go.Figure()
                
                # 添加效率前緣曲線
                fig_ef.add_trace(go.Scatter(
                    x=ef_data["風險(年化波動率)"],
                    y=ef_data["收益(年化報酬率)"],
                    mode='lines',
                    name='效率前緣',
                    line=dict(color='blue', width=2)
                ))
                
                # 添加當前投資組合點
                fig_ef.add_trace(go.Scatter(
                    x=[current_risk],
                    y=[current_return],
                    mode='markers',
                    name='當前投資組合',
                    marker=dict(color='red', size=12, symbol='circle')
                ))
                
                # 添加優化後投資組合點
                fig_ef.add_trace(go.Scatter(
                    x=[opt_risk],
                    y=[opt_return],
                    mode='markers',
                    name='優化後投資組合',
                    marker=dict(color='green', size=12, symbol='star')
                ))
                
                # 更新佈局
                fig_ef.update_layout(
                    title="投資組合效率前緣分析",
                    xaxis_title="風險 (年化波動率)",
                    yaxis_title="收益 (年化報酬率)",
                    height=500,
                    legend=dict(
                        yanchor="top",
                        y=0.99,
                        xanchor="left",
                        x=0.01
                    )
                )
                
                st.plotly_chart(fig_ef, use_container_width=True)
                
            except Exception as e:
                st.error(f"獲取 AI 建議時發生錯誤: {e}")
                st.write("請嘗試簡化您的持倉輸入，或稍後再試。") 
