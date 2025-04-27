# -*- coding: utf-8 -*-
"""
run.py – Streamlit 全方位財務分析工具 (2025-04 rev I)
頁面：
 1. 市場雷達   (產業篩選 + 自動選股 + AI 燈號 + 單檔詳情)
 2. 技術分析   (MA/RSI/MACD/布林 + K 線 + AI 建議)
 3. 進階分析   (估值、競爭、財務指標+AI、淡旺季、財報+AI、籌碼、公司介紹)
 4. 資產配置   (持倉檢視 + AI 配置建議)
"""

import streamlit as st, yfinance as yf, pandas as pd, numpy as np
import plotly.graph_objects as go, plotly.express as px
import openai, json, re, textwrap
import os

# ===== 從環境變數讀取 API key =====
openai.api_key = os.getenv("OPENAI_API_KEY")
if not openai.api_key:
    st.error("請設置 OPENAI_API_KEY 環境變數")
    st.stop()
# ==================================

st.set_page_config(page_title="財務分析工具", layout="wide")
st.title("📊 全方位財務分析工具")

##############################################################################
# 共用：基本函式
##############################################################################
@st.cache_data(ttl=3600)
def get_tw_lists() -> pd.DataFrame:
    """讀取 AP03/05 上市上櫃名單，回傳 (代號, 名稱, 產業別)"""
    def _read(url):
        try:
            return pd.read_json(url)
        except Exception:
            return pd.DataFrame()

    twse = _read("https://openapi.twse.com.tw/v1/opendata/t187ap03_L")
    otc  = _read("https://openapi.twse.com.tw/v1/opendata/t187ap05_L")
    df   = pd.concat([twse, otc], ignore_index=True)

    col_code = [c for c in df.columns if "股票代號" in c or "代號" in c][0]
    col_name = [c for c in df.columns if "公司名稱" in c][0]
    col_ind  = [c for c in df.columns if "產業別"  in c][0]

    df = df[[col_code, col_name, col_ind]].rename(
        columns={col_code: "代號", col_name: "名稱", col_ind: "產業別"}
    )
    df["代號"] = df["代號"].astype(str)
    df["名稱"] = df["名稱"].str.replace("股份有限公司", "").str.strip()
    return df

@st.cache_data(ttl=600)
def fetch_info(sym: str):
    t      = yf.Ticker(sym)
    info   = t.get_info() or {}
    hist   = t.history(period="6mo")["Close"]
    fin_y  = t.financials.T
    fin_q  = t.quarterly_financials.T
    return info, hist, fin_y, fin_q

def ai_resp(prompt: str, model="gpt-4o-mini", maxtok=500, temp=0.4) -> str:
    try:
        rsp = openai.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=maxtok,
            temperature=temp
        )
        return rsp.choices[0].message.content.strip()
    except Exception as e:
        return f"⚠️ AI 失敗：{e}"

##############################################################################
# 左側功能選單
##############################################################################
PAGE = st.sidebar.radio(               # NEW ⬅︎ 一次展開四大功能
    "🔍 直接點選頁面功能",
    ["市場雷達", "技術分析", "進階分析", "資產配置"],
    index=0
)

##############################################################################
# 1. 市場雷達  ----------------------------------------------------------------
##############################################################################
if PAGE == "市場雷達":
    st.header("🚦 市場雷達")

    with st.expander("❓ 如何使用 / 指標解讀", expanded=False):           # NEW ⬅︎
        st.markdown(textwrap.dedent("""
        **步驟：**

        1. 先從左側「選擇產業」下拉選擇想看的產業（或留 **全部**）。
        2. 在「請選擇股票」多選框勾選要追蹤的標的（預設前五檔）。
        3. 點 **⚡ AI 綜合燈號** 以 EPS季增、ROE、殖利率、52週位置快速篩選。
        4. 再到下方 **🔍 單檔詳情** 查看 K 線與 AI 摘要。

        **指標說明：**

        | 指標        | 用途                               | 怎麼看                                                |
        |-------------|------------------------------------|-------------------------------------------------------|
        | EPS季增      | 本季每股盈餘成長率，衡量獲利動能     | **>0** 代表獲利年增；>10% 常用於成長股篩選            |
        | ROE         | 股東權益報酬率，衡量資金運用效率     | **>15%** 被視為優秀，愈高代表公司替股東賺更多          |
        | 殖利率       | 股息 ÷ 價格，衡量現金股利回報        | 每年穩定 >4% 屬高股息；但太高也可能是股價下跌警訊       |
        | RSI14       | 相對強弱指標（14日），判斷超買超賣    | <30 超賣區，>70 超買區                                 |
        | 52週位置    | 現價在 52 週高低點的相對位置 (0~1)   | 越低代表離高點還遠；<0.5 常搭配成長/價值指標找低基期股  |
        """))

    df0 = get_tw_lists()
    inds = ["全部"] + sorted(df0["產業別"].astype(str).unique().tolist())
    sel_ind = st.selectbox("選擇產業", inds, key="radar_ind")
    df_ind  = df0 if sel_ind == "全部" else df0[df0["產業別"] == sel_ind]

    opts = df_ind.apply(lambda r: f"{r['代號']} {r['名稱']}", axis=1).tolist()
    sels = st.multiselect("請選擇股票 (可多選)", opts, default=opts[:5])

    rows = []
    prog_bar = st.progress(0)                               # NEW ⬅︎ 進度條
    for i, item in enumerate(sels, 1):
        code, name = item.split(" ", 1)
        info, hist, *_ = fetch_info(f"{code}.TW")
        price = info.get("regularMarketPrice") or np.nan
        hi, lo = info.get("fiftyTwoWeekHigh", price), info.get("fiftyTwoWeekLow", price)
        pos52  = (price - lo) / (hi - lo + 1e-9)

        # RSI14
        try:
            d = hist.diff()
            rsi14 = (100 - 100 / (1 + d.clip(lower=0).rolling(14).mean() /
                                   (-d.clip(upper=0).rolling(14).mean()))).iloc[-1]
        except Exception:
            rsi14 = np.nan

        rows.append(dict(
            代號=code, 名稱=name, 現價=round(price, 2),
            EPS季增=round(info.get("earningsQuarterlyGrowth", 0), 3),
            ROE=round(info.get("returnOnEquity", 0), 3),
            殖利率=round(info.get("dividendYield", 0), 3),
            RSI14=round(rsi14, 2), _52週位置=round(pos52, 3)
        ))
        prog_bar.progress(i / len(sels))
    prog_bar.empty()

    df = pd.DataFrame(rows)
    st.dataframe(df, use_container_width=True)

    # --- AI 燈號
    if st.button("⚡ AI 綜合燈號"):
        prompt = (
            "以下為股票清單及指標：\n"
            f"{df.to_json(orient='records', force_ascii=False)}\n"
            "請依 EPS季增、ROE、殖利率、52週位置 綜合評等 🟢🟡🔴，回 JSON："
            "[{'symbol':'2330','light':'🟢','reason':'20字'}]"
        )
        txt = ai_resp(prompt, maxtok=700)
        try:
            lights  = pd.DataFrame(json.loads(re.search(r'\[.*\]', txt, re.S).group()))
            merged  = df.merge(lights, left_on="代號", right_on="symbol", how="left")
            merged  = merged[["代號", "名稱", "現價", "light", "reason"]].rename(
                columns={"light": "燈號", "reason": "理由"})
            st.dataframe(merged, use_container_width=True,
                         column_config={"燈號": st.column_config.Column(width="small")})
        except Exception:
            st.error("AI 回傳格式異常👇")
            st.write(txt)

    # --- 單檔詳情
    st.subheader("🔍 單檔詳情")
    detail = st.selectbox("選擇個股", sels)
    if st.button("查看詳情"):
        code, name = detail.split(" ", 1)
        info, hist, *_ = fetch_info(f"{code}.TW")
        st.metric("現價", info.get("regularMarketPrice"))
        kc = yf.Ticker(f"{code}.TW").history(period="6mo")
        st.plotly_chart(go.Figure(go.Candlestick(
            x=kc.index, open=kc["Open"], high=kc["High"],
            low=kc["Low"], close=kc["Close"]
        )), use_container_width=True)

        brief  = {"symbol": code, "name": name, "price": info.get("regularMarketPrice")}
        prompt = ("以下個股概況，請在 intro/fund/tech/suggest 各給 30 字內摘要 (繁中)，"
                  f"回 JSON：{{intro,fund,tech,suggest}}\n{json.dumps(brief, ensure_ascii=False)}")
        js = ai_resp(prompt, maxtok=400)
        try:
            st.json(json.loads(re.search(r'\{.*\}', js, re.S).group()))
        except:
            st.write(js)

##############################################################################
# 2. 技術分析  ---------------------------------------------------------------
##############################################################################
elif PAGE == "技術分析":
    st.header("📈 多因子技術分析")

    with st.expander("❓ 指標用途與閱讀方式", expanded=False):           # NEW ⬅︎
        st.markdown(textwrap.dedent("""
        **四大因子**  

        | 因子         | 用途                    | 讀法                                                         |
        |--------------|-------------------------|--------------------------------------------------------------|
        | **多頭排列**  | MA(短)>MA(中)>MA(長)     | 成立 → 中長期趨勢向上                                        |
        | **MACD 金叉** | 12 與 26 EMA 交叉       | ￥柱線翻正、黃金交叉 → 轉強點                                |
        | **RSI 超賣** | 動能指標 (14 日)         | RSI <30 可能超賣反彈；>70 可能超買                             |
        | **布林下軌** | 估波動、支撐壓力        | 收盤跌破下軌 → 短線過度悲觀，常伴隨反彈（配合量能更佳）        |

        **綜合建議**  

        * ≥3 個因子同時為真 → **🟢 強買**  
        * 2 個為真         → **🟡 觀望**  
        * 0~1 個為真       → **🔴 賣出 / 不建議進場**
        """))

    code = st.sidebar.text_input("股票代號 (無 .TW)", "2330")
    S = st.sidebar.slider("短期 MA", 5, 60, 20)
    M = st.sidebar.slider("中期 MA", 30, 120, 60)
    L = st.sidebar.slider("長期 MA", 60, 240, 120)

    if st.button("執行分析"):
        df_raw = yf.download(f"{code}.TW", start="2015-01-01")
        if df_raw.empty:
            st.error("下載失敗 / 代號可能錯誤")
        else:
            price = df_raw["Adj Close"] if "Adj Close" in df_raw else df_raw["Close"]
            if isinstance(price, pd.DataFrame):
                price = price.squeeze()          # 或 price.iloc[:, 0]

            df2 = price.to_frame(name="price")

            df2["MA_S"] = df2["price"].rolling(S).mean()
            df2["MA_M"] = df2["price"].rolling(M).mean()
            df2["MA_L"] = df2["price"].rolling(L).mean()

            delta = df2["price"].diff()
            df2["RSI"] = 100 - 100 / (1 + delta.clip(lower=0).rolling(14).mean() /
                                           (-delta.clip(upper=0).rolling(14).mean()))
            ema12  = df2["price"].ewm(span=12).mean()
            ema26  = df2["price"].ewm(span=26).mean()
            df2["MACD"]   = ema12 - ema26
            df2["Signal"] = df2["MACD"].ewm(span=9).mean()

            mid = df2["price"].rolling(20).mean()
            std = df2["price"].rolling(20).std()
            df2["BB_LOW"] = mid - 2*std

            last = df2.iloc[-1]
            f1 = bool((last["MA_S"] > last["MA_M"]) and (last["MA_M"] > last["MA_L"]))
            f2 = bool(last["MACD"] > last["Signal"])
            f3 = bool(last["RSI"] < 30)
            f4 = bool(last["price"] < last["BB_LOW"])
            flags = [f1, f2, f3, f4]

            sig = "🟢 強買" if sum(flags) >= 3 else \
                  "🟡 觀望" if sum(flags) == 2 else "🔴 賣出 / 等待轉強"
            st.metric("AI 綜合建議", sig)

            st.write({
                "多頭排列": f1, "MACD 金叉": f2,
                "RSI 超賣": f3, "布林下軌": f4
            })

            fig = go.Figure([
                go.Candlestick(
                    x=df_raw.index, open=df_raw["Open"], high=df_raw["High"],
                    low=df_raw["Low"], close=df_raw["Close"], name="K 線"
                ),
                go.Scatter(x=df2.index, y=df2["MA_S"], name=f"MA{S}"),
                go.Scatter(x=df2.index, y=df2["MA_M"], name=f"MA{M}"),
                go.Scatter(x=df2.index, y=df2["MA_L"], name=f"MA{L}")
            ])
            fig.update_layout(hovermode="x unified", height=600)
            st.plotly_chart(fig, use_container_width=True)

##############################################################################
# 3. 進階分析  ---------------------------------------------------------------
##############################################################################
elif PAGE == "進階分析":
    st.header("🔍 進階分析")
    syms = [x.strip() for x in st.text_input("輸入多檔代號 (逗號分隔, 無 .TW)", "2330,2454").split(",")]
    tabs = st.tabs(["估值", "競爭", "財務指標+AI", "淡旺季", "財報+AI", "籌碼", "公司介紹"])

    # === 3-1 估值 ===========================================================
    with tabs[0]:
        st.subheader("📐 AI 估值 (DDM / DCF / Comps)")
        with st.expander("❓ DDM / DCF / Comps 是什麼？", expanded=False):  # NEW ⬅︎
            st.markdown(textwrap.dedent("""
            * **DDM (Dividend Discount Model)**  
              ➟ 適合成熟、高股利公司，直接折現未來股息。  
            * **DCF (Discounted Cash Flow)**  
              ➟ 用於評估公司整體自由現金流，適用成長或轉型企業。  
            * **Comps (Comparable Multiples)**  
              ➟ 與同業本益比 / EV-EBITDA 比較，快速估出合理區間。  

            > 一般做法：多法並用 → 取平均或加權平均，並觀察差距。差距過大時要檢查假設或資料來源是否偏誤。
            """))

        if st.button("產生估值"):
            results = []
            for s in syms:
                prompt = (
                    f"你是資深估值分析師，請用 DDM、DCF、Comps 三種方法估算 {s}.TW 合理價，"
                    "並只回傳 JSON：{'DDM':x,'DCF':y,'COMPS':z,'note':'30字內說明'}"
                )
                raw = ai_resp(prompt, maxtok=400)
                m = re.search(r'\{.*\}', raw, flags=re.S)
                if not m:
                    st.warning(f"{s} 估值解析失敗：{raw[:50]}…")
                    continue
                try:
                    data = json.loads(m.group())
                    data["symbol"] = s
                    results.append(data)
                except Exception as e:
                    st.warning(f"{s} JSON 載入失敗：{e}")

            if not results:
                st.error("所有股票估值都失敗，請稍後再試。")
            else:
                df_vals = pd.DataFrame(results).set_index("symbol")

                # ⬇︎ NEW：把文字數字轉成 float，無法轉就設 NaN
                for col in ["DDM", "DCF", "COMPS"]:
                    df_vals[col] = pd.to_numeric(df_vals[col], errors="coerce")

                df_vals["平均估值"] = df_vals[["DDM","DCF","COMPS"]].mean(axis=1).round(2)

                st.table(
                    df_vals.style
                    .format({"DDM":"{:.2f}", "DCF":"{:.2f}", "COMPS":"{:.2f}", "平均估值":"{:.2f}"})
                    .set_caption("三法估值與平均值 (元)"))

    # === 3-2 競爭 ===========================================================
    with tabs[1]:
        st.subheader("🤝 同業競爭力")
        if st.button("AI 分析同業"):
            for s in syms:
                ind = fetch_info(f"{s}.TW")[0].get("industry","(無產業資料)")
                st.markdown(f"**{s} – {ind}**")
                st.write(ai_resp(
                    f"列出 {s}.TW 3-5 個主要競爭對手並用表格比較優劣 (繁中)。", maxtok=600))

    # === 3-3 財務指標 + AI ==================================================
    with tabs[2]:
        st.subheader("🩺 核心財務 (%)")
        rows=[]
        for s in syms:
            info,*_=fetch_info(f"{s}.TW")
            rows.append([
                s,
                (info.get("currentRatio") or 0)*100,
                (info.get("debtToEquity") or 0)*100,
                (info.get("returnOnEquity") or 0)*100,
                (info.get("profitMargins") or 0)*100,
                info.get("trailingPE") or np.nan
            ])
        dfF = pd.DataFrame(rows, columns=["代號","流動比","負債比","ROE","淨利率","P/E"]).round(2)
        st.dataframe(dfF,use_container_width=True)
        if st.button("AI 財務洞察"):
            st.info(ai_resp(
                "以下財務指標 JSON，請 80 字內指出亮點與風險：\n"
                + dfF.to_json(orient="records", force_ascii=False),
                maxtok=500))

    # === 3-4 淡旺季 =========================================================
    # ======== 淡旺季 ========
    with tabs[3]:
        st.subheader("📊 淡旺季營收")
        for s in syms:
            tic = yf.Ticker(f"{s}.TW")

            # ★ FIX vH-2 ── 先抓 income statement，若沒有再用舊 qfin
            qis = tic.quarterly_income_stmt
            if qis.empty:
                _, _, _, qis = fetch_info(f"{s}.TW")            # 舊函式備援
            if qis.empty:
                st.info(f"{s} 查無季報財務資料")
                continue

            # transpose：index=日期（字串），columns=財報項目
            qis = qis.T
            qis.index = pd.to_datetime(qis.index, errors="coerce")

            # 找營收欄位：TotalRevenue / Revenue / Sales 擇一
            rev_col = [c for c in qis.columns
                    if re.search(r"revenue|sales", c, re.I)]
            if not rev_col:
                st.info(f"{s} 季報無營收欄")
                continue

            rev = pd.to_numeric(qis[rev_col[0]], errors="coerce")
            qavg = rev.groupby(rev.index.quarter).mean() / 1e6   # 轉百萬元

            fig = px.bar(
                x=[f"Q{q}" for q in qavg.index],
                y=qavg.values,
                title=f"{s} 淡旺季平均營收 (百萬)",
                labels={"x": "季度", "y": "平均營收"})
            st.plotly_chart(fig, use_container_width=True)


    # === 3-5 財報 + AI ======================================================
    with tabs[4]:
        st.subheader("📜 十三年財報 (千元)")
        want = ["Total Revenue", "Gross Profit", "Operating Income",
                "Net Income", "Total Assets", "Total Liab"]       # ★ FIX vH-2
        bundle = {}
        for s in syms:
            fin = yf.Ticker(f"{s}.TW").financials          # 原始欄為行、欄為日期
            if fin.empty:
                st.info(f"{s} 查無年報財務資料")
                continue

            # 只留下想看的 6 行，再把列名簡化
            fin_sel = (fin.loc[fin.index.intersection(want)]
                        .rename(lambda x: x.replace("Total ", "")
                                            .replace("Operating ", "OP ")
                                            .replace("Net ", ""), axis=0))

            df_fin = (fin_sel / 1e3).T.tail(13)            # 轉置成 index=年度
            df_fin.index = df_fin.index.strftime("%Y")

            df_fmt = df_fin.applymap(
                lambda x: f"{int(x):,}" if pd.notna(x) else "")
            st.markdown(f"**{s}**")
            st.dataframe(df_fmt, use_container_width=True)

            bundle[s] = (df_fin.reset_index()
                        .rename(columns={"index": "年度"})
                        .to_dict("records"))

        if st.button("AI 分析財報"):
            st.markdown(ai_resp(
                "請摘要下列公司財報亮點與隱憂，各 120 字：\n"
                + json.dumps(bundle, ensure_ascii=False),
                maxtok=800))


    # === 3-6 籌碼 ===========================================================
    with tabs[5]:
        st.subheader("📈 籌碼概覽")
        if st.button("AI 籌碼摘要"):
            st.write(ai_resp(
                "請摘要下列股票近一年法人買賣超、融資融券："
                + ",".join([f"{s}.TW" for s in syms]), maxtok=600))

    # === 3-7 公司介紹 =======================================================
    with tabs[6]:
        st.subheader("🏢 公司介紹")
        for s in syms:
            info,*_=fetch_info(f"{s}.TW")
            st.markdown(f"**{s}**")
            st.write(info.get("longBusinessSummary","無資料"))

##############################################################################
# 4. 資產配置  ---------------------------------------------------------------
##############################################################################
else:
    st.header("💼 資產配置")

    with st.expander("❓ 如何輸入持倉", expanded=False):          # NEW ⬅︎
        st.markdown("格式：`股票代號:股數;` 用分號或逗號分隔，現金可用 `現金` 或 `cash`")

    txt = st.text_area("輸入持倉", "2330:100; 2603:50; 現金:500000", height=80)

    def parse_portfolio(s: str):
        h, c = {}, 0.0
        for part in re.split(r"[;,]", s):
            if ":" not in part: continue
            k, v = [x.strip() for x in part.split(":",1)]
            if k in ("現金","cash"):
                c = float(v)
            else:
                h[k] = float(v)
        return h, c

    hold, cash = parse_portfolio(txt)
    rows, total = [], cash
    for code, qty in hold.items():
        close = yf.Ticker(f"{code}.TW").history(period="1d")["Close"].iat[-1]
        val   = close * qty
        rows.append((code, qty, close, val)); total += val
    rows.append(("現金", 0, np.nan, cash))
    dfP = pd.DataFrame(rows, columns=["標的","數量","單價","市值"])
    dfP["比例"] = dfP["市值"].div(total).map("{:.2%}".format)
    st.dataframe(dfP, use_container_width=True)

    st.plotly_chart(go.Figure(go.Pie(labels=dfP["標的"],
                                     values=dfP["市值"].fillna(0), hole=.4)),
                    use_container_width=True)

    if st.button("🧮 AI 配置建議"):
        st.info(ai_resp(
            "你是資產配置顧問，以下持倉：\n"
            + dfP.to_json(orient="records", force_ascii=False)
            + "\n請在 120 字內給出回饋並對各標的標示 🟢🟡🔴",
            model="gpt-3.5-turbo", maxtok=600))
