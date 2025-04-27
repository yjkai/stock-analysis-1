# 全方位財務分析工具

這是一個使用 Streamlit 開發的財務分析工具，提供以下功能：

1. 市場雷達：產業篩選、自動選股、AI 燈號、單檔詳情
2. 技術分析：MA/RSI/MACD/布林、K 線、AI 建議
3. 進階分析：估值、競爭、財務指標、淡旺季、財報、籌碼、公司介紹
4. 資產配置：持倉檢視、AI 配置建議

## 本地運行

1. 安裝依賴：
```bash
pip install -r requirements.txt
```

2. 運行應用：
```bash
streamlit run run.py
```

## 部署到 Streamlit Cloud

1. 將代碼推送到 GitHub 倉庫
2. 訪問 [Streamlit Cloud](https://streamlit.io/cloud)
3. 使用 GitHub 帳號登入
4. 點擊 "New app"
5. 選擇倉庫和分支
6. 設置主文件路徑為 `run.py`
7. 在 "Secrets" 中添加 OpenAI API key：
```toml
[openai]
api_key = "your-api-key"
``` 