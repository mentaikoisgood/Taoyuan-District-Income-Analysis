# 桃園市行政區發展潛力分析

本專案分析桃園市13個行政區的發展潛力，通過整合人口、商業、所得、地理和醫療等多維度數據，使用機器學習方法進行聚類分析，最終將行政區分為高、中、低三種發展潛力等級。

## 專案流程

### STEP 1：原始數據收集與載入
- 收集桃園市13個行政區的人口、商業、所得、地理和醫療等數據
- 使用pandas讀取Excel/CSV格式數據
- 處理各種數據格式問題，如編碼、表頭、缺失值等
- 相關檔案：`step1_raw_data_loader.py`

### STEP 2：特徵工程與數據整合
- 從原始數據中提取關鍵特徵
- 處理偏態分布數據（使用對數轉換）
- 創建綜合指標（如醫療服務指數）
- 標準化特徵以便比較
- 最終保留5個關鍵特徵：
  - 人口_working_age_ratio：勞動年齡人口比例
  - 商業_hhi_index：商業集中度指數
  - 所得_median_household_income：家庭收入中位數
  - tertiary_industry_ratio：第三產業比例
  - medical_index：醫療服務綜合指數
- 相關檔案：`step2_feature_engineering.py`

### STEP 3：聚類分析與標籤生成
- 使用t-SNE降維+Ward層次聚類方法
- 將13個行政區分為3個集群
- 基於所得水平和產業結構為集群賦予意義：
  - 高發展潛力：中壢區、桃園區、龜山區、蘆竹區
  - 中發展潛力：八德區、大園區、平鎮區、楊梅區、龍潭區
  - 低發展潛力：大溪區、復興區、新屋區、觀音區
- 相關檔案：`step3_clustering_and_labeling.py`

### STEP 4：模型驗證與解釋
- 使用輪廓係數(Silhouette Score)評估聚類質量：0.718（優秀）
- 使用Davies-Bouldin指數評估：0.545（良好）
- 通過決策樹分析重要特徵：所得是最關鍵的分類依據
- 生成分類規則：
  - 所得≤435,500元 → 低發展潛力
  - 435,500元<所得≤478,000元 → 中發展潛力
  - 所得>478,000元 → 高發展潛力
- 使用箱型圖和雷達圖可視化各群組特徵分布
- 相關檔案：`step4_cluster_validation_interpretation.py`

### STEP 5：互動式網頁儀表板部署
- 創建HTML/CSS/JavaScript網頁儀表板
- 整合分析結果與視覺化圖表
- 提供互動式功能：
  - 行政區潛力等級查詢
  - 特徵對比分析
  - 聚類結果視覺化
  - 政策建議展示
- 部署到GitHub Pages實現線上訪問
- 儀表板網址：https://mentaikoisgood.github.io/Taoyuan-District-Income-Analysis/
- 相關檔案：`docs/` 目錄（包含HTML、CSS、JS文件）

## 安裝與使用

```bash
# 克隆倉庫
git clone https://github.com/mentaikoisgood/Taoyuan-District-Income-Analysis.git
cd Taoyuan-District-Income-Analysis

# 安裝依賴
pip install -r requirements.txt

# 執行各步驟
python step1_raw_data_loader.py  # 數據載入
python step2_feature_engineering.py  # 特徵工程
python step3_clustering_and_labeling.py  # 聚類分析
python step4_cluster_validation_interpretation.py  # 模型驗證

# 查看網頁儀表板
# 訪問：https://mentaikoisgood.github.io/Taoyuan-District-Income-Analysis/
```

## 數據來源

- 人口資料：桃園市政府民政局
- 商業資料：桃園市政府經濟發展局
- 所得資料：財政部財政資訊中心
- 地理資料：內政部國土測繪中心
- 醫療資料：衛生福利部

## 主要發現

1. 桃園市13個行政區可明確分為三類發展潛力群組
2. 中壢區和桃園區為核心商業中心，具有最高的發展潛力
3. 家庭收入中位數是區分發展潛力的最關鍵指標
4. 第三產業比例與醫療服務水平與發展潛力高度相關

## 線上儀表板

🌐 **[立即體驗互動式儀表板](https://mentaikoisgood.github.io/Taoyuan-District-Income-Analysis/)**

儀表板功能包括：
- 📊 行政區發展潛力地圖
- 📈 特徵分析雷達圖
- 🔍 聚類結果互動視覺化
- 📋 詳細數據查詢表格
- 💡 政策建議與洞察

## 專案結構

```
├── docs/                      # GitHub Pages 網頁檔案
│   ├── index.html             # 儀表板主頁
│   ├── css/
│   │   └── style.css          # 樣式檔案
│   ├── js/
│   │   ├── dashboard.js       # 儀表板邏輯
│   │   └── data.js           # 數據處理
│   └── data/                 # 前端數據檔案
├── output/                    # 分析結果
│   ├── taoyuan_features_enhanced.csv  # 增強特徵數據
│   ├── taoyuan_features_numeric.csv   # 基礎數值特徵
│   ├── clustering_results.csv        # 聚類結果
│   ├── taoyuan_enhanced_meta.json     # 元數據
│   └── visualization/                 # 靜態圖表
├── temp_data/                 # 臨時處理數據
├── step1_raw_data_loader.py   # 數據載入腳本
├── step2_feature_engineering.py  # 特徵工程腳本
├── step3_clustering_and_labeling.py  # 聚類分析腳本
├── step4_cluster_validation_interpretation.py  # 模型驗證腳本
└── README.md                  # 本文件
```
