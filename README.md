# 桃園市居民收入與商業發展關聯性分析

## 專案概述

本專案旨在分析桃園市各行政區居民收入與商業活動之間的關聯性。透過結合政府開放資料平台提供的所得稅申報資料 (105-110年) 與商業統計資料 (105-110年, 部分年份缺失)，探討不同行政區域的平均所得、商業規模（公司行號數量、資本額）及主要行業分布之間的關係，並分析其隨時間變化的趨勢。

## 資料來源

### 1. 桃園市商業行業別及行政區域家數統計
- **資料連結**：[桃園市商業行業別及行政區域家數統計](https://data.nat.gov.tw/dataset/160057)
- **資料描述**：包含桃園市各行政區域之不同產業類別 (A-S 類) 的企業家數及資本額季度/年度統計資料。
- **提供機關**：桃園市政府經濟發展局
- **更新頻率**：不定期更新

### 2. 綜稅綜合所得總額全國各縣市鄉鎮村里統計分析表
- **資料連結**：[綜稅綜合所得總額全國各縣市鄉鎮村里統計分析表](https://data.gov.tw/dataset/103066)
- **資料描述**：全國各縣市鄉鎮村里的綜合所得稅申報資料，包含所得總額、平均數、中位數及第一分位數等統計值。
- **提供機關**：財政部財政資訊中心
- **資料單位**：金額 (千元)

## 研究方法與主要發現

本專案採用統計分析與視覺化方法，主要步驟與發現如下：

1.  **資料整合與前處理**：
    *   整合所得稅與商業數據集 (105-110年)，以行政區域與年份為共同鍵值。
    *   計算各行政區的年度**平均所得**、**總公司行號數量**、**總資本額**及關鍵行業（製造業、批發零售、住宿餐飲、專業科學、金融保險）家數。
    *   處理數據格式、缺失值（如109年商業資料）及欄位名稱不一致問題。

2.  **相關性分析 (以110年為例)**：
    *   計算各指標間的皮爾森相關係數，並繪製熱力圖 (`results/correlation_heatmap_110.png`)。
    *   **發現**：平均所得與製造業家數、總資本額、住宿餐飲業家數、總家數呈現較強正相關；與批發零售、金融保險、專業科學服務業也呈中等正相關。

3.  **視覺化分析 (以110年為例)**：
    *   繪製散佈圖 (`results/scatter_*.png`)，視覺化平均所得與總家數、總資本額的關係。

4.  **集群分析 (以110年為例)**：
    *   使用 K-Means 演算法，基於平均所得和總家數將行政區分為三群 (`results/district_clusters_110.png`)。
    *   **發現**：
        *   群組 0 (中等收入-中等商業): 八德、大園、平鎮、楊梅、蘆竹、龍潭、龜山。
        *   群組 1 (低收入-低商業): 大溪、復興、新屋、觀音。
        *   群組 2 (高收入-高商業): 中壢、桃園。

5.  **變化趨勢分析 (105 vs 110年)**：
    *   計算各行政區平均所得、總家數、總資本額的變化率。
    *   繪製所得變化率 vs 總家數變化率散佈圖 (`results/income_vs_business_change_105_110.png`)。
    *   **發現**：期間內所有行政區平均所得均下降，但商業家數普遍增長（龍潭區資本額顯著下降，復興區資本額與家數增長顯著）。

## 實作步驟

1.  **資料收集**：按照 `data/README.md` 的指引手動下載所需 CSV 檔案至對應目錄。
2.  **環境設置**：安裝 `requirements.txt` 中的 Python 依賴套件 (`pip install -r requirements.txt`)。
3.  **執行分析**：運行核心分析腳本 `python3 src/analyze_income_business_correlation.py`。
4.  **查看結果**：所有分析圖表和合併後的數據 (`merged_income_business_105_110.csv`) 將保存在 `results` 資料夾中。

## 使用工具

- **程式語言**：Python
- **資料處理**：pandas、numpy
- **資料分析**：scikit-learn (用於分群)
- **視覺化**：matplotlib、seaborn

## 專案結構

```
├── data/                      # 原始及處理過的資料
│   ├── raw/
│   │   ├── income_tax/        # 所得稅原始資料
│   │   └── taoyuan_business/  # 商業統計原始資料
│   └── README.md              # 資料下載說明
├── src/                       # 源代碼
│   └── analyze_income_business_correlation.py # 核心分析腳本
├── results/                   # 分析結果圖表與數據
├── requirements.txt           # 依賴套件
└── README.md                  # 本專案說明文件
```

## 團隊成員

- [請填入團隊成員資訊]

## 參考資料

1. 財政部財政資訊中心. (n.d.). 綜稅綜合所得總額全國各縣市鄉鎮村里統計分析表. 政府資料開放平臺. https://data.gov.tw/dataset/103066
2. 桃園市政府經濟發展局. (n.d.). 桃園市商業行業別及行政區域家數統計. 政府資料開放平臺. https://data.nat.gov.tw/dataset/160057

## 預期成果

1. 識別桃園市不同行政區域產業發展與居民收入間的時序關係
2. 發現特定產業成長或衰退對收入變化的影響模式
3. 建立預測模型，預測未來的收入趨勢
4. 提供政策制定參考，了解產業發展對居民收入的影響

## 使用工具

- **程式語言**：Python、R
- **資料處理**：pandas、numpy
- **資料探勘**：prefixspan、arulesSequences (R)、SPMF
- **視覺化**：matplotlib、seaborn、plotly、Tableau

## 專案結構

```
├── data/                      # 原始及處理過的資料
│   ├── raw/                   # 原始資料
│   └── processed/             # 處理過的資料
├── notebooks/                 # Jupyter 筆記本
│   ├── 01-data-preprocessing.ipynb
│   ├── 02-exploratory-analysis.ipynb
│   └── 03-sequential-mining.ipynb
├── src/                       # 源代碼
│   ├── data_preprocessing.py
│   ├── sequence_mining.py
│   └── visualization.py
├── results/                   # 結果與視覺化
├── requirements.txt           # 依賴套件
└── README.md                  # 本文件
```

## 團隊成員

- [請填入團隊成員資訊]

## 參考資料

1. 財政部財政資訊中心. (n.d.). 綜稅綜合所得總額全國各縣市鄉鎮村里統計分析表. 政府資料開放平臺. https://data.gov.tw/dataset/103066
2. 桃園市政府經濟發展局. (n.d.). 桃園市商業行業別及行政區域家數統計. 政府資料開放平臺. https://data.nat.gov.tw/dataset/160057
3. Han, J., Pei, J., & Kamber, M. (2011). Data Mining: Concepts and Techniques. Elsevier.
4. Fournier-Viger, P., Lin, J. C. W., Kiran, R. U., Koh, Y. S., & Thomas, R. (2017). A survey of sequential pattern mining. Data Science and Pattern Recognition, 1(1), 54-77. 