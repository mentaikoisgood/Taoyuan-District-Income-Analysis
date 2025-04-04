#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
分析桃園市各行政區105年至110年收入與商業活動的關聯性
"""

import os
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib as mpl
import re
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

# --- 設定 ---
INCOME_DATA_DIR = "data/raw/income_tax"
BUSINESS_DATA_DIR = "data/raw/taoyuan_business"
RESULTS_DIR = "results"
YEARS = range(105, 110 + 1) # 分析105年至110年
KEY_INDUSTRIES = {
    'G': '批發及零售業',
    'C': '製造業',
    'I': '住宿及餐飲業',
    'M': '專業科學及技術服務業',
    'K': '金融及保險業'
}

# 設定中文字體
try:
    plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'Microsoft JhengHei', 'Apple LiGothic Medium']
    plt.rcParams['axes.unicode_minus'] = False
except Exception as e:
    print(f"設定中文字體失敗: {e}. 圖表中的中文可能無法正確顯示。")

# --- 輔助函數 ---
def safe_float_convert(x):
    """安全地將字串轉換為浮點數，處理逗號、前後及內部空格和非數字字符"""
    if pd.isna(x) or isinstance(x, (int, float)):
        return float(x) if not pd.isna(x) else np.nan
    if isinstance(x, str):
        # 移除逗號、所有空格 (包括內部空格)
        x_cleaned = x.replace(',', '').replace(' ', '')
        if x_cleaned == '-' or x_cleaned == '':
            return np.nan
        try:
            return float(x_cleaned)
        except ValueError:
            return np.nan
    return np.nan

# --- 資料讀取與處理 ---

def load_and_process_income_data(data_dir, years):
    """載入並處理指定年份範圍的所得稅資料，計算區級平均所得"""
    all_yearly_data = []
    print("\n--- 開始處理所得稅資料 ---")
    for year in years:
        file_pattern = os.path.join(data_dir, f"{year}_*.csv")
        matching_files = glob.glob(file_pattern)
        if not matching_files:
            print(f"警告: 找不到 {year} 年的所得稅資料檔案 (模式: {file_pattern})。跳過此年份。")
            continue

        file_path = matching_files[0]
        print(f"處理檔案: {file_path} (年份: {year})")
        try:
            try:
                df = pd.read_csv(file_path, encoding='utf-8')
            except UnicodeDecodeError:
                print(f"    使用UTF-8讀取失敗，嘗試Big5...")
                df = pd.read_csv(file_path, encoding='big5')
            print(f"    成功讀取檔案。")

            # 檢查欄位名稱，處理潛在的 '﻿縣市別' 或 '鄉鎮市區'
            first_col_name = df.columns[0]
            if '鄉鎮市區' in first_col_name or '縣市別' in first_col_name:
                 # 清理可能存在的BOM字元
                 first_col_cleaned = first_col_name.lstrip('\ufeff')
                 df.rename(columns={first_col_name: first_col_cleaned}, inplace=True)
                 district_col_name = first_col_cleaned
            else:
                 print(f"    警告: 第一列欄位名稱非預期 ('縣市別' 或 '鄉鎮市區'), 實際為: {first_col_name}. 將直接使用第一列。")
                 district_col_name = first_col_name # 直接使用，可能有風險

            df_taoyuan = df[df[district_col_name].astype(str).str.startswith('桃園市')].copy()

            if df_taoyuan.empty:
                print(f"    警告: 在 {file_path} 中未找到桃園市資料。")
                continue

            df_taoyuan['行政區'] = df_taoyuan[district_col_name].astype(str).str.replace('桃園市', '').str.strip()

            required_cols = ['納稅單位(戶)', '綜合所得總額']
            missing_cols = [col for col in required_cols if col not in df_taoyuan.columns]
            if missing_cols:
                print(f"    錯誤: 檔案 {file_path} 缺少必要欄位: {missing_cols}")
                continue

            df_taoyuan['納稅單位_戶'] = df_taoyuan['納稅單位(戶)'].apply(safe_float_convert)
            df_taoyuan['綜合所得總額_千元'] = df_taoyuan['綜合所得總額'].apply(safe_float_convert)

            # 處理可能存在的NaN值
            df_taoyuan.dropna(subset=['納稅單位_戶', '綜合所得總額_千元'], inplace=True)

            df_taoyuan = df_taoyuan[~df_taoyuan['行政區'].isin(['合計', '其他', ''])]
            # 確保村里欄位存在才進行篩選 (有些檔案可能沒有村里欄)
            if '村里' in df_taoyuan.columns:
                 df_taoyuan = df_taoyuan[~df_taoyuan['村里'].isin(['合計', '其他'])]
            elif df_taoyuan.columns[1] == '村里': # 根據位置判斷
                 df_taoyuan = df_taoyuan[~df_taoyuan.iloc[:, 1].isin(['合計', '其他'])]


            district_agg = df_taoyuan.groupby('行政區').agg(
                總納稅單位_戶=('納稅單位_戶', 'sum'),
                總綜合所得_千元=('綜合所得總額_千元', 'sum')
            ).reset_index()

            district_agg['平均所得_千元'] = district_agg.apply(
                lambda row: (row['總綜合所得_千元'] / row['總納稅單位_戶']) if row['總納稅單位_戶'] > 0 else 0,
                axis=1
            )

            district_agg['年份'] = year
            print(f"    成功處理 {year} 年所得稅資料，共 {len(district_agg)} 個行政區。")
            all_yearly_data.append(district_agg[['行政區', '年份', '平均所得_千元', '總納稅單位_戶']]) # 保留戶數供後續可能的人均計算

        except Exception as e:
            print(f"    處理檔案 {file_path} 時發生錯誤: {e}")

    if not all_yearly_data:
        print("錯誤: 未能成功處理任何年份的所得稅資料。")
        return None

    income_df = pd.concat(all_yearly_data, ignore_index=True)
    print("--- 所得稅資料處理完成 ---")
    return income_df

def load_and_process_business_data(data_dir, years, key_industries):
    """載入並處理指定年份範圍的商業資料，提取總家數、總資本額及關鍵行業家數"""
    all_yearly_data = []
    print("\n--- 開始處理商業資料 ---")

    for year in years:
        # 商業資料的年份可能包含月份，需要更彈性的匹配
        target_year_str = str(year)
        found_file = False
        # 優先尋找包含年份的12月檔案
        file_pattern_dec = os.path.join(data_dir, f"{target_year_str}年12月*.csv")
        matching_files_dec = glob.glob(file_pattern_dec)

        if matching_files_dec:
             file_path = matching_files_dec[0]
             found_file = True
        else:
            # 如果沒有12月，找尋該年任意月份的檔案
            file_pattern_any = os.path.join(data_dir, f"{target_year_str}年*.csv")
            matching_files_any = glob.glob(file_pattern_any)
            if matching_files_any:
                 # 如果有多個月份，選取最新的那個 (假設檔名排序可反映時間)
                 file_path = sorted(matching_files_any)[-1]
                 found_file = True
            else:
                 print(f"警告: 找不到 {year} 年的商業資料檔案。跳過此年份。")
                 continue

        print(f"處理檔案: {file_path} (年份: {year})")
        try:
            df = pd.read_csv(file_path, encoding='utf-8')
            df.columns = [col.strip() for col in df.columns]
            district_col = df.columns[0]
            item_col = df.columns[1]
            total_col = df.columns[-1]

            # --- 修正資本額行政區名稱 --- ##
            df['is_total_row'] = df[district_col].astype(str).str.contains('合計')
            # Replace blanks and common placeholders with NaN
            df[district_col] = df[district_col].replace('^\s*$', np.nan, regex=True)
            df[district_col] = df[district_col].replace(['nan', '-'], np.nan) # Handle 'nan' string and '-' separately
            print(f"    NaNs in district column before ffill: {df[district_col].isna().sum()}")
            # Group and forward-fill
            df[district_col] = df.groupby((df['is_total_row'] == True).cumsum())[district_col].ffill()
            print(f"    NaNs in district column after ffill: {df[district_col].isna().sum()}")
            df.drop(columns=['is_total_row'], inplace=True)
            # --- 填充結束 --- ##

            df['行政區'] = df[district_col].astype(str).str.replace('桃園市', '').str.strip()
            print(f"    原始資料維度: {df.shape}")
            print(f"    填充後行政區 (部分): {df['行政區'].unique()[:20]}...")

            # 篩選家數和資本額
            df_counts = df[df[item_col] == '家數'].copy()
            df_capital = df[df[item_col] == '資本額'].copy()
            print(f"    篩選後 家數維度: {df_counts.shape}, 資本額維度: {df_capital.shape}")
            print(f"    資本額行政區 (篩選後，填充後): {df_capital['行政區'].unique()}")


            # --- 數據清理與篩選 --- ##
            df_counts['行政區'] = df_counts['行政區'].str.strip()
            df_capital['行政區'] = df_capital['行政區'].str.strip()
            valid_districts_mask_counts = ~df_counts['行政區'].isin(['合計', '其他', '', 'nan'])
            df_counts_filtered = df_counts[valid_districts_mask_counts]
            print(f"    家數過濾無效區後 維度: {df_counts_filtered.shape}")

            valid_districts = df_counts_filtered['行政區'].unique()
            valid_districts_mask_capital = ~df_capital['行政區'].isin(['合計', '其他', '', 'nan'])
            df_capital_filtered = df_capital[valid_districts_mask_capital & df_capital['行政區'].isin(valid_districts)]
            print(f"    資本額過濾無效區並匹配有效區後 維度: {df_capital_filtered.shape}")

            if df_counts_filtered.empty:
                 print(f"    警告: 未找到有效的行政區資料(家數)。")
                 continue

            df_counts_filtered = df_counts_filtered.reset_index(drop=True)
            df_capital_filtered = df_capital_filtered.reset_index(drop=True)
            # --- 清理結束 --- ##

            results = {'年份': year}
            districts = df_counts_filtered['行政區'].tolist()
            results['行政區'] = districts

            # 提取總家數
            total_counts_raw = df_counts_filtered[total_col]
            results['總家數'] = total_counts_raw.apply(safe_float_convert).tolist()
            # 還原打印語句
            print(f"    總家數 (轉換後有效數): {pd.Series(results['總家數']).notna().sum()} / {len(results['總家數'])}")

            # 提取總資本額
            if len(df_capital_filtered) == len(districts):
                total_capital_raw = df_capital_filtered[total_col]
                results['總資本額_千元'] = total_capital_raw.apply(safe_float_convert).tolist()
                # 還原打印語句
                print(f"    總資本額 (轉換後有效數): {pd.Series(results['總資本額_千元']).notna().sum()} / {len(results['總資本額_千元'])}")
            else:
                print(f"    警告: 家數({len(districts)})和資本額({len(df_capital_filtered)}) 行數不匹配，嘗試按行政區合併資本額...")
                temp_df = pd.DataFrame({'行政區': districts})
                capital_to_merge = df_capital_filtered[['行政區', total_col]].copy()
                capital_to_merge['總資本額_千元'] = capital_to_merge[total_col].apply(safe_float_convert)
                merged_capital = pd.merge(temp_df, capital_to_merge[['行政區', '總資本額_千元']], on='行政區', how='left')
                results['總資本額_千元'] = merged_capital['總資本額_千元'].tolist()
                # 還原打印語句
                print(f"    總資本額 (Merge後有效數): {pd.Series(results['總資本額_千元']).notna().sum()} / {len(results['總資本額_千元'])}")

            # 提取關鍵行業家數
            for code, name in key_industries.items():
                # 找到包含行業代碼的欄位名稱 (例如 'G批發及零售業')
                industry_col_found = None
                for col in df_counts_filtered.columns: # 從 df_counts_filtered 迭代
                    if col.startswith(code):
                        industry_col_found = col
                        break
                if industry_col_found:
                    col_name = f"{name}_家數"
                    results[col_name] = df_counts_filtered[industry_col_found].apply(safe_float_convert).tolist()
                else:
                    print(f"    警告: 在 {year} 年資料中未找到行業 '{name}' ({code}) 的欄位。")
                    results[f"{name}_家數"] = [np.nan] * len(districts) # 填充NaN

            year_df = pd.DataFrame(results)
            all_yearly_data.append(year_df)
            print(f"    成功處理 {year} 年商業資料，共 {len(year_df)} 個行政區。")

        except Exception as e:
            print(f"    處理檔案 {file_path} 時發生錯誤: {e}")
            import traceback
            traceback.print_exc()

    if not all_yearly_data:
        print("錯誤: 未能成功處理任何年份的商業資料。")
        return None

    business_df = pd.concat(all_yearly_data, ignore_index=True)
    print("--- 商業資料處理完成 ---")
    return business_df


# --- 分析與視覺化 ---

def perform_correlation_analysis(merged_df, year, results_dir):
    """計算指定年份收入與商業指標的相關性"""
    print(f"\n--- {year}年 收入-商業 相關性分析 ---")
    df_year = merged_df[merged_df['年份'] == year].copy()

    if df_year.empty:
        print(f"錯誤: {year}年沒有合併後的資料可供分析。")
        return None

    # 選擇數值型欄位進行相關性分析
    numeric_cols = df_year.select_dtypes(include=np.number).columns.tolist()
    cols_to_analyze = [col for col in numeric_cols if col not in ['年份', '總納稅單位_戶']]
    if '平均所得_千元' not in cols_to_analyze:
         print(f"錯誤: {year}年資料缺少 '平均所得_千元' 欄位。")
         return None

    # --- Debugging: 檢查數據類型 ---
    print(f"\n{year}年 用於相關性分析的欄位及類型:")
    print(df_year[cols_to_analyze].info())
    # --- End Debugging ---

    # 在計算相關性前，再次確保欄位是數值型，並處理潛在的 Inf/-Inf
    df_analyze = df_year[cols_to_analyze].copy()
    for col in df_analyze.columns:
        df_analyze[col] = pd.to_numeric(df_analyze[col], errors='coerce') # 強制轉為數值，無法轉換的變NaN
    df_analyze.replace([np.inf, -np.inf], np.nan, inplace=True) # 替換 Inf

    # 移除全為 NaN 的列，避免相關性計算出錯
    df_analyze.dropna(axis=1, how='all', inplace=True)
    # 重新獲取要分析的欄位名
    cols_to_analyze_cleaned = df_analyze.columns.tolist()
    if not cols_to_analyze_cleaned or '平均所得_千元' not in cols_to_analyze_cleaned:
         print(f"錯誤: {year}年資料在清理後缺少有效數值欄位或平均所得欄位。")
         return None, None

    correlation_matrix = df_analyze.corr()

    print(f"\n{year}年 相關係數矩陣 (清理後):")
    print(correlation_matrix)

    # 提取與平均所得的相關性
    if '平均所得_千元' in correlation_matrix.columns:
        income_correlations = correlation_matrix['平均所得_千元'].drop('平均所得_千元', errors='ignore')
        print(f"\n{year}年 平均所得與各商業指標的相關係數:")
        print(income_correlations.sort_values(ascending=False))
    else:
        income_correlations = None
        print(f"警告: {year}年相關係數矩陣中未找到 '平均所得_千元'。")

    # 繪製熱力圖
    if not correlation_matrix.empty:
        plt.figure(figsize=(12, 10))
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=.5)
        plt.title(f'{year}年 桃園市收入與商業指標相關係數熱力圖', fontsize=16)
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()
        heatmap_path = os.path.join(results_dir, f'correlation_heatmap_{year}.png')
        plt.savefig(heatmap_path)
        print(f"相關係數熱力圖已保存至: {heatmap_path}")
        plt.close()
    else:
        print(f"警告: {year}年無法生成相關係數熱力圖，因為相關係數矩陣為空。")

    return correlation_matrix, income_correlations

def plot_scatter(df_year, x_col, y_col, title, xlabel, ylabel, filename, results_dir, hue_col=None, size_col=None):
    """繪製散佈圖"""
    plt.figure(figsize=(12, 8))
    scatter_plot = sns.scatterplot(data=df_year, x=x_col, y=y_col, hue=hue_col, size=size_col, sizes=(50, 500), legend='auto' if hue_col or size_col else False)

    # 添加文字標籤
    for i in range(df_year.shape[0]):
        plt.text(df_year[x_col].iloc[i] * 1.01, df_year[y_col].iloc[i], # 稍微偏移避免重疊
                 df_year['行政區'].iloc[i], fontdict={'size': 9})

    plt.title(title, fontsize=16)
    plt.xlabel(xlabel, fontsize=12)
    plt.ylabel(ylabel, fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.6)

    # 調整圖例位置 (如果有的話)
    if hue_col or size_col:
        plt.legend(bbox_to_anchor=(1.03, 1), loc='upper left', borderaxespad=0.)

    plt.tight_layout(rect=[0, 0, 0.85, 1] if hue_col or size_col else [0, 0, 1, 1]) # 為圖例留空間
    filepath = os.path.join(results_dir, filename)
    plt.savefig(filepath)
    print(f"散佈圖已保存至: {filepath}")
    plt.close()

def perform_clustering(df_year, features, n_clusters, results_dir):
     """執行K-Means分群"""
     print(f"\n--- {df_year['年份'].iloc[0]}年 基於 {', '.join(features)} 的行政區分群 ---")
     X = df_year[features].copy()
     X.dropna(inplace=True) # 移除包含NaN的行

     if X.empty or len(X) < n_clusters:
          print(f"錯誤: 用於分群的數據不足 (找到 {len(X)} 筆有效數據, 需要至少 {n_clusters} 筆)。")
          return None

     # 標準化數據
     scaler = StandardScaler()
     X_scaled = scaler.fit_transform(X)

     # K-Means
     kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10) # n_init suppresses warning
     clusters = kmeans.fit_predict(X_scaled)

     # 將分群結果加回原數據框 (需要對齊索引)
     df_clustered = df_year.loc[X.index].copy() # 確保索引對應
     df_clustered['分群'] = clusters

     print("\n分群結果:")
     print(df_clustered[['行政區', '分群'] + features].sort_values('分群'))

     # 視覺化分群結果 (使用前兩個特徵繪圖)
     cluster_feature_x = features[0]
     cluster_feature_y = features[1]
     plot_scatter(df_clustered,
                  x_col=cluster_feature_x,
                  y_col=cluster_feature_y,
                  hue_col='分群',
                  title=f"{df_year['年份'].iloc[0]}年 行政區分群 ({cluster_feature_x} vs {cluster_feature_y})",
                  xlabel=cluster_feature_x,
                  ylabel=cluster_feature_y,
                  filename=f'district_clusters_{df_year["年份"].iloc[0]}.png',
                  results_dir=results_dir)

     return df_clustered

def analyze_trends_105_vs_110(merged_df, results_dir):
    """比較105年和110年的變化趨勢"""
    print("\n--- 分析105年至110年變化趨勢 ---")
    df_105 = merged_df[merged_df['年份'] == 105].set_index('行政區')
    df_110 = merged_df[merged_df['年份'] == 110].set_index('行政區')

    # 確保兩個年份都有數據的行政區才進行比較
    common_districts = df_105.index.intersection(df_110.index)
    if len(common_districts) == 0:
        print("錯誤: 105年與110年沒有共同的行政區數據可供比較。")
        return

    df_105 = df_105.loc[common_districts]
    df_110 = df_110.loc[common_districts]

    # 計算變化率
    change_df = pd.DataFrame(index=common_districts)
    change_df['所得變化率_%'] = ((df_110['平均所得_千元'] - df_105['平均所得_千元']) / df_105['平均所得_千元'] * 100).round(2)
    change_df['總家數變化率_%'] = ((df_110['總家數'] - df_105['總家數']) / df_105['總家數'] * 100).round(2)
    change_df['總資本額變化率_%'] = ((df_110['總資本額_千元'] - df_105['總資本額_千元']) / df_105['總資本額_千元'] * 100).round(2)

    # 處理除以零或NaN的情況
    change_df.replace([np.inf, -np.inf], np.nan, inplace=True)

    print("\n105年至110年變化率:")
    print(change_df.sort_values('所得變化率_%', ascending=False))

    # 繪製變化率散佈圖
    change_df_plot = change_df.dropna(subset=['所得變化率_%', '總家數變化率_%']) # 移除NaN值以繪圖
    if not change_df_plot.empty:
        plt.figure(figsize=(12, 8))
        scatter_plot = sns.scatterplot(data=change_df_plot, x='所得變化率_%', y='總家數變化率_%', hue=change_df_plot.index, legend=False) # 用顏色區分行政區，但不顯示圖例

        # 添加文字標籤
        for district in change_df_plot.index:
             plt.text(change_df_plot.loc[district, '所得變化率_%'] * 1.01,
                      change_df_plot.loc[district, '總家數變化率_%'],
                      district, fontdict={'size': 9})

        plt.axhline(0, color='grey', linestyle='--', lw=1)
        plt.axvline(0, color='grey', linestyle='--', lw=1)
        plt.title('桃園市各行政區收入與商業家數變化率 (105 vs 110年)', fontsize=16)
        plt.xlabel('平均所得變化率 (%)', fontsize=12)
        plt.ylabel('總公司行號數量變化率 (%)', fontsize=12)
        plt.grid(True, linestyle='--', alpha=0.6)
        trend_path = os.path.join(results_dir, 'income_vs_business_change_105_110.png')
        plt.savefig(trend_path)
        print(f"變化趨勢散佈圖已保存至: {trend_path}")
        plt.close()
    else:
         print("警告: 無法繪製變化趨勢散佈圖，因為缺少足夠的數據。")


# --- 主函數 ---
def main():
    """主執行流程"""
    # 確保結果目錄存在
    if not os.path.exists(RESULTS_DIR):
        os.makedirs(RESULTS_DIR)
        print(f"已創建結果目錄: {RESULTS_DIR}")

    # 1. 載入並處理資料
    income_df = load_and_process_income_data(INCOME_DATA_DIR, YEARS)
    business_df = load_and_process_business_data(BUSINESS_DATA_DIR, YEARS, KEY_INDUSTRIES)

    if income_df is None or business_df is None:
        print("錯誤: 資料載入或處理失敗，無法繼續分析。")
        return

    # 2. 合併資料
    print("\n--- 合併收入與商業資料 ---")
    merged_df = pd.merge(income_df, business_df, on=['行政區', '年份'], how='inner') # 使用 inner 保留兩個數據集都存在的區/年份
    if merged_df.empty:
         print("錯誤: 合併後的資料框為空，請檢查輸入數據的行政區名稱和年份是否匹配。")
         print("收入數據行政區:", income_df['行政區'].unique())
         print("商業數據行政區:", business_df['行政區'].unique())
         return

    print("合併完成。合併後資料預覽:")
    print(merged_df.head())
    print(f"合併後資料筆數: {len(merged_df)}")
    # 儲存合併後的資料
    merged_file_path = os.path.join(RESULTS_DIR, 'merged_income_business_105_110.csv')
    merged_df.to_csv(merged_file_path, index=False, encoding='utf-8-sig') # 使用 utf-8-sig 確保Excel能正確讀取中文
    print(f"合併後的資料已保存至: {merged_file_path}")


    # 3. 執行相關性分析 (以110年為例)
    latest_year = 110
    corr_matrix, income_corr = perform_correlation_analysis(merged_df, latest_year, RESULTS_DIR)

    # 4. 繪製散佈圖 (以110年為例)
    df_latest = merged_df[merged_df['年份'] == latest_year].copy()
    if not df_latest.empty:
         plot_scatter(df_latest, '平均所得_千元', '總家數',
                      f'{latest_year}年 平均所得 vs 總公司行號數量', '平均所得 (千元)', '總公司行號數量',
                      f'scatter_income_vs_count_{latest_year}.png', RESULTS_DIR)
         plot_scatter(df_latest, '平均所得_千元', '總資本額_千元',
                      f'{latest_year}年 平均所得 vs 總資本額', '平均所得 (千元)', '總資本額 (千元)',
                      f'scatter_income_vs_capital_{latest_year}.png', RESULTS_DIR)
    else:
         print(f"警告: {latest_year}年無數據可繪製散佈圖。")


    # 5. 執行分群分析 (以110年為例，使用平均所得和總家數)
    cluster_features = ['平均所得_千元', '總家數']
    if not df_latest.empty and all(feat in df_latest.columns for feat in cluster_features):
         # 嘗試將區分為3群：高-高, 高-低/低-高, 低-低
         perform_clustering(df_latest, features=cluster_features, n_clusters=3, results_dir=RESULTS_DIR)
    else:
         print(f"警告: {latest_year}年缺少分群所需特徵 ({', '.join(cluster_features)}) 或無數據。")


    # 6. 分析105 vs 110 趨勢
    analyze_trends_105_vs_110(merged_df, RESULTS_DIR)

    print(f"\n--- 分析全部完成 ---")
    print(f"所有結果圖表和數據已保存到 '{RESULTS_DIR}' 資料夾中。")

if __name__ == "__main__":
    main() 