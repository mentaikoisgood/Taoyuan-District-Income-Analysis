import pandas as pd
import numpy as np
import os
import json
from datetime import datetime
try:
    import geopandas as gpd
    GEOPANDAS_AVAILABLE = True
except ImportError:
    GEOPANDAS_AVAILABLE = False

# 桃園市13個行政區
TAOYUAN_DISTRICTS = [
    '中壢', '八德', '大園', '大溪', '平鎮', '復興', 
    '桃園', '新屋', '楊梅', '龍潭', '龜山', '蘆竹', '觀音'
]

# 常量：基礎欄位名稱
POP_TOTAL = '人口_total_population'
POP_WORK_RATIO = '人口_working_age_ratio'
COM_TOTAL_CAP = '商業_total_capital'
COM_TOTAL_CNT = '商業_total_companies'
COM_HHI = '商業_hhi_index'
COM_TERTIARY = '商業_tertiary_industry_count'
INCOME_MEDIAN = '所得_median_household_income'
INCOME_HOUSEHOLDS = '所得_total_households'
GEO_AREA = '地理_area_km2'
FACTORY_COUNT = '工廠_factory_count'
HEALTH_BEDS = '醫療_total_beds'
HEALTH_PERSON = '醫療_medical_personnel_total'
HEALTH_FACILITIES = '醫療_medical_facilities_total'

# 衍生特徵名稱
DERIVED_BEDS_PER_1K = 'beds_per_1k_pop'
DERIVED_STAFF_PER_1K = 'med_staff_per_1k_pop'
DERIVED_CAP_PER_HOUSEHOLD = 'capital_per_household'
DERIVED_FACTORIES_PER_1K_WORK = 'factories_per_1k_working_pop'
DERIVED_TERTIARY_RATIO = 'tertiary_industry_ratio'
DERIVED_MED_DENSITY = 'medical_density_area'
DERIVED_ECON_INDEX = 'economic_index'
DERIVED_MED_INDEX = 'medical_index'
DERIVED_AVG_FACTORY_CAP = 'avg_factory_capital'

def clean_numeric_data(series):
    """清理數值數據的通用函數"""
    return pd.to_numeric(
        series.astype(str)
        .str.replace(',', '')
        .str.replace(' ', '')
        .str.replace('-', '0')
        .str.replace('－', '0'), 
        errors='coerce'
    ).fillna(0)

def create_population_features(raw_population_data):
    """從原始人口數據創建特徵 - 修正版：正確處理Unnamed欄位"""
    print("🔧 創建人口特徵（修正版：包含Unnamed欄位）...")
    
    if raw_population_data.empty:
        print("⚠️  原始人口數據為空，跳過特徵創建")
        return pd.DataFrame()
    
    # 尋找總計欄位
    total_col_name = None
    for col in raw_population_data.columns:
        if '總' in str(col) and '計' in str(col):
            total_col_name = col
            break
    
    if total_col_name is None:
        print("❌ 未找到總計欄位")
        return pd.DataFrame()
    
    print(f"  📊 找到總計欄位: {total_col_name}")
    
    # 🆕 基於數值模式檢測所有年齡相關欄位（包括Unnamed）
    exclude_keywords = ['性別', '區域代碼', '區域別', '總', '計', '小計']
    age_numeric_cols = []
    
    # 先取得樣本數據來測試數值欄位
    sample_row = raw_population_data[raw_population_data['性別'] == '男'].iloc[0] if len(raw_population_data) > 0 else None
    
    if sample_row is None:
        print("❌ 無法找到樣本數據")
        return pd.DataFrame()
    
    for col in raw_population_data.columns:
        col_str = str(col).strip()
        
        # 排除非年齡欄位
        if any(kw in col_str for kw in exclude_keywords):
            continue
        
        # 檢測是否為數值欄位（包含年齡數據）
        try:
            test_val = pd.to_numeric(sample_row[col], errors='coerce')
            if not pd.isna(test_val) and test_val >= 0:  # 有效的非負數值
                age_numeric_cols.append(col)
        except Exception:
            continue
    
    print(f"  📈 檢測到 {len(age_numeric_cols)} 個數值年齡欄位（包含Unnamed）")
    
    # 🆕 智能確定勞動年齡欄位範圍（15-64歲）
    # 策略：基於欄位位置和明確年齡標記來估算範圍
    
    # 找到明確的年齡標記欄位作為參考點
    age_markers = {}
    for i, col in enumerate(age_numeric_cols):
        col_str = str(col)
        if '～' in col_str:
            # 提取年齡數字
            import re
            age_match = re.search(r'(\d+)～(\d+)', col_str)
            if age_match:
                start_age = int(age_match.group(1))
                end_age = int(age_match.group(2))
                age_markers[i] = (start_age, end_age, col)
    
    print(f"  🎯 找到 {len(age_markers)} 個明確年齡標記欄位")
    
    # 確定勞動年齡範圍的欄位索引
    working_age_start_idx = None
    working_age_end_idx = None
    
    if age_markers:
        # 方法1：基於明確標記確定範圍
        for idx, (start_age, end_age, col) in age_markers.items():
            if start_age == 15:  # 找到15歲開始的欄位
                working_age_start_idx = idx
                print(f"    ✅ 找到勞動年齡起點: 索引{idx} ({col})")
                break
        
        for idx, (start_age, end_age, col) in age_markers.items():
            if start_age == 65:  # 找到65歲開始的欄位（勞動年齡結束）
                working_age_end_idx = idx - 1  # 65歲之前的欄位
                print(f"    ✅ 找到勞動年齡終點: 索引{working_age_end_idx} (65歲前)")
                break
    
    # 方法2：如果沒有找到明確標記，使用估算
    if working_age_start_idx is None or working_age_end_idx is None:
        print("  ⚠️  未找到明確年齡標記，使用估算方法")
        total_age_cols = len(age_numeric_cols)
        
        # 估算：假設0-14歲約占前15%，65+歲約占後35%
        estimated_start_idx = int(total_age_cols * 0.15)  # 跳過前15%（0-14歲）
        estimated_end_idx = int(total_age_cols * 0.65)    # 到65%位置（64歲）
        
        working_age_start_idx = working_age_start_idx or estimated_start_idx
        working_age_end_idx = working_age_end_idx or estimated_end_idx
        
        print(f"    📊 估算勞動年齡範圍: 索引 {working_age_start_idx} 到 {working_age_end_idx}")
    
    # 確定最終的勞動年齡欄位
    working_age_cols = age_numeric_cols[working_age_start_idx:working_age_end_idx+1]
    print(f"  ✅ 最終勞動年齡欄位數量: {len(working_age_cols)}")
    print(f"     前3個: {working_age_cols[:3]}")
    print(f"     後3個: {working_age_cols[-3:]}")
    
    # 計算每個行政區的人口特徵
    population_features = []
    taoyuan_district_names = [name + '區' for name in TAOYUAN_DISTRICTS]
    
    print(f"  🏘️  開始計算13個行政區的人口特徵...")
    
    for district_name in taoyuan_district_names:
        male_data = raw_population_data[
            (raw_population_data['區域別'] == district_name) & 
            (raw_population_data['性別'] == '男')
        ]
        female_data = raw_population_data[
            (raw_population_data['區域別'] == district_name) & 
            (raw_population_data['性別'] == '女')
        ]
        
        if male_data.empty or female_data.empty:
            print(f"    ⚠️  {district_name} 缺少完整的男女數據")
            continue
        
        # 計算總人口
        male_total = pd.to_numeric(male_data[total_col_name].iloc[0], errors='coerce') or 0
        female_total = pd.to_numeric(female_data[total_col_name].iloc[0], errors='coerce') or 0
        total_population = male_total + female_total
        
        # 🆕 計算勞動年齡人口（使用修正後的欄位列表）
        male_working_age = 0
        female_working_age = 0
        
        for col in working_age_cols:
            if col in male_data.columns:
                male_val = pd.to_numeric(male_data[col].iloc[0], errors='coerce') or 0
                male_working_age += male_val
            
            if col in female_data.columns:
                female_val = pd.to_numeric(female_data[col].iloc[0], errors='coerce') or 0
                female_working_age += female_val
        
        total_working_age = male_working_age + female_working_age
        working_age_ratio = (total_working_age / total_population * 100) if total_population > 0 else 0
        
        # 🆕 數據驗證：檢查比例是否合理
        if working_age_ratio < 50 or working_age_ratio > 80:
            print(f"    ⚠️  {district_name} 勞動年齡比例異常: {working_age_ratio:.1f}%")
        
        population_features.append({
            '區域別': district_name,
            'total_population': total_population,
            'working_age_ratio': working_age_ratio
        })
        
        # 顯示計算詳情（僅前3個區域）
        if len(population_features) <= 3:
            print(f"    📊 {district_name}: 總人口 {total_population:,.0f}, 勞動年齡 {total_working_age:,.0f}, 比例 {working_age_ratio:.1f}%")
    
    df_population = pd.DataFrame(population_features)
    
    # 🆕 最終驗證和統計
    if not df_population.empty:
        avg_ratio = df_population['working_age_ratio'].mean()
        min_ratio = df_population['working_age_ratio'].min()
        max_ratio = df_population['working_age_ratio'].max()
        
        print(f"  📈 勞動年齡比例統計:")
        print(f"     平均: {avg_ratio:.1f}%")
        print(f"     範圍: {min_ratio:.1f}% ~ {max_ratio:.1f}%")
        
        if 60 <= avg_ratio <= 75:
            print(f"  ✅ 平均比例在合理範圍內 (60-75%)")
        else:
            print(f"  ⚠️  平均比例超出預期範圍，可能需要進一步調整")
    
    print(f"✅ 創建人口特徵完成，包含 {len(df_population)} 個行政區")
    return df_population

def create_commercial_features(raw_commercial_data):
    """從原始商業數據創建綜合經濟指標特徵"""
    print("🔧 創建商業特徵...")
    
    if raw_commercial_data.empty:
        print("⚠️  原始商業數據為空，跳過特徵創建")
        return pd.DataFrame()
    
    # 識別產業欄位 - 排除總計欄位避免重複計算
    exclude_cols = ['區域別', '項目', '各行政區合計家數及資本額']
    
    # 定義產業分類 - 使用完整欄位名稱
    primary_industries = ['A農林漁牧業', 'B礦業及土石採取業']  # 第一級產業
    secondary_industries = ['C製造業', 'D電力及燃氣供應業', 'E用水供應及污染整治業', 'F營造業']  # 第二級產業
    tertiary_industries = ['G批發及零售業', 'H運輸及倉儲業', 'I住宿及餐飲業', 'J資訊及通訊傳播業', 
                          'K金融及保險業', 'L不動產業', 'M專業科學及技術服務業', 'N支援服務業', 
                          'O公共行政及國防；強制性社會安全', 'P教育服務業', 'Q醫療保健及社會服務業', 
                          'R藝術娛樂及休閒服務業', 'S其他服務業']  # 第三級產業
    
    # 篩選桃園市行政區的家數行
    taoyuan_count_rows = raw_commercial_data[
        (raw_commercial_data['區域別'].astype(str).str.contains('桃園市', na=False)) & 
        (raw_commercial_data['項目'] == '家數')
    ]
    
    commercial_features = []
    
    for idx, count_row in taoyuan_count_rows.iterrows():
        # 對應的資本額行在下一行
        capital_idx = idx + 1
        if capital_idx < len(raw_commercial_data) and raw_commercial_data.iloc[capital_idx]['項目'] == '資本額':
            capital_row = raw_commercial_data.iloc[capital_idx]
        else:
            print(f"⚠️  {count_row['區域別']} 缺少對應的資本額行")
            continue
        
        # 清理區域名稱
        district_name = count_row['區域別'].replace('桃園市', '').strip()
        
        # 提取各產業的家數和資本額 - 排除總計欄位
        industry_counts = {}
        industry_capitals = {}
        
        for col in raw_commercial_data.columns:
            if col not in exclude_cols:
                industry_counts[col] = clean_numeric_data(pd.Series([count_row[col]])).iloc[0]
                industry_capitals[col] = clean_numeric_data(pd.Series([capital_row[col]])).iloc[0]
            
        # 計算總家數和總資本額
        total_count = sum(industry_counts.values())
        total_capital = sum(industry_capitals.values())
        
        # 計算 HHI 指數 (Herfindahl Index) - 修正：使用資本份額
        hhi = 0
        if total_capital > 0:
            for capital in industry_capitals.values():
                capital_share = capital / total_capital
                hhi += capital_share ** 2
        hhi = hhi * 10000  # 轉換為標準的 HHI 範圍 (0-10000)
        
        # 計算各級產業總數 - 使用完整欄位名稱，排除總計欄位
        primary_count = 0      # 第一級產業
        secondary_count = 0    # 第二級產業
        tertiary_count = 0     # 第三級產業
        
        # 第一級產業 (A, B)
        for col, count in industry_counts.items():
            if col in primary_industries:
                primary_count += count
        
        # 第二級產業 (C, D, E, F)
        for col, count in industry_counts.items():
            if col in secondary_industries:
                secondary_count += count
        
        # 第三級產業 (G-S) - 確保不包含總計欄位
        for col, count in industry_counts.items():
            if col in tertiary_industries:
                tertiary_count += count
        
        # 移除 secondary_tertiary_ratio 計算
        # secondary_tertiary_ratio = secondary_count / tertiary_count if tertiary_count > 0 else 0
        
        commercial_features.append({
            '區域別': district_name,
            'total_companies': total_count,
            'total_capital': total_capital,
            'hhi_index': hhi,
            'tertiary_industry_count': tertiary_count
        })
    
    df_commercial = pd.DataFrame(commercial_features)
    print(f"✅ 創建商業特徵完成，包含 {len(df_commercial)} 個行政區")
    print(f"  📊 使用標準三級產業分類")
    print(f"  🗑️ 已移除 secondary_tertiary_ratio 特徵")
    print(f"  ⚠️  注意: 排除 '各行政區合計家數及資本額' 欄位避免重複計算")
    print(f"  ⚠️  使用完整欄位名稱進行產業分類")
    return df_commercial

def create_income_features(raw_income_data):
    """從原始所得數據創建特徵"""
    print("🔧 創建所得特徵...")
    
    if raw_income_data.empty:
        print("⚠️  原始所得數據為空，跳過特徵創建")
        return pd.DataFrame()
    
    # 轉換中位數為元（原本是千元單位）
    raw_income_data['median_income_yuan'] = raw_income_data['中位數'] * 1000
    
    # 按區域聚合，計算中位數所得
    def weighted_median(group):
        """計算加權中位數（修正版）"""
        total_households = group['納稅單位(戶)'].sum()
        if total_households == 0:
            print(f"      ⚠️  該區戶數總和為0，返回NaN")
            return np.nan  # 更改：戶數為0時返回NaN而非0
        
        # 按收入水準排序
        sorted_group = group.sort_values('median_income_yuan')
        
        # 計算累積戶數比例
        sorted_group = sorted_group.copy()
        sorted_group['cumulative_households'] = sorted_group['納稅單位(戶)'].cumsum()
        sorted_group['cumulative_ratio'] = sorted_group['cumulative_households'] / total_households
        
        # 找到中位數位置（50%）
        median_idx = sorted_group[sorted_group['cumulative_ratio'] >= 0.5].index[0]
        return sorted_group.loc[median_idx, 'median_income_yuan']
    
    income_features = []
    for district, group in raw_income_data.groupby('區域別'):
        median_income = weighted_median(group)
        total_households = group['納稅單位(戶)'].sum()
        income_features.append({
            '區域別': district,
            'median_household_income': median_income,
            'total_households': total_households
        })
    
    df_income = pd.DataFrame(income_features)
    print(f"✅ 創建所得特徵完成，包含 {len(df_income)} 個行政區")
    return df_income

def create_geo_features(raw_geo_data):
    """從原始地理數據創建特徵"""
    print("🔧 創建地理特徵...")
    
    if not GEOPANDAS_AVAILABLE or raw_geo_data.empty:
        print("⚠️  地理數據不可用，跳過特徵創建")
        return pd.DataFrame()
    
    # 改善座標系統檢測和轉換
    try:
        # 檢查是否有有效的CRS
        if raw_geo_data.crs is None:
            print("  ⚠️  地理數據缺少座標參考系統(CRS)，假設為WGS84")
            raw_geo_data = raw_geo_data.set_crs('EPSG:4326')
        
        # 轉換到適合台灣的座標系統（TWD97 TM2）
        if str(raw_geo_data.crs) != 'EPSG:3826':
            raw_geo_data = raw_geo_data.to_crs('EPSG:3826')
            print("  ✅ 座標系統轉換至 TWD97 TM2 (EPSG:3826)")
    
        # 計算面積（平方公尺轉平方公里）
        raw_geo_data['area_km2'] = raw_geo_data.geometry.area / 1e6
        
    except Exception as e:
        print(f"  ⚠️  座標系統處理出錯: {e}")
        # 嘗試直接計算面積
        try:
            raw_geo_data['area_km2'] = raw_geo_data.geometry.area / 1e6
        except Exception:
            print("  ❌ 無法計算面積，返回空特徵")
            return pd.DataFrame()
    
    # 轉換為普通DataFrame
    geo_features = []
    for idx, row in raw_geo_data.iterrows():
        geo_features.append({
            '區域別': row['區域別'],
            'area_km2': row['area_km2']
        })
    
    df_geo = pd.DataFrame(geo_features)
    print(f"✅ 創建地理特徵完成，包含 {len(df_geo)} 個行政區")
    return df_geo

def create_infrastructure_features(raw_infrastructure_data):
    """從原始公共建設數據創建特徵"""
    print("🔧 創建工廠特徵...")
    
    if raw_infrastructure_data.empty:
        print("⚠️  原始公共建設數據為空，跳過特徵創建")
        return pd.DataFrame()
    
    # 動態搜尋數據起始行 - 改善檢測邏輯
    taoyuan_districts = [name + '區' for name in TAOYUAN_DISTRICTS]
    start_row = None
    
    for i in range(min(50, len(raw_infrastructure_data))):  # 限制搜尋範圍避免誤判
        if raw_infrastructure_data.iloc[i, 0] is not None:
            cell_value = str(raw_infrastructure_data.iloc[i, 0]).strip()
            
            # 更嚴格的行政區檢測條件
            contains_district = any(district in cell_value for district in taoyuan_districts)
            
            # 檢查第二欄是否為有效數字（非0且非空）
            try:
                second_col_value = pd.to_numeric(raw_infrastructure_data.iloc[i, 1], errors='coerce')
                is_valid_number = (second_col_value is not None and 
                                 not pd.isna(second_col_value) and 
                                 second_col_value >= 0)
            except:
                is_valid_number = False
            
            # 避免標題行：檢查是否包含明顯的標題關鍵字
            title_keywords = ['合計', '總計', '小計', '項目', '說明', '備註']
            is_title_row = any(keyword in cell_value for keyword in title_keywords)
            
            # 確保行政區名稱的完整性（避免部分匹配）
            exact_district_match = cell_value in taoyuan_districts or any(
                cell_value.endswith(district) for district in taoyuan_districts
            )
            
            if exact_district_match and is_valid_number and not is_title_row:
                start_row = i
                break
    
    if start_row is None:
        print("❌ 無法找到包含行政區數據的起始行")
        return pd.DataFrame()
    
    # 提取13個行政區資料，只要區域別和總計欄位
    district_data = raw_infrastructure_data.iloc[start_row:start_row+13, [0, 1]].copy()
    district_data.columns = ['區域別', 'factory_count']
    
    # 清理區域名稱
    district_data['區域別'] = district_data['區域別'].astype(str).str.strip()
    district_data['區域別'] = district_data['區域別'].str.replace('\u3000', '').str.replace('　', '')
    district_data['區域別'] = district_data['區域別'].str.extract(r'([^A-Za-z\s]+)')[0]
    district_data['區域別'] = district_data['區域別'].str.strip()
    
    # 篩選有效的行政區
    valid_district_names = [name if name.endswith('區') else name + '區' for name in TAOYUAN_DISTRICTS]
    district_data = district_data[district_data['區域別'].isin(valid_district_names)].copy()
    
    # 清理數值數據
    district_data['factory_count'] = clean_numeric_data(district_data['factory_count'])
    
    print(f"✅ 創建工廠特徵完成，包含 {len(district_data)} 個行政區")
    return district_data

def create_health_features(raw_health_data):
    """從原始醫療衛生數據創建特徵"""
    print("🔧 創建醫療衛生特徵...")
    
    if not raw_health_data:
        print("⚠️  原始醫療衛生數據為空，跳過特徵創建")
        return pd.DataFrame()
    
    taoyuan_districts = [name + '區' for name in TAOYUAN_DISTRICTS]
    health_features = []
    
    # 處理表 9-1：醫事人員總計
    medical_personnel_data = None
    if '9-1' in raw_health_data:
        df_91 = raw_health_data['9-1']
        personnel_rows = []
        for i in range(len(df_91)):
            if df_91.iloc[i, 0] is not None:
                cell_value = str(df_91.iloc[i, 0]).strip()
                if any(district in cell_value for district in taoyuan_districts):
                    personnel_rows.append(i)
        
        if personnel_rows:
            medical_personnel_data = df_91.iloc[personnel_rows, [0, 1]].copy()
            medical_personnel_data.columns = ['區域別', 'medical_personnel_total']
    
    # 處理表 9-2：醫療院所總數和總病床數
    medical_facility_data = None
    if '9-2' in raw_health_data:
        df_92 = raw_health_data['9-2']
        facility_rows = []
        for i in range(len(df_92)):
            if df_92.iloc[i, 0] is not None:
                cell_value = str(df_92.iloc[i, 0]).strip()
                if any(district in cell_value for district in taoyuan_districts):
                    facility_rows.append(i)
        
        if facility_rows:
            medical_facility_data = df_92.iloc[facility_rows, [0, 1, 4]].copy()
            medical_facility_data.columns = ['區域別', 'medical_facilities_total', 'total_beds']
    
    # 合併醫療數據
    if medical_personnel_data is not None and medical_facility_data is not None:
        # 清理區域名稱
        for df in [medical_personnel_data, medical_facility_data]:
            df['區域別'] = df['區域別'].astype(str).str.strip()
            df['區域別'] = df['區域別'].str.extract(r'([^A-Za-z\s]+)')[0]
            df['區域別'] = df['區域別'].str.replace('   ', '').str.strip()
        
        # 合併兩個表
        combined_health_data = pd.merge(medical_personnel_data, medical_facility_data, on='區域別', how='outer')
        
        # 清理數值數據
        numeric_cols = ['medical_personnel_total', 'medical_facilities_total', 'total_beds']
        for col in numeric_cols:
            if col in combined_health_data.columns:
                combined_health_data[col] = clean_numeric_data(combined_health_data[col])
        
        print(f"✅ 創建醫療衛生特徵完成，包含 {len(combined_health_data)} 個行政區")
        return combined_health_data
    
    print("❌ 無法創建醫療衛生特徵")
    return pd.DataFrame()

def standardize_column_names(df, source_name):
    """統一欄名命名規格：來源_指標"""
    new_columns = {}
    for col in df.columns:
        if col == '區域別':
            new_columns[col] = col  # 保持區域別不變
        else:
            # 清理欄位名稱中的特殊字符和空格
            clean_col = str(col).strip().replace(' ', '').replace('　', '').replace('\u3000', '')
            new_columns[col] = f"{source_name}_{clean_col}"
    
    df_renamed = df.rename(columns=new_columns)
    print(f"  標準化 {source_name} 欄位名稱，共 {len(new_columns)-1} 個特徵")
    return df_renamed

def ensure_numeric_types(df, exclude_cols=['區域別']):
    """確保所有數值欄位都轉換為適當的數值類型"""
    for col in df.columns:
        if col not in exclude_cols:
            # 轉換為數值，無法轉換的設為0
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
            
            # 如果是整數，轉為int64，否則保持float64
            if df[col].dtype == 'float64' and (df[col] % 1 == 0).all():
                df[col] = df[col].astype('int64')
            else:
                df[col] = df[col].astype('float64')
    
    return df

def load_raw_data():
    """載入所有原始數據"""
    print("📂 載入原始數據文件...")
    
    raw_data = {}
    
    # 載入各類原始數據
    if os.path.exists('temp_data/raw_population.pkl'):
        raw_data['population'] = pd.read_pickle('temp_data/raw_population.pkl')
        print("  ✅ 載入人口數據")
    
    if os.path.exists('temp_data/raw_commercial.pkl'):
        raw_data['commercial'] = pd.read_pickle('temp_data/raw_commercial.pkl')
        print("  ✅ 載入商業數據")
    
    if os.path.exists('temp_data/raw_income.pkl'):
        raw_data['income'] = pd.read_pickle('temp_data/raw_income.pkl')
        print("  ✅ 載入所得數據")
    
    if os.path.exists('temp_data/raw_geo.pkl'):
        raw_data['geo'] = pd.read_pickle('temp_data/raw_geo.pkl')
        print("  ✅ 載入地理數據")
    
    if os.path.exists('temp_data/raw_infrastructure.pkl'):
        raw_data['infrastructure'] = pd.read_pickle('temp_data/raw_infrastructure.pkl')
        print("  ✅ 載入公共建設數據")
    
    # 載入醫療衛生數據
    health_data = {}
    for sheet in ['9-1', '9-2']:
        file_path = f'temp_data/raw_health_{sheet}.pkl'
        if os.path.exists(file_path):
            health_data[sheet] = pd.read_pickle(file_path)
    
    if health_data:
        raw_data['health'] = health_data
        print("  ✅ 載入醫療衛生數據")
    
    return raw_data

def create_all_features():
    """執行完整的特徵工程流程"""
    print("="*60)
    print("🚀 STEP2: 特徵工程")
    print("📋 功能: 從原始數據創建、標準化、合併特徵")
    print("="*60)
    
    # 檢查是否有原始數據
    if not os.path.exists('temp_data'):
        print("❌ 找不到 temp_data 目錄")
        print("📌 請先執行 step1_raw_data_loader.py")
        return
    
    # 載入原始數據
    raw_data = load_raw_data()
    
    if not raw_data:
        print("❌ 未載入任何原始數據")
        return
    
    # 創建各類特徵
    feature_datasets = {}
    
    if 'population' in raw_data:
        feature_datasets['人口'] = create_population_features(raw_data['population'])
    
    if 'commercial' in raw_data:
        feature_datasets['商業'] = create_commercial_features(raw_data['commercial'])
    
    if 'income' in raw_data:
        feature_datasets['所得'] = create_income_features(raw_data['income'])
    
    if 'geo' in raw_data:
        feature_datasets['地理'] = create_geo_features(raw_data['geo'])
    
    if 'infrastructure' in raw_data:
        feature_datasets['工廠'] = create_infrastructure_features(raw_data['infrastructure'])
    
    if 'health' in raw_data:
        feature_datasets['醫療'] = create_health_features(raw_data['health'])
    
    # 標準化欄位名稱
    print("\n🏷️  標準化欄位名稱...")
    for source_name, df in feature_datasets.items():
        if not df.empty:
            feature_datasets[source_name] = standardize_column_names(df, source_name)
    
    # 合併所有特徵
    print("\n🔗 合併所有特徵...")
    valid_datasets = {name: df for name, df in feature_datasets.items() if not df.empty}
    
    if not valid_datasets:
        print("❌ 沒有有效的特徵數據集")
        return
    
    # 從第一個數據集開始合併
    first_dataset_name = list(valid_datasets.keys())[0]
    merged_df = valid_datasets[first_dataset_name].copy()
    merge_info = {first_dataset_name: f"{merged_df.shape[1]-1} 個特徵"}
    
    print(f"  基準數據集: {first_dataset_name}")
    
    # 逐個合併其他數據集
    for name, df in list(valid_datasets.items())[1:]:
        print(f"  合併 {name} 特徵...")
        before_cols = merged_df.shape[1]
        merged_df = pd.merge(merged_df, df, on='區域別', how='left')
        after_cols = merged_df.shape[1]
        added_cols = after_cols - before_cols
        merge_info[name] = f"{added_cols} 個特徵"
        print(f"    新增 {added_cols} 個特徵")
    
    # 確保數值類型
    print("\n🔢 確保數值類型...")
    merged_df = ensure_numeric_types(merged_df)
    
    # 生成metadata
    print("\n📝 生成metadata...")
    metadata = {
        "dataset_info": {
            "name": "桃園市行政區發展特徵資料",
            "description": "桃園市13個行政區的人口、商業、所得、地理、公共建設、醫療等特徵資料",
            "year": 110,
            "last_updated": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "total_districts": len(merged_df),
            "total_features": merged_df.shape[1] - 1
        },
        "districts": sorted(merged_df['區域別'].tolist()),
        "data_sources": merge_info,
        "features": {}
    }
    
    # 為每個特徵生成描述
    for col in merged_df.columns:
        if col != '區域別':
            if '_' in col:
                source, indicator = col.split('_', 1)
                metadata["features"][col] = {
                    "source": source,
                    "indicator": indicator,
                    "data_type": str(merged_df[col].dtype),
                    "min_value": float(merged_df[col].min()),
                    "max_value": float(merged_df[col].max()),
                    "mean_value": float(merged_df[col].mean()),
                    "missing_count": int(merged_df[col].isnull().sum())
                }
    
    # 保存結果
    print("\n💾 保存結果...")
    os.makedirs('output', exist_ok=True)
    
    # 保存特徵CSV
    csv_path = 'output/taoyuan_features_numeric.csv'
    merged_df.to_csv(csv_path, index=False, encoding='utf-8')
    print(f"  特徵資料已保存至: {csv_path}")
    
    # 保存metadata JSON
    json_path = 'output/taoyuan_meta.json'
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)
    print(f"  Metadata已保存至: {json_path}")
    
    # 輸出最終統計
    print(f"\n✅ STEP2 完成！")
    print(f"📊 最終統計:")
    print(f"  行政區數量: {len(merged_df)}")
    print(f"  特徵數量: {merged_df.shape[1]-1}")
    print(f"  資料來源: {len(valid_datasets)} 個")
    print(f"  缺失值: {merged_df.isnull().sum().sum()} 個")
    
    # 檢查數據完整性
    if merged_df.isnull().sum().sum() == 0:
        print("  ✅ 數據完整，無缺失值")
    else:
        print("  ⚠️  存在缺失值，請檢查")

    # 九、缺失值處理策略
    print("\n🔧 九、缺失值處理策略")
    
    missing_count = merged_df.isnull().sum().sum()
    if missing_count > 0:
        print(f"  發現 {missing_count} 個缺失值")
        
        # 檢查econ_to_med_ratio的缺失值
        if DERIVED_ECON_MED_RATIO in merged_df.columns and merged_df[DERIVED_ECON_MED_RATIO].isnull().any():
            missing_districts = merged_df[merged_df[DERIVED_ECON_MED_RATIO].isnull()]['區域別'].values
            print(f"  📍 {DERIVED_ECON_MED_RATIO} 缺失值行政區: {', '.join(missing_districts)}")
            
            # 提供多種處理策略
            print("  💡 處理策略選項:")
            print("     1. 設為極大值 (表示醫療資源稀缺)")
            print("     2. 設為經濟指數值 (保守估計)")
            print("     3. 使用其他行政區中位數")
            print("     4. 保留NaN (分析時排除)")
            
            # 策略1: 設為極大值 (推薦用於該場景)
            max_ratio = merged_df[DERIVED_ECON_MED_RATIO].max()
            merged_df[DERIVED_ECON_MED_RATIO] = merged_df[DERIVED_ECON_MED_RATIO].fillna(max_ratio * 2)
            print(f"  ✅ 採用策略1: 設為 {max_ratio * 2:.1f} (表示醫療資源極度稀缺)")
            
        print(f"  ✅ 處理後缺失值數量: {merged_df.isnull().sum().sum()}")
    else:
        print("  ✅ 無缺失值")
    
    # 使用數值特徵（排除區域別）
    X = merged_df.drop(columns=['區域別']).fillna(0)

def enhanced_feature_engineering():
    """增強版特徵工程：簡化版"""
    print("="*80)
    print("🔬 開始增強版特徵工程")
    print("="*80)
    
    # 載入基礎特徵
    df = pd.read_csv('output/taoyuan_features_numeric.csv')
    print(f"📂 載入基礎特徵數據: {df.shape[0]} 行 {df.shape[1]} 列")
    
    # 確認行政區數量
    district_count = len(df)
    assert district_count == 13, f"應有13個行政區，實際有 {district_count} 個"
    print(f"  ✅ 確認13個行政區: {', '.join(df['區域別'].tolist())}")
    
    # 創建密度與人均特徵
    print("\n🏗️ 創建特徵...")
    
    if '地理_area_km2' in df.columns:
        # 每千人特徵
        if HEALTH_BEDS in df.columns and POP_TOTAL in df.columns:
            df[DERIVED_BEDS_PER_1K] = df[HEALTH_BEDS] / (df[POP_TOTAL] / 1000)
            print("  ✅ 每千人病床數")
        
        if HEALTH_PERSON in df.columns and POP_TOTAL in df.columns:
            df[DERIVED_STAFF_PER_1K] = df[HEALTH_PERSON] / (df[POP_TOTAL] / 1000)
            print("  ✅ 每千人醫療人員數")
        
        # 人均特徵
        if COM_TOTAL_CAP in df.columns and INCOME_HOUSEHOLDS in df.columns:
            df[DERIVED_CAP_PER_HOUSEHOLD] = df[COM_TOTAL_CAP] / df[INCOME_HOUSEHOLDS]
            print("  ✅ 資本額/戶")
        
        # 勞動力相關特徵
        if POP_WORK_RATIO in df.columns and FACTORY_COUNT in df.columns and POP_TOTAL in df.columns:
            working_age_population = df[POP_TOTAL] * (df[POP_WORK_RATIO] / 100)
            df[DERIVED_FACTORIES_PER_1K_WORK] = df[FACTORY_COUNT] / (working_age_population / 1000)
            print("  ✅ 每千勞動人口工廠數")
        
        # 產業結構特徵
        if COM_TERTIARY in df.columns and COM_TOTAL_CNT in df.columns:
            df[DERIVED_TERTIARY_RATIO] = df[COM_TERTIARY] / df[COM_TOTAL_CNT] * 100
            print("  ✅ 第三級產業占比")
        
        # 醫療密度
        if HEALTH_FACILITIES in df.columns:
            df[DERIVED_MED_DENSITY] = df[HEALTH_FACILITIES] / df[GEO_AREA]
            print("  ✅ 醫療密度")
    
    # 創建綜合指數
    print("\n🎯 創建綜合指數...")
    
    from sklearn.preprocessing import StandardScaler
    
    # 經濟發展指數
    economic_cols = [col for col in [COM_TOTAL_CNT, COM_TOTAL_CAP] if col in df.columns]
    if len(economic_cols) >= 2:
        economic_data = df[economic_cols].fillna(0)
        econ_scaler = StandardScaler()
        economic_scaled = econ_scaler.fit_transform(economic_data)
        df[DERIVED_ECON_INDEX] = np.mean(economic_scaled, axis=1)
        print(f"  ✅ 經濟發展指數")
    
    # 🆕 醫療服務子指標 - 方案A: 保持原始數據到STEP3
    medical_cols = [col for col in [DERIVED_BEDS_PER_1K, DERIVED_STAFF_PER_1K, DERIVED_MED_DENSITY] if col in df.columns]
    if len(medical_cols) >= 3:
        print(f"  📊 醫療子指標數量: {len(medical_cols)}")
        
        # 🔄 方案A: 保留原始醫療子指標，不進行標準化
        # 重命名為更簡潔的名稱，供STEP3使用
        df['medical_beds_per_1k'] = df[DERIVED_BEDS_PER_1K]
        df['medical_staff_per_1k'] = df[DERIVED_STAFF_PER_1K] 
        df['medical_facility_density'] = df[DERIVED_MED_DENSITY]
        
        print(f"  ✅ 保留醫療子指標原始數據 (方案A統一Z-score策略)")
        print(f"    醫療床位密度範圍: {df['medical_beds_per_1k'].min():.2f} - {df['medical_beds_per_1k'].max():.2f}")
        print(f"    醫療人員密度範圍: {df['medical_staff_per_1k'].min():.2f} - {df['medical_staff_per_1k'].max():.2f}")
        print(f"    醫療設施密度範圍: {df['medical_facility_density'].min():.4f} - {df['medical_facility_density'].max():.4f}")
        print(f"    ⚡ 這些指標將在STEP3進行統一Z-score標準化")
    else:
        print(f"  ⚠️  醫療子指標不足，跳過醫療指標保留")
    
    # 處理偏態分布
    print("\n📊 處理偏態分布...")
    
    numeric_cols_for_skew = df.select_dtypes(include=[np.number]).columns.tolist()
    predefined_skewed = [
        GEO_AREA, COM_TOTAL_CAP, COM_TOTAL_CNT, 
        HEALTH_FACILITIES, HEALTH_BEDS, HEALTH_PERSON, 
        FACTORY_COUNT, COM_TERTIARY, COM_HHI, INCOME_HOUSEHOLDS, POP_TOTAL
    ]
    
    existing_skewed = [col for col in predefined_skewed if col in df.columns]
    
    for feature in existing_skewed:
        if feature in df.columns:
            df[feature] = np.log1p(df[feature])
    
    print(f"  ✅ 處理了 {len(existing_skewed)} 個偏態特徵")
    
    # 創建新的衍生特徵
    if COM_TOTAL_CAP in df.columns and FACTORY_COUNT in df.columns:
        df[DERIVED_AVG_FACTORY_CAP] = np.where(
            df[FACTORY_COUNT] == 0,
            np.nan,
            df[COM_TOTAL_CAP] / df[FACTORY_COUNT]
        )
        print("  ✅ 平均工廠資本額")
    
    # 刪除不必要的特徵
    print("\n🗑️ 刪除不必要的特徵...")
    
    features_to_drop = [
        COM_TOTAL_CNT, COM_TOTAL_CAP, COM_TERTIARY, INCOME_HOUSEHOLDS, 
        GEO_AREA, POP_TOTAL, HEALTH_PERSON, HEALTH_FACILITIES, HEALTH_BEDS, 
        DERIVED_ECON_INDEX, DERIVED_CAP_PER_HOUSEHOLD, DERIVED_BEDS_PER_1K, 
        DERIVED_STAFF_PER_1K, DERIVED_MED_DENSITY
    ]
    # 🔄 方案A: 保留醫療子指標，不刪除
    # medical_beds_per_1k, medical_staff_per_1k, medical_facility_density 將保留
    
    existing_drop_features = [col for col in features_to_drop if col in df.columns]
    df = df.drop(columns=existing_drop_features)
    print(f"  ✅ 刪除了 {len(existing_drop_features)} 個特徵")
    
    # 數據檢查
    print("\n📏 數據檢查...")
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    print(f"  總特徵數: {len(numeric_cols)}")
    print(f"  缺失值: {df.isnull().sum().sum()}")
    
    # 保存結果
    df.to_csv('output/taoyuan_features_enhanced.csv', index=False, encoding='utf-8-sig')
    print(f"\n💾 保存增強版特徵數據: output/taoyuan_features_enhanced.csv")
    
    # 生成元數據
    metadata = {
        'total_features': len(numeric_cols),
        'total_samples': df.shape[0],
        'missing_values': int(df.isnull().sum().sum()),
        'kept_features': [col for col in df.columns if col != '區域別']
    }
    
    import json
    with open('output/taoyuan_enhanced_meta.json', 'w', encoding='utf-8') as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)
    
    print(f"✅ 增強版特徵工程完成！")
    print(f"📊 最終特徵數量: {len(numeric_cols)}")
    print("="*80)
    
    return df

# 修改主程序，讓增強版特徵工程成為預設行為
def main():
    """主程序 - 預設執行增強版特徵工程"""
    import sys
    
    # 先執行基礎特徵工程（如果還沒有）
    if not os.path.exists('output/taoyuan_features_numeric.csv'):
        print("📌 首次執行，先進行基礎特徵工程...")
    create_all_features() 
    
    # 執行增強版特徵工程
    enhanced_feature_engineering()

if __name__ == "__main__":
    main() 