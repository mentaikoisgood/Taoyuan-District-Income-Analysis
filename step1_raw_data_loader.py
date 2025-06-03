import pandas as pd
import numpy as np
import os
import traceback
from datetime import datetime
try:
    import geopandas as gpd
    GEOPANDAS_AVAILABLE = True
except ImportError:
    GEOPANDAS_AVAILABLE = False
    print("警告: 無法導入 geopandas，地理資料功能將受限")

# 數據文件路徑
POPULATION_DATA_PATH = 'data/110年桃園市人口數按性別及年齡分.xlsx'
INCOME_DATA_PATH = 'data/110_income_by_district.csv'
GEOJSON_PATH = 'data/taoyuan_districts.geojson'
COMMERCIAL_CSV_PATH = 'data/110年12月底商業行業別及行政區域家數.csv'
PUBLIC_INFRASTRUCTURE_PATH = 'data/110桃園市公共建設.xlsx'
HEALTH_DATA_PATH = 'data/110桃園市衛生.xlsx'

# 桃園市13個行政區
TAOYUAN_DISTRICTS = [
    '中壢', '八德', '大園', '大溪', '平鎮', '復興', 
    '桃園', '新屋', '楊梅', '龍潭', '龜山', '蘆竹', '觀音'
]

def clean_numeric_data(series):
    """
    清理數值數據的通用函數
    """
    return pd.to_numeric(
        series.astype(str)
        .str.replace(',', '')
        .str.replace(' ', '')
        .str.replace('-', '0')
        .str.replace('－', '0'), 
        errors='coerce'
    ).fillna(0)

def load_raw_population_data():
    """載入原始人口資料 - 只進行基本清理，不計算特徵"""
    print("📂 載入原始人口資料...")
    try:
        if not os.path.exists(POPULATION_DATA_PATH):
            print(f"❌ 人口資料檔案未找到: {POPULATION_DATA_PATH}")
            return pd.DataFrame()
        
        # 讀取Excel文件
        df_raw = pd.read_excel(POPULATION_DATA_PATH, header=3)
        df_raw.columns = df_raw.columns.str.strip()
        
        # 篩選桃園市13個行政區的男女數據
        taoyuan_district_names = [name + '區' for name in TAOYUAN_DISTRICTS]
        df_taoyuan = df_raw[
            df_raw['區域別'].isin(taoyuan_district_names) & 
            df_raw['性別'].isin(['男', '女'])
        ].copy()
        
        if df_taoyuan.empty:
            print("❌ 未找到桃園市行政區人口數據")
            return pd.DataFrame()
        
        print(f"✅ 成功載入 {len(df_taoyuan['區域別'].unique())} 個行政區的人口數據")
        return df_taoyuan
        
    except Exception as e:
        print(f"❌ 載入人口資料時發生錯誤: {e}")
        return pd.DataFrame()

def load_raw_commercial_data():
    """載入原始商業資料 - 只進行基本清理，不計算指標"""
    print("📂 載入原始商業資料...")
    try:
        if not os.path.exists(COMMERCIAL_CSV_PATH):
            print(f"❌ 商業資料檔案未找到: {COMMERCIAL_CSV_PATH}")
            return pd.DataFrame()
        
        # 嘗試不同編碼讀取CSV
        encodings = ['utf-8', 'big5', 'cp950', 'gbk', 'latin1']
        df_commercial = None
        
        for encoding in encodings:
            try:
                df_commercial = pd.read_csv(COMMERCIAL_CSV_PATH, encoding=encoding)
                print(f"  使用 {encoding} 編碼成功讀取")
                break
            except UnicodeDecodeError:
                continue
        
        if df_commercial is None:
            print("❌ 無法讀取商業資料CSV文件")
            return pd.DataFrame()
        
        # 重命名第一個欄位
        if '行政區別行業別' in df_commercial.columns:
            df_commercial = df_commercial.rename(columns={'行政區別行業別': '區域別'})
        
        print(f"✅ 成功載入商業資料，形狀: {df_commercial.shape}")
        return df_commercial
        
    except Exception as e:
        print(f"❌ 載入商業資料時發生錯誤: {e}")
        return pd.DataFrame()

def load_raw_income_data():
    """載入原始所得資料 - 只進行基本清理"""
    print("📂 載入原始所得資料...")
    try:
        if not os.path.exists(INCOME_DATA_PATH):
            print(f"❌ 所得資料檔案未找到: {INCOME_DATA_PATH}")
            return pd.DataFrame()
        
        df_income = pd.read_csv(INCOME_DATA_PATH, encoding='utf-8')
        df_income.columns = [col.lstrip('\ufeff') for col in df_income.columns]
        
        # 篩選桃園市資料
        taoyuan_mask = df_income['縣市別'].str.contains('桃園市', na=False)
        df_taoyuan_income = df_income[taoyuan_mask].copy()
        
        # 提取區域資訊
        df_taoyuan_income['區域別'] = df_taoyuan_income['縣市別'].str.extract(r'桃園市(.+?)區')[0] + '區'
        df_taoyuan_income = df_taoyuan_income.dropna(subset=['區域別'])
        
        print(f"✅ 成功載入所得資料，形狀: {df_taoyuan_income.shape}")
        return df_taoyuan_income
        
    except Exception as e:
        print(f"❌ 載入所得資料時發生錯誤: {e}")
        return pd.DataFrame()

def load_raw_geo_data():
    """載入原始地理資料 - 只進行基本清理"""
    print("📂 載入原始地理資料...")
    try:
        if not GEOPANDAS_AVAILABLE:
            print("⚠️  geopandas未安裝，跳過地理資料")
            return pd.DataFrame()
            
        if not os.path.exists(GEOJSON_PATH):
            print(f"❌ GeoJSON檔案未找到: {GEOJSON_PATH}")
            return pd.DataFrame()
        
        gdf = gpd.read_file(GEOJSON_PATH)
        
        # 尋找名稱欄位
        name_column = None
        for col in ['名稱', 'name', 'NAME', '區域', '區域別']:
            if col in gdf.columns:
                name_column = col
                break
        
        if name_column is None:
            print("⚠️  未找到名稱欄位，使用索引作為區域別")
            gdf['區域別'] = f"未知區域_{gdf.index}"
        else:
            gdf['區域別'] = gdf[name_column].astype(str).str.replace('臺', '台').str.strip()
            gdf['區域別'] = gdf['區域別'].str.replace('桃園市', '').str.strip()
            # 確保區域名稱以'區'結尾
            gdf['區域別'] = gdf['區域別'].apply(
                lambda x: x + '區' if isinstance(x, str) and x in TAOYUAN_DISTRICTS and not x.endswith('區') else x
            )
        
        print(f"✅ 成功載入地理資料，形狀: {gdf.shape}")
        return gdf
        
    except Exception as e:
        print(f"❌ 載入地理資料時發生錯誤: {e}")
        return pd.DataFrame()

def load_raw_infrastructure_data():
    """載入原始公共建設資料 - 只進行基本清理"""
    print("📂 載入原始公共建設資料...")
    try:
        if not os.path.exists(PUBLIC_INFRASTRUCTURE_PATH):
            print(f"❌ 公共建設資料檔案未找到: {PUBLIC_INFRASTRUCTURE_PATH}")
            return pd.DataFrame()
        
        df_infrastructure = pd.read_excel(PUBLIC_INFRASTRUCTURE_PATH, header=None)
        
        print(f"✅ 成功載入公共建設資料，形狀: {df_infrastructure.shape}")
        return df_infrastructure
        
    except Exception as e:
        print(f"❌ 載入公共建設資料時發生錯誤: {e}")
        return pd.DataFrame()

def load_raw_health_data():
    """載入原始醫療衛生資料 - 只進行基本清理"""
    print("📂 載入原始醫療衛生資料...")
    try:
        if not os.path.exists(HEALTH_DATA_PATH):
            print(f"❌ 醫療衛生資料檔案未找到: {HEALTH_DATA_PATH}")
            return pd.DataFrame()
        
        # 獲取所有工作表
        excel_file = pd.ExcelFile(HEALTH_DATA_PATH)
        health_data = {}
        
        # 載入主要工作表
        for sheet_name in ['9-1', '9-2']:
            if sheet_name in excel_file.sheet_names:
                health_data[sheet_name] = pd.read_excel(HEALTH_DATA_PATH, sheet_name=sheet_name, header=None)
                print(f"  載入工作表 {sheet_name}: {health_data[sheet_name].shape}")
        
        print(f"✅ 成功載入醫療衛生資料，包含 {len(health_data)} 個工作表")
        return health_data
        
    except Exception as e:
        print(f"❌ 載入醫療衛生資料時發生錯誤: {e}")
        return {}

def save_raw_data():
    """載入所有原始數據並保存到臨時文件"""
    print("="*60)
    print("🚀 STEP1: 原始數據載入器")
    print("📋 功能: 載入、基本清理、保存原始數據")
    print("="*60)
    
    # 確保輸出目錄存在
    os.makedirs('temp_data', exist_ok=True)
    
    raw_data = {}
    
    # 載入各類原始數據
    raw_data['population'] = load_raw_population_data()
    raw_data['commercial'] = load_raw_commercial_data()
    raw_data['income'] = load_raw_income_data()
    raw_data['geo'] = load_raw_geo_data()
    raw_data['infrastructure'] = load_raw_infrastructure_data()
    raw_data['health'] = load_raw_health_data()
    
    # 保存非空的數據到pickle文件
    saved_count = 0
    for data_type, data in raw_data.items():
        if isinstance(data, dict):
            # 健康數據是字典格式
            if data:
                for sheet_name, sheet_data in data.items():
                    file_path = f'temp_data/raw_{data_type}_{sheet_name}.pkl'
                    sheet_data.to_pickle(file_path)
                    print(f"💾 保存 {data_type}_{sheet_name} 到 {file_path}")
                    saved_count += 1
        elif not data.empty:
            file_path = f'temp_data/raw_{data_type}.pkl'
            data.to_pickle(file_path)
            print(f"💾 保存 {data_type} 到 {file_path}")
            saved_count += 1
        else:
            print(f"⚠️  跳過空的 {data_type} 數據")
    
    print(f"\n✅ STEP1 完成！成功保存 {saved_count} 個原始數據文件到 temp_data/ 目錄")
    print("📌 下一步: 執行 step2_feature_engineering.py 進行特徵工程")

if __name__ == "__main__":
    save_raw_data() 