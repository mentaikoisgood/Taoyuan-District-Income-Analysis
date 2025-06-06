"""
STEP 5 - 3級Jenks分級網頁儀表板數據生成

生成用於網頁DASHBOARD的JSON數據，包含：
1. 分級結果統計
2. 區域詳細信息
3. 特徵雷達圖數據
4. 排名和分數數據
5. 可視化圖表數據

輸出：docs/data/ 目錄下的JSON文件供網頁使用
"""

import pandas as pd
import numpy as np
import json
import os
from sklearn.preprocessing import StandardScaler
import warnings
import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import folium
from folium import plugins
import shapely.geometry
from shapely.geometry import mapping
from scipy import stats
import matplotlib.font_manager as fm
import matplotlib.patheffects as path_effects

warnings.filterwarnings('ignore')

# 設定中文字體
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

HAS_GEO = True

def load_classification_results():
    """載入3級Jenks分級結果"""
    print("📂 載入3級Jenks分級結果...")
    
    try:
        results_df = pd.read_csv('output/3_level_jenks_results.csv')
        
        with open('output/3_level_jenks_config.json', 'r', encoding='utf-8') as f:
            config = json.load(f)
        
        df = pd.read_csv('output/taoyuan_features_enhanced.csv')
        
        print(f"✅ 成功載入 {len(results_df)} 個行政區的分級結果")
        return results_df, config, df
        
    except FileNotFoundError as e:
        print(f"❌ 無法載入分級結果: {e}")
        return None, None, None

def prepare_dashboard_data(results_df, config, df):
    """準備儀表板數據"""
    print("\n🎨 準備儀表板數據...")
    
    # 基本統計 (scores are now 0-10 from results_df)
    level_stats = {
        'high_potential': {
            'count': int(sum(results_df['3級Jenks分級'] == '高潛力')),
            'districts': results_df[results_df['3級Jenks分級'] == '高潛力']['區域別'].tolist(),
            'scores': [round(s, 1) for s in results_df[results_df['3級Jenks分級'] == '高潛力']['綜合分數'].tolist()],
            'avg_score': float(round(results_df[results_df['3級Jenks分級'] == '高潛力']['綜合分數'].mean(), 1))
        },
        'medium_potential': {
            'count': int(sum(results_df['3級Jenks分級'] == '中潛力')),
            'districts': results_df[results_df['3級Jenks分級'] == '中潛力']['區域別'].tolist(),
            'scores': [round(s, 1) for s in results_df[results_df['3級Jenks分級'] == '中潛力']['綜合分數'].tolist()],
            'avg_score': float(round(results_df[results_df['3級Jenks分級'] == '中潛力']['綜合分數'].mean(), 1))
        },
        'low_potential': {
            'count': int(sum(results_df['3級Jenks分級'] == '低潛力')),
            'districts': results_df[results_df['3級Jenks分級'] == '低潛力']['區域別'].tolist(),
            'scores': [round(s, 1) for s in results_df[results_df['3級Jenks分級'] == '低潛力']['綜合分數'].tolist()],
            'avg_score': float(round(results_df[results_df['3級Jenks分級'] == '低潛力']['綜合分數'].mean(), 1))
        }
    }
    
    # 特徵名稱映射
    feature_mapping = {
        '人口_working_age_ratio': '工作年齡人口比例',
        '商業_hhi_index': '商業集中度指數',
        '所得_median_household_income': '家戶中位數所得',
        'tertiary_industry_ratio': '第三產業比例',
        'medical_index': '醫療指數'
    }
    
    # 準備區域詳細數據
    district_details = []
    for _, row in results_df.iterrows():
        district_data = {
            'name': row['區域別'],
            'rank': int(row['排名']),
            'score': float(round(row['綜合分數'], 1)),
            'level': row['3級Jenks分級'],
            'level_en': {
                '高潛力': 'high',
                '中潛力': 'medium', 
                '低潛力': 'low'
            }[row['3級Jenks分級']],
            'features': {}
        }
        
        # 添加特徵數據
        for feature, display_name in feature_mapping.items():
            if feature in row:
                district_data['features'][display_name] = float(row[feature])
        
        district_details.append(district_data)
    
    return level_stats, district_details, feature_mapping

def create_radar_chart_data(results_df, feature_mapping):
    """創建雷達圖數據"""
    print("📡 創建雷達圖數據...")
    
    # 標準化特徵數據
    feature_cols = list(feature_mapping.keys())
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(results_df[feature_cols])
    
    # 轉換為0-100範圍
    X_normalized = ((X_scaled - X_scaled.min(axis=0)) / (X_scaled.max(axis=0) - X_scaled.min(axis=0))) * 100
    
    radar_data = {}
    for i, row in results_df.iterrows():
        district_name = row['區域別']
        radar_data[district_name] = {
            'features': list(feature_mapping.values()),
            'values': [float(val) for val in X_normalized[i]],
            'level': row['3級Jenks分級'],
            'score': float(round(row['綜合分數'], 1))
        }
    
    return radar_data

def create_scatter_chart_data(results_df):
    """創建散點圖數據"""
    print("📊 創建散點圖數據...")
    
    scatter_data = []
    color_map = {
        '高潛力': '#eb7062',      # 紅色 (Red)
        '中潛力': '#f5b041',      # 橙色 (Orange)  
        '低潛力': '#5cace2',      # 藍色 (Blue)
    }
    
    for _, row in results_df.iterrows():
        scatter_data.append({
            'x': float(row['排名']),
            'y': float(round(row['綜合分數'], 1)),
            'name': row['區域別'],
            'level': row['3級Jenks分級'],
            'color': color_map[row['3級Jenks分級']]
        })
    
    return scatter_data

def create_feature_importance_data(config):
    """創建特徵重要性數據"""
    print("⚖️ 創建特徵重要性數據...")
    
    feature_weights = config['feature_properties']
    importance_data = []
    
    feature_mapping = {
        '人口_working_age_ratio': '工作年齡人口比例',
        '商業_hhi_index': '商業集中度指數',
        '所得_median_household_income': '家戶中位數所得',
        'tertiary_industry_ratio': '第三產業比例',
        'medical_index': '醫療指數'
    }
    
    for feature, props in feature_weights.items():
        importance_data.append({
            'feature': feature_mapping.get(feature, feature),
            'weight': props['weight'],
            'direction': props['direction'],
            'description': props.get('description', '')
        })
    
    # 按權重排序
    importance_data.sort(key=lambda x: x['weight'], reverse=True)
    
    return importance_data

def create_method_info(config):
    """創建方法說明信息"""
    print("📝 創建方法說明...")
    
    method_info = {
        'title': '3級Jenks自然斷點分級',
        'description': '使用Jenks自然斷點方法將桃園市13個行政區分為3個潛力等級',
        'breaks': config['breaks'],
        'total_districts': config['n_districts'],
        'score_range': [float(round(s, 1)) for s in config['score_range']],
        'features_count': len(config['feature_properties']),
        'advantages': [
            '基於數據驅動的自然分割點',
            '最大化組間差異，最小化組內差異',
            '適合小樣本分析',
            '穩定性優秀',
            '政策解釋性強'
        ],
        'steps': [
            {
                'step': 'STEP 1: 數據準備',
                'description': '整合人口、商業、所得、產業、醫療等5個關鍵指標'
            },
            {
                'step': 'STEP 2: 特徵標準化', 
                'description': '使用Z-score標準化處理所有特徵'
            },
            {
                'step': 'STEP 3: 權重計算',
                'description': '根據政策重要性設定權重 (詳細權重見選定方案)'
            },
            {
                'step': 'STEP 4: Jenks分級',
                'description': '使用自然斷點方法找到最優分割點，形成3個潛力等級'
            },
            {
                'step': 'STEP 5: 驗證分析',
                'description': '通過統計指標和質量指標驗證分級效果'
            }
        ]
    }
    
    return method_info

# Map generation functions (integrated from former step6)

def load_map_data():
    """載入地圖數據和分級結果"""
    print("📂 載入地圖數據和分級結果 (for map generation)...")
    
    try:
        # 載入GeoJSON地理數據
        gdf = gpd.read_file('data/taoyuan_districts.geojson')
        print(f"✅ 成功載入 {len(gdf)} 個行政區的地理數據")
        
        return gdf
        
    except FileNotFoundError as e:
        print(f"❌ 無法載入地理數據: {e}")
        print("請確保 taoyuan_districts.geojson 文件位於 data/ 目錄下")
        return None

def merge_geodata_with_results(gdf, results_df):
    """將地理數據與分級結果合併"""
    print("🔗 合併地理數據與分級結果...")
    
    # 修正 GeoDataFrame 中的欄位名稱不一致問題
    if '名稱' in gdf.columns and '區域別' not in gdf.columns:
        gdf = gdf.rename(columns={'名稱': '區域別'})

    # 確保 '區域別' 欄位存在
    if '區域別' not in gdf.columns or '區域別' not in results_df.columns:
        print("❌ '區域別' 欄位在其中一個數據源中不存在")
        return None
        
    merged_gdf = gdf.merge(results_df, on='區域別', how='left')
    
    if merged_gdf['3級Jenks分級'].isnull().any():
        print("⚠️ 警告: 部分行政區沒有匹配的分級結果")
    
    # 動態添加顏色列
    color_map = {
        '高潛力': '#EB7062',
        '中潛力': '#F5B041',
        '低潛力': '#5CACE2'
    }
    merged_gdf['color'] = merged_gdf['3級Jenks分級'].map(color_map).fillna('#cccccc') # 未匹配的為灰色

    print(f"✅ 成功合併 {len(merged_gdf)} 個行政區")
    return merged_gdf

def create_static_map(merged_gdf, config):
    """生成靜態PNG格式的地圖"""
    print("🗺️  生成靜態PNG地圖...")
    
    # 創建畫布
    fig, ax = plt.subplots(1, 1, figsize=(16, 12))
    ax.set_aspect('equal')
    
    # 配色方案
    colors = {
        '高潛力': '#EB7062', 
        '中潛力': '#F5B041', 
        '低潛力': '#5CACE2'
    }
    
    # 繪製地圖
    for _, row in merged_gdf.iterrows():
        geom = row.geometry
        
        # 處理 MultiPolygon 的情況
        if isinstance(geom, shapely.geometry.MultiPolygon):
            for poly in geom.geoms:
                ax.add_patch(mpatches.Polygon(
                    np.array(poly.exterior.coords),
                    facecolor=row['color'],
                    edgecolor='white',
                    linewidth=1.5,
                    alpha=0.9,
                    transform=ax.transData
                ))
        else: # 處理單一 Polygon
            ax.add_patch(mpatches.Polygon(
                np.array(geom.exterior.coords),
                facecolor=row['color'],
                edgecolor='white',
                linewidth=1.5,
                alpha=0.9,
                transform=ax.transData
            ))
        
    # 添加背景，移除坐標軸
    ax.set_facecolor('#f0f2f5')
    ax.axis('off')
    
    # 添加標籤
    for idx, row in merged_gdf.iterrows():
        centroid = row.geometry.centroid
        ax.text(centroid.x, centroid.y, row['區域別'].replace('區', ''), 
                ha='center', va='center', fontsize=14, color='white',
                bbox=dict(facecolor='black', alpha=0.4, edgecolor='none', boxstyle='round,pad=0.4'),
                path_effects=[path_effects.withStroke(linewidth=3, foreground='black')])
    
    # 添加標題和副標題
    plt.suptitle("桃園市行政區發展潛力分級地圖", fontsize=28, fontweight='bold', color='#333333', y=0.95)
    plt.title("基於多維度數據與3級Jenks自然斷點分析", fontsize=18, color='#666666', y=1.0)
    
    # 創建圖例
    legend_patches = [mpatches.Patch(color=color, label=label) for label, color in colors.items()]
    legend = plt.legend(handles=legend_patches, 
                        title="潛力等級", 
                        fontsize=14, 
                        title_fontsize=16, 
                        loc='upper right',
                        bbox_to_anchor=(0.95, 0.95),
                        fancybox=True,
                        shadow=True,
                        frameon=True,
                        framealpha=0.9,
                        facecolor='white',
                        edgecolor='#cccccc'
                       )
    legend.get_title().set_fontweight('bold')
    
    # 添加底部說明
    footer_text = "數據來源：桃園市政府開放數據 | 分析方法：Jenks自然斷點分級 (3級) | 分數範圍：0.0 - 10.0 (0-10分制)"
    fig.text(0.5, 0.05, footer_text, ha='center', va='bottom', fontsize=12, color='#888888')
    
    # 調整佈局並保存
    plt.tight_layout(rect=[0, 0.05, 1, 0.95]) # 留出空間給底部文字和標題
    
    output_path = 'output/taoyuan_potential_map.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight', pad_inches=0.1, facecolor=ax.get_facecolor())
    
    print(f"✅ 靜態地圖已保存至: {output_path}")
    return output_path

def generate_interactive_map(df, jenks_data):
    """生成互動式地圖"""
    print("🗺️ 生成互動式地圖...")
    
    if not HAS_GEO:
        print("⚠️ 跳過互動地圖生成（缺少地理套件）")
        return

    gdf_map = load_map_data() # Load GeoJSON data
    if gdf_map is None:
        print("❌ 地理數據載入失敗，無法生成互動地圖。")
        # Create a simple map if data load fails, indicating the error
        center_lat, center_lng = 24.9936, 121.3010
        m_error = folium.Map(location=[center_lat, center_lng], zoom_start=10, tiles='OpenStreetMap')
        title_html = '<h3 align="center" style="font-size:20px; color:red;"><b>桃園市行政區發展潛力地圖 (地理數據載入失敗)</b></h3>'
        m_error.get_root().html.add_child(folium.Element(title_html))
        os.makedirs('docs', exist_ok=True)
        m_error.save('docs/map_interactive.html')
        print("⚠️ 已生成一個指示錯誤的基本地圖。")
        return

    # 標準化 jenks_data 中的區域名稱欄位
    if '區域別' not in jenks_data.columns:
        if 'district' in jenks_data.columns:
            print("ℹ️ 'jenks_data' 中使用 'district' 作為區域名稱，將其重命名為 '區域別'")
            jenks_data = jenks_data.rename(columns={'district': '區域別'})
        else:
            print("❌ 'jenks_data' 中缺少 '區域別' 或 'district' 欄位，無法合併。")
            # Create a simple map if key column is missing
            center_lat, center_lng = 24.9936, 121.3010
            m_error = folium.Map(location=[center_lat, center_lng], zoom_start=10, tiles='OpenStreetMap')
            title_html = '<h3 align="center" style="font-size:20px; color:red;"><b>桃園市行政區發展潛力地圖 (分級數據欄位缺失)</b></h3>'
            m_error.get_root().html.add_child(folium.Element(title_html))
            os.makedirs('docs', exist_ok=True)
            m_error.save('docs/map_interactive.html')
            print("⚠️ 已生成一個指示錯誤的基本地圖。")
            return
            
    # 合併地理數據與分級結果
    merged_gdf = merge_geodata_with_results(gdf_map, jenks_data)
    
    if merged_gdf is None or merged_gdf.empty or 'geometry' not in merged_gdf.columns:
        print("❌ 地理數據與分級結果合併失敗、為空或缺少'geometry'欄位，無法生成互動地圖。")
        center_lat, center_lng = 24.9936, 121.3010
        m_error = folium.Map(location=[center_lat, center_lng], zoom_start=10, tiles='OpenStreetMap')
        title_html = '<h3 align="center" style="font-size:20px; color:red;"><b>桃園市行政區發展潛力地圖 (數據合併失敗)</b></h3>'
        m_error.get_root().html.add_child(folium.Element(title_html))
        os.makedirs('docs', exist_ok=True)
        m_error.save('docs/map_interactive.html')
        print("⚠️ 已生成一個指示錯誤的基本地圖。")
        return
        
    # 檢查必要欄位是否存在
    required_fields = ['區域別', '3級Jenks分級', '綜合分數', '排名']
    missing_fields = [field for field in required_fields if field not in merged_gdf.columns]
    if missing_fields:
        print(f"⚠️ 缺少必要欄位: {missing_fields}")
        for field in missing_fields:
            merged_gdf[field] = 'N/A'
    
    # 創建英文潛力等級欄位用於顏色映射
    level_mapping = {
        '高潛力': 'High Potential',
        '中潛力': 'Medium Potential',
        '低潛力': 'Low Potential'
    }
    merged_gdf['level'] = merged_gdf['3級Jenks分級'].map(level_mapping).fillna('N/A')

    # 創建基本地圖
    if not merged_gdf.empty and not merged_gdf[merged_gdf.geometry.is_valid].empty:
        valid_centroids_gdf = merged_gdf[merged_gdf.geometry.is_valid]
        center_lat = float(valid_centroids_gdf.geometry.centroid.y.mean())
        center_lng = float(valid_centroids_gdf.geometry.centroid.x.mean())
    else:
        center_lat, center_lng = 24.9936, 121.3010

    m = folium.Map(
        location=[center_lat, center_lng],
        zoom_start=10,
        tiles='CartoDB dark_matter'
    )

    # 定義顏色映射
    color_map = {
        'High Potential': '#eb7062',  # 紅色
        'Medium Potential': '#f5b041', # 橙色
        'Low Potential': '#5cace2',   # 藍色
        'N/A': '#757575' # Grey for N/A
    }
    
    # 添加 GeoJSON 圖層
    geojson_layer = folium.GeoJson(
        merged_gdf,
        name='桃園市發展潛力分級',
        style_function=lambda feature: {
            'fillColor': color_map.get(feature['properties'].get('level', 'N/A'), '#757575'),
            'color': '#333333',
            'weight': 1.5, # Slightly thicker border
            'fillOpacity': 0.85, # More opaque for better visibility on dark background
        },
        tooltip=folium.GeoJsonTooltip(
            fields=['區域別', '3級Jenks分級', '綜合分數'],
            aliases=['行政區:', '潛力等級:', '綜合分數:'],
            localize=True,
            sticky=False,
            style=("background-color: rgba(30,30,30,0.9); color: #ffffff; font-family: 'Helvetica Neue', Arial, sans-serif; font-size: 13px; padding: 8px; border-radius: 4px; box-shadow: 0 0 5px rgba(0,0,0,0.5); border: 1px solid #333333;")
        ),
        popup=folium.GeoJsonPopup(
            fields=['區域別', '3級Jenks分級', '綜合分數', '排名'],
            aliases=['行政區:', '潛力等級:', '綜合分數:', '排名:'],
            localize=True,
            style=("width: 200px; background-color: rgba(30,30,30,0.9); color: #ffffff; font-family: 'Helvetica Neue', Arial, sans-serif; font-size: 13px; padding: 10px; border-radius: 4px; border: 1px solid #333333;")
        )
    )
    geojson_layer.add_to(m)

    # 添加行政區名稱標註
    for idx, row in merged_gdf.iterrows():
        if pd.notna(row.geometry) and row.geometry.centroid:
            centroid = row.geometry.centroid
            district_name = row['區域別']  # 保留完整區域名稱
            
            # 創建文字標籤
            folium.Marker(
                location=[centroid.y, centroid.x],
                icon=folium.DivIcon(
                    html=f'''<div style="
                        font-family: 'Microsoft JhengHei', 'Arial Unicode MS', Arial, sans-serif;
                        font-size: 14px;
                        font-weight: bold;
                        color: #ffffff;
                        text-shadow: 1px 1px 2px rgba(0,0,0,0.8), -1px -1px 2px rgba(0,0,0,0.8), 1px -1px 2px rgba(0,0,0,0.8), -1px 1px 2px rgba(0,0,0,0.8);
                        text-align: center;
                        white-space: nowrap;
                        pointer-events: none;
                        user-select: none;
                    ">{district_name}</div>''',
                    icon_size=(60, 20),
                    icon_anchor=(30, 10),
                    class_name='district-label'
                )
            ).add_to(m)

    # 添加圖例
    legend_html = """
     <div style="position: fixed; 
                 bottom: 30px; left: 30px; width: 150px;  
                 border:1px solid #333333; z-index:9999; font-size:14px;
                 background-color:rgba(30,30,30,0.9); border-radius: 5px; padding: 10px; box-shadow: 0 0 8px rgba(0,0,0,0.3);">
       <h4 style="margin-top:0; margin-bottom:8px; font-weight:bold; color:#ffffff;">圖例</h4>
       <div style="margin-bottom: 5px; color:#ffffff;"><i style="background:#eb7062; color:#eb7062; border-radius:50%; margin-right:5px;">__</i> 高潛力</div>
       <div style="margin-bottom: 5px; color:#ffffff;"><i style="background:#f5b041; color:#f5b041; border-radius:50%; margin-right:5px;">__</i> 中潛力</div>
       <div style="margin-bottom: 5px; color:#ffffff;"><i style="background:#5cace2; color:#5cace2; border-radius:50%; margin-right:5px;">__</i> 低潛力</div>
       <div style="color:#ffffff;"><i style="background:#757575; color:#757575; border-radius:50%; margin-right:5px;">__</i> 無數據</div>
     </div>
    """
    m.get_root().html.add_child(folium.Element(legend_html))
    
    # 添加標題
    title_html = '''
    <div style="position: fixed; 
                top: 10px; left: 50%; transform: translateX(-50%); 
                width: auto; padding: 8px 15px; 
                background-color: rgba(30,30,30,0.9); 
                border: 1px solid #333333; border-radius: 5px; 
                z-index:9990; box-shadow: 0 2px 4px rgba(0,0,0,0.3);">
        <h3 align="center" style="font-size:18px; font-family: 'Microsoft JhengHei', 'Segoe UI', sans-serif; color: #ffffff; margin:0;">
            <b>桃園市行政區發展潛力地圖</b>
        </h3>
    </div>
    '''
    m.get_root().html.add_child(folium.Element(title_html))

    # 添加圖層控制器
    folium.LayerControl().add_to(m)
    
    # 保存地圖
    os.makedirs('docs', exist_ok=True)
    m.save('docs/map_interactive.html')
    print("✅ 互動式地圖生成完成: docs/map_interactive.html")

def create_map_data_for_web(merged_gdf, config):
    """為網頁創建地圖數據 (e.g. for a custom JS map component if not using iframe)"""
    if merged_gdf is None or config is None:
        print("❌ 無法創建網頁地圖數據：數據不完整。")
        return None
    print("📊 創建網頁地圖數據 (GeoJSON)...")
    
    map_data = {
        'type': 'FeatureCollection',
        'features': []
    }
    
    valid_gdf = merged_gdf[merged_gdf.geometry.notna() & pd.notna(merged_gdf['綜合分數'])]

    for idx, row in valid_gdf.iterrows():
        feature = {
            'type': 'Feature',
            'properties': {
                'name': row['區域別'],
                'level': row.get('3級Jenks分級', 'N/A'),
                'score': float(round(row['綜合分數'], 1)),
                'rank': int(row['排名']),
                'color': {
                    '高潛力': '#eb7062',
                    '中潛力': '#f5b041',
                    '低潛力': '#5cace2'
                }.get(row.get('3級Jenks分級'), '#808080')
            },
            'geometry': mapping(row.geometry) # Uses shapely.geometry.mapping
        }
        map_data['features'].append(feature)
    
    # 添加配置信息
    center_lat, center_lon = 24.9937, 121.2988 # Default
    if not valid_gdf.empty and not valid_gdf[valid_gdf.geometry.is_valid].empty: # Check for valid geometries before centroid calculation
        valid_centroids_gdf = valid_gdf[valid_gdf.geometry.is_valid]
        center_lat = float(valid_centroids_gdf.geometry.centroid.y.mean())
        center_lon = float(valid_centroids_gdf.geometry.centroid.x.mean())

    map_config_output = {
        'title': '桃園市行政區發展潛力分級地圖',
        'subtitle': '基於3級Jenks自然斷點分析',
        'score_range': [float(round(s, 1)) for s in config['score_range']],
        'total_districts': len(map_data['features']),
        'center': {
            'lat': center_lat,
            'lng': center_lon
        },
        'zoom_level': 10 # Adjusted zoom
    }
    
    output_data = {
        'config': map_config_output,
        'geojson': map_data
    }
    
    os.makedirs('docs/data', exist_ok=True)
    output_path = 'docs/data/map_data.json'
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)
    
    print(f"✅ 網頁地圖數據 (GeoJSON) 已保存: {output_path}")
    return output_data

def generate_map_statistics(merged_gdf, config):
    """生成地圖統計數據 (e.g., F-statistic for Jenks breaks goodness of fit)"""
    if merged_gdf is None or merged_gdf[pd.notna(merged_gdf['綜合分數'])].empty:
        print("❌ 無法生成地圖統計：數據不完整或無評分數據。")
        # Create a dummy stats file if needed by other parts, or handle this case upstream
        stats_output = {
            'f_statistic': 'N/A', 'p_value': 'N/A', 'effect_size': 'N/A',
            'level_statistics': {}, 'total_districts': 0,
            'score_distribution': {'mean': 'N/A', 'std': 'N/A', 'min': 'N/A', 'max': 'N/A'}
        }
        os.makedirs('docs/data', exist_ok=True)
        output_path = 'docs/data/map_statistics.json'
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(stats_output, f, ensure_ascii=False, indent=2)
        print(f"⚠️  生成了包含 N/A 值的統計數據文件: {output_path}")
        return stats_output

    print("📈 生成地圖相關統計數據 (F-statistic, etc.)...")
    
    # Filter for rows with scores for statistical analysis
    scored_gdf = merged_gdf.dropna(subset=['綜合分數', '3級Jenks分級'])
    if len(scored_gdf) < 3 or scored_gdf['3級Jenks分級'].nunique() < 2 : # Need at least 2 groups for ANOVA
        print(f"⚠️ 分數數據不足 ({len(scored_gdf)} 筆) 或少於2個潛力等級 ({scored_gdf['3級Jenks分級'].nunique()} 個) 無法計算F統計量。")
        f_stat_val, p_value_val, eta_squared_val = 'N/A', 'N/A', 'N/A'
    else:
        high_scores = scored_gdf[scored_gdf['3級Jenks分級'] == '高潛力']['綜合分數'].values
        medium_scores = scored_gdf[scored_gdf['3級Jenks分級'] == '中潛力']['綜合分數'].values
        low_scores = scored_gdf[scored_gdf['3級Jenks分級'] == '低潛力']['綜合分數'].values
        
        # Ensure there are at least two groups with data to compare
        groups_for_anova = [g for g in [high_scores, medium_scores, low_scores] if len(g) > 0]
        if len(groups_for_anova) >= 2:
            f_stat_val, p_value_val = stats.f_oneway(*groups_for_anova) # Use only groups with data
            f_stat_val = float(round(f_stat_val, 3))
            p_value_val = float(round(p_value_val, 6))

            # Calculate eta squared (effect size)
            all_scores = scored_gdf['綜合分數'].values
            ss_total = np.sum((all_scores - np.mean(all_scores))**2)
            
            ss_between = 0
            for group_scores in groups_for_anova:
                ss_between += len(group_scores) * (np.mean(group_scores) - np.mean(all_scores))**2
            
            eta_squared_val = (ss_between / ss_total) if ss_total > 0 else 0
            eta_squared_val = float(round(eta_squared_val, 3))
        else:
            print(f"⚠️ 少於2個有效數據組 ({len(groups_for_anova)} 組) 無法計算F統計量。")
            f_stat_val, p_value_val, eta_squared_val = 'N/A', 'N/A', 'N/A'

    level_stats_output = {}
    for level in ['高潛力', '中潛力', '低潛力']:
        subset = scored_gdf[scored_gdf['3級Jenks分級'] == level]
        if not subset.empty:
            level_stats_output[level] = {
                'count': len(subset),
                'districts': subset['區域別'].tolist(),
                'avg_score': float(round(subset['綜合分數'].mean(), 2)),
                'score_range': [
                    float(round(subset['綜合分數'].min(), 2)),
                    float(round(subset['綜合分數'].max(), 2))
                ]
            }
        else: # Ensure all levels are present in output even if empty
            level_stats_output[level] = {
                'count': 0, 'districts': [], 'avg_score': 'N/A', 'score_range': ['N/A', 'N/A']
            }
            
    statistics_output = {
        'f_statistic': f_stat_val,
        'p_value': p_value_val,
        'effect_size': eta_squared_val,
        'level_statistics': level_stats_output,
        'total_districts': len(scored_gdf), # Count of districts with scores used for stats
        'score_distribution': {
            'mean': float(round(scored_gdf['綜合分數'].mean(), 2)) if not scored_gdf.empty else 'N/A',
            'std': float(round(scored_gdf['綜合分數'].std(), 2)) if not scored_gdf.empty else 'N/A',
            'min': float(round(scored_gdf['綜合分數'].min(), 2)) if not scored_gdf.empty else 'N/A',
            'max': float(round(scored_gdf['綜合分數'].max(), 2)) if not scored_gdf.empty else 'N/A'
        }
    }
    
    os.makedirs('docs/data', exist_ok=True)
    output_path_stats = 'docs/data/map_statistics.json' # This file is used by index.html's load_statistics.js
    with open(output_path_stats, 'w', encoding='utf-8') as f:
        json.dump(statistics_output, f, ensure_ascii=False, indent=2)
    
    print(f"✅ 地圖統計數據已保存: {output_path_stats}")
    if f_stat_val != 'N/A':
        print(f"   F統計量: {statistics_output['f_statistic']}")
        print(f"   效應大小: {statistics_output['effect_size']}")
    
    return statistics_output

# End of Map generation functions

def generate_web_data():
    """生成網頁所需的所有數據"""
    print("🌐 生成網頁DASHBOARD數據")
    print("="*50)
    
    # 載入數據
    results_df, config, df = load_classification_results()
    if results_df is None:
        print("❌ 無法載入數據，請先運行 step3_ranking_classification.py")
        return

    # 標準化 results_df 中的欄位，以供地圖和其他部分使用
    level_mapping_to_en = {'高潛力': 'High Potential', '中潛力': 'Medium Potential', '低潛力': 'Low Potential'}
    results_df['level'] = results_df['3級Jenks分級'].map(level_mapping_to_en).fillna('N/A')
    
    if '綜合分數' in results_df.columns:
        results_df['comprehensive_score'] = results_df['綜合分數'].round(1)
    else:
        print("⚠️ 'results_df' 中缺少 '綜合分數' 欄位。地圖提示訊息可能不正確。")
        results_df['comprehensive_score'] = 'N/A'
        
    if '3級Jenks分級' in results_df.columns:
        results_df['level_chinese'] = results_df['3級Jenks分級']
    else:
        print("⚠️ 'results_df' 中缺少 '3級Jenks分級' 欄位。地圖提示訊息可能不正確。")
        results_df['level_chinese'] = '無數據'

    if '区域别' in results_df.columns and '區域別' not in results_df.columns:
        results_df.rename(columns={'区域别': '區域別'}, inplace=True)
    elif 'district' in results_df.columns and '區域別' not in results_df.columns:
         results_df.rename(columns={'district': '區域別'}, inplace=True)

    # 確保docs/data目錄存在
    os.makedirs('docs/data', exist_ok=True)
    
    # 準備各種數據
    level_stats, district_details, feature_mapping = prepare_dashboard_data(results_df, config, df)
    radar_data = create_radar_chart_data(results_df, feature_mapping)
    scatter_data = create_scatter_chart_data(results_df)
    importance_data = create_feature_importance_data(config)
    method_info = create_method_info(config)
    
    # 🗺️ 生成地圖相關數據
    print("\n🗺️ 生成地圖可視化數據...")
    gdf = load_map_data()
    if gdf is not None:
        merged_gdf = merge_geodata_with_results(gdf, results_df)
        if merged_gdf is not None:
            # 生成各種地圖輸出
            create_static_map(merged_gdf, config)
            generate_interactive_map(df, results_df)
            create_map_data_for_web(merged_gdf, config)
            generate_map_statistics(merged_gdf, config)
    
    # 生成主數據文件
    dashboard_data = {
        'title': '桃園市行政區3級Jenks分級分析',
        'subtitle': '基於Jenks自然斷點的13個行政區發展潛力評估 (0-10分制)',
        'last_updated': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S'),
        'summary': {
            'total_districts': len(results_df),
            'method': config['method'],
            'score_range': [float(round(s, 1)) for s in config['score_range']]
        },
        'level_statistics': level_stats,
        'districts': district_details,
        'radar_data': radar_data,
        'scatter_data': scatter_data,
        'feature_importance': importance_data,
        'method_info': method_info
    }
    
    # 保存數據文件
    data_files = {
        'dashboard_data.json': dashboard_data,
        'districts.json': district_details,
        'radar_data.json': radar_data,
        'level_stats.json': level_stats,
        'method_info.json': method_info
    }
    
    for filename, data in data_files.items():
        filepath = f'docs/data/{filename}'
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        print(f"✅ 已生成: {filepath}")
    
    # 生成JavaScript數據文件
    js_data_content = f"""// 3級Jenks分級分析數據
// 自動生成於 {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}

const DASHBOARD_DATA = {json.dumps(dashboard_data, ensure_ascii=False, indent=2)};

// 導出數據供其他模塊使用
if (typeof module !== 'undefined' && module.exports) {{
    module.exports = DASHBOARD_DATA;
}}
"""
    
    js_filepath = 'docs/js/jenks_data.js'
    with open(js_filepath, 'w', encoding='utf-8') as f:
        f.write(js_data_content)
    print(f"✅ 已生成: {js_filepath}")
    
    # 顯示統計摘要
    print(f"\n📈 數據統計摘要:")
    print(f"  總行政區數: {len(results_df)}")
    print(f"  高潛力區域: {level_stats['high_potential']['count']} 個")
    print(f"  中潛力區域: {level_stats['medium_potential']['count']} 個") 
    print(f"  低潛力區域: {level_stats['low_potential']['count']} 個")
    print(f"  分數範圍: {config['score_range'][0]:.1f} - {config['score_range'][1]:.1f} (0-10分制)")
    
    return dashboard_data

def update_html_for_jenks():
    """更新HTML文件以適配3級Jenks分級 (及10分制) 並加入互動地圖"""
    print("\n🔄 檢查HTML文件 (docs/index.html)...")
    
    # 檢查是否已存在 index.html
    if os.path.exists('docs/index.html'):
        print("✅ docs/index.html 文件已存在，跳過重新生成")
        return
    
    print("⚠️ 未找到 docs/index.html，將生成新文件...")
    
    # 生成新的HTML內容 (確保繁體中文和10分制範例)
    html_content = """<!DOCTYPE html>
<html lang="zh-TW">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>桃園市行政區3級Jenks分級分析儀表板</title>
    <link rel="stylesheet" href="css/style.css">
    <script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.js"></script>
</head>
<body>
    <div class="container">
        <!-- 標題區域 -->
        <header class="header">
            <h1>桃園市行政區發展潛力分析</h1>
            <p class="subtitle">基於3級Jenks自然斷點的13個行政區發展潛力評估 (0-10分制)</p>
        </header>

        <!-- 主要指標卡片 -->
        <section class="metrics-cards">
            <div class="card high-potential">
                <h3>高潛力</h3>
                <div class="metric-value" id="highCount"> 加載中... </div>
                <div class="metric-label">個行政區</div>
                <div class="districts-list" id="highDistricts"> 加載中... </div>
            </div>
            <div class="card medium-potential">
                <h3>中潛力</h3>
                <div class="metric-value" id="mediumCount"> 加載中... </div>
                <div class="metric-label">個行政區</div>
                <div class="districts-list" id="mediumDistricts"> 加載中... </div>
            </div>
            <div class="card low-potential">
                <h3>低潛力</h3>
                <div class="metric-value" id="lowCount"> 加載中... </div>
                <div class="metric-label">個行政區</div>
                <div class="districts-list" id="lowDistricts"> 加載中... </div>
            </div>
        </section>

        <!-- 主要內容區域 -->
        <main class="main-content">
            <!-- 左側：分級散點圖 -->
            <section class="chart-section">
                <h2>3級Jenks分級結果</h2>
                <div class="chart-container">
                    <canvas id="jenksChart"></canvas>
                </div>
                <div class="chart-info">
                    <p><strong>F統計量:</strong> <span id="fStatistic">加載中...</span></p>
                    <p><strong>效應大小:</strong> <span id="effectSize">加載中...</span></p>
                    <p><strong>分級方法:</strong> Jenks自然斷點 (3級)</p>
                    <p><strong>分數範圍:</strong> 0-10分制</p> 
                </div>
            </section>

            <!-- 右側：特徵雷達圖 -->
            <section class="chart-section">
                <h2>特徵分析雷達圖</h2>
                <div class="radar-controls">
                    <select id="districtSelect">
                        <option value="">選擇行政區</option>
                    </select>
                </div>
                <div class="chart-container">
                    <canvas id="radarChart"></canvas>
                </div>
            </section>
        </main>

        <!-- 詳細數據表格 -->
        <section class="data-table-section">
            <h2>詳細分級結果</h2>
            <div class="table-container">
                <table id="dataTable">
                    <thead>
                        <tr>
                            <th>排名</th>
                            <th>行政區</th>
                            <th>潛力等級</th>
                            <th>綜合分數 (0-10)</th>
                            <th>工作年齡比例</th>
                            <th>家戶中位數所得</th>
                            <th>第三產業比例</th>
                            <th>醫療指數</th>
                            <th>商業集中度</th>
                        </tr>
                    </thead>
                    <tbody id="tableBody">
                        <!-- 動態生成 -->
                    </tbody>
                </table>
            </div>
        </section>

        <!-- Interactive Map Section -->
        <section class="map-section content-section">
            <h2>互動式發展潛力地圖</h2>
            <div class="map-container-iframe">
                <iframe src="map_interactive.html" width="100%" height="600px" style="border:1px solid #ddd; border-radius: 8px;" title="桃園市互動式發展潛力地圖"></iframe>
            </div>
        </section>

        <!-- 方法說明 -->
        <section class="methodology-section">
            <h2>Jenks自然斷點分級方法</h2>
            <div class="method-steps">
                <div class="step">
                    <h4>STEP 1: 數據準備</h4>
                    <p>整合人口、商業、所得、產業、醫療等5個關鍵指標</p>
                </div>
                <div class="step">
                    <h4>STEP 2: 特徵標準化</h4>
                    <p>使用Z-score標準化處理所有特徵</p>
                </div>
                <div class="step">
                    <h4>STEP 3: 權重計算</h4>
                    <p>依據政策重要性設定權重 (詳細權重見選定方案)</p>
                </div>
                <div class="step">
                    <h4>STEP 4: Jenks分級</h4>
                    <p>使用自然斷點方法找到最優分割點，形成3個潛力等級</p>
                </div>
                <div class="step">
                    <h4>STEP 5: 驗證分析</h4>
                    <p>通過F統計量、效應大小等指標驗證分級效果</p>
                </div>
            </div>
        </section>

        <!-- 頁腳 -->
        <footer class="footer">
            <p>&copy; 桃園市行政區發展潛力分析 | 
               <a href="https://github.com/mentaikoisgood/Taoyuan-District-Income-Analysis" target="_blank">
                   GitHub Repository
               </a>
            </p>
        </footer>
    </div>

    <script src="js/jenks_data.js"></script>
    <script src="js/jenks_dashboard.js"></script>
    <script src="js/load_statistics.js"></script>
</body>
</html>"""
    
    # 保存更新的HTML
    os.makedirs('docs', exist_ok=True)
    with open('docs/index.html', 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    print("✅ HTML文件已生成")

def main():
    """主函數"""
    print("🌐 STEP 5 - 3級Jenks分級網頁儀表板生成")
    print("="*60)
    
    # 生成數據
    dashboard_data = generate_web_data()
    
    if dashboard_data:
        # 更新HTML
        update_html_for_jenks()
        
        print(f"\n✅ 網頁儀表板數據生成完成!")
        print("="*60)
        print("📁 生成的文件:")
        print("  - docs/data/dashboard_data.json")
        print("  - docs/data/districts.json") 
        print("  - docs/data/radar_data.json")
        print("  - docs/data/level_stats.json")
        print("  - docs/data/method_info.json")
        print("  - docs/js/jenks_data.js")
        print("  - docs/index.html (若不存在則生成)")
        # Add new map files to the list
        print("  - output/taoyuan_potential_map.png (靜態地圖)")
        print("  - docs/taoyuan_potential_map.png (網頁用靜態地圖)")
        print("  - docs/map_interactive.html (互動式地圖)")
        print("  - docs/data/map_data.json (網頁地圖 GeoJSON數據)")
        print("  - docs/data/map_statistics.json (地圖統計數據)")
        
        print(f"\n🚀 網頁數據已生成於 docs/ 目錄下，可用於部署至 GitHub Pages。")
        print(f"   主儀表板: docs/index.html")
        print(f"   互動地圖: docs/map_interactive.html")
        print(f"   注意: 若要修改UI，可直接編輯 docs/ 目錄下的文件，無需重新運行此腳本")

if __name__ == "__main__":
    main()
