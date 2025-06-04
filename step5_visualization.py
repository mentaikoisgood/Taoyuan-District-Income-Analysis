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
warnings.filterwarnings('ignore')

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
        '高潛力': '#2E8B57',
        '中潛力': '#FFA500', 
        '低潛力': '#DC143C'
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
                'description': '通過Bootstrap穩定性測試和質量指標驗證分級效果'
            }
        ]
    }
    
    return method_info

def generate_web_data():
    """生成網頁所需的所有數據"""
    print("🌐 生成網頁DASHBOARD數據")
    print("="*50)
    
    # 載入數據
    results_df, config, df = load_classification_results()
    if results_df is None:
        print("❌ 無法載入數據，請先運行 step3_ranking_classification.py")
        return
    
    # 確保docs/data目錄存在
    os.makedirs('docs/data', exist_ok=True)
    
    # 準備各種數據
    level_stats, district_details, feature_mapping = prepare_dashboard_data(results_df, config, df)
    radar_data = create_radar_chart_data(results_df, feature_mapping)
    scatter_data = create_scatter_chart_data(results_df)
    importance_data = create_feature_importance_data(config)
    method_info = create_method_info(config)
    
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
    """更新HTML文件以適配3級Jenks分級 (及10分制)"""
    print("\n🔄 更新HTML文件...")
    
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
            <h1>🏙️ 桃園市行政區發展潛力分析</h1>
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
                <h2>📊 3級Jenks分級結果</h2>
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
                <h2>🎯 特徵分析雷達圖</h2>
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
            <h2>📋 詳細分級結果</h2>
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

        <!-- 關鍵洞察 -->
        <section class="insights-section">
            <h2>💡 關鍵洞察與政策建議</h2>
            <div class="insights-grid">
                <div class="insight-card">
                    <h3>🏢 高潛力區域</h3>
                    <ul>
                        <li>桃園核心都會區重點區域</li>
                        <li>平均綜合分數 <span id="highAvgScore">加載中...</span> 分 (0-10制)</li>
                        <li>醫療資源充足，產業發達，所得水平高</li>
                        <li><strong>建議:</strong> 持續強化核心競爭力與創新產業發展</li>
                    </ul>
                </div>
                <div class="insight-card">
                    <h3>🌱 中潛力區域</h3>
                    <ul>
                        <li>發展中區域群組</li>
                        <li>平均綜合分數 <span id="mediumAvgScore">加載中...</span> 分 (0-10制)</li>
                        <li>發展程度適中，具有成長潛力</li>
                        <li><strong>建議:</strong> 因地制宜發展特色產業，加強基礎建設</li>
                    </ul>
                </div>
                <div class="insight-card">
                    <h3>🌾 低潛力區域</h3>
                    <ul>
                        <li>需關注區域，可能需要政策扶持</li>
                        <li>平均綜合分數 <span id="lowAvgScore">加載中...</span> 分 (0-10制)</li>
                        <li>發展條件相對較弱，但有特色資源</li>
                        <li><strong>建議:</strong> 重點投入基礎建設，發展觀光與特色農業</li>
                    </ul>
                </div>
            </div>
        </section>

        <!-- 方法說明 -->
        <section class="methodology-section">
            <h2>🔬 Jenks自然斷點分級方法</h2>
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
                    <p>依據選定方案設定各指標權重</p>
                </div>
                <div class="step">
                    <h4>STEP 4: Jenks分級</h4>
                    <p>使用自然斷點方法找到最優分割點，最大化組間差異、最小化組內差異</p>
                </div>
                <div class="step">
                    <h4>STEP 5: 驗證分析</h4>
                    <p>通過F統計量、效應大小等指標驗證分級質量</p>
                </div>
            </div>
        </section>

        <!-- 頁腳 -->
        <footer class="footer">
            <p>&copy; 2024 桃園市行政區發展潛力分析 - 3級Jenks分級 | 
               <a href="https://github.com/mentaikoisgood/Taoyuan-District-Income-Analysis" target="_blank">
                   GitHub Repository
               </a>
            </p>
        </footer>
    </div>

    <script src="js/jenks_data.js"></script>
    <script src="js/jenks_dashboard.js"></script>
</body>
</html>"""
    
    # 保存更新的HTML
    with open('docs/index.html', 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    print("✅ HTML文件已更新")

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
        print("  - docs/index.html (已更新)")
        print(f"\n🚀 網頁數據已生成於 docs/ 目錄下，可用於部署至 GitHub Pages。")
        print(f"   目標網址: @https://mentaikoisgood.github.io/Taoyuan-District-Income-Analysis/")

if __name__ == "__main__":
    main()
