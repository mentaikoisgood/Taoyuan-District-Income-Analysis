"""
桃園市行政區3級Jenks分級分析
使用3級Jenks自然斷點進行分級

專注於核心分級任務：
1. 對5個指標進行z-score標準化
2. 設置權重計算綜合分數
3. 使用3級Jenks自然斷點分級
4. 保存分級結果

詳細驗證和視覺化請參考 step4_validation.py
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# 嘗試導入Jenks自然斷點
try:
    import jenkspy
    JENKS_AVAILABLE = True
    print("✅ Jenks自然斷點可用")
except ImportError:
    JENKS_AVAILABLE = False
    print("❌ Jenks未安裝，無法執行3級分析")
    exit(1)

def load_data():
    """載入特徵數據"""
    print("📂 載入桃園市行政區特徵數據...")
    
    df = pd.read_csv('output/taoyuan_features_enhanced.csv')
    districts = df['區域別'].tolist()
    feature_names = df.columns[1:].tolist()
    X = df[feature_names].values
    
    print(f"✅ 數據載入成功: {len(districts)} 個行政區, {len(feature_names)} 個特徵")
    print(f"特徵列表: {feature_names}")
    
    return df, districts, feature_names, X

def get_feature_properties():
    """获取特徵属性和權重 - 方案A: 統一Z-score策略"""
    # 🆕 方案A: 醫療權重30%分配給3個子指標
    feature_properties = {
        '人口_working_age_ratio': {'direction': 'positive', 'weight': 0.15, 'description': '工作年齡人口比例'},
        '商業_hhi_index': {'direction': 'negative', 'weight': 0.10, 'description': '商業集中度 (轉分散度)'},
        '所得_median_household_income': {'direction': 'positive', 'weight': 0.40, 'description': '家戶中位數所得'},
        'tertiary_industry_ratio': {'direction': 'positive', 'weight': 0.05, 'description': '服務業比例'},
        # 🏥 醫療30%權重分配給3個子指標
        'medical_beds_per_1k': {'direction': 'positive', 'weight': 0.10, 'description': '每千人病床數'},
        'medical_staff_per_1k': {'direction': 'positive', 'weight': 0.10, 'description': '每千人醫療人員'},
        'medical_facility_density': {'direction': 'positive', 'weight': 0.10, 'description': '醫療設施密度'}
    }
    
    # 验证权重总和
    total_weight = sum(props['weight'] for props in feature_properties.values())
    print(f"\n🔍 權重配置驗證:")
    print(f"  權重總和: {total_weight:.3f} (目標: 1.000)")
    print(f"  配置說明: 方案A統一Z-score策略 - 醫療30%權重分解為3個子指標")
    
    if abs(total_weight - 1.0) > 0.001:
        print(f"  ⚠️ 權重總和不等於1，進行標準化...")
        for feature in feature_properties:
            feature_properties[feature]['weight'] /= total_weight
        
        print(f"  ✅ 權重已標準化")
    
    print(f"  詳細權重設定:")
    medical_total = 0
    for feature, props in feature_properties.items():
        direction_symbol = "+" if props['direction'] == 'positive' else "-"
        weight_pct = f"{props['weight']:.1%}"
        if feature.startswith('medical_'):
            weight_pct += " 🏥"
            medical_total += props['weight']
        elif props['weight'] >= 0.3:
            weight_pct += " 🎯高權重"
        elif props['weight'] <= 0.1:
            weight_pct += " 🔽低權重"
        print(f"    {feature}: {weight_pct} ({direction_symbol}) - {props['description']}")
    
    print(f"  📊 醫療總權重: {medical_total:.1%} (3個子指標)")
    
    return feature_properties

def calculate_composite_scores(df, feature_names, feature_properties):
    """計算綜合分數"""
    print("\n🧮 計算綜合分數...")
    
    # Z-score標準化
    scaler = StandardScaler()
    X_standardized = scaler.fit_transform(df[feature_names])
    
    # 計算加權綜合分數
    composite_scores = np.zeros(len(df))
    
    for i, feature in enumerate(feature_names):
        weight = feature_properties[feature]['weight']
        direction = feature_properties[feature]['direction']
        
        if direction == 'positive':
            feature_score = X_standardized[:, i] * weight
        else:
            feature_score = -X_standardized[:, i] * weight
        
        composite_scores += feature_score
    
    # 正規化到0-10分
    min_score = np.min(composite_scores)
    max_score = np.max(composite_scores)
    normalized_scores = ((composite_scores - min_score) / (max_score - min_score)) * 10
    
    print(f"分數統計: 範圍[{np.min(normalized_scores):.1f}, {np.max(normalized_scores):.1f}], 平均{np.mean(normalized_scores):.1f} (0-10分制)")
    
    return normalized_scores

def jenks_3_level_classification(scores, districts):
    """使用3級Jenks自然斷點分級"""
    print(f"\n📊 執行3級Jenks自然斷點分級...")
    
    # 3級標籤
    labels_names = ['低潛力', '中潛力', '高潛力']
    
    try:
        # 使用Jenks自然斷點
        breaks = jenkspy.jenks_breaks(scores, n_classes=3)
        labels = []
        
        for score in scores:
            if score <= breaks[1]:
                labels.append('低潛力')
            elif score <= breaks[2]:
                labels.append('中潛力')
            else:
                labels.append('高潛力')
        
        labels = pd.Categorical(labels, categories=labels_names)
        
        print(f"分割點: {[f'{cut:.2f}' for cut in breaks]}")
        
        # 統計各級別
        level_counts = pd.Series(labels).value_counts()
        print(f"分級結果:")
        for level in labels_names:
            if level in level_counts:
                districts_in_level = [districts[i] for i, label in enumerate(labels) if label == level]
                print(f"  {level}: {level_counts[level]} 個區域 - {', '.join(districts_in_level)}")
        
        return labels, breaks
                
    except Exception as e:
        print(f"❌ 3級Jenks分級失敗: {e}")
        return None, None

def save_results(df, districts, normalized_scores, labels, breaks, feature_names, feature_properties):
    """保存分級結果"""
    print("\n💾 保存分級結果...")
    
    # 創建結果DataFrame
    results_df = pd.DataFrame({
        '區域別': districts,
        '綜合分數': normalized_scores,
        '3級Jenks分級': labels,
    })
    
    # 添加原始特徵
    for feature in feature_names:
        results_df[feature] = df[feature].values
    
    # 按分數排序
    results_df = results_df.sort_values('綜合分數', ascending=False).reset_index(drop=True)
    results_df['排名'] = range(1, len(results_df) + 1)
    
    # 重新排列列順序
    cols = ['排名', '區域別', '綜合分數', '3級Jenks分級']
    cols.extend(feature_names)
    results_df = results_df[cols]
    
    # 保存CSV
    csv_path = 'output/3_level_jenks_results.csv'
    results_df.to_csv(csv_path, index=False, encoding='utf-8')
    
    # 保存配置信息
    config = {
        'method': '3級Jenks自然斷點',
        'breaks': breaks if breaks is not None else None,
        'feature_properties': feature_properties,
        'n_districts': len(districts),
        'score_range': [float(np.min(normalized_scores)), float(np.max(normalized_scores))]
    }
    
    import json
    config_path = 'output/3_level_jenks_config.json'
    with open(config_path, 'w', encoding='utf-8') as f:
        json.dump(config, f, ensure_ascii=False, indent=2)
    
    print(f"✅ 結果已保存: {csv_path}")
    print(f"✅ 配置已保存: {config_path}")
    
    return csv_path, config_path

def main():
    """主函數"""
    print("🎯 STEP 3 - 桃園市行政區3級Jenks分級分析")
    print("="*60)
    
    # 載入數據
    df, districts, feature_names, X = load_data()
    
    # 獲取特徵屬性
    feature_properties = get_feature_properties()
    
    # 計算綜合分數
    normalized_scores = calculate_composite_scores(df, feature_names, feature_properties)
    
    # 3級Jenks分級
    labels, breaks = jenks_3_level_classification(normalized_scores, districts)
    
    if labels is None:
        print("❌ 分級失敗")
        return
    
    # 保存結果
    csv_path, config_path = save_results(df, districts, normalized_scores, labels, breaks, 
                                       feature_names, feature_properties)
    
    # 顯示最終排名
    sorted_indices = np.argsort(normalized_scores)[::-1]
    print(f"\n🏆 最終排名 (0-10分制):")
    for i in range(len(districts)):
        idx = sorted_indices[i]
        print(f"  {i+1:2d}. {districts[idx]:4s}: {normalized_scores[idx]:4.1f}分 ({labels[idx]})")
    
    print(f"\n✅ STEP 3 完成! 請執行 step4_validation.py 進行詳細驗證分析")

if __name__ == "__main__":
    main() 