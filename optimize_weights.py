"""
權重優化分析
找到更穩定的權重配置以改善3級Jenks分級的敏感度問題

基於敏感度分析結果，測試不同的權重組合
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import json
import itertools
import warnings
warnings.filterwarnings('ignore')

try:
    import jenkspy
    JENKS_AVAILABLE = True
except ImportError:
    JENKS_AVAILABLE = False
    print("❌ Jenks未安裝")
    exit(1)

def load_data():
    """載入數據"""
    df = pd.read_csv('output/taoyuan_features_enhanced.csv')
    districts = df['區域別'].tolist()
    feature_names = df.columns[1:].tolist()
    return df, districts, feature_names

def calculate_scores_and_ranking(df, feature_names, weights):
    """計算分數和排名"""
    scaler = StandardScaler()
    X_standardized = scaler.fit_transform(df[feature_names])
    
    # 計算綜合分數
    composite_scores = np.zeros(len(df))
    
    for i, feature in enumerate(feature_names):
        weight = weights[feature]['weight']
        direction = weights[feature]['direction']
        
        if direction == 'positive':
            feature_score = X_standardized[:, i] * weight
        else:
            feature_score = -X_standardized[:, i] * weight
        
        composite_scores += feature_score
    
    # 正規化到0-100分
    min_score = np.min(composite_scores)
    max_score = np.max(composite_scores)
    normalized_scores = ((composite_scores - min_score) / (max_score - min_score)) * 100
    
    # 計算排名
    ranking = len(normalized_scores) + 1 - pd.Series(normalized_scores).rank(method='min')
    
    return normalized_scores, ranking

def test_weight_stability(df, feature_names, weights, districts, n_tests=20):
    """測試權重配置的穩定性"""
    base_scores, base_ranking = calculate_scores_and_ranking(df, feature_names, weights)
    
    # 生成權重變動
    correlations = []
    top5_stabilities = []
    
    weight_delta = 0.05
    feature_list = list(weights.keys())
    
    for _ in range(n_tests):
        # 隨機選擇一個特徵進行變動
        target_feature = np.random.choice(feature_list)
        direction = np.random.choice([-1, 1])
        
        # 創建變動後的權重
        modified_weights = {}
        for feature, props in weights.items():
            modified_weights[feature] = props.copy()
        
        # 調整權重
        new_weight = weights[target_feature]['weight'] + (direction * weight_delta)
        if new_weight < 0:
            new_weight = 0.01
            
        modified_weights[target_feature]['weight'] = new_weight
        
        # 重新標準化
        total_weight = sum(props['weight'] for props in modified_weights.values())
        for feature in modified_weights:
            modified_weights[feature]['weight'] /= total_weight
        
        try:
            # 計算新分數和排名
            new_scores, new_ranking = calculate_scores_and_ranking(df, feature_names, modified_weights)
            
            # 計算相關性
            corr = np.corrcoef(base_ranking, new_ranking)[0, 1]
            if not np.isnan(corr):
                correlations.append(corr)
            
            # Top 5穩定性
            base_top5 = set(np.argsort(base_ranking)[:5])
            new_top5 = set(np.argsort(new_ranking)[:5])
            overlap = len(base_top5.intersection(new_top5)) / 5
            top5_stabilities.append(overlap)
            
        except:
            continue
    
    if len(correlations) == 0:
        return 0, 0
    
    avg_corr = np.mean(correlations)
    avg_top5 = np.mean(top5_stabilities)
    
    return avg_corr, avg_top5

def generate_weight_combinations():
    """生成不同的權重組合進行測試"""
    # 特徵列表
    features = ['人口_working_age_ratio', '商業_hhi_index', '所得_median_household_income', 
               'tertiary_industry_ratio', 'medical_index']
    
    # 權重候選值 (必須總和為1)
    weight_configs = [
        # 配置1: 均等權重
        {
            '人口_working_age_ratio': 0.20,
            '商業_hhi_index': 0.20, 
            '所得_median_household_income': 0.20,
            'tertiary_industry_ratio': 0.20,
            'medical_index': 0.20
        },
        # 配置2: 所得主導
        {
            '人口_working_age_ratio': 0.15,
            '商業_hhi_index': 0.10,
            '所得_median_household_income': 0.40,
            'tertiary_industry_ratio': 0.20,
            'medical_index': 0.15
        },
        # 配置3: 醫療和所得並重
        {
            '人口_working_age_ratio': 0.15,
            '商業_hhi_index': 0.10,
            '所得_median_household_income': 0.30,
            'tertiary_industry_ratio': 0.15,
            'medical_index': 0.30
        },
        # 配置4: 產業主導
        {
            '人口_working_age_ratio': 0.20,
            '商業_hhi_index': 0.15,
            '所得_median_household_income': 0.20,
            'tertiary_industry_ratio': 0.35,
            'medical_index': 0.10
        },
        # 配置5: 人口和所得重視
        {
            '人口_working_age_ratio': 0.30,
            '商業_hhi_index': 0.10,
            '所得_median_household_income': 0.35,
            'tertiary_industry_ratio': 0.15,
            'medical_index': 0.10
        },
        # 配置6: 平衡型(降低所得權重)
        {
            '人口_working_age_ratio': 0.22,
            '商業_hhi_index': 0.18,
            '所得_median_household_income': 0.20,
            'tertiary_industry_ratio': 0.20,
            'medical_index': 0.20
        },
        # 配置7: 保守型(降低變動最大的特徵權重)
        {
            '人口_working_age_ratio': 0.15,  # 降低最敏感的特徵
            '商業_hhi_index': 0.20,
            '所得_median_household_income': 0.25,
            'tertiary_industry_ratio': 0.20,
            'medical_index': 0.20
        }
    ]
    
    # 轉換為完整格式
    full_configs = []
    for i, config in enumerate(weight_configs):
        full_config = {}
        for feature, weight in config.items():
            full_config[feature] = {
                'weight': weight,
                'direction': 'negative' if feature == '商業_hhi_index' else 'positive',
                'description': f"配置{i+1}"
            }
        full_configs.append(full_config)
    
    return full_configs

def evaluate_configuration(df, districts, feature_names, weights):
    """評估權重配置"""
    # 計算基本分數
    scores, ranking = calculate_scores_and_ranking(df, feature_names, weights)
    
    # 使用Jenks分級
    try:
        breaks = jenkspy.jenks_breaks(scores, n_classes=3)
        labels = []
        
        for score in scores:
            if score <= breaks[1]:
                labels.append('低潛力')
            elif score <= breaks[2]:
                labels.append('中潛力')
            else:
                labels.append('高潛力')
    except:
        return None
    
    # 計算質量指標
    def calculate_f_statistic(scores, labels):
        unique_labels = np.unique(labels)
        overall_mean = np.mean(scores)
        
        between_var = 0
        within_var = 0
        
        for label in unique_labels:
            mask = np.array(labels) == label
            group_scores = scores[mask]
            group_mean = np.mean(group_scores)
            group_size = len(group_scores)
            
            between_var += (group_mean - overall_mean) ** 2 * group_size
            if group_size > 1:
                within_var += np.var(group_scores) * group_size
        
        between_var /= len(scores)
        within_var /= len(scores)
        
        return between_var / within_var if within_var > 0 else 0
    
    f_stat = calculate_f_statistic(scores, labels)
    
    # 測試穩定性
    avg_corr, avg_top5 = test_weight_stability(df, feature_names, weights, districts)
    
    # 計算級別分布
    level_counts = pd.Series(labels).value_counts()
    balance_score = 1 - np.std([level_counts.get(level, 0) for level in ['高潛力', '中潛力', '低潛力']]) / np.mean([level_counts.get(level, 0) for level in ['高潛力', '中潛力', '低潛力']])
    
    return {
        'f_statistic': f_stat,
        'ranking_correlation': avg_corr,
        'top5_stability': avg_top5,
        'balance_score': balance_score,
        'scores': scores.tolist(),
        'ranking': ranking.tolist(),
        'labels': labels,
        'breaks': breaks
    }

def main():
    """主函數"""
    print("⚖️ 權重優化分析")
    print("="*50)
    
    # 載入數據
    df, districts, feature_names = load_data()
    
    # 生成權重配置
    weight_configs = generate_weight_combinations()
    
    print(f"📊 測試 {len(weight_configs)} 種權重配置...")
    
    results = []
    
    for i, weights in enumerate(weight_configs):
        print(f"\n🔍 測試配置 {i+1}:")
        
        # 顯示權重
        for feature, props in weights.items():
            direction_symbol = "+" if props['direction'] == 'positive' else "-"
            print(f"  {feature}: {props['weight']:.3f} ({direction_symbol})")
        
        # 評估配置
        result = evaluate_configuration(df, districts, feature_names, weights)
        
        if result:
            result['config_id'] = i + 1
            result['weights'] = weights
            results.append(result)
            
            print(f"  F統計量: {result['f_statistic']:.3f}")
            print(f"  排名穩定性: {result['ranking_correlation']:.3f}")
            print(f"  前5名穩定性: {result['top5_stability']:.3f}")
            print(f"  級別平衡性: {result['balance_score']:.3f}")
        else:
            print("  ❌ 評估失敗")
    
    if not results:
        print("❌ 沒有成功的配置")
        return
    
    # 計算綜合評分
    for result in results:
        # 綜合評分 = F統計量 * 0.3 + 排名穩定性 * 0.4 + 前5名穩定性 * 0.2 + 平衡性 * 0.1
        result['composite_score'] = (
            result['f_statistic'] * 0.3 +
            max(0, result['ranking_correlation']) * 0.4 +
            result['top5_stability'] * 0.2 +
            result['balance_score'] * 0.1
        )
    
    # 排序結果
    results.sort(key=lambda x: x['composite_score'], reverse=True)
    
    print(f"\n🏆 權重配置排名:")
    print("="*50)
    
    for i, result in enumerate(results):
        print(f"{i+1}. 配置{result['config_id']} - 綜合評分: {result['composite_score']:.3f}")
        print(f"   F統計量: {result['f_statistic']:.3f} | 排名穩定性: {result['ranking_correlation']:.3f}")
        print(f"   前5名穩定性: {result['top5_stability']:.3f} | 平衡性: {result['balance_score']:.3f}")
    
    # 推薦最佳配置
    best_config = results[0]
    print(f"\n✅ 推薦配置: 配置{best_config['config_id']}")
    print("推薦權重:")
    for feature, props in best_config['weights'].items():
        direction_symbol = "+" if props['direction'] == 'positive' else "-"
        print(f"  '{feature}': {props['weight']:.3f} ({direction_symbol})")
    
    # 保存結果
    output_path = 'output/weight_optimization_results.json'
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    print(f"\n💾 優化結果已保存: {output_path}")
    
    return best_config

if __name__ == "__main__":
    best_config = main() 