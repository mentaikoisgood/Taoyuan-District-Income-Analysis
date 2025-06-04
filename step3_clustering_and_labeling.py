"""
桃園市行政區分群及標籤 (最佳方法)

使用最佳特征組合(收入+醫療+第三產業) + t-SNE降維和Ward層次聚類進行分群
基於潛力綜合分數和多數決進行集群標籤分配

性能指標:
- Silhouette分數: 0.717 (優秀)
- 語義一致性: 0.794 (優秀)
- 分群數量: 3

輸出:
- 分群結果CSV
- 分群視覺化圖
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import warnings
from sklearn.manifold import TSNE
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score
import matplotlib
from sklearn.metrics import pairwise_distances
from sklearn.preprocessing import StandardScaler
from scipy import stats

# 忽略警告
warnings.filterwarnings('ignore')

# 設定中文字體
matplotlib.rcParams['font.family'] = ['sans-serif']
matplotlib.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False

# 參數設定
OUTPUT_DIR = 'output'
RANDOM_STATE = 42

def load_data():
    """載入特徵數據並選擇最佳特徵組合"""
    print("📂 載入特徵數據並選擇最佳特徵組合...")
    
    # 載入特徵數據
    df = pd.read_csv('output/taoyuan_features_enhanced.csv')
    
    # 最佳特徵組合: 3個核心特徵
    optimal_features = [
        '所得_median_household_income',  # 經濟水平指標
        'medical_index',                # 醫療服務指標
        'tertiary_industry_ratio'       # 產業結構指標
    ]
    
    # 提取行政區和特徵
    districts = df['區域別'].tolist()
    X = df[optimal_features].values
    
    print(f"✅ 數據載入成功: {len(districts)} 個行政區, {X.shape[1]} 個最佳特徵")
    print(f"  最佳特徵組合: {', '.join(optimal_features)}")
    print(f"  特徵選擇理由:")
    print(f"    1. 經濟基礎 - 家庭收入中位數")
    print(f"    2. 公共服務 - 醫療服務指數")
    print(f"    3. 產業發展 - 第三產業比例")
    
    return X, districts, optimal_features, df

def calculate_potential_score(df):
    """計算潛力綜合分數"""
    print("\n📊 計算潛力綜合分數...")
    
    # 選定潛力指標（包含聚類特徵+額外人口指標以增強評估）
    potential_indicators = [
        '所得_median_household_income',  # 經濟水平
        'medical_index',                # 醫療服務  
        'tertiary_industry_ratio',      # 產業結構
        '人口_working_age_ratio'        # 人口結構（額外指標）
    ]
    
    # 檢查指標是否存在
    available_indicators = [col for col in potential_indicators if col in df.columns]
    print(f"  可用潛力指標: {', '.join(available_indicators)}")
    
    # 對選定指標進行z-score標準化
    z_scores = pd.DataFrame()
    z_scores['區域別'] = df['區域別']
    
    for indicator in available_indicators:
        # 計算z-score
        z_score = stats.zscore(df[indicator])
        z_scores[f'{indicator}_zscore'] = z_score
        print(f"  {indicator}: 平均={df[indicator].mean():.2f}, 標準差={df[indicator].std():.2f}")
    
    # 計算潛力綜合分數（z-score的平均）
    z_score_cols = [col for col in z_scores.columns if col.endswith('_zscore')]
    z_scores['potential_score'] = z_scores[z_score_cols].mean(axis=1)
    
    print(f"✅ 潛力綜合分數計算完成")
    print(f"  潛力分數範圍: {z_scores['potential_score'].min():.3f} ~ {z_scores['potential_score'].max():.3f}")
    
    return z_scores

def assign_district_labels(potential_scores):
    """使用分位數分箱分配行政區標籤"""
    print("\n🏷️ 分配行政區發展潛力標籤...")
    
    # 使用qcut按分位數切成三等份
    district_labels = pd.qcut(
        potential_scores['potential_score'], 
        q=3, 
        labels=['低潛力', '中潛力', '高潛力']
    )
    
    potential_scores['district_label'] = district_labels
    
    # 顯示分組結果
    for label in ['低潛力', '中潛力', '高潛力']:
        districts_in_group = potential_scores[potential_scores['district_label'] == label]['區域別'].tolist()
        score_range = potential_scores[potential_scores['district_label'] == label]['potential_score']
        print(f"  {label}: {len(districts_in_group)} 個行政區")
        print(f"    行政區: {', '.join(districts_in_group)}")
        print(f"    潛力分數範圍: {score_range.min():.3f} ~ {score_range.max():.3f}")
    
    return potential_scores

def assign_cluster_labels_by_majority(cluster_labels, district_labels_df):
    """使用多數決分配集群標籤"""
    print("\n🗳️ 使用多數決分配集群標籤...")
    
    # 創建包含集群和行政區標籤的DataFrame
    mapping_df = pd.DataFrame({
        'cluster_id': cluster_labels,
        'district_label': district_labels_df['district_label'].values,
        'district': district_labels_df['區域別'].values
    })
    
    # 為每個集群分配標籤
    cluster_potential_mapping = {}
    
    for cluster_id in range(3):
        cluster_data = mapping_df[mapping_df['cluster_id'] == cluster_id]
        
        # 計算各標籤的出現次數
        label_counts = cluster_data['district_label'].value_counts()
        majority_label = label_counts.index[0]  # 出現次數最多的標籤
        
        cluster_potential_mapping[cluster_id] = majority_label
        
        print(f"  集群 {cluster_id}: 多數決結果 = {majority_label}")
        print(f"    集群成員: {', '.join(cluster_data['district'].tolist())}")
        print(f"    標籤分布: {dict(label_counts)}")
    
    print(f"✅ 集群潛力等級映射: {cluster_potential_mapping}")
    
    return cluster_potential_mapping

def evaluate_semantic_consistency(cluster_labels, district_labels_df):
    """評估語義一致性"""
    mapping_df = pd.DataFrame({
        'cluster_id': cluster_labels,
        'potential_score': district_labels_df['potential_score'].values
    })
    
    # 計算各集群潛力分數統計
    consistency_scores = []
    for cluster_id in range(3):
        cluster_data = mapping_df[mapping_df['cluster_id'] == cluster_id]
        potential_scores = cluster_data['potential_score'].values
        
        if len(potential_scores) > 1:
            std = potential_scores.std()
            consistency = 1 / (1 + std)
        else:
            consistency = 1.0
        
        consistency_scores.append(consistency)
    
    return np.mean(consistency_scores)

def perform_clustering(X, districts, df):
    """執行最佳聚類方法"""
    print("\n🔍 執行最佳聚類方法...")
    print("方法: 3核心特徵 + t-SNE降維 + Ward層次聚類")
    
    # t-SNE降維
    tsne = TSNE(n_components=2, random_state=RANDOM_STATE, perplexity=3, max_iter=1000)
    X_tsne = tsne.fit_transform(X)
    print(f"✅ t-SNE降維完成: {X.shape} → {X_tsne.shape}")
    
    # Ward層次聚類
    ward = AgglomerativeClustering(n_clusters=3, linkage='ward')
    cluster_labels = ward.fit_predict(X_tsne)
    
    # 計算輪廓係數
    silhouette = silhouette_score(X_tsne, cluster_labels)
    print(f"✅ Ward層次聚類完成: 3個集群, 輪廓係數={silhouette:.3f}")
    
    # 計算每個集群的中心點
    cluster_centers = np.array([X_tsne[cluster_labels == i].mean(axis=0) for i in range(3)])
    
    # 計算每個點到每個集群中心的距離
    distances = pairwise_distances(X_tsne, cluster_centers)
    
    # 使用更合理的機率計算方法
    def calculate_probabilities(distances):
        # 計算到最近集群的相對距離
        min_distances = distances.min(axis=1, keepdims=True)
        relative_distances = distances / (min_distances + 1e-6)
        
        # 使用溫度調節的softmax（溫度較高，機率分布較平滑）
        temperature = 3.0
        exp_values = np.exp(-relative_distances / temperature)
        probabilities = exp_values / exp_values.sum(axis=1, keepdims=True)
        return probabilities
    
    cluster_proba = calculate_probabilities(distances)
    max_probas = np.max(cluster_proba, axis=1)
    
    # 計算不確定度：使用熵的概念
    def calculate_uncertainty(probabilities):
        # 避免log(0)
        prob_safe = np.clip(probabilities, 1e-10, 1.0)
        entropy = -np.sum(probabilities * np.log(prob_safe), axis=1)
        # 正規化entropy到[0,1]範圍（最大entropy為log(3)）
        max_entropy = np.log(3)
        normalized_entropy = entropy / max_entropy
        return normalized_entropy
    
    uncertainties = calculate_uncertainty(cluster_proba)
    
    # 分析各集群
    for cluster_id in range(3):
        cluster_districts = [districts[i] for i in range(len(districts)) if cluster_labels[i] == cluster_id]
        print(f"  集群 {cluster_id}: {len(cluster_districts)} 個行政區 - {', '.join(cluster_districts)}")
    
    # 🔧 基於潛力綜合分數進行標籤分配
    
    # 1. 計算潛力綜合分數
    potential_scores = calculate_potential_score(df)
    
    # 2. 分配行政區標籤（使用分位數分箱）
    district_labels_df = assign_district_labels(potential_scores)
    
    # 3. 使用多數決分配集群標籤
    potential_mapping = assign_cluster_labels_by_majority(cluster_labels, district_labels_df)
    
    # 4. 評估語義一致性
    consistency_score = evaluate_semantic_consistency(cluster_labels, district_labels_df)
    print(f"✅ 語義一致性: {consistency_score:.3f}")
    
    # 創建結果DataFrame
    results_df = pd.DataFrame({
        '行政區': districts,
        '集群編號': cluster_labels,
        '潛力等級': [potential_mapping[label] for label in cluster_labels],
        'tsne_x': X_tsne[:, 0],
        'tsne_y': X_tsne[:, 1],
        '分群機率': max_probas,
        '不確定度': uncertainties
    })
    
    # 添加潛力分數信息
    potential_score_dict = dict(zip(district_labels_df['區域別'], district_labels_df['potential_score']))
    results_df['潛力分數'] = [potential_score_dict[district] for district in districts]
    
    # 顯示性能總結
    print(f"\n🎯 性能指標總結:")
    print(f"  - Silhouette分數: {silhouette:.3f} ({'優秀' if silhouette > 0.7 else '良好' if silhouette > 0.4 else '一般'})")
    print(f"  - 語義一致性: {consistency_score:.3f} ({'優秀' if consistency_score > 0.7 else '良好' if consistency_score > 0.6 else '一般'})")
    
    # 顯示前10筆檢查
    print(f"\n📋 前10筆結果檢查:")
    print(results_df[['行政區', '潛力等級', '集群編號', '潛力分數']].head(10).to_string(index=False))
    
    return results_df, X_tsne, cluster_labels

def save_results(results_df):
    """保存分群結果"""
    print("\n💾 保存分群結果...")
    
    # 確保輸出目錄存在
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # 保存分群結果CSV
    output_path = os.path.join(OUTPUT_DIR, 'clustering_results.csv')
    results_df.to_csv(output_path, index=False, encoding='utf-8')
    print(f"✅ 分群結果已保存: {output_path}")
    
    return output_path

def create_visualization(results_df, X_tsne):
    """創建分群視覺化"""
    print("\n🎨 創建分群視覺化...")
    
    # 顏色映射
    level_colors = {'高潛力': '#e74c3c', '中潛力': '#f39c12', '低潛力': '#3498db'}
    
    # 創建畫布
    plt.figure(figsize=(12, 8))
    
    # 繪製散點圖
    for level in ['高潛力', '中潛力', '低潛力']:
        level_data = results_df[results_df['潛力等級'] == level]
        if len(level_data) > 0:
            plt.scatter(
                level_data['tsne_x'], level_data['tsne_y'],
                c=level_colors.get(level, 'gray'),
                label=level,
                s=150,
                alpha=0.8,
                edgecolors='white',
                linewidth=2
            )
    
    # 標註行政區名稱
    for i, row in results_df.iterrows():
        plt.annotate(
            row['行政區'],
            (row['tsne_x'], row['tsne_y']),
            xytext=(5, 5),
            textcoords='offset points',
            fontsize=11,
            ha='left',
            fontweight='bold'
        )
    
    # 添加標題和圖例
    plt.title('桃園市行政區分群結果 (最佳方法: 3核心特徵)', fontsize=16, fontweight='bold', pad=20)
    plt.legend(fontsize=12, title='發展潛力等級', title_fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.xlabel('t-SNE 維度 1', fontsize=12)
    plt.ylabel('t-SNE 維度 2', fontsize=12)
    
    # 保存圖片
    viz_path = os.path.join(OUTPUT_DIR, 'clustering_visualization.png')
    plt.savefig(viz_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ 分群視覺化已保存: {viz_path}")
    
    return viz_path

def main():
    """主函數"""
    print("="*60)
    print("🚀 桃園市行政區分群及標籤 (最佳方法)")
    print("特徵組合: 收入+醫療+第三產業")
    print("目標: Silhouette > 0.6 且 一致性 > 0.7")
    print("="*60)
    
    # 1. 載入數據
    X, districts, features, df = load_data()
    
    # 2. 執行分群
    results_df, X_tsne, cluster_labels = perform_clustering(X, districts, df)
    
    # 3. 保存結果
    output_path = save_results(results_df)
    
    # 4. 創建視覺化
    viz_path = create_visualization(results_df, X_tsne)
    
    print("\n✅ 分群及標籤完成!")
    print(f"  - 分群結果: {output_path}")
    print(f"  - 視覺化圖: {viz_path}")

if __name__ == "__main__":
    main() 