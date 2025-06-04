"""
桃園市行政區分群及標籤

使用t-SNE降維和Ward層次聚類進行分群
輪廓係數>0.7且分群數=3

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
    """載入特徵數據"""
    print("📂 載入特徵數據...")
    
    # 載入特徵數據
    df = pd.read_csv('output/taoyuan_features_enhanced.csv')
    
    # 提取行政區和特徵
    districts = df['區域別'].tolist()
    X = df.drop('區域別', axis=1).values
    
    print(f"✅ 數據載入成功: {len(districts)} 個行政區, {X.shape[1]} 個特徵")
    print(f"  特徵列表: {', '.join(df.columns[1:])}")
    
    return X, districts, df.columns[1:].tolist()

def perform_clustering(X, districts):
    """執行t-SNE降維和Ward層次聚類"""
    print("\n🔍 執行t-SNE降維和Ward層次聚類...")
    
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
    
    # 🔧 修正：基於所得水平進行潛力等級映射
    # 載入特徵數據以獲取所得信息
    df = pd.read_csv('output/taoyuan_features_enhanced.csv')
    income_data = df.set_index('區域別')['所得_median_household_income'].to_dict()
    
    # 計算各集群的平均所得
    cluster_incomes = []
    for cluster_id in range(3):
        cluster_mask = cluster_labels == cluster_id
        cluster_districts_list = [districts[i] for i in range(len(districts)) if cluster_mask[i]]
        cluster_income = np.mean([income_data[district] for district in cluster_districts_list])
        cluster_incomes.append(cluster_income)
        print(f"  集群 {cluster_id} 平均所得: {cluster_income:,.0f} 元")
    
    # 按平均所得排序（降序：高所得=高潛力）
    cluster_order = np.argsort(cluster_incomes)[::-1]  # 降序排列
    potential_mapping = {
        cluster_order[0]: '高潛力',
        cluster_order[1]: '中潛力',
        cluster_order[2]: '低潛力'
    }
    
    print(f"✅ 基於所得的潛力等級映射: {potential_mapping}")
    print(f"  各集群平均所得排序: {[f'{cluster_incomes[i]:,.0f}元' for i in cluster_order]}")
    
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
    level_colors = {'高潛力': 'red', '中潛力': 'orange', '低潛力': 'blue'}
    
    # 創建畫布
    plt.figure(figsize=(10, 8))
    
    # 繪製散點圖
    for level in ['高潛力', '中潛力', '低潛力']:
        level_data = results_df[results_df['潛力等級'] == level]
        if len(level_data) > 0:
            plt.scatter(
                level_data['tsne_x'], level_data['tsne_y'],
                c=level_colors.get(level, 'gray'),
                label=level,
                s=120,
                alpha=0.8
            )
    
    # 標註行政區名稱
    for i, row in results_df.iterrows():
        plt.annotate(
            row['行政區'],
            (row['tsne_x'], row['tsne_y']),
            xytext=(5, 5),
            textcoords='offset points',
            fontsize=10,
            ha='left'
        )
    
    # 添加標題和圖例
    plt.title('桃園市行政區分群結果 (t-SNE降維)', fontsize=14, fontweight='bold')
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    
    # 保存圖片
    viz_path = os.path.join(OUTPUT_DIR, 'clustering_visualization.png')
    plt.savefig(viz_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ 分群視覺化已保存: {viz_path}")
    
    return viz_path

def main():
    """主函數"""
    print("="*60)
    print("🚀 桃園市行政區分群及標籤")
    print("="*60)
    
    # 1. 載入數據
    X, districts, features = load_data()
    
    # 2. 執行分群
    results_df, X_tsne, cluster_labels = perform_clustering(X, districts)
    
    # 3. 保存結果
    output_path = save_results(results_df)
    
    # 4. 創建視覺化
    viz_path = create_visualization(results_df, X_tsne)
    
    print("\n✅ 分群及標籤完成!")
    print(f"  - 分群結果: {output_path}")
    print(f"  - 視覺化圖: {viz_path}")

if __name__ == "__main__":
    main() 