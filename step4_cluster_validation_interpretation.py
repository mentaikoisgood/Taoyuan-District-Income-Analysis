"""
STEP 4 - 群集驗證與解讀

包含：
1. 群內/群間距離分析
2. Silhouette分數計算
3. 箱型圖和雷達圖視覺化
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score, silhouette_samples
from sklearn.metrics import pairwise_distances
import warnings
import os
import matplotlib

warnings.filterwarnings('ignore')

# 設定中文字體
matplotlib.rcParams['font.family'] = ['sans-serif']
matplotlib.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False

OUTPUT_DIR = 'output'
RANDOM_STATE = 42

def load_data_and_cluster():
    """載入數據並執行聚類"""
    print("📂 載入數據並執行聚類...")
    
    # 載入特徵數據
    df = pd.read_csv('output/taoyuan_features_enhanced.csv')
    districts = df['區域別'].tolist()
    X = df.drop('區域別', axis=1).values
    feature_names = df.columns[1:].tolist()
    
    # t-SNE降維
    tsne = TSNE(n_components=2, random_state=RANDOM_STATE, perplexity=3, max_iter=1000)
    X_tsne = tsne.fit_transform(X)
    
    # Ward層次聚類
    ward = AgglomerativeClustering(n_clusters=3, linkage='ward')
    cluster_labels = ward.fit_predict(X_tsne)
    
    return X, X_tsne, cluster_labels, districts, feature_names, df

def analyze_cluster_distances(X_tsne, cluster_labels, districts):
    """分析群內/群間距離"""
    print("\n📐 群內/群間距離分析")
    print("="*40)
    
    # 計算群內距離（Within-cluster distance）
    within_cluster_distances = []
    for cluster_id in range(3):
        cluster_mask = cluster_labels == cluster_id
        cluster_points = X_tsne[cluster_mask]
        
        if len(cluster_points) > 1:
            # 計算群內點對距離的平均值
            intra_distances = pairwise_distances(cluster_points)
            # 取上三角矩陣（避免重複計算和對角線）
            upper_triangle = np.triu(intra_distances, k=1)
            within_dist = upper_triangle[upper_triangle > 0].mean()
        else:
            within_dist = 0
        
        within_cluster_distances.append(within_dist)
        cluster_districts = [districts[i] for i in range(len(districts)) if cluster_labels[i] == cluster_id]
        print(f"  集群 {cluster_id}: 群內平均距離 = {within_dist:.2f}")
        print(f"    包含行政區: {', '.join(cluster_districts)}")
    
    # 計算群間距離（Between-cluster distance）
    print(f"\n群間距離:")
    cluster_centers = []
    for cluster_id in range(3):
        cluster_mask = cluster_labels == cluster_id
        center = X_tsne[cluster_mask].mean(axis=0)
        cluster_centers.append(center)
    
    between_distances = pairwise_distances(cluster_centers)
    for i in range(3):
        for j in range(i+1, 3):
            dist = between_distances[i, j]
            print(f"  集群 {i} ↔ 集群 {j}: {dist:.2f}")
    
    # 計算 Davies-Bouldin 指數
    db_scores = []
    for i in range(3):
        max_ratio = 0
        for j in range(3):
            if i != j:
                ratio = (within_cluster_distances[i] + within_cluster_distances[j]) / between_distances[i, j]
                max_ratio = max(max_ratio, ratio)
        db_scores.append(max_ratio)
    
    db_index = np.mean(db_scores)
    print(f"\n📊 Davies-Bouldin 指數: {db_index:.3f} (越小越好)")
    
    return within_cluster_distances, between_distances

def analyze_silhouette_scores(X_tsne, cluster_labels, districts):
    """詳細的Silhouette分析"""
    print("\n🎯 Silhouette 分數分析")
    print("="*40)
    
    # 整體Silhouette分數
    overall_silhouette = silhouette_score(X_tsne, cluster_labels)
    print(f"整體 Silhouette 分數: {overall_silhouette:.3f}")
    
    # 各點的Silhouette分數
    sample_silhouettes = silhouette_samples(X_tsne, cluster_labels)
    
    print(f"\n各行政區 Silhouette 分數:")
    for i, (district, score) in enumerate(zip(districts, sample_silhouettes)):
        cluster = cluster_labels[i]
        print(f"  {district} (集群{cluster}): {score:.3f}")
    
    # 各集群平均Silhouette分數
    print(f"\n各集群平均 Silhouette 分數:")
    for cluster_id in range(3):
        cluster_mask = cluster_labels == cluster_id
        cluster_silhouettes = sample_silhouettes[cluster_mask]
        avg_silhouette = cluster_silhouettes.mean()
        print(f"  集群 {cluster_id}: {avg_silhouette:.3f}")
    
    return sample_silhouettes

def create_boxplots(df, cluster_labels, feature_names):
    """創建箱型圖比較各群特徵分布"""
    print("\n📊 創建箱型圖...")
    
    # 準備數據
    plot_data = df.copy()
    plot_data['集群'] = cluster_labels
    
    # 創建子圖
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    for i, feature in enumerate(feature_names):
        sns.boxplot(data=plot_data, x='集群', y=feature, ax=axes[i])
        axes[i].set_title(f'{feature}', fontsize=10)
        axes[i].tick_params(axis='both', labelsize=8)
    
    # 隱藏多餘的子圖
    if len(feature_names) < len(axes):
        axes[-1].set_visible(False)
    
    plt.tight_layout()
    plt.suptitle('各集群特徵分布箱型圖', fontsize=14, y=1.02)
    
    # 保存圖片
    boxplot_path = os.path.join(OUTPUT_DIR, 'cluster_boxplots.png')
    plt.savefig(boxplot_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ 箱型圖已保存: {boxplot_path}")
    
    return boxplot_path

def create_radar_chart(X, cluster_labels, feature_names, districts):
    """創建雷達圖展示各群特徵平均值"""
    print("\n🕸️ 創建雷達圖...")
    
    # 標準化特徵值到0-1範圍
    from sklearn.preprocessing import MinMaxScaler
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)
    
    # 計算各群特徵平均值
    cluster_means = []
    for cluster_id in range(3):
        cluster_mask = cluster_labels == cluster_id
        means = X_scaled[cluster_mask].mean(axis=0)
        cluster_means.append(means)
    
    # 設定雷達圖
    angles = np.linspace(0, 2 * np.pi, len(feature_names), endpoint=False).tolist()
    angles += angles[:1]  # 閉合圖形
    
    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))
    
    colors = ['red', 'blue', 'green']
    labels = ['高潛力', '低潛力', '中潛力']  # 根據我們的映射
    
    for cluster_id, (means, color, label) in enumerate(zip(cluster_means, colors, labels)):
        values = means.tolist()
        values += values[:1]  # 閉合圖形
        
        ax.plot(angles, values, 'o-', linewidth=2, label=f'集群{cluster_id} ({label})', color=color)
        ax.fill(angles, values, alpha=0.25, color=color)
    
    # 設定標籤
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(feature_names, fontsize=10)
    ax.set_ylim(0, 1)
    ax.set_title('各集群特徵雷達圖', fontsize=14, pad=20)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
    ax.grid(True)
    
    # 保存圖片
    radar_path = os.path.join(OUTPUT_DIR, 'cluster_radar_chart.png')
    plt.savefig(radar_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ 雷達圖已保存: {radar_path}")
    
    return radar_path

def main():
    """主函數"""
    print("="*60)
    print("🚀 STEP 4 - 群集驗證與解讀")
    print("="*60)
    
    # 確保輸出目錄存在
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # 1. 載入數據並執行聚類
    X, X_tsne, cluster_labels, districts, feature_names, df = load_data_and_cluster()
    
    # 2. 距離分析
    within_distances, between_distances = analyze_cluster_distances(X_tsne, cluster_labels, districts)
    
    # 3. Silhouette分析
    silhouette_scores = analyze_silhouette_scores(X_tsne, cluster_labels, districts)
    
    # 4. 視覺化分析
    boxplot_path = create_boxplots(df, cluster_labels, feature_names)
    radar_path = create_radar_chart(X, cluster_labels, feature_names, districts)
    
    print(f"\n✅ 群集驗證與解讀完成!")
    print(f"  - 箱型圖: {boxplot_path}")
    print(f"  - 雷達圖: {radar_path}")
    print(f"  - 整體 Silhouette 分數: {silhouette_score(X_tsne, cluster_labels):.3f}")

if __name__ == "__main__":
    main() 