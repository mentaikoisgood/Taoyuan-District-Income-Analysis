"""
STEP 4 - 群集驗證與解讀 (綜合評估版)

包含：
1. 群內/群間距離分析
2. Silhouette分數計算 (幾何質量)
3. 潛力分數一致性評估 (語義質量)
4. 箱型圖和雷達圖視覺化
5. 綜合評估報告
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score, silhouette_samples
from sklearn.metrics import pairwise_distances
from scipy import stats
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
    
    # 🔧 使用與STEP3相同的最佳特征組合
    optimal_features = [
        '所得_median_household_income',  # 經濟水平指標
        'medical_index',                # 醫療服務指標
        'tertiary_industry_ratio'       # 產業結構指標
    ]
    
    X = df[optimal_features].values
    feature_names = optimal_features
    
    print(f"✅ 使用最佳特征組合: {len(districts)} 個行政區, {X.shape[1]} 個特征")
    print(f"  特征列表: {', '.join(optimal_features)}")
    
    # t-SNE降維
    tsne = TSNE(n_components=2, random_state=RANDOM_STATE, perplexity=3, max_iter=1000)
    X_tsne = tsne.fit_transform(X)
    
    # Ward層次聚類
    ward = AgglomerativeClustering(n_clusters=3, linkage='ward')
    cluster_labels = ward.fit_predict(X_tsne)
    
    # 載入聚類結果（包含潛力等級）
    cluster_results = pd.read_csv('output/clustering_results.csv')
    potential_labels = []
    for district in districts:
        row = cluster_results[cluster_results['行政區'] == district]
        potential_labels.append(row['潛力等級'].iloc[0])
    
    return X, X_tsne, cluster_labels, districts, feature_names, df, potential_labels, cluster_results

def analyze_cluster_distances(X_tsne, cluster_labels, districts):
    """分析群內/群間距離"""
    print("\n📐 群內/群間距離分析 (幾何質量)")
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
    print(f"\n📊 Davies-Bouldin 指數: {db_index:.3f} (越小越好，<1.5為良好)")
    
    return within_cluster_distances, between_distances

def analyze_silhouette_scores(X_tsne, cluster_labels, districts):
    """詳細的Silhouette分析"""
    print("\n🎯 Silhouette 分數分析 (幾何質量)")
    print("="*40)
    
    # 整體Silhouette分數
    overall_silhouette = silhouette_score(X_tsne, cluster_labels)
    print(f"整體 Silhouette 分數: {overall_silhouette:.3f}")
    
    # 解釋分數
    if overall_silhouette > 0.7:
        quality = "優秀"
    elif overall_silhouette > 0.4:
        quality = "良好"  
    elif overall_silhouette > 0.2:
        quality = "一般"
    else:
        quality = "較差"
    print(f"幾何聚類質量: {quality}")
    
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

def evaluate_potential_consistency(cluster_results):
    """評估潛力分數與聚類結果的一致性"""
    print("\n🧮 潛力分數一致性評估 (語義質量)")
    print("="*40)
    
    # 計算各集群的潛力分數統計
    cluster_stats = {}
    for cluster_id in range(3):
        cluster_data = cluster_results[cluster_results['集群編號'] == cluster_id]
        potential_scores = cluster_data['潛力分數'].values
        
        cluster_stats[cluster_id] = {
            'count': len(cluster_data),
            'districts': cluster_data['行政區'].tolist(),
            'potential_level': cluster_data['潛力等級'].iloc[0],
            'mean_score': potential_scores.mean(),
            'std_score': potential_scores.std(),
            'min_score': potential_scores.min(),
            'max_score': potential_scores.max()
        }
        
        print(f"集群 {cluster_id} ({cluster_stats[cluster_id]['potential_level']}):")
        print(f"  行政區: {', '.join(cluster_stats[cluster_id]['districts'])}")
        print(f"  潛力分數: {cluster_stats[cluster_id]['mean_score']:.3f} ± {cluster_stats[cluster_id]['std_score']:.3f}")
        print(f"  分數範圍: [{cluster_stats[cluster_id]['min_score']:.3f}, {cluster_stats[cluster_id]['max_score']:.3f}]")
    
    # 計算集群間分離度 (潛力分數)
    print(f"\n集群間潛力分數分離度:")
    cluster_means = [cluster_stats[i]['mean_score'] for i in range(3)]
    separation_score = max(cluster_means) - min(cluster_means)
    print(f"  分數範圍跨度: {separation_score:.3f}")
    
    # 檢查集群內一致性
    print(f"\n集群內一致性檢查:")
    consistency_scores = []
    for cluster_id in range(3):
        std = cluster_stats[cluster_id]['std_score']
        consistency = 1 / (1 + std)  # 標準差越小，一致性越高
        consistency_scores.append(consistency)
        print(f"  集群 {cluster_id}: 一致性分數 = {consistency:.3f} (越接近1越好)")
    
    overall_consistency = np.mean(consistency_scores)
    print(f"\n🎯 總體語義一致性: {overall_consistency:.3f}")
    
    return cluster_stats, overall_consistency

def calculate_comprehensive_score(silhouette_score, consistency_score, db_index):
    """計算綜合評估分數"""
    print("\n🏆 綜合評估分數")
    print("="*40)
    
    # 正規化各指標 (0-1範圍)
    # Silhouette: 0.463 -> 約 0.6 (在 0-1 範圍內)
    norm_silhouette = max(0, min(1, (silhouette_score + 1) / 2))  # 從 [-1,1] 轉換到 [0,1]
    
    # 一致性: 已經在 0-1 範圍
    norm_consistency = consistency_score
    
    # Davies-Bouldin: 越小越好，1.112 -> 約 0.47 (1/(1+DB))
    norm_db = 1 / (1 + db_index)
    
    print(f"標準化指標:")
    print(f"  幾何質量 (Silhouette): {norm_silhouette:.3f}")
    print(f"  語義一致性: {norm_consistency:.3f}")
    print(f"  分離度 (Davies-Bouldin): {norm_db:.3f}")
    
    # 計算加權綜合分數
    weights = [0.3, 0.5, 0.2]  # 更重視語義一致性
    comprehensive_score = (
        weights[0] * norm_silhouette + 
        weights[1] * norm_consistency + 
        weights[2] * norm_db
    )
    
    print(f"\n🎯 綜合評估分數: {comprehensive_score:.3f}")
    
    if comprehensive_score > 0.8:
        grade = "優秀"
    elif comprehensive_score > 0.6:
        grade = "良好"
    elif comprehensive_score > 0.4:
        grade = "一般"
    else:
        grade = "需改進"
    
    print(f"綜合評級: {grade}")
    
    return comprehensive_score

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

def create_radar_chart(X, cluster_labels, feature_names, districts, potential_labels):
    """創建雷達圖展示各群特徵平均值"""
    print("\n🕸️ 創建雷達圖...")
    
    # 標準化特徵值到0-1範圍
    from sklearn.preprocessing import MinMaxScaler
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)
    
    # 計算各群特徵平均值
    cluster_means = []
    cluster_labels_unique = []
    for cluster_id in range(3):
        cluster_mask = cluster_labels == cluster_id
        means = X_scaled[cluster_mask].mean(axis=0)
        cluster_means.append(means)
        
        # 獲取該集群的潛力等級
        cluster_districts = [districts[i] for i in range(len(districts)) if cluster_labels[i] == cluster_id]
        cluster_potential = potential_labels[cluster_labels.tolist().index(cluster_id)]
        cluster_labels_unique.append(f'集群{cluster_id} ({cluster_potential})')
    
    # 設定雷達圖
    angles = np.linspace(0, 2 * np.pi, len(feature_names), endpoint=False).tolist()
    angles += angles[:1]  # 閉合圖形
    
    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))
    
    colors = ['#e74c3c', '#3498db', '#f39c12']  # 紅、藍、橙
    
    for cluster_id, (means, color, label) in enumerate(zip(cluster_means, colors, cluster_labels_unique)):
        values = means.tolist()
        values += values[:1]  # 閉合圖形
        
        ax.plot(angles, values, 'o-', linewidth=2, label=label, color=color)
        ax.fill(angles, values, alpha=0.25, color=color)
    
    # 設定標籤
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(feature_names, fontsize=10)
    ax.set_ylim(0, 1)
    ax.set_title('各集群特徵雷達圖 (基於潛力綜合分數)', fontsize=14, pad=20)
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
    print("🚀 STEP 4 - 群集驗證與解讀 (綜合評估版)")
    print("="*60)
    
    # 確保輸出目錄存在
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # 1. 載入數據並執行聚類
    X, X_tsne, cluster_labels, districts, feature_names, df, potential_labels, cluster_results = load_data_and_cluster()
    
    # 2. 幾何質量分析
    within_distances, between_distances = analyze_cluster_distances(X_tsne, cluster_labels, districts)
    silhouette_scores = analyze_silhouette_scores(X_tsne, cluster_labels, districts)
    overall_silhouette = silhouette_score(X_tsne, cluster_labels)
    
    # 計算Davies-Bouldin指數
    db_scores = []
    for i in range(3):
        max_ratio = 0
        for j in range(3):
            if i != j:
                ratio = (within_distances[i] + within_distances[j]) / between_distances[i, j]
                max_ratio = max(max_ratio, ratio)
        db_scores.append(max_ratio)
    db_index = np.mean(db_scores)
    
    # 3. 語義質量分析
    cluster_stats, consistency_score = evaluate_potential_consistency(cluster_results)
    
    # 4. 綜合評估
    comprehensive_score = calculate_comprehensive_score(overall_silhouette, consistency_score, db_index)
    
    # 5. 視覺化分析
    boxplot_path = create_boxplots(df, cluster_labels, feature_names)
    radar_path = create_radar_chart(X, cluster_labels, feature_names, districts, potential_labels)
    
    print(f"\n✅ 群集驗證與解讀完成!")
    print(f"  - 箱型圖: {boxplot_path}")
    print(f"  - 雷達圖: {radar_path}")
    print(f"\n📊 關鍵指標總結:")
    print(f"  - 幾何 Silhouette 分數: {overall_silhouette:.3f} (良好)")
    print(f"  - 語義一致性分數: {consistency_score:.3f}")
    print(f"  - Davies-Bouldin 指數: {db_index:.3f}")
    print(f"  - 綜合評估分數: {comprehensive_score:.3f}")

if __name__ == "__main__":
    main() 