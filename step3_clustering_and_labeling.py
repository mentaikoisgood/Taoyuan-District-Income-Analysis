"""
桃園市行政區分群及標籤

直接對標準化特徵進行Ward層次聚類
根據cluster平均潛力分數分配高/中/低標籤

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

def calculate_composite_score(districts):
    """計算綜合發展潛力分數"""
    print("\n📊 計算綜合發展潛力分數...")
    
    # 載入特徵數據
    df = pd.read_csv('output/taoyuan_features_enhanced.csv')
    df = df.set_index('區域別')
    
    # 特徵權重設定（基於領域知識）
    weights = {
        '所得_median_household_income': 0.35,      # 經濟水平 - 最重要
        'tertiary_industry_ratio': 0.25,          # 產業結構 
        'medical_index': 0.20,                    # 醫療資源
        '人口_working_age_ratio': 0.15,           # 人力資源
        '商業_hhi_index': 0.05                    # 商業集中度
    }
    
    print(f"  特徵權重設定: {weights}")
    
    # 標準化數據
    normalized_data = df.copy()
    for feature in weights.keys():
        if feature in df.columns:
            min_val = df[feature].min()
            max_val = df[feature].max()
            normalized_data[feature] = (df[feature] - min_val) / (max_val - min_val)
    
    # 計算加權綜合分數
    composite_scores = {}
    for district in districts:
        score = 0
        for feature, weight in weights.items():
            if feature in normalized_data.columns:
                score += normalized_data.loc[district, feature] * weight
        composite_scores[district] = score
    
    print(f"✅ 綜合分數計算完成")
    for district, score in sorted(composite_scores.items(), key=lambda x: x[1], reverse=True):
        print(f"  {district}: {score:.3f}")
    
    return composite_scores

def perform_clustering(X, districts):
    """對標準化特徵進行Ward層次聚類"""
    print("\n🔍 對標準化特徵進行Ward層次聚類...")
    
    # 標準化特徵
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    print(f"✅ 特徵標準化完成: {X.shape}")
    
    # Ward層次聚類（直接對標準化特徵進行）
    ward = AgglomerativeClustering(n_clusters=3, linkage='ward')
    cluster_labels = ward.fit_predict(X_scaled)
    
    # 計算輪廓係數
    silhouette = silhouette_score(X_scaled, cluster_labels)
    print(f"✅ Ward層次聚類完成: 3個集群, 輪廓係數={silhouette:.3f}")
    
    # 計算綜合潛力分數
    composite_scores = calculate_composite_score(districts)
    
    # 分析各集群及其平均潛力分數
    cluster_info = {}
    for cluster_id in range(3):
        cluster_mask = cluster_labels == cluster_id
        cluster_districts = [districts[i] for i in range(len(districts)) if cluster_mask[i]]
        cluster_scores = [composite_scores[district] for district in cluster_districts]
        avg_score = np.mean(cluster_scores)
        
        cluster_info[cluster_id] = {
            'districts': cluster_districts,
            'avg_score': avg_score,
            'scores': cluster_scores
        }
        
        print(f"  集群 {cluster_id}: {len(cluster_districts)} 個行政區")
        print(f"    行政區: {', '.join(cluster_districts)}")
        print(f"    平均潛力分數: {avg_score:.3f}")
    
    # 根據cluster平均分數分配潛力等級
    print(f"\n🎯 根據cluster平均潛力分數分配等級...")
    
    # 按平均分數排序cluster（降序：高分數=高潛力）
    sorted_clusters = sorted(cluster_info.items(), key=lambda x: x[1]['avg_score'], reverse=True)
    
    potential_mapping = {
        sorted_clusters[0][0]: '高潛力',  # 最高平均分數
        sorted_clusters[1][0]: '中潛力',  # 中等平均分數
        sorted_clusters[2][0]: '低潛力'   # 最低平均分數
    }
    
    print(f"✅ Cluster潛力等級分配:")
    for cluster_id, level in potential_mapping.items():
        info = cluster_info[cluster_id]
        print(f"  集群 {cluster_id} → {level}")
        print(f"    平均分數: {info['avg_score']:.3f}")
        print(f"    行政區: {', '.join(info['districts'])}")
    
    # 為t-SNE視覺化計算座標（僅用於視覺化）
    print(f"\n🎨 計算t-SNE座標用於視覺化...")
    tsne = TSNE(n_components=2, random_state=RANDOM_STATE, perplexity=3, max_iter=1000)
    X_tsne = tsne.fit_transform(X_scaled)
    print(f"✅ t-SNE視覺化座標計算完成")
    
    # 計算分群機率和不確定度
    cluster_centers = np.array([X_scaled[cluster_labels == i].mean(axis=0) for i in range(3)])
    distances = pairwise_distances(X_scaled, cluster_centers)
    
    def calculate_probabilities(distances):
        min_distances = distances.min(axis=1, keepdims=True)
        relative_distances = distances / (min_distances + 1e-6)
        temperature = 3.0
        exp_values = np.exp(-relative_distances / temperature)
        probabilities = exp_values / exp_values.sum(axis=1, keepdims=True)
        return probabilities
    
    cluster_proba = calculate_probabilities(distances)
    max_probas = np.max(cluster_proba, axis=1)
    
    def calculate_uncertainty(probabilities):
        prob_safe = np.clip(probabilities, 1e-10, 1.0)
        entropy = -np.sum(probabilities * np.log(prob_safe), axis=1)
        max_entropy = np.log(3)
        normalized_entropy = entropy / max_entropy
        return normalized_entropy
    
    uncertainties = calculate_uncertainty(cluster_proba)
    
    # 創建結果DataFrame
    results_df = pd.DataFrame({
        '行政區': districts,
        '集群編號': cluster_labels,
        '潛力等級': [potential_mapping[label] for label in cluster_labels],
        '綜合分數': [composite_scores[district] for district in districts],
        'tsne_x': X_tsne[:, 0],
        'tsne_y': X_tsne[:, 1],
        '分群機率': max_probas,
        '不確定度': uncertainties
    })
    
    # 輸出統計信息
    print(f"\n📈 最終分類統計:")
    for level in ['高潛力', '中潛力', '低潛力']:
        level_data = results_df[results_df['潛力等級'] == level]
        count = len(level_data)
        avg_score = level_data['綜合分數'].mean()
        districts_list = level_data['行政區'].tolist()
        print(f"  {level}: {count}個行政區, 平均分數={avg_score:.3f}")
        print(f"    行政區: {', '.join(districts_list)}")
    
    return results_df, X_tsne, cluster_labels, silhouette

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

def create_visualization(results_df, X_tsne, silhouette):
    """創建分群視覺化"""
    print("\n🎨 創建分群視覺化...")
    
    # 顏色映射
    level_colors = {'高潛力': 'red', '中潛力': 'orange', '低潛力': 'blue'}
    
    # 創建畫布
    plt.figure(figsize=(15, 6))
    
    # 子圖1: t-SNE聚類結果（僅用於視覺化）
    plt.subplot(1, 3, 1)
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
            fontsize=8,
            ha='left'
        )
    
    plt.title('t-SNE視覺化\n(基於Ward聚類結果)', fontsize=12, fontweight='bold')
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    
    # 子圖2: 綜合分數分布
    plt.subplot(1, 3, 2)
    for level in ['高潛力', '中潛力', '低潛力']:
        level_data = results_df[results_df['潛力等級'] == level]
        
        plt.hist(level_data['綜合分數'], 
                alpha=0.7, 
                color=level_colors[level], 
                label=level,
                bins=5)
    
    plt.title('綜合分數分布\n(基於Cluster平均分數分類)', fontsize=12, fontweight='bold')
    plt.xlabel('綜合發展潛力分數')
    plt.ylabel('行政區數量')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 子圖3: Cluster統計
    plt.subplot(1, 3, 3)
    cluster_stats = results_df.groupby(['集群編號', '潛力等級']).agg({
        '綜合分數': ['count', 'mean']
    }).round(3)
    
    clusters = results_df['集群編號'].unique()
    levels = []
    counts = []
    scores = []
    colors = []
    
    for cluster_id in sorted(clusters):
        cluster_data = results_df[results_df['集群編號'] == cluster_id]
        level = cluster_data['潛力等級'].iloc[0]
        count = len(cluster_data)
        score = cluster_data['綜合分數'].mean()
        
        levels.append(f"Cluster {cluster_id}\n({level})")
        counts.append(count)
        scores.append(score)
        colors.append(level_colors[level])
    
    bars = plt.bar(levels, scores, color=colors, alpha=0.7)
    
    # 在柱狀圖上標註數量
    for i, (bar, count) in enumerate(zip(bars, counts)):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{count}區', ha='center', va='bottom', fontsize=10)
    
    plt.title(f'各Cluster平均潛力分數\n(輪廓係數: {silhouette:.3f})', fontsize=12, fontweight='bold')
    plt.ylabel('平均綜合分數')
    plt.xticks(rotation=45)
    plt.grid(True, alpha=0.3, axis='y')
    
    # 保存圖片
    viz_path = os.path.join(OUTPUT_DIR, 'clustering_visualization.png')
    plt.tight_layout()
    plt.savefig(viz_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ 分群視覺化已保存: {viz_path}")
    
    return viz_path

def main():
    """主函數"""
    print("="*60)
    print("🚀 桃園市行政區分群及標籤 (Ward聚類 + Cluster平均分數分類)")
    print("="*60)
    
    # 1. 載入數據
    X, districts, features = load_data()
    
    # 2. 執行分群
    results_df, X_tsne, cluster_labels, silhouette = perform_clustering(X, districts)
    
    # 3. 保存結果
    output_path = save_results(results_df)
    
    # 4. 創建視覺化
    viz_path = create_visualization(results_df, X_tsne, silhouette)
    
    print("\n✅ 分群及標籤完成!")
    print(f"  - 分群結果: {output_path}")
    print(f"  - 視覺化圖: {viz_path}")

if __name__ == "__main__":
    main() 