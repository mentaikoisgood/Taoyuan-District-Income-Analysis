"""
实现最佳聚类方法
特征组合2: 所得_median_household_income, medical_index, tertiary_industry_ratio
结果: Silhouette=0.613, 一致性=0.836

这个组合选择了最具代表性的三个特征：
1. 经济指标：家庭收入中位数
2. 服务指标：医疗服务指数  
3. 产业指标：第三产业比例
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
from scipy import stats
import warnings
import os
import matplotlib

warnings.filterwarnings('ignore')

# 设定中文字体
matplotlib.rcParams['font.family'] = ['sans-serif']
matplotlib.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False

OUTPUT_DIR = 'output'
RANDOM_STATE = 42

def load_and_prepare_data():
    """载入并准备最佳特征组合数据"""
    print("📂 载入数据并准备最佳特征组合...")
    
    # 载入原始数据
    df = pd.read_csv('output/taoyuan_features_enhanced.csv')
    districts = df['區域別'].tolist()
    
    # 最佳特征组合2: 3个核心特征
    optimal_features = [
        '所得_median_household_income',  # 经济水平
        'medical_index',                # 医疗服务
        'tertiary_industry_ratio'       # 产业结构
    ]
    
    print(f"✅ 最佳特征组合: {', '.join(optimal_features)}")
    
    # 提取特征数据
    X_optimal = df[optimal_features].values
    
    print(f"数据维度: {X_optimal.shape}")
    print(f"行政区数量: {len(districts)}")
    
    return df, X_optimal, districts, optimal_features

def calculate_potential_score_optimal(df):
    """使用最佳特征计算潜力综合分数"""
    print("\n📊 计算潜力综合分数 (最佳特征)...")
    
    # 潜力指标（与聚类特征一致以确保一致性）
    potential_indicators = [
        '所得_median_household_income',
        'medical_index', 
        'tertiary_industry_ratio',
        '人口_working_age_ratio'  # 额外添加人口结构指标
    ]
    
    # 检查指标是否存在
    available_indicators = [col for col in potential_indicators if col in df.columns]
    print(f"  使用潜力指标: {', '.join(available_indicators)}")
    
    # 对选定指标进行z-score标准化
    z_scores = pd.DataFrame()
    z_scores['區域別'] = df['區域別']
    
    for indicator in available_indicators:
        z_score = stats.zscore(df[indicator])
        z_scores[f'{indicator}_zscore'] = z_score
        print(f"  {indicator}: 平均={df[indicator].mean():.2f}, 标准差={df[indicator].std():.2f}")
    
    # 计算潜力综合分数（z-score的平均）
    z_score_cols = [col for col in z_scores.columns if col.endswith('_zscore')]
    z_scores['potential_score'] = z_scores[z_score_cols].mean(axis=1)
    
    print(f"✅ 潜力综合分数计算完成")
    print(f"  分数范围: {z_scores['potential_score'].min():.3f} ~ {z_scores['potential_score'].max():.3f}")
    
    return z_scores

def assign_district_labels_optimal(potential_scores):
    """分配行政区标签"""
    print("\n🏷️ 分配行政区发展潜力标签...")
    
    # 使用qcut按分位数切成三等份
    district_labels = pd.qcut(
        potential_scores['potential_score'], 
        q=3, 
        labels=['低潛力', '中潛力', '高潛力']
    )
    
    potential_scores['district_label'] = district_labels
    
    # 显示分组结果
    for label in ['低潛力', '中潛力', '高潛力']:
        districts_in_group = potential_scores[potential_scores['district_label'] == label]['區域別'].tolist()
        score_range = potential_scores[potential_scores['district_label'] == label]['potential_score']
        print(f"  {label}: {len(districts_in_group)} 个行政区")
        print(f"    行政区: {', '.join(districts_in_group)}")
        print(f"    潜力分数范围: {score_range.min():.3f} ~ {score_range.max():.3f}")
    
    return potential_scores

def perform_optimal_clustering(X_optimal, districts, df):
    """执行最佳聚类方法"""
    print("\n🔍 执行最佳聚类方法...")
    print("方法: 3核心特征 + t-SNE降维 + Ward层次聚类")
    
    # t-SNE降维（使用最佳特征）
    tsne = TSNE(n_components=2, random_state=RANDOM_STATE, perplexity=3, max_iter=1000)
    X_tsne = tsne.fit_transform(X_optimal)
    print(f"✅ t-SNE降维完成: {X_optimal.shape} → {X_tsne.shape}")
    
    # Ward层次聚类
    ward = AgglomerativeClustering(n_clusters=3, linkage='ward')
    cluster_labels = ward.fit_predict(X_tsne)
    
    # 计算Silhouette分数
    silhouette = silhouette_score(X_tsne, cluster_labels)
    print(f"✅ Ward层次聚类完成: 3个集群, Silhouette分数={silhouette:.3f}")
    
    # 计算潜力分数和语义一致性
    potential_scores = calculate_potential_score_optimal(df)
    district_labels_df = assign_district_labels_optimal(potential_scores)
    
    # 使用多数决分配集群标签
    cluster_potential_mapping = assign_cluster_labels_by_majority(cluster_labels, district_labels_df, districts)
    
    # 计算语义一致性
    consistency_score = evaluate_semantic_consistency(cluster_labels, district_labels_df, districts)
    
    print(f"✅ 语义一致性: {consistency_score:.3f}")
    print(f"🎯 综合表现: Silhouette={silhouette:.3f}, 一致性={consistency_score:.3f}")
    
    # 创建结果DataFrame
    results_df = pd.DataFrame({
        '行政区': districts,
        '集群编号': cluster_labels,
        '潜力等级': [cluster_potential_mapping[label] for label in cluster_labels],
        'tsne_x': X_tsne[:, 0],
        'tsne_y': X_tsne[:, 1]
    })
    
    # 添加潜力分数信息
    potential_score_dict = dict(zip(district_labels_df['區域別'], district_labels_df['potential_score']))
    results_df['潜力分数'] = [potential_score_dict[district] for district in districts]
    
    # 添加原始特征值
    for i, feature in enumerate(['家庭收入中位数', '医疗指数', '第三产业比例']):
        results_df[feature] = X_optimal[:, i]
    
    return results_df, X_tsne, cluster_labels, silhouette, consistency_score

def assign_cluster_labels_by_majority(cluster_labels, district_labels_df, districts):
    """使用多数决分配集群标签"""
    mapping_df = pd.DataFrame({
        'cluster_id': cluster_labels,
        'district_label': district_labels_df['district_label'].values,
        'district': districts
    })
    
    cluster_potential_mapping = {}
    for cluster_id in range(3):
        cluster_data = mapping_df[mapping_df['cluster_id'] == cluster_id]
        label_counts = cluster_data['district_label'].value_counts()
        majority_label = label_counts.index[0]
        cluster_potential_mapping[cluster_id] = majority_label
        
        print(f"  集群 {cluster_id}: {majority_label}")
        print(f"    成员: {', '.join(cluster_data['district'].tolist())}")
    
    return cluster_potential_mapping

def evaluate_semantic_consistency(cluster_labels, district_labels_df, districts):
    """评估语义一致性"""
    mapping_df = pd.DataFrame({
        'cluster_id': cluster_labels,
        'potential_score': district_labels_df['potential_score'].values
    })
    
    # 计算各集群潜力分数统计
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

def create_optimal_visualization(results_df, X_tsne):
    """创建最佳聚类结果的视觉化"""
    print("\n🎨 创建最佳聚类视觉化...")
    
    # 颜色映射
    level_colors = {'高潛力': '#e74c3c', '中潛力': '#f39c12', '低潛力': '#3498db'}
    
    # 创建画布
    plt.figure(figsize=(12, 8))
    
    # 绘制散点图
    for level in ['高潛力', '中潛力', '低潛力']:
        level_data = results_df[results_df['潜力等级'] == level]
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
    
    # 标注行政区名称
    for i, row in results_df.iterrows():
        plt.annotate(
            row['行政区'],
            (row['tsne_x'], row['tsne_y']),
            xytext=(5, 5),
            textcoords='offset points',
            fontsize=11,
            ha='left',
            fontweight='bold'
        )
    
    # 添加标题和图例
    plt.title('桃园市行政区分群结果 (最佳方法: 3核心特征)', fontsize=16, fontweight='bold', pad=20)
    plt.legend(fontsize=12, title='发展潜力等级', title_fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.xlabel('t-SNE 维度 1', fontsize=12)
    plt.ylabel('t-SNE 维度 2', fontsize=12)
    
    # 保存图片
    viz_path = os.path.join(OUTPUT_DIR, 'optimal_clustering_visualization.png')
    plt.savefig(viz_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ 最佳聚类视觉化已保存: {viz_path}")
    
    return viz_path

def save_optimal_results(results_df):
    """保存最佳聚类结果"""
    print("\n💾 保存最佳聚类结果...")
    
    # 确保输出目录存在
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # 保存聚类结果CSV
    output_path = os.path.join(OUTPUT_DIR, 'optimal_clustering_results.csv')
    results_df.to_csv(output_path, index=False, encoding='utf-8')
    print(f"✅ 最佳聚类结果已保存: {output_path}")
    
    # 显示结果预览
    print(f"\n📋 结果预览:")
    print(results_df[['行政区', '潜力等级', '集群编号', '潜力分数']].to_string(index=False))
    
    return output_path

def main():
    """主函数"""
    print("="*60)
    print("🚀 实现最佳聚类方法")
    print("特征组合2: 所得+医疗+第三产业")
    print("目标: Silhouette > 0.6 且 一致性 > 0.7")
    print("="*60)
    
    # 1. 载入并准备数据
    df, X_optimal, districts, optimal_features = load_and_prepare_data()
    
    # 2. 执行最佳聚类
    results_df, X_tsne, cluster_labels, silhouette, consistency = perform_optimal_clustering(X_optimal, districts, df)
    
    # 3. 创建视觉化
    viz_path = create_optimal_visualization(results_df, X_tsne)
    
    # 4. 保存结果
    output_path = save_optimal_results(results_df)
    
    # 5. 总结
    print(f"\n✅ 最佳聚类方法实现完成!")
    print(f"📊 性能指标:")
    print(f"  - Silhouette分数: {silhouette:.3f} {'✅' if silhouette > 0.6 else '❌'}")
    print(f"  - 语义一致性: {consistency:.3f} {'✅' if consistency > 0.7 else '❌'}")
    print(f"  - 分群数量: 3 ✅")
    print(f"📁 输出文件:")
    print(f"  - 聚类结果: {output_path}")
    print(f"  - 视觉化图: {viz_path}")

if __name__ == "__main__":
    main() 