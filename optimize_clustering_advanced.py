"""
高级聚类优化实验
目标: Silhouette > 0.6 且 语义一致性 > 0.7 且 分群数 >= 3

测试策略:
1. 多种聚类算法组合
2. 特征选择优化
3. 降维方法对比
4. 参数调优
5. 集成方法
"""

import pandas as pd
import numpy as np
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.cluster import AgglomerativeClustering, KMeans, DBSCAN
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from scipy import stats
import itertools
import warnings
warnings.filterwarnings('ignore')

RANDOM_STATE = 42

def load_data():
    """载入数据"""
    df = pd.read_csv('output/taoyuan_features_enhanced.csv')
    districts = df['區域別'].tolist()
    
    # 所有可用特征
    all_features = df.columns[1:].tolist()
    X_full = df.drop('區域別', axis=1).values
    
    return X_full, districts, all_features, df

def calculate_potential_score(df, features_subset=None):
    """计算潜力综合分数"""
    if features_subset is None:
        potential_indicators = [
            '人口_working_age_ratio',
            '所得_median_household_income', 
            'tertiary_industry_ratio',
            'medical_index'
        ]
    else:
        potential_indicators = features_subset
    
    # 检查指标是否存在
    available_indicators = [col for col in potential_indicators if col in df.columns]
    
    # 计算z-score
    z_scores = pd.DataFrame()
    z_scores['區域別'] = df['區域別']
    
    for indicator in available_indicators:
        z_score = stats.zscore(df[indicator])
        z_scores[f'{indicator}_zscore'] = z_score
    
    # 计算潜力综合分数
    z_score_cols = [col for col in z_scores.columns if col.endswith('_zscore')]
    z_scores['potential_score'] = z_scores[z_score_cols].mean(axis=1)
    
    return z_scores

def assign_district_labels(potential_scores):
    """分配行政区标签"""
    district_labels = pd.qcut(
        potential_scores['potential_score'], 
        q=3, 
        labels=['低潛力', '中潛力', '高潛力']
    )
    potential_scores['district_label'] = district_labels
    return potential_scores

def evaluate_semantic_consistency(cluster_labels, district_labels_df, districts):
    """评估语义一致性"""
    # 创建映射DataFrame
    mapping_df = pd.DataFrame({
        'cluster_id': cluster_labels,
        'district_label': district_labels_df['district_label'].values,
        'district': districts,
        'potential_score': district_labels_df['potential_score'].values
    })
    
    # 使用多数决分配集群标签
    cluster_potential_mapping = {}
    for cluster_id in range(len(np.unique(cluster_labels))):
        cluster_data = mapping_df[mapping_df['cluster_id'] == cluster_id]
        if len(cluster_data) > 0:
            label_counts = cluster_data['district_label'].value_counts()
            majority_label = label_counts.index[0]
            cluster_potential_mapping[cluster_id] = majority_label
    
    # 计算各集群潜力分数统计
    cluster_stats = {}
    for cluster_id in range(len(np.unique(cluster_labels))):
        cluster_data = mapping_df[mapping_df['cluster_id'] == cluster_id]
        if len(cluster_data) > 0:
            potential_scores = cluster_data['potential_score'].values
            cluster_stats[cluster_id] = {
                'mean_score': potential_scores.mean(),
                'std_score': potential_scores.std() if len(potential_scores) > 1 else 0,
                'count': len(cluster_data)
            }
    
    # 计算一致性分数
    consistency_scores = []
    for cluster_id in cluster_stats:
        std = cluster_stats[cluster_id]['std_score']
        consistency = 1 / (1 + std)
        consistency_scores.append(consistency)
    
    overall_consistency = np.mean(consistency_scores) if consistency_scores else 0
    
    return overall_consistency, cluster_potential_mapping, cluster_stats

class ClusteringExperiment:
    def __init__(self, X, districts, df):
        self.X = X
        self.districts = districts
        self.df = df
        self.results = []
    
    def experiment_1_algorithm_comparison(self):
        """实验1: 不同聚类算法对比"""
        print("🔬 实验1: 聚类算法对比")
        
        # t-SNE降维
        tsne = TSNE(n_components=2, random_state=RANDOM_STATE, perplexity=3, max_iter=1000)
        X_tsne = tsne.fit_transform(self.X)
        
        algorithms = [
            ('Ward', AgglomerativeClustering(n_clusters=3, linkage='ward')),
            ('Complete', AgglomerativeClustering(n_clusters=3, linkage='complete')), 
            ('Average', AgglomerativeClustering(n_clusters=3, linkage='average')),
            ('KMeans', KMeans(n_clusters=3, random_state=RANDOM_STATE)),
            ('GMM', GaussianMixture(n_components=3, random_state=RANDOM_STATE))
        ]
        
        for name, algorithm in algorithms:
            if name == 'GMM':
                cluster_labels = algorithm.fit_predict(X_tsne)
            else:
                cluster_labels = algorithm.fit_predict(X_tsne)
            
            if len(np.unique(cluster_labels)) >= 3:
                silhouette = silhouette_score(X_tsne, cluster_labels)
                
                # 计算语义一致性
                potential_scores = calculate_potential_score(self.df)
                district_labels_df = assign_district_labels(potential_scores)
                consistency, _, _ = evaluate_semantic_consistency(cluster_labels, district_labels_df, self.districts)
                
                self.results.append({
                    'method': f't-SNE + {name}',
                    'silhouette': silhouette,
                    'consistency': consistency,
                    'n_clusters': len(np.unique(cluster_labels)),
                    'details': f'算法: {name}'
                })
                
                print(f"  {name:10s}: Silhouette={silhouette:.3f}, 一致性={consistency:.3f}")
    
    def experiment_2_dimensionality_reduction(self):
        """实验2: 不同降维方法"""
        print("\n🔬 实验2: 降维方法对比")
        
        # 标准化数据
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(self.X)
        
        reduction_methods = [
            ('t-SNE_p3', TSNE(n_components=2, random_state=RANDOM_STATE, perplexity=3, max_iter=1000)),
            ('t-SNE_p5', TSNE(n_components=2, random_state=RANDOM_STATE, perplexity=5, max_iter=1000)),
            ('PCA', PCA(n_components=2, random_state=RANDOM_STATE)),
        ]
        
        for name, reducer in reduction_methods:
            X_reduced = reducer.fit_transform(X_scaled)
            
            # 使用Ward聚类
            ward = AgglomerativeClustering(n_clusters=3, linkage='ward')
            cluster_labels = ward.fit_predict(X_reduced)
            
            silhouette = silhouette_score(X_reduced, cluster_labels)
            
            potential_scores = calculate_potential_score(self.df)
            district_labels_df = assign_district_labels(potential_scores)
            consistency, _, _ = evaluate_semantic_consistency(cluster_labels, district_labels_df, self.districts)
            
            self.results.append({
                'method': f'{name} + Ward',
                'silhouette': silhouette,
                'consistency': consistency,
                'n_clusters': len(np.unique(cluster_labels)),
                'details': f'降维: {name}'
            })
            
            print(f"  {name:10s}: Silhouette={silhouette:.3f}, 一致性={consistency:.3f}")
    
    def experiment_3_feature_selection(self):
        """实验3: 特征选择优化"""
        print("\n🔬 实验3: 特征选择优化")
        
        # 定义不同的特征组合
        feature_combinations = [
            # 原始组合
            ['人口_working_age_ratio', '所得_median_household_income', 'tertiary_industry_ratio', 'medical_index', '商業_concentration_index'],
            
            # 高相关性组合
            ['所得_median_household_income', 'medical_index', 'tertiary_industry_ratio'],
            
            # 平衡组合
            ['人口_working_age_ratio', '所得_median_household_income', '商業_concentration_index', 'medical_index'],
            
            # 全特征
            self.df.columns[1:].tolist()
        ]
        
        for i, features in enumerate(feature_combinations):
            # 检查特征是否存在
            available_features = [f for f in features if f in self.df.columns]
            if len(available_features) < 3:
                continue
                
            # 提取特征子集
            feature_indices = [self.df.columns.get_loc(f) - 1 for f in available_features]
            X_subset = self.X[:, feature_indices]
            
            # 标准化
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X_subset)
            
            # t-SNE降维
            tsne = TSNE(n_components=2, random_state=RANDOM_STATE, perplexity=3, max_iter=1000)
            X_reduced = tsne.fit_transform(X_scaled)
            
            # Ward聚类
            ward = AgglomerativeClustering(n_clusters=3, linkage='ward')
            cluster_labels = ward.fit_predict(X_reduced)
            
            silhouette = silhouette_score(X_reduced, cluster_labels)
            
            # 使用特征子集计算潜力分数
            potential_indicators = [f for f in available_features if f in ['人口_working_age_ratio', '所得_median_household_income', 'tertiary_industry_ratio', 'medical_index']]
            potential_scores = calculate_potential_score(self.df, potential_indicators if potential_indicators else available_features[:4])
            district_labels_df = assign_district_labels(potential_scores)
            consistency, _, _ = evaluate_semantic_consistency(cluster_labels, district_labels_df, self.districts)
            
            self.results.append({
                'method': f'特征组合{i+1}',
                'silhouette': silhouette,
                'consistency': consistency,
                'n_clusters': len(np.unique(cluster_labels)),
                'details': f'特征数: {len(available_features)}'
            })
            
            print(f"  组合{i+1:2d}: Silhouette={silhouette:.3f}, 一致性={consistency:.3f} (特征数:{len(available_features)})")
    
    def experiment_4_scaling_methods(self):
        """实验4: 不同标准化方法"""
        print("\n🔬 实验4: 标准化方法对比")
        
        scalers = [
            ('StandardScaler', StandardScaler()),
            ('MinMaxScaler', MinMaxScaler()),
            ('RobustScaler', RobustScaler()),
            ('NoScaling', None)
        ]
        
        for name, scaler in scalers:
            if scaler is not None:
                X_scaled = scaler.fit_transform(self.X)
            else:
                X_scaled = self.X.copy()
            
            # t-SNE降维
            tsne = TSNE(n_components=2, random_state=RANDOM_STATE, perplexity=3, max_iter=1000)
            X_reduced = tsne.fit_transform(X_scaled)
            
            # Ward聚类
            ward = AgglomerativeClustering(n_clusters=3, linkage='ward')
            cluster_labels = ward.fit_predict(X_reduced)
            
            silhouette = silhouette_score(X_reduced, cluster_labels)
            
            potential_scores = calculate_potential_score(self.df)
            district_labels_df = assign_district_labels(potential_scores)
            consistency, _, _ = evaluate_semantic_consistency(cluster_labels, district_labels_df, self.districts)
            
            self.results.append({
                'method': f'{name} + t-SNE + Ward',
                'silhouette': silhouette,
                'consistency': consistency,
                'n_clusters': len(np.unique(cluster_labels)),
                'details': f'标准化: {name}'
            })
            
            print(f"  {name:15s}: Silhouette={silhouette:.3f}, 一致性={consistency:.3f}")
    
    def experiment_5_ensemble_method(self):
        """实验5: 集成聚类方法"""
        print("\n🔬 实验5: 集成聚类方法")
        
        # 多种方法组合
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(self.X)
        
        # 方法1: PCA + KMeans
        pca = PCA(n_components=2, random_state=RANDOM_STATE)
        X_pca = pca.fit_transform(X_scaled)
        kmeans = KMeans(n_clusters=3, random_state=RANDOM_STATE)
        labels_1 = kmeans.fit_predict(X_pca)
        
        # 方法2: t-SNE + Ward
        tsne = TSNE(n_components=2, random_state=RANDOM_STATE, perplexity=3, max_iter=1000)
        X_tsne = tsne.fit_transform(X_scaled)
        ward = AgglomerativeClustering(n_clusters=3, linkage='ward')
        labels_2 = ward.fit_predict(X_tsne)
        
        # 集成投票
        ensemble_labels = []
        for i in range(len(self.districts)):
            votes = [labels_1[i], labels_2[i]]
            # 简单投票机制
            unique, counts = np.unique(votes, return_counts=True)
            ensemble_labels.append(unique[np.argmax(counts)])
        
        ensemble_labels = np.array(ensemble_labels)
        
        # 如果集群数量不足3，使用原始方法2
        if len(np.unique(ensemble_labels)) < 3:
            ensemble_labels = labels_2
        
        silhouette = silhouette_score(X_tsne, ensemble_labels)  # 使用t-SNE空间计算
        
        potential_scores = calculate_potential_score(self.df)
        district_labels_df = assign_district_labels(potential_scores)
        consistency, _, _ = evaluate_semantic_consistency(ensemble_labels, district_labels_df, self.districts)
        
        self.results.append({
            'method': 'Ensemble (PCA+KMeans & t-SNE+Ward)',
            'silhouette': silhouette,
            'consistency': consistency,
            'n_clusters': len(np.unique(ensemble_labels)),
            'details': '集成方法'
        })
        
        print(f"  集成方法: Silhouette={silhouette:.3f}, 一致性={consistency:.3f}")
    
    def run_all_experiments(self):
        """运行所有实验"""
        print("🚀 开始高级聚类优化实验")
        print("目标: Silhouette > 0.6 且 一致性 > 0.7 且 分群数 >= 3")
        print("="*60)
        
        self.experiment_1_algorithm_comparison()
        self.experiment_2_dimensionality_reduction()
        self.experiment_3_feature_selection()
        self.experiment_4_scaling_methods()
        self.experiment_5_ensemble_method()
        
        return self.results
    
    def analyze_results(self):
        """分析结果"""
        print("\n📊 实验结果分析")
        print("="*60)
        
        # 转换为DataFrame便于分析
        df_results = pd.DataFrame(self.results)
        
        # 筛选满足条件的结果
        target_silhouette = 0.6
        target_consistency = 0.7
        target_clusters = 3
        
        print(f"筛选条件: Silhouette > {target_silhouette}, 一致性 > {target_consistency}, 分群数 >= {target_clusters}")
        
        good_results = df_results[
            (df_results['silhouette'] > target_silhouette) & 
            (df_results['consistency'] > target_consistency) & 
            (df_results['n_clusters'] >= target_clusters)
        ]
        
        if len(good_results) > 0:
            print(f"\n✅ 找到 {len(good_results)} 个满足条件的方法:")
            for _, row in good_results.iterrows():
                print(f"  {row['method']:30s}: Silhouette={row['silhouette']:.3f}, 一致性={row['consistency']:.3f}")
        else:
            print(f"\n❌ 没有找到完全满足条件的方法")
            print(f"\n📈 最佳结果 (按综合分数排序):")
            
            # 计算综合分数
            df_results['combined_score'] = (
                0.4 * df_results['silhouette'] + 
                0.6 * df_results['consistency']
            )
            
            best_results = df_results.nlargest(5, 'combined_score')
            for _, row in best_results.iterrows():
                print(f"  {row['method']:30s}: Silhouette={row['silhouette']:.3f}, 一致性={row['consistency']:.3f}, 综合={row['combined_score']:.3f}")
        
        return good_results if len(good_results) > 0 else best_results

def main():
    """主函数"""
    print("🎯 高级聚类优化实验")
    print("="*60)
    
    # 载入数据
    X, districts, features, df = load_data()
    
    # 运行实验
    experiment = ClusteringExperiment(X, districts, df)
    results = experiment.run_all_experiments()
    
    # 分析结果
    best_methods = experiment.analyze_results()
    
    print(f"\n✅ 实验完成! 共测试了 {len(results)} 种方法")
    
    return best_methods

if __name__ == "__main__":
    best_methods = main() 