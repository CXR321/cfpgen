import pickle
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
import seaborn as sns
from scipy.spatial.distance import cdist
import pandas as pd
from skbio.stats.distance import permanova  # 添加PERMANOVA
from statsmodels.stats.multitest import multipletests  # 添加多重检验校正

# 设置样式
plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
sns.set_style("whitegrid")

def load_embeddings(file_path):
    """加载嵌入数据"""
    with open(file_path, "rb") as f:
        go_terms_emb = pickle.load(f)
    return go_terms_emb

# def calculate_intra_class_similarity(embeddings_dict):
#     """计算类内相似度"""
#     intra_similarities = {}
    
#     for term, embeddings in embeddings_dict.items():
#         if len(embeddings) > 1:
#             # 计算余弦相似度
#             similarity_matrix = cosine_similarity(embeddings)
#             # 获取上三角矩阵（不包括对角线）
#             upper_triangle = similarity_matrix[np.triu_indices_from(similarity_matrix, k=1)]
#             intra_similarities[term] = {
#                 'mean_similarity': np.mean(upper_triangle),
#                 'std_similarity': np.std(upper_triangle),
#                 'min_similarity': np.min(upper_triangle),
#                 'max_similarity': np.max(upper_triangle),
#                 'count': len(embeddings)
#             }
#         else:
#             intra_similarities[term] = {
#                 'mean_similarity': 1.0,
#                 'std_similarity': 0.0,
#                 'min_similarity': 1.0,
#                 'max_similarity': 1.0,
#                 'count': 1
#             }
    
#     return intra_similarities

def calculate_intra_class_similarity(embeddings_dict):
    """计算类内相似度和欧式距离"""
    intra_similarities = {}
    
    # 用于存储所有类的统计信息
    all_cosine_means = []
    all_euclidean_means = []
    
    print("=== 类内距离统计 ===")
    print(f"{'Term':<20} {'Samples':<8} {'Cos Mean':<10} {'Cos Std':<10} {'Euc Mean':<12} {'Euc Std':<10}")
    print("-" * 70)
    
    for term, embeddings in embeddings_dict.items():
        if len(embeddings) > 1:
            embeddings_array = np.array(embeddings)
            
            # 计算余弦相似度
            similarity_matrix = cosine_similarity(embeddings_array)
            upper_triangle_cosine = similarity_matrix[np.triu_indices_from(similarity_matrix, k=1)]
            
            # 计算欧式距离
            distance_matrix = euclidean_distances(embeddings_array)
            upper_triangle_euclidean = distance_matrix[np.triu_indices_from(distance_matrix, k=1)]
            
            intra_similarities[term] = {
                'mean_similarity': np.mean(upper_triangle_cosine),
                'std_similarity': np.std(upper_triangle_cosine),
                'min_similarity': np.min(upper_triangle_cosine),
                'max_similarity': np.max(upper_triangle_cosine),
                'mean_euclidean': np.mean(upper_triangle_euclidean),
                'std_euclidean': np.std(upper_triangle_euclidean),
                'min_euclidean': np.min(upper_triangle_euclidean),
                'max_euclidean': np.max(upper_triangle_euclidean),
                'count': len(embeddings)
            }
            
            all_cosine_means.append(np.mean(upper_triangle_cosine))
            all_euclidean_means.append(np.mean(upper_triangle_euclidean))
            
        else:
            intra_similarities[term] = {
                'mean_similarity': 1.0,
                'std_similarity': 0.0,
                'min_similarity': 1.0,
                'max_similarity': 1.0,
                'mean_euclidean': 0.0,
                'std_euclidean': 0.0,
                'min_euclidean': 0.0,
                'max_euclidean': 0.0,
                'count': 1
            }
            
            # 输出单个样本的term信息
            # print(f"{term:<20} {1:<8} {'1.000000':<10} {'0.000000':<10} {'0.000000':<12} {'0.000000':<10}")
    
    # 输出总体统计信息
    print("-" * 70)
    if all_cosine_means:
        print(f"{'Overall Average':<20} {'-':<8} {np.mean(all_cosine_means):<10.6f} "
              f"{np.std(all_cosine_means):<10.6f} {np.mean(all_euclidean_means):<12.6f} "
              f"{np.std(all_euclidean_means):<10.6f}")
        
        # 输出分位数信息
        print(f"\n=== 类内距离分位数统计 ===")
        print(f"余弦相似度分位数:")
        print(f"  5%: {np.percentile(all_cosine_means, 5):.6f}, "
              f"25%: {np.percentile(all_cosine_means, 25):.6f}, "
              f"50%: {np.percentile(all_cosine_means, 50):.6f}, "
              f"75%: {np.percentile(all_cosine_means, 75):.6f}, "
              f"95%: {np.percentile(all_cosine_means, 95):.6f}")
        
        print(f"欧式距离分位数:")
        print(f"  5%: {np.percentile(all_euclidean_means, 5):.6f}, "
              f"25%: {np.percentile(all_euclidean_means, 25):.6f}, "
              f"50%: {np.percentile(all_euclidean_means, 50):.6f}, "
              f"75%: {np.percentile(all_euclidean_means, 75):.6f}, "
              f"95%: {np.percentile(all_euclidean_means, 95):.6f}")
    
    return intra_similarities


def calculate_inter_class_distance(embeddings_dict):
    """计算类间距离"""
    # 获取每个类的平均嵌入向量
    class_centroids = {}
    for term, embeddings in embeddings_dict.items():
        class_centroids[term] = np.mean(embeddings, axis=0)
    
    # 转换为矩阵形式
    terms = list(class_centroids.keys())
    centroid_matrix = np.array([class_centroids[term] for term in terms])
    
    # 计算余弦距离（1 - 余弦相似度）
    cosine_dist_matrix = 1 - cosine_similarity(centroid_matrix)
    
    # 计算欧几里得距离
    euclidean_dist_matrix = euclidean_distances(centroid_matrix)
    
    inter_distances = {}
    n_terms = len(terms)
    
    for i in range(n_terms):
        for j in range(i + 1, n_terms):
            pair_name = f"{terms[i]}-{terms[j]}"
            inter_distances[pair_name] = {
                'cosine_distance': cosine_dist_matrix[i, j],
                'euclidean_distance': euclidean_dist_matrix[i, j]
            }
    
    return inter_distances, terms, centroid_matrix

def visualize_embeddings(embeddings_dict, method='tsne', title_suffix=""):
    """可视化嵌入向量（无标签版本）"""
    # 准备数据
    all_embeddings = []
    all_labels = []
    
    for i, (term, embeddings) in enumerate(embeddings_dict.items()):
        all_embeddings.extend(embeddings)
        all_labels.extend([i] * len(embeddings))
    
    all_embeddings = np.array(all_embeddings)
    all_labels = np.array(all_labels)
    
    # 降维
    if method == 'tsne':
        reducer = TSNE(n_components=2, random_state=42, perplexity=30)
    else:
        reducer = PCA(n_components=2, random_state=42)
    
    reduced_embeddings = reducer.fit_transform(all_embeddings)
    
    # 创建图表
    fig, axes = plt.subplots(1, 2, figsize=(20, 8))
    
    # 散点图 - 所有样本点
    scatter = axes[0].scatter(reduced_embeddings[:, 0], reduced_embeddings[:, 1], 
                             c=all_labels, cmap='tab20', alpha=0.6, s=20)
    axes[0].set_title(f'GO Terms Embeddings - {method.upper()} {title_suffix}')
    axes[0].set_xlabel('Component 1')
    axes[0].set_ylabel('Component 2')
    
    # 类中心点的散点图
    class_centroids = []
    unique_labels = np.unique(all_labels)
    for label in unique_labels:
        label_indices = np.where(all_labels == label)[0]
        if len(label_indices) > 0:
            centroid = np.mean(reduced_embeddings[label_indices], axis=0)
            class_centroids.append(centroid)
    
    class_centroids = np.array(class_centroids)
    axes[1].scatter(class_centroids[:, 0], class_centroids[:, 1], 
                   c=range(len(class_centroids)), cmap='tab20', s=80, alpha=0.8)
    
    axes[1].set_title(f'Class Centroids - {method.upper()} {title_suffix}')
    axes[1].set_xlabel('Component 1')
    axes[1].set_ylabel('Component 2')
    
    plt.tight_layout()
    plt.savefig(f'go_terms_embeddings_{method}{title_suffix.replace(" ", "_")}.png', 
                dpi=300, bbox_inches='tight')
    plt.show()
    
    return reduced_embeddings, all_labels

def plot_statistics(intra_similarities, inter_distances):
    """绘制统计图表（无标签版本）"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # 类内相似度分布
    intra_means = [stats['mean_similarity'] for stats in intra_similarities.values()]
    axes[0, 0].hist(intra_means, bins=30, alpha=0.7, color='skyblue', edgecolor='black')
    axes[0, 0].set_title('Distribution of Intra-class Similarity Means')
    axes[0, 0].set_xlabel('Mean Cosine Similarity')
    axes[0, 0].set_ylabel('Frequency')
    
    # 类间距离分布
    inter_cosine = [dist['cosine_distance'] for dist in inter_distances.values()]
    axes[0, 1].hist(inter_cosine, bins=30, alpha=0.7, color='lightcoral', edgecolor='black')
    axes[0, 1].set_title('Distribution of Inter-class Cosine Distances')
    axes[0, 1].set_xlabel('Cosine Distance')
    axes[0, 1].set_ylabel('Frequency')
    
    # 样本数量与相似度的关系
    sample_counts = [stats['count'] for stats in intra_similarities.values()]
    axes[1, 0].scatter(sample_counts, intra_means, alpha=0.6, s=30)
    axes[1, 0].set_title('Sample Count vs Intra-class Similarity')
    axes[1, 0].set_xlabel('Number of Samples per Class')
    axes[1, 0].set_ylabel('Mean Intra-class Similarity')
    
    # 类内相似度与类间距离的关系
    class_inter_dist = {}
    for class_idx, term in enumerate(intra_similarities.keys()):
        relevant_dists = []
        for pair_name, dist in inter_distances.items():
            if term in pair_name:
                relevant_dists.append(dist['cosine_distance'])
        class_inter_dist[term] = np.mean(relevant_dists) if relevant_dists else 0
    
    inter_means = list(class_inter_dist.values())
    axes[1, 1].scatter(intra_means, inter_means, alpha=0.6, s=30)
    axes[1, 1].set_title('Intra-class Similarity vs Inter-class Distance')
    axes[1, 1].set_xlabel('Mean Intra-class Similarity')
    axes[1, 1].set_ylabel('Mean Inter-class Distance')
    
    plt.tight_layout()
    plt.savefig('go_terms_statistics.png', dpi=300, bbox_inches='tight')
    plt.show()

def plot_density_visualization(embeddings_dict, method='tsne'):
    """绘制密度可视化图"""
    # 准备数据
    all_embeddings = []
    all_labels = []
    
    for i, (term, embeddings) in enumerate(embeddings_dict.items()):
        all_embeddings.extend(embeddings)
        all_labels.extend([i] * len(embeddings))
    
    all_embeddings = np.array(all_embeddings)
    
    # 降维
    if method == 'tsne':
        reducer = TSNE(n_components=2, random_state=42, perplexity=30)
    else:
        reducer = PCA(n_components=2, random_state=42)
    
    reduced_embeddings = reducer.fit_transform(all_embeddings)
    
    # 创建密度图
    plt.figure(figsize=(10, 8))
    
    # 使用hexbin进行密度可视化
    hb = plt.hexbin(reduced_embeddings[:, 0], reduced_embeddings[:, 1], 
                   gridsize=50, cmap='viridis', alpha=0.8)
    plt.colorbar(hb, label='Point density')
    plt.title(f'GO Terms Embeddings Density - {method.upper()}')
    plt.xlabel('Component 1')
    plt.ylabel('Component 2')
    
    plt.tight_layout()
    plt.savefig(f'go_terms_density_{method}.png', dpi=300, bbox_inches='tight')
    plt.show()

def analyze_embeddings(go_terms_emb):
    """深入分析嵌入向量特性"""
    
    # 1. 检查嵌入向量的范数分布
    all_embeddings = []
    for term, embeddings in go_terms_emb.items():
        all_embeddings.extend(embeddings)
    all_embeddings = np.array(all_embeddings)
    
    norms = np.linalg.norm(all_embeddings, axis=1)
    print(f"嵌入向量范数范围: {norms.min():.4f} - {norms.max():.4f}")
    print(f"平均范数: {norms.mean():.4f}")
    
    # 2. 检查是否经过归一化
    normalized_embeddings = all_embeddings / norms[:, np.newaxis]
    normalized_norms = np.linalg.norm(normalized_embeddings, axis=1)
    print(f"归一化后范数: {normalized_norms.mean():.4f} ± {normalized_norms.std():.4f}")
    
    # 3. 分析原始样本间的距离（不仅仅是类中心）
    all_similarities = cosine_similarity(all_embeddings)
    print(f"所有样本间平均余弦相似度: {all_similarities.mean():.4f}")
    
    return all_embeddings, norms

def calculate_alternative_metrics(embeddings_dict):
    """计算替代的相似度/距离指标"""
    
    # 1. 使用中位数而不是均值作为类代表
    class_medians = {}
    for term, embeddings in embeddings_dict.items():
        class_medians[term] = np.median(embeddings, axis=0)
    
    terms = list(class_medians.keys())
    median_matrix = np.array([class_medians[term] for term in terms])
    median_cosine_dist = 1 - cosine_similarity(median_matrix)
    
    # 2. 计算类间最小距离（最近邻距离）
    min_inter_distances = {}
    terms_list = list(embeddings_dict.keys())
    
    for i, term1 in enumerate(terms_list):
        for j, term2 in enumerate(terms_list[i+1:], i+1):
            # 计算两个类所有样本对之间的最小距离
            emb1 = np.array(embeddings_dict[term1])
            emb2 = np.array(embeddings_dict[term2])
            
            # 使用欧几里得距离
            min_euclidean = np.min(cdist(emb1, emb2, metric='euclidean'))
            
            # 使用余弦距离
            cosine_dists = 1 - cosine_similarity(emb1, emb2)
            min_cosine = np.min(cosine_dists)
            
            pair_name = f"{term1}-{term2}"
            min_inter_distances[pair_name] = {
                'min_euclidean': min_euclidean,
                'min_cosine': min_cosine
            }
    
    return median_cosine_dist, min_inter_distances

def calculate_separation_metrics(embeddings_dict):
    """计算类别分离度指标"""
    
    separation_results = {}
    
    for term, embeddings in embeddings_dict.items():
        if len(embeddings) < 2:
            continue
            
        intra_embeddings = np.array(embeddings)
        
        # 类内紧密度
        centroid = np.mean(intra_embeddings, axis=0)
        intra_distances = np.linalg.norm(intra_embeddings - centroid, axis=1)
        intra_tightness = np.mean(intra_distances)
        
        # 计算到最近类别的距离
        min_inter_distance = float('inf')
        for other_term, other_embeddings in embeddings_dict.items():
            if other_term == term:
                continue
                
            other_centroid = np.mean(other_embeddings, axis=0)
            inter_distance = np.linalg.norm(centroid - other_centroid)
            min_inter_distance = min(min_inter_distance, inter_distance)
        
        # 分离度比率
        separation_ratio = min_inter_distance / intra_tightness if intra_tightness > 0 else float('inf')
        
        separation_results[term] = {
            'intra_tightness': intra_tightness,
            'min_inter_distance': min_inter_distance,
            'separation_ratio': separation_ratio
        }
    
    return separation_results

def plot_detailed_analysis(embeddings_dict, intra_similarities, inter_distances):
    """绘制详细分析图表"""
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # 1. 类内相似度分布
    intra_means = [stats['mean_similarity'] for stats in intra_similarities.values()]
    axes[0, 0].hist(intra_means, bins=30, alpha=0.7, color='skyblue', edgecolor='black')
    axes[0, 0].set_title('Intra-class Similarity Distribution')
    axes[0, 0].set_xlabel('Cosine Similarity')
    axes[0, 0].set_ylabel('Frequency')
    
    # 2. 类间距离分布
    inter_cosine = [dist['cosine_distance'] for dist in inter_distances.values()]
    axes[0, 1].hist(inter_cosine, bins=30, alpha=0.7, color='lightcoral', edgecolor='black')
    axes[0, 1].set_title('Inter-class Distance Distribution')
    axes[0, 1].set_xlabel('Cosine Distance')
    axes[0, 1].set_ylabel('Frequency')
    
    # 3. 嵌入向量范数分布
    all_norms = []
    for embeddings in embeddings_dict.values():
        norms = np.linalg.norm(embeddings, axis=1)
        all_norms.extend(norms)
    
    axes[0, 2].hist(all_norms, bins=30, alpha=0.7, color='lightgreen', edgecolor='black')
    axes[0, 2].set_title('Embedding Norm Distribution')
    axes[0, 2].set_xlabel('Vector Norm')
    axes[0, 2].set_ylabel('Frequency')
    
    # 4. 样本数量分布
    sample_counts = [len(embeddings) for embeddings in embeddings_dict.values()]
    axes[1, 0].hist(sample_counts, bins=30, alpha=0.7, color='orange', edgecolor='black')
    axes[1, 0].set_title('Sample Count Distribution')
    axes[1, 0].set_xlabel('Samples per Class')
    axes[1, 0].set_ylabel('Frequency')
    
    # 5. 类内相似度 vs 样本数量
    axes[1, 1].scatter(sample_counts, intra_means, alpha=0.6, s=30)
    axes[1, 1].set_title('Intra-similarity vs Sample Count')
    axes[1, 1].set_xlabel('Sample Count')
    axes[1, 1].set_ylabel('Intra-class Similarity')
    
    # 6. 距离矩阵热图（前20个类别）
    terms = list(embeddings_dict.keys())[:20]
    centroids = [np.mean(embeddings_dict[term], axis=0) for term in terms]
    distance_matrix = 1 - cosine_similarity(centroids)
    
    im = axes[1, 2].imshow(distance_matrix, cmap='viridis', aspect='auto')
    axes[1, 2].set_title('Distance Matrix (Top 20 Classes)')
    plt.colorbar(im, ax=axes[1, 2])
    
    plt.tight_layout()
    plt.savefig('detailed_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()


def perform_permanova_analysis(embeddings_dict, permutations=999):
    """
    执行PERMANOVA分析来检验不同标签间的分布差异
    """
    print("\n" + "="*60)
    print("PERMANOVA 统计分析")
    print("="*60)
    
    # 准备数据
    all_embeddings = []
    all_labels = []
    label_mapping = {}
    
    # 创建标签映射
    for i, (term, embeddings) in enumerate(embeddings_dict.items()):
        all_embeddings.extend(embeddings)
        all_labels.extend([i] * len(embeddings))
        label_mapping[i] = term
    
    all_embeddings = np.array(all_embeddings)
    all_labels = np.array(all_labels)

    print(all_embeddings.shape)

    # all_embeddings = all_embeddings[:10000]
    
    # 计算距离矩阵
    print("计算距离矩阵...")
    distance_matrix = euclidean_distances(all_embeddings)

    distance_matrix = (distance_matrix + distance_matrix.T) / 2

    print(distance_matrix.shape)

    # 检查distance_matrix是否有NaN元素
    print("检查距离矩阵中的NaN值...")
    nan_count = np.sum(np.isnan(distance_matrix))
    inf_count = np.sum(np.isinf(distance_matrix))
    negative_count = np.sum(distance_matrix < 0)

    print(f"NaN值数量: {nan_count}")
    print(f"Inf值数量: {inf_count}")
    print(f"负值数量: {negative_count}")    

# 再次检查矩阵是否对称
    is_symmetric = np.allclose(distance_matrix, distance_matrix.T)
    print(f"矩阵是否对称: {is_symmetric}")
    
    # 检查对角线是否为0
    diagonal_zeros = np.allclose(np.diag(distance_matrix), 0)
    print(f"对角线是否为0: {diagonal_zeros}")
    # 执行PERMANOVA
    print("执行PERMANOVA检验...")
    try:
        from skbio import DistanceMatrix
        distance_matrix = DistanceMatrix(distance_matrix)
        print("成功创建DistanceMatrix对象")

        permanova_result = permanova(distance_matrix, all_labels, permutations=permutations)
        
        print(f"PERMANOVA 结果:")
        print(f"F-statistic: {permanova_result['test statistic']:.4f}")
        print(f"p-value: {permanova_result['p-value']:.6f}")
        print(f" permutations: {permutations}")
        
        # 计算效应量 (η²)
        print("\n计算效应量...")
        unique_labels = np.unique(all_labels)
        overall_mean = np.mean(all_embeddings, axis=0)
        
        ss_total = np.sum((all_embeddings - overall_mean)**2)
        
        ss_between = 0
        for label in unique_labels:
            group_data = all_embeddings[all_labels == label]
            group_mean = np.mean(group_data, axis=0)
            ss_between += len(group_data) * np.sum((group_mean - overall_mean)**2)
        
        eta_squared = ss_between / ss_total  # 效应量
        
        print(f"效应量 (η²): {eta_squared:.4f}")
        print(f"解释方差比例: {eta_squared * 100:.2f}%")
        
        # 解释结果
        print(f"\n结果解释:")
        if permanova_result['p-value'] < 0.05:
            if eta_squared > 0.14:
                print("✅ 强证据支持组间存在显著差异 (大效应量)")
            elif eta_squared > 0.06:
                print("✅ 中等证据支持组间存在显著差异 (中等效应量)")
            else:
                print("✅ 弱证据支持组间存在显著差异 (小效应量但统计显著)")
        else:
            print("❌ 无统计显著证据支持组间差异")
            
        return {
            'f_statistic': permanova_result['test statistic'],
            'p_value': permanova_result['p-value'],
            'effect_size': eta_squared,
            'ss_total': ss_total,
            'ss_between': ss_between,
            'is_significant': permanova_result['p-value'] < 0.05
        }
        
    except Exception as e:
        print(f"PERMANOVA分析出错: {e}")
        return None

def perform_pairwise_permanova(embeddings_dict, permutations=999):
    """
    执行成对PERMANOVA分析
    """
    print("\n" + "="*60)
    print("成对 PERMANOVA 分析")
    print("="*60)
    
    terms = list(embeddings_dict.keys())
    pairwise_results = []
    
    for i in range(len(terms)):
        for j in range(i + 1, len(terms)):
            term1, term2 = terms[i], terms[j]
            
            # 合并两个组的数据
            group1_data = np.array(embeddings_dict[term1])
            group2_data = np.array(embeddings_dict[term2])
            
            combined_data = np.vstack([group1_data, group2_data])
            combined_labels = np.array([0] * len(group1_data) + [1] * len(group2_data))
            
            # 计算距离矩阵
            distance_matrix = euclidean_distances(combined_data)
            
            # 执行PERMANOVA
            try:
                from skbio import DistanceMatrix
                distance_matrix = DistanceMatrix(distance_matrix)
                print("成功创建DistanceMatrix对象")
                result = permanova(distance_matrix, combined_labels, permutations=permutations)
                
                pairwise_results.append({
                    'term1': term1,
                    'term2': term2,
                    'f_statistic': result['test statistic'],
                    'p_value': result['p-value'],
                    'count1': len(group1_data),
                    'count2': len(group2_data)
                })
                
            except Exception as e:
                print(f"成对分析 {term1}-{term2} 出错: {e}")
    
    # 多重检验校正
    p_values = [result['p_value'] for result in pairwise_results]
    rejected, corrected_p, _, _ = multipletests(p_values, method='fdr_bh')
    
    for i, result in enumerate(pairwise_results):
        result['corrected_p'] = corrected_p[i]
        result['significant'] = corrected_p[i] < 0.05
    
    # 输出显著的结果
    significant_pairs = [r for r in pairwise_results if r['significant']]
    print(f"\n发现 {len(significant_pairs)} 对显著差异的GO term组合:")
    
    for result in significant_pairs[:10]:  # 只显示前10个
        print(f"{result['term1']} vs {result['term2']}: "
              f"p={result['p_value']:.6f}, adj_p={result['corrected_p']:.6f}, "
              f"F={result['f_statistic']:.4f}")
    
    if len(significant_pairs) > 10:
        print(f"... 还有 {len(significant_pairs) - 10} 对显著组合")
    
    return pairwise_results

def calculate_effect_size_distribution(embeddings_dict):
    """
    计算每个类别的效应量分布
    """
    print("\n" + "="*60)
    print("各类别效应量分析")
    print("="*60)
    
    effect_sizes = {}
    all_embeddings = np.vstack(list(embeddings_dict.values()))
    overall_mean = np.mean(all_embeddings, axis=0)
    ss_total = np.sum((all_embeddings - overall_mean)**2)
    
    for term, embeddings in embeddings_dict.items():
        if len(embeddings) > 1:
            group_data = np.array(embeddings)
            group_mean = np.mean(group_data, axis=0)
            
            # 计算该组对总变异的贡献
            ss_group = len(group_data) * np.sum((group_mean - overall_mean)**2)
            eta_squared = ss_group / ss_total
            
            effect_sizes[term] = {
                'effect_size': eta_squared,
                'sample_count': len(embeddings),
                'contribution_per_sample': eta_squared / len(embeddings) if len(embeddings) > 0 else 0
            }
    
    # 输出效应量统计
    es_values = [es['effect_size'] for es in effect_sizes.values()]
    print(f"效应量范围: {min(es_values):.6f} - {max(es_values):.6f}")
    print(f"平均效应量: {np.mean(es_values):.6f}")
    print(f"效应量分位数:")
    print(f"  5%: {np.percentile(es_values, 5):.6f}")
    print(f" 25%: {np.percentile(es_values, 25):.6f}")
    print(f" 50%: {np.percentile(es_values, 50):.6f}")
    print(f" 75%: {np.percentile(es_values, 75):.6f}")
    print(f" 95%: {np.percentile(es_values, 95):.6f}")
    
    return effect_sizes

def plot_statistical_results(permanova_result, effect_sizes, pairwise_results):
    """
    绘制统计分析结果
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # 1. 效应量分布
    es_values = [es['effect_size'] for es in effect_sizes.values()]
    axes[0, 0].hist(es_values, bins=30, alpha=0.7, color='lightblue', edgecolor='black')
    axes[0, 0].axvline(np.mean(es_values), color='red', linestyle='--', label=f'Mean: {np.mean(es_values):.4f}')
    axes[0, 0].set_title('Distribution of Effect Sizes (η²) by GO Term')
    axes[0, 0].set_xlabel('Effect Size (η²)')
    axes[0, 0].set_ylabel('Frequency')
    axes[0, 0].legend()
    
    # 2. 效应量与样本量的关系
    sample_counts = [es['sample_count'] for es in effect_sizes.values()]
    axes[0, 1].scatter(sample_counts, es_values, alpha=0.6, s=30)
    axes[0, 1].set_title('Effect Size vs Sample Count')
    axes[0, 1].set_xlabel('Sample Count')
    axes[0, 1].set_ylabel('Effect Size (η²)')
    
    # 3. 每样本贡献度
    contrib_per_sample = [es['contribution_per_sample'] for es in effect_sizes.values()]
    axes[1, 0].hist(contrib_per_sample, bins=30, alpha=0.7, color='lightgreen', edgecolor='black')
    axes[1, 0].set_title('Contribution per Sample')
    axes[1, 0].set_xlabel('η² per Sample')
    axes[1, 0].set_ylabel('Frequency')
    
    # 4. 显著成对比较的p值分布
    sig_p_values = [r['corrected_p'] for r in pairwise_results if r['significant']]
    if sig_p_values:
        axes[1, 1].hist(sig_p_values, bins=20, alpha=0.7, color='salmon', edgecolor='black')
        axes[1, 1].set_title('Distribution of Significant p-values\n(After Multiple Testing Correction)')
        axes[1, 1].set_xlabel('Adjusted p-value')
        axes[1, 1].set_ylabel('Frequency')
    else:
        axes[1, 1].text(0.5, 0.5, 'No significant pairs found', 
                       ha='center', va='center', transform=axes[1, 1].transAxes)
        axes[1, 1].set_title('No Significant Pairwise Comparisons')
    
    plt.tight_layout()
    plt.savefig('statistical_analysis_results.png', dpi=300, bbox_inches='tight')
    plt.show()

def main():
    # 加载数据
    # file_path = "data-bin/uniprotKB/cfpgen_general_dataset/train_go_terms_emb.pkl"
    file_path = "data-bin/uniprotKB/cfpgen_general_dataset/train_go_terms_cls_emb.pkl"
    # file_path = "data-bin/uniprotKB/cfpgen_general_dataset/train_go_terms_mean_emb.pkl"
    # file_path =  "data-bin/uniprotKB/cfpgen_general_dataset/train_go_terms_lfq_mean_emb.pkl"

    go_terms_emb = load_embeddings(file_path)
    
    print(f"总共加载了 {len(go_terms_emb)} 个GO terms")
    
    # 显示每个term的样本数量
    sample_counts = [len(embeddings) for embeddings in go_terms_emb.values()]
    print(f"总样本数: {sum(sample_counts)}")
    print(f"每个term平均样本数: {np.mean(sample_counts):.2f} ± {np.std(sample_counts):.2f}")
    
    # 执行PERMANOVA分析
    permanova_result = perform_permanova_analysis(go_terms_emb, permutations=999)
    
    # 执行成对分析
    # pairwise_results = perform_pairwise_permanova(go_terms_emb, permutations=999)
    pairwise_results = None
    
    # 计算效应量分布
    effect_sizes = calculate_effect_size_distribution(go_terms_emb)
    
    # 绘制统计结果
    plot_statistical_results(permanova_result, effect_sizes, pairwise_results)
    
    # 计算类内相似度
    print("\n计算类内相似度...")
    intra_similarities = calculate_intra_class_similarity(go_terms_emb)

    # exit()
    
    # 计算类间距离
    print("计算类间距离...")
    inter_distances, terms, centroids = calculate_inter_class_distance(go_terms_emb)
    
    # 输出统计结果
    print("\n=== 类内相似度统计 ===")
    intra_mean_all = np.mean([stats['mean_similarity'] for stats in intra_similarities.values()])
    intra_std_all = np.std([stats['mean_similarity'] for stats in intra_similarities.values()])
    print(f"平均类内相似度: {intra_mean_all:.4f} ± {intra_std_all:.4f}")
    
    print("\n=== 类间距离统计 ===")
    inter_cosine_all = np.mean([dist['cosine_distance'] for dist in inter_distances.values()])
    inter_std_all = np.std([dist['cosine_distance'] for dist in inter_distances.values()])
    print(f"平均类间余弦距离: {inter_cosine_all:.4f} ± {inter_std_all:.4f}")

    inter_euclidean_all = np.mean([dist['euclidean_distance'] for dist in inter_distances.values()])
    inter_std_euclidean_all = np.std([dist['euclidean_distance'] for dist in inter_distances.values()])
    print(f"平均类间欧氏距离: {inter_euclidean_all:.4f} ± {inter_std_euclidean_all:.4f}")
    
    # 深入分析
    print("\n=== 深入分析 ===")
    all_embeddings, norms = analyze_embeddings(go_terms_emb)
    
    # 计算替代指标
    # median_cosine_dist, min_inter_distances = calculate_alternative_metrics(go_terms_emb)
    # print(f"使用中位数的平均类间距离: {np.mean(median_cosine_dist):.4f}")
    
    # min_cosine_dists = [dist['min_cosine'] for dist in min_inter_distances.values()]
    # print(f"类间最小余弦距离: {np.mean(min_cosine_dists):.4f}")
    
    # 计算分离度指标
    separation_metrics = calculate_separation_metrics(go_terms_emb)
    separation_ratios = [metrics['separation_ratio'] for metrics in separation_metrics.values()]
    print(f"平均分离度比率: {np.mean(separation_ratios):.4f}")
    
    # 绘制详细分析图
    plot_detailed_analysis(go_terms_emb, intra_similarities, inter_distances)

    # # 可视化
    # print("\n生成可视化图表...")
    
    # # 使用t-SNE可视化
    # print("生成t-SNE可视化...")
    # reduced_tsne, labels_tsne = visualize_embeddings(
    #     go_terms_emb, method='tsne', title_suffix="GO Terms"
    # )
    
    # # 使用PCA可视化
    print("生成PCA可视化...")
    reduced_pca, labels_pca = visualize_embeddings(
        go_terms_emb, method='pca', title_suffix="GO Terms"
    )
    
    # # 生成密度图
    # print("生成密度可视化...")
    # plot_density_visualization(go_terms_emb, method='tsne')
    # plot_density_visualization(go_terms_emb, method='pca')
    
    # 绘制统计图表
    print("生成统计图表...")
    plot_statistics(intra_similarities, inter_distances)
    
    # 保存详细结果
    save_detailed_results(intra_similarities, inter_distances)
    
    # return intra_similarities, inter_distances, reduced_tsne, reduced_pca
    return intra_similarities, inter_distances

def save_detailed_results(intra_similarities, inter_distances):
    """保存详细结果到CSV文件"""
    # 保存类内相似度
    intra_df = pd.DataFrame.from_dict(intra_similarities, orient='index')
    intra_df.index.name = 'GO_Term'
    intra_df.to_csv('intra_class_similarity.csv')
    
    # 保存类间距离
    inter_df = pd.DataFrame.from_dict(inter_distances, orient='index')
    inter_df.to_csv('inter_class_distance.csv')
    
    print("详细结果已保存到 CSV 文件")

if __name__ == "__main__":
    main()