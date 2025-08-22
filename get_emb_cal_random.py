import pickle
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import seaborn as sns
import pandas as pd

def analyze_random_embeddings(go_terms_emb):
    """专门分析random嵌入数据"""
    
    if 'random' not in go_terms_emb:
        print("未找到 'random' 键")
        return
    
    random_embeddings = go_terms_emb['random']
    print(f"Random embeddings shape: {len(random_embeddings)} samples, each {random_embeddings[0].shape}")
    
    # 转换为numpy数组
    random_array = np.array(random_embeddings)
    
    # 1. 基本统计信息
    norms = np.linalg.norm(random_array, axis=1)
    print(f"\n=== Random Embeddings 基本统计 ===")
    print(f"范数范围: {norms.min():.4f} - {norms.max():.4f}")
    print(f"平均范数: {norms.mean():.4f} ± {norms.std():.4f}")
    
    # 2. 随机性检验 - 计算所有样本间的相似度
    cosine_sim_matrix = cosine_similarity(random_array)
    np.fill_diagonal(cosine_sim_matrix, np.nan)  # 忽略对角线
    
    print(f"\n=== 随机性分析 ===")
    print(f"所有样本间平均余弦相似度: {np.nanmean(cosine_sim_matrix):.6f}")
    print(f"所有样本间余弦相似度标准差: {np.nanstd(cosine_sim_matrix):.6f}")
    print(f"所有样本间最小余弦相似度: {np.nanmin(cosine_sim_matrix):.6f}")
    print(f"所有样本间最大余弦相似度: {np.nanmax(cosine_sim_matrix):.6f}")
    
    # 3. 欧式距离分析
    euclidean_dist_matrix = euclidean_distances(random_array)
    np.fill_diagonal(euclidean_dist_matrix, np.nan)  # 忽略对角线
    
    print(f"\n=== 欧式距离分析 ===")
    print(f"所有样本间平均欧式距离: {np.nanmean(euclidean_dist_matrix):.6f}")
    print(f"所有样本间欧式距离标准差: {np.nanstd(euclidean_dist_matrix):.6f}")
    print(f"所有样本间最小欧式距离: {np.nanmin(euclidean_dist_matrix):.6f}")
    print(f"所有样本间最大欧式距离: {np.nanmax(euclidean_dist_matrix):.6f}")
    
    # 4. 与理论期望值比较（对于高维随机向量）
    # 对于d维随机向量，期望的欧式距离大约是 sqrt(2d) * σ，其中σ是每个维度的标准差
    d = random_array.shape[1]  # 维度数
    avg_std_per_dim = np.mean(np.std(random_array, axis=0))
    expected_euclidean = np.sqrt(2 * d) * avg_std_per_dim
    print(f"理论期望欧式距离 (sqrt(2d)*σ): {expected_euclidean:.6f}")
    print(f"实际/期望比值: {np.nanmean(euclidean_dist_matrix)/expected_euclidean:.6f}")
    
    # 5. 距离分布的分位数
    euclidean_flat = euclidean_dist_matrix[~np.isnan(euclidean_dist_matrix)]
    quantiles = np.percentile(euclidean_flat, [5, 25, 50, 75, 95])
    print(f"欧式距离分位数:")
    print(f"  5%: {quantiles[0]:.6f}, 25%: {quantiles[1]:.6f}, 中位数: {quantiles[2]:.6f}")
    print(f"  75%: {quantiles[3]:.6f}, 95%: {quantiles[4]:.6f}")
    
    # 6. 距离与余弦相似度的关系分析
    # 对于归一化向量，欧式距离 = sqrt(2*(1-cosine_similarity))
    # 检查这个关系是否成立
    cosine_flat = cosine_sim_matrix[~np.isnan(cosine_sim_matrix)]
    theoretical_euclidean = np.sqrt(2 * (1 - cosine_flat))
    actual_euclidean_flat = euclidean_dist_matrix[~np.isnan(euclidean_dist_matrix)]
    
    # 计算理论值和实际值的相关性
    from scipy.stats import pearsonr
    corr, p_value = pearsonr(theoretical_euclidean, actual_euclidean_flat)
    print(f"\n=== 距离关系验证 ===")
    print(f"理论欧式距离与实际欧式距离的相关系数: {corr:.6f} (p-value: {p_value:.6e})")
    
    # 7. 检查是否归一化
    normalized_norms = np.linalg.norm(random_array / norms[:, np.newaxis], axis=1)
    is_normalized = np.allclose(normalized_norms, 1.0, atol=1e-6)
    print(f"向量是否归一化: {is_normalized}")
    if is_normalized:
        print(f"归一化验证 - 平均范数: {np.mean(normalized_norms):.6f}")
    
    return random_array, cosine_sim_matrix, euclidean_dist_matrix

def visualize_random_embeddings(random_array, method='tsne'):
    """可视化random嵌入"""
    
    # 降维
    if method == 'tsne':
        reducer = TSNE(n_components=2, random_state=42, perplexity=30)
    else:
        reducer = PCA(n_components=2, random_state=42)
    
    reduced_embeddings = reducer.fit_transform(random_array)
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # 1. 散点图
    axes[0].scatter(reduced_embeddings[:, 0], reduced_embeddings[:, 1], 
                   alpha=0.6, s=20, c='blue')
    axes[0].set_title(f'Random Embeddings - {method.upper()}')
    axes[0].set_xlabel('Component 1')
    axes[0].set_ylabel('Component 2')
    
    # 2. 密度图
    hb = axes[1].hexbin(reduced_embeddings[:, 0], reduced_embeddings[:, 1], 
                       gridsize=30, cmap='viridis', alpha=0.8)
    plt.colorbar(hb, ax=axes[1])
    axes[1].set_title(f'Density - {method.upper()}')
    axes[1].set_xlabel('Component 1')
    axes[1].set_ylabel('Component 2')
    
    # 3. 距离分布直方图
    pairwise_distances = euclidean_distances(reduced_embeddings).flatten()
    # 移除对角线元素（距离为0）
    pairwise_distances = pairwise_distances[pairwise_distances > 1e-10]
    
    axes[2].hist(pairwise_distances, bins=50, alpha=0.7, color='green', edgecolor='black')
    axes[2].set_title('Pairwise Distance Distribution')
    axes[2].set_xlabel('Euclidean Distance')
    axes[2].set_ylabel('Frequency')
    
    plt.tight_layout()
    plt.savefig(f'random_embeddings_{method}.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return reduced_embeddings

def plot_random_statistics(random_array, cosine_sim_matrix):
    """绘制random数据的统计图表"""
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # 1. 范数分布
    norms = np.linalg.norm(random_array, axis=1)
    axes[0, 0].hist(norms, bins=30, alpha=0.7, color='skyblue', edgecolor='black')
    axes[0, 0].set_title('Embedding Norm Distribution')
    axes[0, 0].set_xlabel('Vector Norm')
    axes[0, 0].set_ylabel('Frequency')
    
    # 2. 余弦相似度分布
    cosine_vals = cosine_sim_matrix[~np.isnan(cosine_sim_matrix)]
    axes[0, 1].hist(cosine_vals, bins=50, alpha=0.7, color='lightcoral', edgecolor='black')
    axes[0, 1].set_title('Cosine Similarity Distribution')
    axes[0, 1].set_xlabel('Cosine Similarity')
    axes[0, 1].set_ylabel('Frequency')
    axes[0, 1].axvline(x=np.mean(cosine_vals), color='red', linestyle='--', 
                      label=f'Mean: {np.mean(cosine_vals):.6f}')
    axes[0, 1].legend()
    
    # 3. 相关性热图（前50个样本）
    n_samples = min(50, len(random_array))
    small_cosine_matrix = cosine_similarity(random_array[:n_samples])
    im = axes[0, 2].imshow(small_cosine_matrix, cmap='coolwarm', vmin=-1, vmax=1)
    axes[0, 2].set_title(f'Cosine Similarity Matrix (First {n_samples} samples)')
    plt.colorbar(im, ax=axes[0, 2])
    
    # 4. 维度方差
    var_per_dim = np.var(random_array, axis=0)
    axes[1, 0].plot(var_per_dim, alpha=0.7)
    axes[1, 0].set_title('Variance per Dimension')
    axes[1, 0].set_xlabel('Dimension')
    axes[1, 0].set_ylabel('Variance')
    
    # 5. 累积方差解释
    pca = PCA().fit(random_array)
    axes[1, 1].plot(np.cumsum(pca.explained_variance_ratio_), alpha=0.7)
    axes[1, 1].set_title('Cumulative Explained Variance')
    axes[1, 1].set_xlabel('Number of Components')
    axes[1, 1].set_ylabel('Cumulative Variance Ratio')
    axes[1, 1].axhline(y=0.95, color='red', linestyle='--', label='95% variance')
    axes[1, 1].legend()
    
    # 6. 随机性检验 - QQ图（与高斯分布比较）
    from scipy import stats
    flattened = random_array.flatten()
    stats.probplot(flattened, dist="norm", plot=axes[1, 2])
    axes[1, 2].set_title('QQ Plot vs Normal Distribution')
    
    plt.tight_layout()
    plt.savefig('random_embeddings_statistics.png', dpi=300, bbox_inches='tight')
    plt.show()

def compare_with_other_terms(go_terms_emb, random_array):
    """将random数据与其他GO terms比较"""
    
    other_terms = [key for key in go_terms_emb.keys() if key != 'random']
    
    if not other_terms:
        print("没有其他GO terms用于比较")
        return
    
    # 随机选择几个其他terms进行比较
    comparison_terms = other_terms[:min(5, len(other_terms))]
    
    print(f"\n=== 与其他GO terms比较 ===")
    
    comparison_results = {}
    for term in comparison_terms:
        term_embeddings = np.array(go_terms_emb[term])
        term_centroid = np.mean(term_embeddings, axis=0)
        
        # 计算random样本到该term中心的平均距离
        distances_to_centroid = np.linalg.norm(random_array - term_centroid, axis=1)
        avg_distance = np.mean(distances_to_centroid)
        
        # 计算余弦相似度
        cosine_sims = cosine_similarity(random_array, term_centroid.reshape(1, -1))
        avg_cosine_sim = np.mean(cosine_sims)
        
        comparison_results[term] = {
            'avg_distance': avg_distance,
            'avg_cosine_sim': avg_cosine_sim,
            'sample_count': len(term_embeddings)
        }
        
        print(f"Term {term}: 平均距离={avg_distance:.4f}, 平均余弦相似度={avg_cosine_sim:.4f}")
    
    return comparison_results

def main_random_analysis():
    """主函数：专门分析random数据"""
    
    # 加载数据
    file_path = "data-bin/uniprotKB/cfpgen_general_dataset/train_go_terms_cls_emb_500_random.pkl"
    with open(file_path, "rb") as f:
        go_terms_emb_random = pickle.load(f)
    
    file_path = "data-bin/uniprotKB/cfpgen_general_dataset/train_go_terms_cls_emb.pkl"
    with open(file_path, "rb") as f:
        go_terms_emb = pickle.load(f)

    go_terms_emb['random'] = go_terms_emb_random['random']
    print("开始分析random嵌入数据...")
    
    # 分析random数据
    random_array, cosine_sim_matrix, euclidean_dist_matrix = analyze_random_embeddings(go_terms_emb)
    
    # # 可视化
    # print("\n生成t-SNE可视化...")
    # reduced_tsne = visualize_random_embeddings(random_array, method='tsne')
    
    # print("生成PCA可视化...")
    # reduced_pca = visualize_random_embeddings(random_array, method='pca')
    
    # 统计图表
    print("生成统计图表...")
    # plot_random_statistics(random_array, cosine_sim_matrix)
    
    # 与其他terms比较
    comparison_results = compare_with_other_terms(go_terms_emb, random_array)
    
    # 保存结果
    results = {
        'random_array': random_array,
        'cosine_sim_matrix': cosine_sim_matrix,
        # 'reduced_tsne': reduced_tsne,
        # 'reduced_pca': reduced_pca,
        'comparison_results': comparison_results
    }
    
    with open('random_analysis_results.pkl', 'wb') as f:
        pickle.dump(results, f)
    
    print("\n分析完成！结果已保存到 random_analysis_results.pkl")
    
    return results

if __name__ == "__main__":
    results = main_random_analysis()