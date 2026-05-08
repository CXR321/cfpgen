import os
import pickle
import numpy as np
import networkx as nx
from numpy.core.fromnumeric import trace
from tqdm import tqdm

# ================= 配置 =================
TRAIN_PATH = 'data-bin/uniprotKB/cfpgen_general_dataset/train_all_old_motif_added_pfamMotif_esmfold_pfamEmb.pkl'
GO_MAPPING_PATH = 'go_mapping.pkl'
GO_OBO_PATH = 'go-basic.obo'

# 目标 ID 列表
TARGET_IDS = ['GO:0005212', 'GO:0042802'] 

# 输出文件
OUTPUT_CSV = 'target_go_farthest_distance.csv'

# ================= 1. 构建 GO 图 =================
def load_go_graph_undirected(obo_path):
    print(f"Loading GO Ontology from {obo_path}...")
    if not os.path.exists(obo_path):
        raise FileNotFoundError(f"请下载 go-basic.obo 文件并放置在 {obo_path}")
    
    # 使用 DiGraph 读取，以便获取 descendants
    G_dir = nx.DiGraph()
    # 使用 Graph (无向) 计算语义距离
    G_undir = nx.Graph() 
    
    with open(obo_path, 'r') as f:
        current_id = ""
        for line in f:
            line = line.strip()
            if line.startswith("id: GO:"):
                current_id = line.split("id: ")[1]
                G_dir.add_node(current_id)
                G_undir.add_node(current_id)
            elif line.startswith("is_a:"):
                parent_id = line.split("is_a: ")[1].split(" ! ")[0]
                if current_id:
                    G_dir.add_edge(current_id, parent_id) # Child -> Parent
                    G_undir.add_edge(current_id, parent_id)
            elif line.startswith("relationship: part_of"):
                parent_id = line.split("relationship: part_of ")[1].split(" ! ")[0]
                if current_id:
                    G_dir.add_edge(current_id, parent_id)
                    G_undir.add_edge(current_id, parent_id)
    
    print(f"GO Graph loaded. Nodes: {len(G_undir)}")
    return G_dir, G_undir

# ================= 2. 主逻辑 =================
def main():
    # 1. 加载资源
    G_dir, G_undir = load_go_graph_undirected(GO_OBO_PATH)
    
    print(f"Loading mapping from {GO_MAPPING_PATH}...")
    with open(GO_MAPPING_PATH, 'rb') as f:
        go_mapping = pickle.load(f)
    index_to_go = {v: k for k, v in go_mapping.items()}
    
    print(f"Loading training data from {TRAIN_PATH}...")
    with open(TRAIN_PATH, 'rb') as f:
        train_data = pickle.load(f)

    # print(train_data[0])
    # exit()

    # 2. 扩展目标集合 (包含所有子节点)
    # 因为蛋白可能标记的是具体的子功能，而不是根节点
    print("Expanding target IDs to include all descendants...")
    target_family = set()
    for tid in TARGET_IDS:
        if tid in G_dir:
            target_family.add(tid)
            # descendants 返回的是所有子孙节点
            target_family.update(nx.ancestors(G_dir, tid)) 
            # 注意: nx.ancestors 在 Child->Parent 的图中实际上是获取 Parent (祖先)
            # 但在 networkx 读取 OBO 时通常边是 Term -> Parent
            # 让我们确认方向:
            # 在 OBO 中 "A is_a B" 意味着 A 是子, B 是父。
            # 如果我们存的是 edge(A, B)，那么 ancestors(A) 是 B, C... (泛化)
            # 我们需要的是 "specific terms" (descendants)。
            # 在 edge(Child, Parent) 的图中，子节点实际上是 source 指向 target
            # 所以我们需要反转图或者用 predecessors
            
    # 更正：获取所有属于这些类别的具体子项
    # 为了保险，我们用最笨的办法：遍历全图找到所有指向 Target 的节点
    # 或者简单点：如果 G_dir 是 Child -> Parent，我们需要 descendants (x->Target)
    # 也就是 nx.ancestors(G_dir_reversed, Target)
    
    # 重新构建一个 Parent -> Child 的图方便查找子节点
    G_parent_to_child = nx.DiGraph()
    for u, v in G_dir.edges():
        G_parent_to_child.add_edge(v, u) # Parent -> Child
    
    valid_target_terms = set(TARGET_IDS)
    for tid in TARGET_IDS:
        if tid in G_parent_to_child:
            valid_target_terms.update(nx.descendants(G_parent_to_child, tid))
            
    print(f"Total valid target terms (including children): {len(valid_target_terms)}")

    # 3. 搜索与评分
    results = []
    
    print("Scanning training set...")
    for entry in tqdm(train_data):
        pid = entry['uniprot_id']
        go_ids = [index_to_go[i] for i in entry['go_f_mapped']]
        go_set = set(go_ids)
        
        # 筛选条件：必须包含至少一个目标家族的标签
        # (即这个蛋白确实属于这两个类别之一)
        intersection = go_set.intersection(valid_target_terms)
        if not intersection:
            continue
            
        # 计算距离指标
        # 这里的定义：该蛋白身上 *所有* 标签，距离目标根节点 (TARGET_IDS) 的平均距离。
        # 如果蛋白很纯 (只有目标子类)，距离就是它在树里的深度 (通常较小，或者就是0如果它就是根)。
        # 如果蛋白很杂 (含有完全无关的功能)，那个无关功能的距离会非常大。
        
        dists = []
        for go_term in go_ids:
            # 计算该词到最近的一个 Target Root 的距离
            term_dists = []
            for target_root in TARGET_IDS:
                if go_term in G_undir and target_root in G_undir:
                    try:
                        d = nx.shortest_path_length(G_undir, source=go_term, target=target_root)
                        term_dists.append(d)
                    except nx.NetworkXNoPath:
                        term_dists.append(100) # 无路径惩罚
                else:
                    term_dists.append(100) # 缺失节点惩罚
            
            if term_dists:
                dists.append(min(term_dists)) # 取到两个目标中较近的那个距离
        
        avg_dist = np.mean(dists) if dists else 0
        
        results.append({
            'id': pid,
            'avg_dist': avg_dist,
            'labels': go_ids,
            'matched_targets': list(intersection),
            'motif': entry['motif']
        })

    # 4. 排序 (越远越好 -> Reverse=True)
    # 距离大 = 含有语义上离目标很远的其他标签 = "不纯/多功能"
    results.sort(key=lambda x: x['avg_dist'], reverse=True)
    
    # 5. 输出
    print("\n" + "="*80)
    print(f"Top 20 Proteins (Farthest semantic distance from targets)")
    print("These proteins contain the target function BUT also have very distant/unrelated functions.")
    print("="*80)
    
    print(f"{'UniProt ID':<15} | {'Avg Dist':<10} | {'Labels Count':<12} | {'Example Labels'}")
    print("-" * 80)
    
    for res in results[:20]:
        labels_str = ', '.join(res['labels'])
        if len(labels_str) > 50: labels_str = labels_str[:47] + "..."
        print(f"{res['id']:<15} | {res['avg_dist']:<10.2f} | {len(res['labels']):<12} | {labels_str}")

    # 保存 CSV
    import csv
    with open(OUTPUT_CSV, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['uniprot_id', 'avg_semantic_distance', 'num_labels', 'all_labels', 'matched_target_terms', 'motif'])
        for res in results:
            writer.writerow([res['id'], res['avg_dist'], len(res['labels']), ','.join(res['labels']), ','.join(res['matched_targets']), ','.join(str(res['motif']))])
            
    print(f"\nSaved all {len(results)} entries to {OUTPUT_CSV}")

if __name__ == '__main__':
    main()