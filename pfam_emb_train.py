import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random

from get_emb_cal import load_embeddings
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

file_path = "data-bin/uniprotKB/cfpgen_general_dataset/train_pfam_cls_emb_pfamMotif.pkl"

go_terms_emb = load_embeddings(file_path)

train_pfam_data = load_embeddings("data-bin/uniprotKB/cfpgen_general_dataset/train_all_old_motif_added_pfamMotif_esmfold.pkl")

go_term_pfam_name_list = []

for data in train_pfam_data:
    for pfam in data['pfam_motif']:
        if len(pfam['strong_go_id']) > 0:
            if pfam['pfam_id'] not in go_term_pfam_name_list:
                go_term_pfam_name_list.append(pfam['pfam_id'])

print(len(go_term_pfam_name_list))

go_terms_emb = {key: value for key, value in go_terms_emb.items() if key in go_term_pfam_name_list}

filtered_go_terms_emb = {}

for key, value_list in go_terms_emb.items():
    filtered_values = []
    for value_set in value_list:

        feature_vector, e_value = value_set 
        if e_value <= 0.05:
            filtered_values.append(value_set)
            # 如果数据结构不同，可以根据实际情况调整
    
    # 只有当过滤后还有数据时才保留这个key
    if filtered_values:
        filtered_go_terms_emb[key] = filtered_values

# 使用过滤后的数据
go_terms_emb = filtered_go_terms_emb    

# print(go_terms_emb[list[go_terms_emb.keys()][0]])

sorted_keys = sorted(go_terms_emb.keys(), 
                    key=lambda x: len(go_terms_emb[x]), 
                    reverse=True)[:200]

# sorted_keys = random.choices(sorted_keys, k=10)

# print(sorted_keys)

# sorted_keys

# exit()



sub_dataset = {}

for key in sorted_keys:
    value_list = go_terms_emb[key]
    feature_vectors = []
    
    # 遍历list中的每个set
    for value_set in value_list:
        # 遍历set中的每个tuple
        feature_vector, e_value = value_set
        # 只保留特征向量，去掉e-value
        feature_vectors.append(feature_vector)
    
    # 将特征向量列表存入子数据集
    sub_dataset[key] = feature_vectors

# exit()

go_terms_emb = sub_dataset


# ==== 1. 准备数据 ====
X, y, label2id = [], [], {}
for i, (key, value_list) in enumerate(go_terms_emb.items()):
    label2id[key] = i
    for vec in value_list:
        X.append(vec)
        y.append(i)

X = torch.tensor(np.array(X), dtype=torch.float32).to(device)
y = torch.tensor(np.array(y), dtype=torch.long).to(device)

# 划分 train / valid (8:2)
perm = torch.randperm(len(X))
split = int(0.8 * len(X))
train_idx, valid_idx = perm[:split], perm[split:]

X_train, y_train = X[train_idx], y[train_idx]
X_valid, y_valid = X[valid_idx], y[valid_idx]

input_dim = X.shape[1]
num_classes = len(label2id)

# ==== 2. 定义 embedding 网络 ====
# 简单 MLP，也可以换成更复杂的 encoder
class Encoder(nn.Module):
    def __init__(self, input_dim, hidden_dim=1280, out_dim=1280):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            # nn.ReLU(),
            # nn.Linear(hidden_dim, out_dim)
        )

    def forward(self, x):
        return self.net(x)

encoder = Encoder(input_dim)

encoder.to(device)

# ==== 3. ProtoNet 分类函数 ====
def prototypical_loss(encoder, support_x, support_y, query_x, query_y):
    # 1. 计算 support embeddings
    z_support = encoder(support_x)  # [Ns, d]
    z_query = encoder(query_x)      # [Nq, d]

    # 2. 计算每类 prototype
    prototypes = []
    for c in torch.unique(support_y):
        prototypes.append(z_support[support_y == c].mean(dim=0))
    prototypes = torch.stack(prototypes)  # [C, d]

    # 3. 计算 query 到 prototype 的距离
    dists = torch.cdist(z_query, prototypes)  # [Nq, C]

    # 4. 交叉熵 loss（最近的 prototype）
    log_p_y = torch.log_softmax(-dists, dim=1)
    target_inds = torch.arange(len(query_y))
    loss = nn.NLLLoss()(log_p_y, query_y)
    acc = (log_p_y.argmax(dim=1) == query_y).float().mean().item()
    return loss, acc

# ==== 4. 训练 ====
optimizer = optim.Adam(encoder.parameters(), lr=1e-3)

for epoch in range(40):
    # --- Train ---
    idx = torch.randperm(len(X_train), device=device)
    support_size = int(0.5 * len(idx))
    support_idx, query_idx = idx[:support_size], idx[support_size:]
    loss_train, acc_train = prototypical_loss(
        encoder, X_train[support_idx], y_train[support_idx],
        X_train[query_idx], y_train[query_idx]
    )

    optimizer.zero_grad()
    loss_train.backward()
    optimizer.step()

    # --- Valid ---
    with torch.no_grad():
        idx = torch.randperm(len(X_valid), device=device)
        support_size = int(0.5 * len(idx))
        support_idx, query_idx = idx[:support_size], idx[support_size:]
        loss_val, acc_val = prototypical_loss(
            encoder, X_valid[support_idx], y_valid[support_idx],
            X_valid[query_idx], y_valid[query_idx]
        )

    if (epoch+1) % 1 == 0:
        print(f"Epoch {epoch+1}: "
              f"Train Loss={loss_train.item():.4f}, Acc={acc_train:.4f} | "
              f"Valid Loss={loss_val.item():.4f}, Acc={acc_val:.4f}")


# ==== 5. 得到最终 prototype ====
with torch.no_grad():
    Z = encoder(X)  # 全部样本的 embedding
    class_embeddings = {}
    proto_vecs, proto_labels = [], []

    for key, idx in label2id.items():
        proto = Z[y == idx].mean(dim=0)
        class_embeddings[key] = proto.cpu().numpy()
        proto_vecs.append(proto.cpu().numpy())
        proto_labels.append(key)

# ==== 6. PCA 可视化 ====
pca = PCA(n_components=2)
proto_2d = pca.fit_transform(np.array(proto_vecs))

plt.figure(figsize=(8,6))
for i, label in enumerate(proto_labels):
    x, y_ = proto_2d[i]
    plt.scatter(x, y_, marker='o', s=60)
    plt.text(x+0.02, y_, label, fontsize=8)  # 标上 GO term
plt.title("Prototypical Network Class Embeddings (PCA)")
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.grid(True)

plt.savefig("proto_net_class_embeddings.png", dpi=300)