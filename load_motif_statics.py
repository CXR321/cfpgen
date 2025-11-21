import os
import pickle
import torch
import numpy as np



file_path = os.path.join('./data-bin/uniprotKB/cfpgen_general_dataset/', 'train' + '_all_old_motif_added_pfamMotif_esmfold_pfamEmb.pkl')

assert os.path.isfile(file_path)
with open(file_path, 'rb') as f:
    train_data = pickle.load(f)

motif_labels_lens = [0 for i in range(375)]

for data in train_data:
    motif_labels = data['motif_go_number']
    for label in motif_labels:
        motif_labels_lens[label] += 1

print(motif_labels_lens)
print(sum(motif_labels_lens))

NUM_CLASSES = len(motif_labels_lens) # 375

# -----------------
# 1. 转换为 Numpy 数组进行计算
# -----------------
counts = np.array(motif_labels_lens)

# -----------------
# 2. 计算权重
# -----------------

# (A) 找出计数大于0的类别索引
valid_indices = counts > 0

# (B) 初始化权重：所有类别权重默认为 0
# 计数为 0 的类别最终权重将保持为 0，从而排除它们对 Loss 的贡献。
weights = np.zeros(NUM_CLASSES, dtype=np.float32)

# (C) 计算有效类别的逆频率权重
# 逆频率： 1 / N_c
# 这样做能保证样本越少的类别，获得的权重越大。
weights[valid_indices] = 1.0 / counts[valid_indices]

# -----------------
# 3. 归一化 (可选但推荐)
# -----------------
# 归一化可以防止权重过大或过小，使总的损失值保持在一个合理的区间。
# 我们将权重归一化到它们的平均值，或者直接归一化到最大值。

# 方式一：归一化到平均值 (常用)
# 仅对有效类别进行归一化
normalized_weights = weights.copy()
mean_valid_weight = np.mean(weights[valid_indices])
normalized_weights[valid_indices] /= mean_valid_weight
# 零计数类别的权重仍保持为 0

# -----------------
# 4. 转换为 PyTorch Tensor
# -----------------

# 构建最终的 PyTorch 权重 Tensor
class_weights_tensor = torch.tensor(normalized_weights, dtype=torch.float32)

with open('go_number_class_weights.pkl', 'wb') as f:
    pickle.dump(class_weights_tensor, f)

print(f"总类别数: {NUM_CLASSES}")
print(f"计数大于0的类别数: {np.sum(valid_indices)}")
print(f"权重Tensor形状: {class_weights_tensor.shape}")
print("前几个有效权重示例:", class_weights_tensor[class_weights_tensor > 0][:5])
print("零计数类别的权重 (保持为0):", class_weights_tensor[counts == 0][:5])