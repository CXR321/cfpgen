import math
from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np

class SimpleCrossAttention(nn.Module):
    """
    一个极简的 Cross-Attention 模块。
    它只包含 Q, K 的线性层和 scaled dot-product attention。
    """
    def __init__(self, d_model, d_k, d_v):
        super().__init__()
        # 简化：我们只用一个头
        self.d_k = d_k

        # Q 和 K 的线性投影层
        self.query_proj = nn.Linear(d_model, d_k, bias=False)
        self.key_proj = nn.Linear(d_model, d_k, bias=False)

    def forward(self, query_input, key_input):
        """
        Args:
            query_input (Tensor): (B, N_Q, D_Model)  # e.g., Decoder input
            key_input (Tensor):   (B, N_K, D_Model)  # e.g., Encoder output

        Returns:
            attn_weights (Tensor): (B, N_Q, N_K)
        """
        # 1. 线性投影
        Q = self.query_proj(query_input)  # (B, N_Q, D_K)
        K = self.key_proj(key_input)      # (B, N_K, D_K)

        # 2. 相似度计算 (Scaled Dot Product)
        # Q @ K^T / sqrt(D_K)
        # K.transpose(-2, -1) 得到 (B, D_K, N_K)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / np.sqrt(self.d_k)
        # scores 形状为 (B, N_Q, N_K)

        # 3. Softmax 得到 Attention Weights
        attn_weights = F.softmax(scores, dim=-1) # 对 Key 的维度 (N_K) 求 softmax

        return attn_weights

MASK_VALUE = -1e9 


class MultiHeadCrossAttention(nn.Module):
    """
    一个功能完整的 Multi-Head Cross-Attention 模块，支持 Key Padding Mask。
    """
    def __init__(self, d_model, n_heads):
        super().__init__()
        
        assert d_model % n_heads == 0
        
        self.n_heads = n_heads
        self.d_k = d_model // n_heads # 每个头的 Q/K 维度
        self.d_v = d_model // n_heads # 每个头的 V 维度
        self.d_model = d_model

        # Q, K, V 的总线性投影层
        self.query_proj = nn.Linear(d_model, d_model, bias=False)
        self.key_proj = nn.Linear(d_model, d_model, bias=False)
        self.value_proj = nn.Linear(d_model, d_model, bias=False) 

        # 最终输出投影层
        self.output_proj = nn.Linear(d_model, d_model, bias=False)

    def forward(self, query_input, key_input, value_input, key_padding_mask=None):
        """
        Args:
            query_input (Tensor):   (B, N_Q, D_Model)  # Q 的输入
            key_input (Tensor):     (B, N_K, D_Model)  # K 的输入
            value_input (Tensor):   (B, N_K, D_Model)  # V 的输入
            key_padding_mask (Tensor): (B, N_K)      # 掩码张量，True/1 表示要忽略的位置 (Padding)
                                                     # 默认为 None (无掩码)

        Returns:
            output (Tensor):       (B, N_Q, D_Model) - 最终的上下文向量
            attn_weights (Tensor): (B, N_Q, N_K)     - 所有头的平均注意力权重
        """
        B, N_Q, D = query_input.shape
        _, N_K, _ = key_input.shape
        
        # 1. 线性投影和分头 (与之前相同)
        Q = self.query_proj(query_input).view(B, N_Q, self.n_heads, self.d_k).transpose(1, 2)
        K = self.key_proj(key_input).view(B, N_K, self.n_heads, self.d_k).transpose(1, 2)
        V = self.value_proj(value_input).view(B, N_K, self.n_heads, self.d_v).transpose(1, 2)
        
        # 2. 相似度计算 (Scaled Dot Product)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        # scores 形状为 (B, H, N_Q, N_K)

        # 3. ⭐️ 应用 Key Padding Mask 逻辑 ⭐️
        if key_padding_mask is not None:
            # Mask 形状为 (B, N_K)。我们需要将其扩展以匹配 scores 的形状 (B, H, N_Q, N_K)
            # 扩展到 (B, 1, 1, N_K)，然后广播到所有头 (H) 和所有 Query 位置 (N_Q)
            
            # Key Padding Mask 应该是一个布尔张量 (True 表示 mask) 或 0/1 张量 (1 表示 mask)
            # 我们假设输入的是 True/False (或 1/0)
            
            # unsqueeze(1) 增加 Head 维度 (H=1)
            # unsqueeze(2) 增加 Query 维度 (N_Q=1)
            mask_view = key_padding_mask.unsqueeze(1).unsqueeze(2) # (B, 1, 1, N_K)

            # 使用 mask_view 填充 scores。True/1 的位置会被填充为 -1e9
            # 这样 Softmax(MASK_VALUE) 就会趋近于 0
            scores.masked_fill_(mask_view, MASK_VALUE)

        # 4. Softmax 得到 Attention Weights (A)
        attn_weights_all_heads = F.softmax(scores, dim=-1) # (B, H, N_Q, N_K)

        # 5. 计算 Context Vector (C)
        context_vectors = torch.matmul(attn_weights_all_heads, V) # (B, H, N_Q, D_V)

        # 6. 拼接 Context Vector 和最终输出投影 (与之前相同)
        context_vectors = context_vectors.transpose(1, 2).contiguous().view(B, N_Q, self.d_model)
        output = self.output_proj(context_vectors) 
        
        # 返回平均注意力权重
        avg_attn_weights = torch.mean(attn_weights_all_heads, dim=1) 

        return output, avg_attn_weights

class NonSoftmaxMultiheadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0.0, bias=True):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == self.embed_dim, "embed_dim 必须能被 num_heads 整除"
        
        # 线性投影层 (W_Q, W_K, W_V)
        # 我们将所有 QKV 投影合并到一个层中，简化操作（与 nn.MultiheadAttention 类似）
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        
        # 最后的输出投影层 (W_O)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        
        self.dropout = nn.Dropout(dropout)
        self.scaling = self.head_dim ** -0.5 # 缩放因子 1/sqrt(d_k)
        
        # 初始化权重 (使用与Transformer一致的初始化)
        self._reset_parameters()

    def _reset_parameters(self):
        # 经典的 Transformer 初始化
        nn.init.xavier_uniform_(self.q_proj.weight)
        nn.init.xavier_uniform_(self.k_proj.weight)
        nn.init.xavier_uniform_(self.v_proj.weight)
        nn.init.xavier_uniform_(self.out_proj.weight)
        if self.q_proj.bias is not None:
            nn.init.constant_(self.q_proj.bias, 0.)
            nn.init.constant_(self.k_proj.bias, 0.)
            nn.init.constant_(self.v_proj.bias, 0.)
            nn.init.constant_(self.out_proj.bias, 0.)

    def forward(
            self,
            query: torch.Tensor,
            key: torch.Tensor,
            value: torch.Tensor,
            key_padding_mask: Optional[torch.Tensor] = None,
            attn_mask: Optional[torch.Tensor] = None, # <-- 新增 attn_mask 参数
            need_weights: bool = True,
        ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
            
            B, Lq, D = query.shape
            Lk = key.shape[1]
            
            # 1-3. Q, K, V 投影与多头分割 (不变)
            # ... (略去 QKV 投影和分割代码)
            q = self.q_proj(query) # [B, Lq, D]
            k = self.k_proj(key)   # [B, Lk, D]
            v = self.v_proj(value) # [B, Lk, D]
            
            q = q.view(B, Lq, self.num_heads, self.head_dim).transpose(1, 2) # [B, H, Lq, head_dim]
            k = k.view(B, Lk, self.num_heads, self.head_dim).transpose(1, 2) # [B, H, Lk, head_dim]
            v = v.view(B, Lk, self.num_heads, self.head_dim).transpose(1, 2) # [B, H, Lk, head_dim]

            # 4. 计算注意力得分
            attn_scores = torch.matmul(q, k.transpose(-2, -1)) * self.scaling 
            # attn_scores 形状: [B, H, Lq, Lk]


            # print(f"key_padding_mask: {key_padding_mask}")
            
            # 5. 应用 Key Padding Mask (如果存在)
            if key_padding_mask is not None:
                # key_padding_mask shape: [B, Lk]
                # 扩展 mask 以匹配 attn_scores [B, H, Lq, Lk]
                attn_scores.masked_fill_(
                    key_padding_mask.unsqueeze(1).unsqueeze(2), 
                    -math.inf # 用一个非常小的负数填充，保证该位置的 score 极低
                )
            
            # 6. ⚠️ 关键修改：应用 Attention Mask (如果存在)
            if attn_mask is not None:
                raise NotImplementedError("Attention mask is not supported for NonSoftmaxMultiheadAttention")

            # 7. 应用 Sigmoid 激活（非 Softmax）
            # 低得分（如 -1e9）经过 Sigmoid 后会非常接近 0，从而实现忽略 Mask 位置的目的。
            # attn_weights = torch.sigmoid(attn_scores) 
            attn_weights = F.softmax(attn_scores, dim=-1)

            # print(f"attn_weights: {attn_weights}")
            
            # 8-10. 加权求和，合并多头，最终输出投影 (不变)
            attn_output = torch.matmul(attn_weights, v) # [B, H, Lq, head_dim]
            attn_output = attn_output.transpose(1, 2).contiguous().view(B, Lq, D) # [B, Lq, D]
            attn_output = self.out_proj(attn_output)
            attn_output = self.dropout(attn_output)

            final_scores = None
            if True:
                # 返回所有头的平均 Sigmoid 权重，[B, Lq, Lk]
                final_scores = attn_weights[:,0,:,:] 
                
            return attn_output, final_scores

def attention_weight_matching_loss(current_weights, target_weights):
    """
    计算当前注意力权重和目标注意力权重之间的均方误差 (MSE)。
    
    Args:
        current_weights (Tensor): (B, N_Q, N_K)
        target_weights (Tensor): (B, N_Q, N_K)
        
    Returns:
        loss (Tensor): 标量
    """
    return F.mse_loss(current_weights, target_weights)

# --- 实验参数 ---
D_MODEL = 16    # 模型的特征维度
D_K = 8         # Q/K 的维度
N_Q = 4         # Query 的序列长度
N_K = 5         # Key 的序列长度
BATCH_SIZE = 1  # 每次只用一个 batch
LR = 1e-4       # 学习率
EPOCHS = 1500    # 训练轮数

key_padding_mask = torch.tensor([
    # [False, False, False, False, True],  # Batch 0: 忽略索引 4
    [False, False, False, True, True]   # Batch 1: 忽略索引 3, 4
], dtype=torch.bool)


def init_attention_with_strong_bias(multihead_attn):
    hidden_size = multihead_attn.embed_dim
    num_heads = multihead_attn.num_heads
    
    with torch.no_grad():
        # 1. Query投影：使用较大的初始方差
        nn.init.normal_(multihead_attn.in_proj_weight[:hidden_size], 
                       mean=0.0, std=0.02)
        
        # 2. Key投影：使用不同的初始化，创造不对称性
        nn.init.normal_(multihead_attn.in_proj_weight[hidden_size:2*hidden_size], 
                       mean=0.0, std=0.025)
        
        # 3. Value投影：标准初始化
        nn.init.xavier_uniform_(multihead_attn.in_proj_weight[2*hidden_size:])
        
        # 4. 为query和key添加故意的不对称偏置
        if multihead_attn.in_proj_bias is not None:
            # 创建交替的正负偏置模式，促进稀疏性
            bias_pattern = torch.zeros(hidden_size * 2)
            for i in range(0, hidden_size * 2, num_heads):
                bias_pattern[i:i+num_heads] = torch.linspace(-0.2, 0.2, num_heads)
            multihead_attn.in_proj_bias[:hidden_size*2] = bias_pattern
    
    # 5. 输出投影
    nn.init.xavier_uniform_(multihead_attn.out_proj.weight, gain=0.5)
    nn.init.constant_(multihead_attn.out_proj.bias, 0.0)

def init_cross_attn_with_bias(multihead_attn):
    # 初始化query和key的投影矩阵，增加区分度
    nn.init.xavier_uniform_(multihead_attn.in_proj_weight, gain=1.0)
    nn.init.xavier_uniform_(multihead_attn.out_proj.weight, gain=1.0)
    
    # 为in_proj_weight添加偏置，强化query-key的差异
    with torch.no_grad():
        # 分离query, key, value的权重 (前2/3是query+key，后1/3是value)
        hidden_size = multihead_attn.embed_dim
        qk_weight = multihead_attn.in_proj_weight[:2*hidden_size]
        
        # 添加一个促进稀疏性的偏置
        # 这会让某些attention head更关注特定的位置
        bias_matrix = torch.randn_like(qk_weight) * 0.1
        multihead_attn.in_proj_weight[:2*hidden_size] += bias_matrix
    
    # 初始化偏置项
    if multihead_attn.in_proj_bias is not None:
        nn.init.constant_(multihead_attn.in_proj_bias, 0.1)
    nn.init.constant_(multihead_attn.out_proj.bias, 0.0)

# --- 1. 实例化模型、优化器 ---
# model = SimpleCrossAttention(D_MODEL, D_K, D_K)
# model = MultiHeadCrossAttention(D_MODEL, 4)
# model = NonSoftmaxMultiheadAttention(D_MODEL, 4)
model = nn.MultiheadAttention(
            embed_dim=D_MODEL,
            num_heads=4,
            dropout=0.0,
            batch_first=True,
            )

# init_cross_attn_with_bias(model)
init_attention_with_strong_bias(model)

optimizer = optim.Adam(model.parameters(), lr=LR)

# --- 2. 准备数据 (一个 batch) ---
# 随机生成 Query 和 Key 的输入
query_input = torch.randn(BATCH_SIZE, N_Q, D_MODEL) # (1, 4, 16)
key_input = torch.randn(BATCH_SIZE, N_K, D_MODEL)   # (1, 6, 16)
value_input = key_input # 这里假设 Key 和 Value 相同

# --- 3. 定义目标注意力权重 (Target Weights) ---
# 目标权重必须满足 Softmax 的约束：行和为 1，且元素非负。
# 我们随机生成一个 (N_Q, N_K) 的矩阵，然后手动 Softmax 来保证合法性。

# 目标 Softmax 的 logits (随机)
# target_logits = torch.randn(N_Q, N_K) 
target_logits = torch.zeros(N_Q, N_K) 

for i in range(N_Q):
    target_logits[i][0] = 1e10

# 对 Key 的维度 (dim=1) 求 Softmax
target_attn_weights = F.softmax(target_logits, dim=1).unsqueeze(0) # (1, 4, 6)



print("--- 实验配置 ---")
print(f"Query 序列长度 N_Q: {N_Q}")
print(f"Key 序列长度 N_K: {N_K}")
print(f"Q/K 维度 D_K: {D_K}")
print(f"目标权重形状: {target_attn_weights.shape}")
print("-" * 20)

# --- 4. 训练前评估 ---
print("## 🚀 训练前 (Epoch 0) 注意力权重:")
# 只需要训练前的 Q/K 投影层的参数是随机的，所以初始的 attn weights 是随机的
with torch.no_grad():
    _, initial_weights = model(query_input, key_input, value_input, key_padding_mask)
    initial_loss = attention_weight_matching_loss(initial_weights, target_attn_weights)

# 打印初始权重（取 batch 0）
print(initial_weights.squeeze(0).numpy().round(4))
print(f"初始损失: {initial_loss.item():.6f}\n")

print("## 🎯 目标注意力权重:")
print(target_attn_weights.squeeze(0).numpy().round(4))
print("-" * 20)


# --- 5. 训练循环 ---
print("## ⚙️ 开始训练...")
for epoch in range(1, EPOCHS + 1):
    model.train()
    optimizer.zero_grad()

    # 前向传播
    _, current_weights = model(query_input, key_input, value_input, key_padding_mask)

    # 计算损失
    loss = attention_weight_matching_loss(current_weights, target_attn_weights)

    # 反向传播和优化
    loss.backward()
    optimizer.step()

    if epoch % 100 == 0 or epoch == EPOCHS:
        print(f"Epoch {epoch:4d} | Loss: {loss.item():.6f}")

print("## ✅ 训练完成。\n")

# --- 6. 训练后评估 ---
print("## 📉 训练后 (Epoch 500) 注意力权重:")
model.eval()
with torch.no_grad():
    _, final_weights = model(query_input, key_input, value_input, key_padding_mask)
    final_loss = attention_weight_matching_loss(final_weights, target_attn_weights)

# 打印最终权重（取 batch 0）
print(final_weights.squeeze(0).numpy().round(4))
print(f"最终损失: {final_loss.item():.6f}")

# 最终对比
print("\n--- 最终对比 ---")
print("初始损失:", f"{initial_loss.item():.6f}")
print("最终损失:", f"{final_loss.item():.6f}")