import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# ================= 1. 数据准备 =================
# Data: [Baseline, Ours]
labels = ['Baseline', 'Ours']

# Metric 1: F1 Score
data_f1 = [0.1744, 0.3295]

# Metric 2: Match Rate (Definition: Recall / Coverage of Target Labels)
# 强调：Baseline 在这里非常低，Ours 有显著提升
data_match = [0.0277, 0.2160]

# ================= 2. 绘图风格设置 (ICML Standard) =================
# 使用 Seaborn 的 paper 上下文，字体稍微调大一点以保证清晰度
sns.set_theme(style="whitegrid", context="paper", font_scale=1.4)
plt.rcParams['font.family'] = 'serif'
# plt.rcParams['font.serif'] = ['Times New Roman']

# 配色：Baseline (蓝色), Ours (红色/深红) - 保持一致性
colors = ["#4c72b0", "#c44e52"]

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# ================= 3. 辅助绘图函数 =================
def plot_bars(ax, data, title, ylabel):
    x = np.arange(len(labels))
    width = 0.6
    
    # 绘制柱状图，带轻微透明度和边框
    rects = ax.bar(x, data, width, color=colors, alpha=0.9, edgecolor='black', linewidth=0.8)
    
    # 设置标题和标签
    ax.set_title(title, fontsize=16, weight='bold', pad=15)
    ax.set_ylabel(ylabel, fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=14)
    
    # 动态调整 Y 轴范围，留出顶部空间给数字标签
    # 对于 Match Rate 图，由于 Baseline 特别小，需要确保它可见且上方有空间
    max_val = max(data)
    ax.set_ylim(0, max_val * 1.3) # 留出30%空间
    
    # 美化：去除顶部和右侧边框，保留Y轴网格
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(axis='y', linestyle='--', alpha=0.5)
    ax.grid(axis='x', visible=False) # 不显示X轴网格

    # 添加数值标签 (使用4位小数以精确显示小数值)
    for rect in rects:
        height = rect.get_height()
        ax.annotate(f'{height:.4f}',
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 5),  # 垂直偏移 5 points
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=13, weight='bold', color='black')

# ================= 4. 绘制子图 =================

# Subplot 1: F1 Score
plot_bars(axes[0], data_f1, 
          title="F1 Score Comparison", 
          ylabel="F1 Score")

# Subplot 2: Match Rate (Recall)
# 这里的标题和Y轴标签准确反映了您的定义
plot_bars(axes[1], data_match, 
          title="Target Function Match Rate", 
          ylabel="Match Rate")

# ================= 5. 全局布局与标题 =================
# 主标题：核心修改点。强调“假设的/非自然的”组合，区别于之前的 OOD/Unseen。
fig.suptitle("Performance on Hypothetical Functional Combinations (Non-Natural Sets)", 
             fontsize=18, weight='bold', y=1.05)

plt.tight_layout()

# 保存图片
filename = "non_natural_combinations_performance.png"
plt.savefig(filename, dpi=300, bbox_inches='tight')
print(f"Plot saved as {filename}")

plt.show()