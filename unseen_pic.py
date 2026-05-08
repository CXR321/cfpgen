import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# 设置样式
plt.style.use('default')
sns.set_style("whitegrid")

# 数据准备
recall_metrics = ['Recall (Micro)', 'Recall (Macro)']
f1_metrics = ['F1 Score (Micro)', 'F1 Score (Macro)']

our_recall = [0.2496, 0.2729]
cfpgen_recall = [0.2258, 0.2388]

our_f1 = [0.0802, 0.1001]
cfpgen_f1 = [0.0690, 0.0867]

# 创建图形
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# 设置颜色
color_our = '#2E86AB'  # OUR Model的蓝色
color_cfpgen = '#A23B72'  # CFPGEN的紫色

# ========== 第一张图：Recall对比 ==========
x_recall = np.arange(len(recall_metrics))
width = 0.4

# 绘制柱状图
bars_our_recall = ax1.bar(x_recall - width/2, our_recall, width, 
                          label='OUR Model', alpha=0.9, color=color_our, 
                          edgecolor='black', linewidth=1.8, zorder=3)

bars_cfpgen_recall = ax1.bar(x_recall + width/2, cfpgen_recall, width, 
                             label='CFPGEN', alpha=0.9, color=color_cfpgen, 
                             edgecolor='black', linewidth=1.8, zorder=3)

# 计算Recall的最小值和最大值，设置合适的y轴范围
recall_min = min(min(our_recall), min(cfpgen_recall)) - 0.01
recall_max = max(max(our_recall), max(cfpgen_recall)) + 0.01

# 设置图表属性
# ax1.set_xlabel('Recall Metrics', fontsize=13, fontweight='bold')
ax1.set_ylabel('Score', fontsize=13, fontweight='bold')
# ax1.set_title('Recall Performance: OOD Strict Unseen Label', 
#               fontsize=15, fontweight='bold', pad=18)
ax1.set_xticks(x_recall)
ax1.set_xticklabels(recall_metrics, fontsize=12, rotation=0)
ax1.legend(loc='upper left', fontsize=11, framealpha=0.95, shadow=True)
ax1.grid(True, alpha=0.25, axis='y', linestyle='-', zorder=0)

# 设置y轴范围以凸显差异
ax1.set_ylim([0.20, 0.30])  # 仅显示0.20-0.30范围，凸显OUR Model的优势

# 美化边框
ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)
ax1.spines['left'].set_linewidth(1.5)
ax1.spines['bottom'].set_linewidth(1.5)

# 添加差异指示箭头
# for i in range(len(recall_metrics)):
#     diff = our_recall[i] - cfpgen_recall[i]
#     if diff > 0:
#         ax1.annotate(f'+{diff:.4f}', 
#                     xy=(x_recall[i], max(our_recall[i], cfpgen_recall[i]) + 0.002),
#                     xytext=(0, 5), textcoords='offset points',
#                     ha='center', va='bottom', fontsize=10, fontweight='bold',
#                     color='green')

# ========== 第二张图：F1 Score对比 ==========
x_f1 = np.arange(len(f1_metrics))

# 绘制柱状图
bars_our_f1 = ax2.bar(x_f1 - width/2, our_f1, width, 
                      label='OUR Model', alpha=0.9, color=color_our, 
                      edgecolor='black', linewidth=1.8, zorder=3)

bars_cfpgen_f1 = ax2.bar(x_f1 + width/2, cfpgen_f1, width, 
                         label='CFPGEN', alpha=0.9, color=color_cfpgen, 
                         edgecolor='black', linewidth=1.8, zorder=3)

# 计算F1的最小值和最大值
f1_min = min(min(our_f1), min(cfpgen_f1)) - 0.002
f1_max = max(max(our_f1), max(cfpgen_f1)) + 0.002

# 设置图表属性
# ax2.set_xlabel('F1 Score Metrics', fontsize=13, fontweight='bold')
ax2.set_ylabel('Score', fontsize=13, fontweight='bold')
# ax2.set_title('F1 Score Performance: OOD Strict Unseen Label', 
#               fontsize=15, fontweight='bold', pad=18)
ax2.set_xticks(x_f1)
ax2.set_xticklabels(f1_metrics, fontsize=12, rotation=0)
ax2.legend(loc='upper left', fontsize=11, framealpha=0.95, shadow=True)
ax2.grid(True, alpha=0.25, axis='y', linestyle='-', zorder=0)

# 设置y轴范围以凸显差异
ax2.set_ylim([0.06, 0.11])  # 仅显示0.06-0.11范围，凸显OUR Model的优势

# 美化边框
ax2.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)
ax2.spines['left'].set_linewidth(1.5)
ax2.spines['bottom'].set_linewidth(1.5)

# # 添加差异指示箭头
# for i in range(len(f1_metrics)):
#     diff = our_f1[i] - cfpgen_f1[i]
#     if diff > 0:
#         ax2.annotate(f'+{diff:.4f}', 
#                     xy=(x_f1[i], max(our_f1[i], cfpgen_f1[i]) + 0.0005),
#                     xytext=(0, 5), textcoords='offset points',
#                     ha='center', va='bottom', fontsize=10, fontweight='bold',
#                     color='green')

# 调整布局
plt.tight_layout()

# 添加整体标题
fig.suptitle('Model Performance Comparison on OOD Strict Unseen Label', 
             fontsize=16, fontweight='bold', y=1.02)

plt.show()

# 可选：单独保存两张图
fig.savefig('unseen_recall_f1_comparison.png', dpi=300, bbox_inches='tight')