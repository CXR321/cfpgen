import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# ================= 1. 数据准备 =================
# --- 雷达图数据 (Micro F1 和 Micro Precision 已互换位置) ---
radar_labels = ['Micro Precision', 'Macro F1', 'Micro Recall', 'Micro F1']

# Baseline: Precision(0.4551), MacroF1(0.3392), Recall(0.4993), MicroF1(0.4762)
radar_baseline = [0.4551, 0.3392, 0.4993, 0.4762]

# Ours: Precision(0.4945), MacroF1(0.4304), Recall(0.5866), MicroF1(0.5366)
radar_ours = [0.4945, 0.4304, 0.5866, 0.5366]

# --- 柱状图数据 (Exact Match) ---
bar_labels = ['Baseline\n(CFP-Gen)', 'Ours']
bar_values = [10.00, 14.47]  # percentages

# ================= 2. 绘图配置 =================
sns.set_theme(style="whitegrid", context="paper")
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman']
plt.rcParams['font.size'] = 11
plt.rcParams['axes.labelsize'] = 11
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 9
plt.rcParams['legend.fontsize'] = 11

colors = ["#4c72b0", "#c44e52"]  # Blue vs Red

# 创建画布
fig = plt.figure(figsize=(10, 4.5))

# ================= 3. 左图：雷达图 =================
ax1 = fig.add_subplot(121, polar=True)

# 数据闭环
N = len(radar_labels)
angles = [n / float(N) * 2 * np.pi for n in range(N)]
angles += angles[:1]
radar_baseline += radar_baseline[:1]
radar_ours += radar_ours[:1]

# 绘制
ax1.plot(angles, radar_baseline, linewidth=2, linestyle='-', color=colors[0], label='Baseline (CFP-Gen)')
ax1.fill(angles, radar_baseline, color=colors[0], alpha=0.15)

ax1.plot(angles, radar_ours, linewidth=2, linestyle='-', color=colors[1], label='Ours')
ax1.fill(angles, radar_ours, color=colors[1], alpha=0.15)

# --- 雷达图样式 ---
ax1.set_xticks(angles[:-1])
ax1.set_xticklabels(radar_labels, fontsize=11, weight='bold')

# 设置 Y 轴刻度：以 0.1 为间隔
ax1.set_ylim(0, 0.65)
yticks = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
ax1.set_yticks(yticks)
ax1.set_yticklabels([str(y) for y in yticks], color="grey", size=9)

ax1.grid(color='grey', linestyle='--', linewidth=0.5, alpha=0.4)
ax1.spines['polar'].set_visible(False)

# 设置方向 (12点钟开始顺时针)
ax1.set_theta_offset(np.pi / 2)
ax1.set_theta_direction(-1)

# ================= 4. 右图：柱状图 (恢复之前的样式) =================
ax2 = fig.add_subplot(122)

x_pos = np.arange(len(bar_labels))
width = 0.5  # 保持之前的宽度

# 绘制柱子 (恢复样式参数: linewidth=0.5)
rects = ax2.bar(x_pos, bar_values, width, color=colors, alpha=0.9, edgecolor='black', linewidth=0.5)

# --- 柱状图样式 (恢复之前的设置) ---
ax2.set_ylabel('Exact Match Rate (%)', fontsize=12)
# 标题样式恢复：fontsize=14, bold, pad=15
ax2.set_title('Exact Match Consistency', fontsize=14, weight='bold', pad=15)
ax2.set_xticks(x_pos)
ax2.set_xticklabels(bar_labels, fontsize=11)
# Y轴范围恢复：0-20
ax2.set_ylim(0, 20)

ax2.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)
ax2.grid(axis='y', linestyle='--', alpha=0.5)
ax2.grid(axis='x', visible=False)

# 添加数值标签 (恢复之前的样式)
for rect in rects:
    height = rect.get_height()
    ax2.annotate(f'{height:.2f}%',
                xy=(rect.get_x() + rect.get_width() / 2, height),
                xytext=(0, 3),
                textcoords="offset points",
                ha='center', va='bottom', fontsize=11, weight='bold', color='black')

# ================= 5. 图例与保存 =================
handles, labels = ax1.get_legend_handles_labels()
fig.legend(handles, labels, loc='lower center', 
           bbox_to_anchor=(0.5, -0.05), ncol=2, 
           frameon=False, columnspacing=2.0)

plt.tight_layout()
plt.subplots_adjust(bottom=0.15)

filename = "combined_radar_bar_restored_style.png"
plt.savefig(filename, dpi=300, bbox_inches='tight')
print(f"Plot saved as {filename}")

plt.show()