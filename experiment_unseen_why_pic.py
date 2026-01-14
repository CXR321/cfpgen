import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# --- 1. Data Setup ---
# Categories
categories = ['Exact Match', 'Partial Match', 'Failure']

# Metric 1: GT Internal Dist (Ground Truth 标签的内部语义距离)
# Data: [Exact, Partial, Fail]
gt_internal_dist = [5.57, 6.73, 6.30]

# Metric 2: Pred Internal Dist (预测标签的内部语义距离)
# Data: [Exact, Partial, Fail]
pred_internal_dist = [3.73, 3.53, 3.20]

# Metric 3: Avg Depth (平均层级深度)
# Data: [Exact, Partial, Fail]
avg_depth = [3.46, 3.91, 4.01]

# --- 2. Plotting Configuration (ICML Style) ---
sns.set_theme(style="whitegrid", context="paper", font_scale=1.4)
plt.rcParams['font.family'] = 'serif'
# plt.rcParams['font.serif'] = ['Times New Roman']

# Define a 3-color palette for the categories
# Blue (Exact), Orange (Partial), Red (Fail)
colors = ["#4c72b0", "#dd8452", "#c44e52"] 

fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))

# --- Helper Function for Labels ---
def autolabel(rects, ax, fmt='{:.2f}'):
    for rect in rects:
        height = rect.get_height()
        ax.annotate(fmt.format(height),
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 5),
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=12, weight='bold')

# --- 3. Subplot 1: GT Internal Semantic Distance ---
x = np.arange(len(categories))
rects1 = axes[0].bar(categories, gt_internal_dist, color=colors, alpha=0.9, edgecolor='black', linewidth=0.5)

axes[0].set_ylabel('Distance Score')
axes[0].set_title('Mean Intra-Set Semantic Distance (GT)', fontsize=14, weight='bold', pad=15)
axes[0].set_ylim(4, 8) # Adapted for max value ~6.73
autolabel(rects1, axes[0])

# --- 4. Subplot 2: Pred Internal Semantic Distance ---
rects2 = axes[1].bar(categories, pred_internal_dist, color=colors, alpha=0.9, edgecolor='black', linewidth=0.5)

axes[1].set_ylabel('Distance Score')
axes[1].set_title('Mean Intra-Set Semantic Distance (Pred)', fontsize=14, weight='bold', pad=15)
axes[1].set_ylim(3, 4) # Adapted for max value ~3.73
autolabel(rects2, axes[1])

# --- 5. Subplot 3: Average Depth ---
rects3 = axes[2].bar(categories, avg_depth, color=colors, alpha=0.9, edgecolor='black', linewidth=0.5)

axes[2].set_ylabel('Avg. Hierarchical Depth')
axes[2].set_title('Mean GO Term Depth', fontsize=14, weight='bold', pad=15)
axes[2].set_ylim(2.5, 4.5) # Adapted for max value ~4.01
autolabel(rects3, axes[2])

# --- 6. Global Formatting ---
# Common adjustments
for ax in axes:
    ax.grid(axis='y', linestyle='--', alpha=0.6)
    # Rotate x-labels slightly if needed, or keep straight
    ax.tick_params(axis='x', labelsize=12)
    # Remove top and right spines for cleaner look
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

# Main Title
fig.suptitle("Attribute Analysis of OOD Generalization Performance", fontsize=18, weight='bold', y=1.02)

plt.tight_layout()

# Save
filename = "experiment_ood_performance_attribute_analysis_updated.png"
plt.savefig(filename, dpi=300, bbox_inches='tight')
print(f"Plot saved as {filename}")

plt.show()