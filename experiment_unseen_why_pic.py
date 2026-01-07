import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# --- 1. Data Setup ---
# Categories
categories = ['Exact Match', 'Partial Match', 'Failure']

# Metric 1: GT Internal Dist -> "Mean Intra-Set Semantic Distance"
# (Represents how semantically diverse the ground truth labels are)
semantic_dist = [6.5862, 6.2092, 3.0897]

# Metric 2: Avg Train Frequency -> "Mean Training Frequency"
# (Represents how common these labels were in the training set)
train_freq = [585.8333, 768.5267, 414.8333]

# Metric 3: Avg Label Count -> "Mean Label Cardinality"
# (Represents the complexity/number of targets per sample)
label_cardinality = [2.2414, 3.3472, 2.8974]

# --- 2. Plotting Configuration (ICML Style) ---
sns.set_theme(style="whitegrid", context="paper", font_scale=1.4)
plt.rcParams['font.family'] = 'serif'

# Define a 3-color palette for the categories (Distinct but professional)
# Blue (Exact), Orange (Partial), Red (Fail) - Muted versions
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

# --- 3. Subplot 1: Semantic Distance ---
x = np.arange(len(categories))
rects1 = axes[0].bar(categories, semantic_dist, color=colors, alpha=0.9, edgecolor='black', linewidth=0.5)

axes[0].set_ylabel('Distance Score')
axes[0].set_title('Mean Intra-Set Semantic Distance', fontsize=14, weight='bold', pad=15)
axes[0].set_ylim(1, 7)
# Insight: "Failures occur on semantically simpler/tighter sets"
autolabel(rects1, axes[0])

# --- 4. Subplot 2: Training Frequency ---
rects2 = axes[1].bar(categories, train_freq, color=colors, alpha=0.9, edgecolor='black', linewidth=0.5)

axes[1].set_ylabel('Frequency Count')
axes[1].set_title('Mean Training Frequency', fontsize=14, weight='bold', pad=15)
axes[1].set_ylim(100, 900)
# Insight: "Partial matches are driven by high-frequency head labels"
autolabel(rects2, axes[1], fmt='{:.0f}')

# --- 5. Subplot 3: Label Cardinality ---
rects3 = axes[2].bar(categories, label_cardinality, color=colors, alpha=0.9, edgecolor='black', linewidth=0.5)

axes[2].set_ylabel('Avg. Number of Labels')
axes[2].set_title('Mean Label Cardinality', fontsize=14, weight='bold', pad=15)
axes[2].set_ylim(0, 4.0)
# Insight: "Partial matches often have more labels to predict"
autolabel(rects3, axes[2])

# --- 6. Global Formatting ---
# Common adjustments
for ax in axes:
    ax.grid(axis='y', linestyle='--', alpha=0.6)
    # Rotate x-labels slightly if needed, or keep straight
    ax.tick_params(axis='x', labelsize=12)

# Main Title
fig.suptitle("Attribute Analysis of OOD Generalization Performance", fontsize=18, weight='bold', y=1.05)

plt.tight_layout()

# Save
filename = "experiment_ood_performance_attribute_analysis.png"
plt.savefig(filename, dpi=300, bbox_inches='tight')
print(f"Plot saved as {filename}")

plt.show()