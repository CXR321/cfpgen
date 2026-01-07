import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# --- 1. Data Setup ---
# Metrics from your log
metrics = ['Micro F1', 'Macro F1', 'Micro Recall', 'Micro Precision']

# Baseline: cfpgen_650m
baseline_scores = [0.4686, 0.3401, 0.5557, 0.4050]
baseline_exact = 199
total_samples = 1400

# Ours: DPLM2
ours_scores = [0.5057, 0.4243, 0.6062, 0.4338]
ours_exact = 233

# Calculate Exact Match Percentages
baseline_em_pct = (baseline_exact / total_samples) * 100
ours_em_pct = (ours_exact / total_samples) * 100

# --- 2. Plotting Configuration (ICML Style) ---
# Set a clean, professional style suitable for academic publication
sns.set_theme(style="whitegrid", context="paper", font_scale=1.2)
plt.rcParams['font.family'] = 'serif'  # Serif fonts are standard for ICML/NeurIPS
colors = ["#4c72b0", "#c44e52"]  # Professional Blue vs Red/Maroon

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# --- 3. Subplot 1: Metrics Report (Grouped Bar Chart) ---
x = np.arange(len(metrics))
width = 0.35

# Plot bars
rects1 = axes[0].bar(x - width/2, baseline_scores, width, label='Baseline (CFPGen-650M)', color=colors[0], alpha=0.9, edgecolor='black', linewidth=0.5)
rects2 = axes[0].bar(x + width/2, ours_scores, width, label='Ours', color=colors[1], alpha=0.9, edgecolor='black', linewidth=0.5)

# Formatting Subplot 1
axes[0].set_ylabel('Score')
axes[0].set_title('Performance Metrics on OOD Test Set', fontsize=14, weight='bold', pad=15)
axes[0].set_xticks(x)
axes[0].set_xticklabels(metrics)
axes[0].set_ylim(0.2, 0.7)  # Set limit to make room for labels
axes[0].legend(loc='upper left', frameon=True, framealpha=0.9)

# Helper function to add labels on bars
def autolabel(rects, ax):
    for rect in rects:
        height = rect.get_height()
        ax.annotate(f'{height:.3f}',
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=10)

autolabel(rects1, axes[0])
autolabel(rects2, axes[0])

# --- 4. Subplot 2: Exact Match Rate (Bar Chart) ---
em_labels = ['Baseline (CFPGen-650M)', 'Ours']
em_values = [baseline_em_pct, ours_em_pct]
x_em = np.arange(len(em_labels))

rects_em = axes[1].bar(x_em, em_values, width=0.5, color=colors, alpha=0.9, edgecolor='black', linewidth=0.5)

# Formatting Subplot 2
axes[1].set_ylabel('Exact Match Rate (%)')
axes[1].set_title('Exact Match Consistency', fontsize=14, weight='bold', pad=15)
axes[1].set_xticks(x_em)
axes[1].set_xticklabels(em_labels)
axes[1].set_ylim(0, 20)  # Adjust based on data (highest is ~16%)

# Add labels for Exact Matches
for i, rect in enumerate(rects_em):
    height = rect.get_height()
    raw_count = baseline_exact if i == 0 else ours_exact
    # Display Percentage and Raw Count
    label_text = f"{height:.2f}%"
    axes[1].annotate(label_text,
                xy=(rect.get_x() + rect.get_width() / 2, height),
                xytext=(0, 3),
                textcoords="offset points",
                ha='center', va='bottom', fontsize=11, weight='bold')

# --- 5. Final Layout Adjustments ---
# Main Super Title
fig.suptitle("Performance Comparison: OOD Conditional Generation", fontsize=18, weight='bold', y=1.02)

plt.tight_layout()

# Save the figure
filename = "experiment_ood_conditional_generation_metrics.png"
plt.savefig(filename, dpi=300, bbox_inches='tight')
print(f"Plot saved as {filename}")

plt.show()