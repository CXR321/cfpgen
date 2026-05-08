import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy import stats

# 假设 df_res 是你之前的 DataFrame，且包含 'match_status' 和 'Avg_Train_Freq'
# 如果需要从文件读取：
df_res = pd.read_csv('analysis_semantic_distances_raw.csv')

def investigate_distribution_anomaly(df, col='Avg_Train_Freq'):
    print(f"\n{'='*20} Deep Dive into {col} {'='*20}")
    
    # 1. 打印详细统计量 (Descriptive Statistics)
    # 重点看 50% (Median) 和 Mean 的差距，以及 Std (波动大小)
    stats = df.groupby('match_status')[col].describe()
    print(stats[['count', 'mean', 'std', '25%', '50%', '75%']])
    
    # 2. 绘制箱线图 (Boxplot) - 最直观的工具
    # 箱线图能显示中位数（箱子中间的线）和离群值（黑点）
    plt.figure(figsize=(12, 6))
    
    # 子图1: 普通坐标轴
    plt.subplot(1, 2, 1)
    sns.boxplot(x='match_status', y=col, data=df, order=['Exact Match', 'Partial Match', 'Fail'])
    plt.title(f'{col} Distribution (Linear Scale)')
    
    # 子图2: 对数坐标轴 (Log Scale)
    # 因为频率数据通常是长尾的，用 Log 才能看清低频区的区别
    plt.subplot(1, 2, 2)
    sns.boxplot(x='match_status', y=col, data=df, order=['Exact Match', 'Partial Match', 'Fail'])
    plt.yscale('log')
    plt.title(f'{col} Distribution (Log Scale)')
    
    plt.tight_layout()
    plt.savefig(f'investigation_{col}.png')
    plt.show()
    
    # 3. 绘制密度图 (KDE) - 看峰值在哪里
    plt.figure(figsize=(10, 6))
    sns.kdeplot(data=df, x=col, hue='match_status', common_norm=False, fill=True, 
                hue_order=['Exact Match', 'Partial Match', 'Fail'])
    plt.title(f'{col} Density Plot (Where is the bulk of data?)')
    plt.xlim(0, df[col].quantile(0.95)) # 只看前95%的数据，忽略极端长尾，否则图会很丑
    plt.savefig(f'investigation_{col}_density.png')
    plt.show()

# 运行深究函数
if 'Avg_Train_Freq' in df_res.columns:
    investigate_distribution_anomaly(df_res, 'Avg_Train_Freq')

def analyze_freq_depth_relationship(df):
    print("\n" + "="*80)
    print("DEEP DIVE: Frequency vs. Depth Correlation Analysis")
    print("="*80)

    # 1. 准备数据
    # 只关注 Exact 和 Fail，也可以加入 Partial 做对比
    target_groups = ['Exact Match', 'Fail']
    df_subset = df[df['match_status'].isin(target_groups)].copy()
    
    # 2. 计算相关性系数 (Spearman Correlation)
    # 因为频率是非线性的，Spearman (秩相关) 比 Pearson 更准确
    print(f"{'Group':<15} | {'Spearman Corr (Freq vs Depth)':<30} | {'P-value'}")
    print("-" * 60)
    
    for group in target_groups:
        sub = df_subset[df_subset['match_status'] == group]
        corr, p = stats.spearmanr(sub['Avg_Train_Freq'], sub['Avg_Depth'])
        print(f"{group:<15} | {corr:<30.4f} | {p:.2e}")

    # 3. 可视化：散点图 + 回归线 (Scatter + Regression)
    # 使用 Log Scale Y轴，因为频率是长尾的
    plt.figure(figsize=(10, 6))
    
    # 小技巧：为了让回归线在对数轴上看起来直，我们先对频率取 Log
    df_subset['Log10_Freq'] = np.log10(df_subset['Avg_Train_Freq'] + 1)
    
    # 绘制散点和拟合线
    g = sns.lmplot(
        data=df_subset, 
        x='Avg_Depth', 
        y='Log10_Freq', 
        hue='match_status',
        height=6, 
        aspect=1.5,
        scatter_kws={'alpha': 0.5, 's': 30},
        palette={'Exact Match': '#2ca02c', 'Fail': '#d62728'} # 绿色Exact, 红色Fail
    )
    
    plt.title('Relationship between Label Depth and Training Frequency', fontsize=14)
    plt.xlabel('Average Depth (Structure Specificity)', fontsize=12)
    plt.ylabel('Log10(Average Training Frequency)', fontsize=12)
    plt.grid(True, linestyle=':', alpha=0.5)
    plt.savefig('investigation_freq_vs_depth_scatter.png')
    plt.show()

    # 4. 可视化：二维密度图 (2D Density Plot / KDE)
    # 散点图如果点太多会重叠，密度图能看到“重心”在哪里
    plt.figure(figsize=(10, 6))
    
    # 绘制 Fail 的密度 (红色)
    sns.kdeplot(
        data=df_subset[df_subset['match_status'] == 'Fail'],
        x='Avg_Depth',
        y='Log10_Freq',
        color='red',
        fill=False,
        levels=5,
        alpha=0.7,
        label='Fail Density'
    )
    
    # 绘制 Exact 的密度 (绿色)
    sns.kdeplot(
        data=df_subset[df_subset['match_status'] == 'Exact Match'],
        x='Avg_Depth',
        y='Log10_Freq',
        color='green',
        fill=True,
        alpha=0.3, # 半透明填充
        levels=5,
        label='Exact Density'
    )
    
    plt.title('2D Density: Where are the "Centers of Gravity"?', fontsize=14)
    plt.xlabel('Average Depth', fontsize=12)
    plt.ylabel('Log10(Avg Train Frequency)', fontsize=12)
    # 手动添加图例说明
    plt.legend(['Fail (Contour)', 'Exact (Filled)'], loc='upper right')
    
    plt.tight_layout()
    plt.savefig('investigation_freq_vs_depth_kde.png')
    plt.show()

# 执行分析
if 'Avg_Train_Freq' in df_res.columns and 'Avg_Depth' in df_res.columns:
    analyze_freq_depth_relationship(df_res)