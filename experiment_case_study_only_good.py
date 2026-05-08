import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.gridspec as gridspec
from PIL import Image

# ================= 配置区域 =================
# 文件路径
IMG_PATH_OURS = 'our_Q48KZ8.png'
IMG_PATH_BASE = 'Q48KZ8_baseline.png'

# 元数据
TARGET_ID = "Q48KZ8"
GO_CONDITIONS = ["GO:0004488", "GO:0004477"]

# 详细指标
METRICS_OURS = {
    "pLDDT": 94.37,
    "pTM": 0.941
}

METRICS_BASE = {
    "pLDDT": 36.94,
    "pTM": 0.181
}

# ================= 绘图设置 (ICML Style) =================
plt.rcParams['font.family'] = 'serif'
# plt.rcParams['font.serif'] = ['Times New Roman']
plt.rcParams['font.size'] = 12
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['figure.dpi'] = 300

def create_case_study():
    # 创建画布
    fig = plt.figure(figsize=(10, 6))
    gs = gridspec.GridSpec(1, 2, width_ratios=[1, 1], wspace=0.1)

    # --- 1. 加载并绘制 Ours ---
    ax1 = plt.subplot(gs[0])
    try:
        img_ours = Image.open(IMG_PATH_OURS)
        ax1.imshow(img_ours)
    except FileNotFoundError:
        # 如果找不到图片，生成空白占位符（方便调试）
        import numpy as np
        ax1.imshow(np.ones((500, 500, 3)) * 0.9) 
        ax1.text(250, 250, "Image Not Found\n(Place 'our_Q48KZ8.png' here)", ha='center')

    ax1.axis('off')
    
    # Ours 标题与指标
    title_text = "Ours"
    ax1.set_title(title_text, fontweight='bold', pad=10)
    
    # 指标文本 (Ours 表现好，可以使用深绿色或者纯黑加粗)
    metrics_text = (f"pLDDT: {METRICS_OURS['pLDDT']:.2f} | pTM: {METRICS_OURS['pTM']:.3f}\n"
                    )
    ax1.text(0.5, -0.15, metrics_text, transform=ax1.transAxes, 
             ha='center', va='top', fontsize=12, linespacing=1.5)

    # --- 2. 加载并绘制 Baseline ---
    ax2 = plt.subplot(gs[1])
    try:
        img_base = Image.open(IMG_PATH_BASE)
        ax2.imshow(img_base)
    except FileNotFoundError:
        import numpy as np
        ax2.imshow(np.ones((500, 500, 3)) * 0.9)
        ax2.text(250, 250, "Image Not Found\n(Place 'Q48KZ8_baseline.png' here)", ha='center')

    ax2.axis('off')

    # Baseline 标题与指标
    ax2.set_title("Baseline (CFPGen)", fontweight='bold', pad=10)
    
    # 指标文本 (Baseline 表现差)
    metrics_text_base = (f"pLDDT: {METRICS_BASE['pLDDT']:.2f} | pTM: {METRICS_BASE['pTM']:.3f}\n"
                         )
                         
    ax2.text(0.5, -0.15, metrics_text_base, transform=ax2.transAxes, 
             ha='center', va='top', fontsize=12, linespacing=1.5)

    # --- 3. 全局标题和条件信息 ---
    # 主标题
    plt.suptitle(f"Case Study: Conditional Generation for Target {TARGET_ID}", 
                 y=0.95, fontsize=16, fontweight='bold')

    # 条件展示 (在底部)
    condition_str = "Conditioning GO Terms: " + ", ".join(GO_CONDITIONS)
    # 可以在这里补充GO的文字描述，如果需要的话
    # condition_str += " (Methylenetetrahydrofolate dehydrogenase activity)"
    
    fig.text(0.5, 0.05, condition_str, ha='center', fontsize=11, 
             bbox=dict(facecolor='#f0f0f0', edgecolor='none', boxstyle='round,pad=0.5'))

    # 保存
    output_filename = f"case_study_{TARGET_ID}.png"
    plt.savefig(output_filename, dpi=300, bbox_inches='tight')
    print(f"Figure saved as {output_filename}")
    plt.show()

if __name__ == "__main__":
    create_case_study()