from PIL import Image, ImageOps
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import os

# 1. 定义只需要处理的三个文件
files_map = {
    'nature_cat': "Q4X1A4_natural.png",
    'our_cat': "Q4X1A4_gen.png",
    'baseline': "Q4X1A4_bad.png"
}

# 2. 找出这三张图片中最大的宽度和高度 (以此为基准进行Padding)
max_w, max_h = 0, 0
for fpath in files_map.values():
    if os.path.exists(fpath):
        with Image.open(fpath) as img:
            max_w = max(max_w, img.width)
            max_h = max(max_h, img.height)

# 设定统一的目标尺寸
TARGET_SIZE = (max_w, max_h)
print(f"统一目标尺寸: {TARGET_SIZE}")

# 3. 图像处理函数：将图片填充到目标尺寸 (保持不变)
def pad_to_size(image_path, target_size):
    if not os.path.exists(image_path):
        print(f"警告: 文件不存在 {image_path}")
        return None
    
    img = Image.open(image_path)
    target_w, target_h = target_size
    
    delta_w = target_w - img.width
    delta_h = target_h - img.height
    
    padding = (delta_w // 2, delta_h // 2, delta_w - (delta_w // 2), delta_h - (delta_h // 2))
    new_img = ImageOps.expand(img, padding, fill='white')
    return new_img

# 4. 预处理图片
padded_imgs = {}
for key, fpath in files_map.items():
    padded_imgs[key] = pad_to_size(fpath, TARGET_SIZE)

# 5. 开始绘图
# 注意：调整了画布尺寸 (figsize)
fig = plt.figure(figsize=(18, 6), dpi=300)

# -------------------------------------------------------------------------
# 修改点：将 wspace 设置为负数以允许重叠
# 尝试 -0.05, -0.1, -0.2 等值，数值越小（负得越多），重叠越多
# -------------------------------------------------------------------------
gs = GridSpec(1, 3, figure=fig, wspace=-0.4) 

# 字体设置
title_font = {'fontsize': 18, 'fontweight': 'bold'}
label_font = {'fontsize': 18, 'fontweight': 'bold'}

# 绘图辅助函数 (建议增加 zorder 控制遮挡关系)
def plot_cell(ax_pos, key, title_text, label_text, z_order=1):
    ax = fig.add_subplot(ax_pos)
    
    # 设置 zorder，数值大的图会盖在数值小的图上面
    # 如果你想让左边的压住右边的，或者右边的压住左边的，可以在调用时调整
    ax.set_zorder(z_order) 
    
    if padded_imgs.get(key):
        ax.imshow(padded_imgs[key])
    else:
        ax.text(0.5, 0.5, "Image Missing", ha='center', va='center')
        
    ax.set_title(title_text, pad=15, **title_font)
    ax.axis('off')
    # 标签位置
    ax.text(0.5, -0.1, label_text, transform=ax.transAxes, ha='center', va='top', **label_font)

# --- 单行绘制三张图 ---

# 这里可以通过 z_order 控制谁压谁
# 例如：设置递增的 z_order (1, 2, 3)，则右边的图会压住左边的图的边缘
# 如果设置递减 (3, 2, 1)，则左边的图会压住右边的图
plot_cell(gs[0, 0], 'nature_cat', "Natural Protein", "(a)", z_order=3)
plot_cell(gs[0, 1], 'our_cat',    "Ours",            "(b)", z_order=2)
plot_cell(gs[0, 2], 'baseline',   "Baseline (CFP-Gen)", "(c)", z_order=1)

# 6. 保存结果
output_path = 'icml_figure_bottom_row.png'
plt.savefig(output_path, bbox_inches='tight')
print(f"成功保存到: {output_path}")