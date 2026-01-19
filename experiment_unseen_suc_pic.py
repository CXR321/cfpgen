from PIL import Image, ImageOps
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import os

# 1. 定义所有需要处理的文件
files_map = {
    'nature_nad': "nature_NAD.png",
    'our_nad': "our_NAD.png",
    'nature_cat': "nature_catalytic.png",
    'our_cat': "our_catalytic.png",
    'baseline': "cfp_bad_unseen.png"
}

# 2. 找出所有图片中最大的宽度和高度
max_w, max_h = 0, 0
for fpath in files_map.values():
    if os.path.exists(fpath):
        with Image.open(fpath) as img:
            max_w = max(max_w, img.width)
            max_h = max(max_h, img.height)

# 设定统一的目标尺寸
TARGET_SIZE = (max_w, max_h)
print(f"统一目标尺寸: {TARGET_SIZE}")

# 3. 图像处理函数：将图片填充到目标尺寸
def pad_to_size(image_path, target_size):
    if not os.path.exists(image_path):
        return None
    
    img = Image.open(image_path)
    target_w, target_h = target_size
    
    # 计算需要填充的像素数
    delta_w = target_w - img.width
    delta_h = target_h - img.height
    
    # padding = (left, top, right, bottom)
    # 左右对半分，上下对半分
    padding = (delta_w // 2, delta_h // 2, delta_w - (delta_w // 2), delta_h - (delta_h // 2))
    
    # 使用白色填充
    new_img = ImageOps.expand(img, padding, fill='white')
    return new_img

# 4. 预处理所有图片
padded_imgs = {}
for key, fpath in files_map.items():
    padded_imgs[key] = pad_to_size(fpath, TARGET_SIZE)

# 5. 开始绘图
# figsize=(16, 12) 是画布的物理尺寸
fig = plt.figure(figsize=(16, 12), dpi=300)
# hspace=0.3 拉大上下两行的间距
gs = GridSpec(2, 6, figure=fig, hspace=0.3, wspace=0.1)

# 字体设置
title_font = {'fontsize': 16, 'fontweight': 'bold'}
label_font = {'fontsize': 16, 'fontweight': 'bold'}

# 绘图辅助函数
def plot_cell(ax_pos, key, title_text, label_text):
    ax = fig.add_subplot(ax_pos)
    if padded_imgs.get(key):
        ax.imshow(padded_imgs[key])
        
    ax.set_title(title_text, pad=15, **title_font)
    ax.axis('off')
    # 标签位置：y=-0.1
    ax.text(0.5, -0.1, label_text, transform=ax.transAxes, ha='center', va='top', **label_font)

# --- 第一行 (上半部分) ---
# 两个图，平分宽度 (各占3列)
plot_cell(gs[0, 0:3], 'nature_nad', "Nature (NAD binding)", "(a)")
plot_cell(gs[0, 3:6], 'our_nad',    "Ours (NAD binding)",   "(b)")

# --- 第二行 (下半部分) ---
# 三个图，平分宽度 (各占2列)
plot_cell(gs[1, 0:2], 'nature_cat', "Nature", "(c)")
plot_cell(gs[1, 2:4], 'our_cat',    "Ours",   "(d)")
plot_cell(gs[1, 4:6], 'baseline',   "Baseline",     "(e)")

# 6. 保存结果
output_path = 'icml_figure_final_padded.png'
plt.savefig(output_path, bbox_inches='tight')
print(f"成功保存到: {output_path}")