import matplotlib.pyplot as plt

def generate_partial_circle_image(filename, share_proportion, hex_color, outline_width=2.0):
    """
    生成一个只包含部分圆（饼图切片）的透明背景图片，并为彩色部分添加黑线轮廓。

    Args:
        filename (str): 保存的文件名.
        share_proportion (float): 需要着色的比例.
        hex_color (str): 十六进制颜色代码.
        outline_width (float): 轮廓线的宽度.
    """
    remaining_proportion = 1.0 - share_proportion
    sizes = [share_proportion, remaining_proportion]
    # (0, 0, 0, 0) 代表完全透明
    colors = [hex_color, (0, 0, 0, 0)]

    # 创建画布
    fig, ax = plt.subplots(figsize=(4, 4))

    # --- 核心修改部分开始 ---

    # 绘制饼图，并捕获返回的 'wedges' (楔形对象列表)
    # ax.pie 返回一个元组，第一个元素是我们需要的楔形列表
    wedges, text_labels = ax.pie(sizes, colors=colors, startangle=90, counterclock=False)

    # wedges[0] 是我们需要的彩色扇形
    # wedges[1] 是透明的辅助扇形

    # 只设置第一个扇形的边框属性
    target_wedge = wedges[0]
    target_wedge.set_edgecolor('black')    # 设置边框颜色为黑色
    target_wedge.set_linewidth(outline_width) # 设置边框宽度

    # --- 核心修改部分结束 ---


    # 确保饼图是正圆
    ax.axis('equal')
    # 关闭坐标轴
    ax.axis('off')

    # 保存图片 (背景透明，紧凑裁剪)
    plt.savefig(filename, transparent=True, bbox_inches='tight', pad_inches=0, dpi=300)
    plt.close(fig)
    print(f"已生成带轮廓图片: {filename} (比例: {share_proportion:.2f}, 颜色: {hex_color})")

# --- 主程序 ---
if __name__ == "__main__":
    print("开始生成带黑线轮廓的图片...")

    # 任务 1: 1/3 大小，颜色 #AFAFDA，带黑边
    generate_partial_circle_image(
        filename="circle_outline_2_3_AFAFDA.png",
        share_proportion=2/3,
        hex_color="#AFAFDA"
    )

    # 任务 2: 3/5 大小，颜色 #E3AAC5，带黑边
    generate_partial_circle_image(
        filename="circle_outline_3_5_E3AAC5.png",
        share_proportion=3/5,
        hex_color="#E3AAC5"
    )

    print("\n所有图片生成完毕。")