import matplotlib.pyplot as plt
import matplotlib.patches as patches

def draw_architecture():
    # 1. 设置画布 (ICML 宽比例)
    fig, ax = plt.subplots(figsize=(14, 8))
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 10)
    ax.axis('off')  # 关闭坐标轴

    # --- 样式定义 ---
    # 颜色
    c_seq = '#1a237e'      # 深蓝 (Sequence)
    c_struct = '#d1c4e9'   # 浅紫 (Structure)
    c_func = '#ffab91'     # 橙色 (Function/GO)
    c_block = '#f5f5f5'    # 灰色 (Transformer Block背景)
    c_layer = '#ffffff'    # 白色 (层内部)
    
    # 字体
    font_title = {'family': 'sans-serif', 'weight': 'bold', 'size': 10}
    font_label = {'family': 'sans-serif', 'size': 9}
    
    # --- 辅助绘图函数 ---
    def draw_box(xy, w, h, text, fc, ec='black', text_color='black', style='round,pad=0.1'):
        # 绘制圆角矩形
        box = patches.FancyBboxPatch(
            xy, w, h, boxstyle=style, 
            linewidth=1.5, edgecolor=ec, facecolor=fc
        )
        ax.add_patch(box)
        # 添加文字
        ax.text(
            xy[0] + w/2, xy[1] + h/2, text, 
            ha='center', va='center', color=text_color, **font_title
        )
        return box

    def draw_arrow(start, end, color='black', style='->', connection='arc3,rad=0'):
        ax.annotate(
            '', xy=end, xytext=start,
            arrowprops=dict(arrowstyle=style, color=color, lw=1.5, connectionstyle=connection)
        )

    # ================= 1. 左侧：输入 (Inputs) =================
    
    # Sequence Path
    draw_box((1, 8.5), 2.5, 0.8, "Sequence Tokens\n(Masked)", fc='white', ec=c_seq)
    draw_box((1, 7.0), 2.5, 0.8, "Seq Embedding", fc='white', ec=c_seq)
    draw_arrow((2.25, 8.5), (2.25, 7.8), color=c_seq)
    
    # Structure Path
    draw_box((4, 8.5), 2.5, 0.8, "Structure Tokens\n(Noised)", fc=c_struct, ec='purple')
    draw_box((4, 7.0), 2.5, 0.8, "Struct Embedding", fc=c_struct, ec='purple')
    draw_arrow((5.25, 8.5), (5.25, 7.8), color='purple')

    # Sum Operation
    draw_box((3.25, 5.8), 1, 0.6, "+", fc='white', style='circle')
    ax.text(4.4, 6.1, "+ Pos Embed", fontsize=8, color='gray')
    
    # 连接到 Sum
    draw_arrow((2.25, 7.0), (3.25, 6.1), color=c_seq, connection="angle,angleA=-90,angleB=180,rad=5")
    draw_arrow((5.25, 7.0), (4.25, 6.1), color='purple', connection="angle,angleA=-90,angleB=0,rad=5")

    # ================= 2. 下方：功能条件 (GO/FSR) =================
    
    draw_box((1, 2.5), 2.5, 0.8, "GO Label Input", fc=c_func, ec='#d84315')
    
    # FSR 模块
    draw_box((4, 2.5), 2.5, 0.8, "FSR Module\n(Retrieval)", fc='#ffe0b2', ec='#ef6c00')
    
    # Sum Condition
    draw_box((7, 2.5), 0.8, 0.8, "+", fc='white', style='circle')
    ax.text(7.0, 1.8, "Condition Embedding", ha='center', fontsize=8)

    # 连线
    draw_arrow((3.5, 2.9), (4.0, 2.9), color='#ef6c00') # GO -> FSR
    draw_arrow((6.5, 2.9), (7.0, 2.9), color='#ef6c00') # FSR -> Sum
    # One-hot 分支
    draw_arrow((2.25, 2.5), (7.0, 2.5), color='#ef6c00', connection="angle,angleA=-90,angleB=-90,rad=10")
    ax.text(4.5, 1.5, "One-hot / Learnable", ha='center', fontsize=8, color='#ef6c00')


    # ================= 3. 中间：Transformer Block =================
    
    # 大背景框 (N x Layers)
    block_rect = patches.FancyBboxPatch((6, 3.5), 4, 5.5, boxstyle="round,pad=0.2", fc=c_block, ec='gray', ls='--')
    ax.add_patch(block_rect)
    ax.text(6.2, 8.8, "N x Transformer Layers", fontsize=10, fontweight='bold', color='gray')

    # 内部组件
    # Self Attention
    draw_box((6.5, 8.0), 3, 0.6, "Self-Attention", fc=c_layer)
    ax.text(8.0, 7.7, "Add & Norm", ha='center', fontsize=7)
    
    # Cross Attention
    draw_box((6.5, 6.5), 3, 0.6, "Cross-Attention", fc=c_layer)
    ax.text(8.0, 6.2, "Add & Norm", ha='center', fontsize=7)
    
    # FFN
    draw_box((6.5, 5.0), 3, 0.6, "Feed-Forward (FFN)", fc=c_layer)
    ax.text(8.0, 4.7, "Add & Norm", ha='center', fontsize=7)

    # 内部连线 (主干流)
    # Input -> Self-Attn
    draw_arrow((3.75, 5.8), (6.5, 8.3), color='black', connection="angle,angleA=0,angleB=180,rad=5")
    
    # Self -> Cross (Q)
    draw_arrow((8, 8.0), (8, 7.1), color='black')
    ax.text(8.1, 7.5, "Q", fontsize=9, fontweight='bold')
    
    # Cross -> FFN
    draw_arrow((8, 6.5), (8, 5.6), color='black')
    
    # Condition Injection (K, V) - 关键线
    draw_arrow((7.8, 2.9), (9.5, 6.8), color='#ef6c00', connection="angle,angleA=0,angleB=0,rad=5")
    ax.text(9.6, 6.5, "K, V", fontsize=9, fontweight='bold', color='#ef6c00')


    # ================= 4. 右侧：输出 & Loss =================
    
    # 主输出
    draw_box((11.5, 5.0), 2.5, 0.8, "Prediction Head", fc='white', ec='black')
    draw_box((11.5, 7.0), 2.5, 0.8, "Reconstructed\nProtein", fc='white', ec='black')
    
    # 主连线
    draw_arrow((8.0, 5.0), (11.5, 5.4), color='black', connection="angle,angleA=-90,angleB=180,rad=5")
    draw_arrow((12.75, 5.8), (12.75, 7.0), color='black')
    
    # Recon Loss
    ax.text(15.0, 7.4, "L_recon", fontsize=12, fontweight='bold')
    draw_arrow((14.0, 7.4), (14.8, 7.4), style='-|>')

    # LSFS (局部监督) - 从 FFN 前面或者中间引出
    draw_box((11.5, 3.5), 2.5, 0.8, "LSFS Head\n(Function Cls)", fc='#ffe0b2', ec='#ef6c00')
    
    # LSFS 连线 (从 Cross-Attn 输出引出)
    draw_arrow((9.5, 6.0), (11.5, 3.9), color='purple', style='->', connection="angle,angleA=0,angleB=180,rad=5")
    ax.text(10.5, 4.5, "Latent Rep.", fontsize=8, color='purple')
    
    # LSFS Loss
    ax.text(15.0, 3.9, "L_LSFS", fontsize=12, fontweight='bold', color='#ef6c00')
    draw_arrow((14.0, 3.9), (14.8, 3.9), style='-|>')

    # 标题
    plt.suptitle("CodeFP Architecture: Co-generation with Structural Priors", fontsize=14, y=0.98)
    
    # 保存与显示
    plt.tight_layout()
    plt.savefig('architecture_icml.png', dpi=300, bbox_inches='tight')
    plt.show()

if __name__ == "__main__":
    draw_architecture()