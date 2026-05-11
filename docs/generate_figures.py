"""
VCP 论文三张图片生成脚本
运行: pip install matplotlib numpy
      python generate_figures.py
输出: fig1_framework.jpg, fig2_shot_map50.jpg, fig3_pr_comparison.jpg
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np

# ============================================================
# 全局设置
# ============================================================
plt.rcParams.update({
    'font.family': 'DejaVu Sans',
    'font.size': 11,
    'axes.titlesize': 13,
    'axes.labelsize': 11,
    'savefig.dpi': 200,
    'savefig.bbox': 'tight',
})

# ============================================================
# 图1: VCP 整体框架图
# ============================================================
def draw_fig1():
    fig, ax = plt.subplots(1, 1, figsize=(18, 12.5))
    ax.set_xlim(0, 18)
    ax.set_ylim(0, 12)
    ax.axis('off')

    # ================================================================
    # 辅助绘图函数
    # ================================================================
    def box(ax, x, y, w, h, text, color='#E8F0FE', fontsize=9, bold=False,
            edgecolor='#333333', lw=1.0):
        rect = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.08",
                              facecolor=color, edgecolor=edgecolor, linewidth=lw)
        ax.add_patch(rect)
        weight = 'bold' if bold else 'normal'
        ax.text(x + w/2, y + h/2, text, ha='center', va='center',
                fontsize=fontsize, fontweight=weight)

    def small_box(ax, x, y, w, h, text, color='#F0F0F0', fontsize=7, bold=False):
        """更小的子模块框"""
        rect = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.05",
                              facecolor=color, edgecolor='#666666', linewidth=0.8)
        ax.add_patch(rect)
        weight = 'bold' if bold else 'normal'
        ax.text(x + w/2, y + h/2, text, ha='center', va='center',
                fontsize=fontsize, fontweight=weight)

    def arrow(ax, x1, y1, x2, y2, color='#555555', lw=1.2, style='->'):
        ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle=style, color=color, lw=lw))

    def dashed_arrow(ax, x1, y1, x2, y2, color='#888888', lw=1.0):
        ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle='->', color=color, lw=lw,
                                    linestyle='dashed'))

    def draw_path(ax, points, color='#555555', lw=1.2, style='-'):
        """画折线路径"""
        xs, ys = zip(*points)
        ax.plot(xs, ys, color=color, lw=lw, linestyle=style)
        # 最后一个线段加箭头
        ax.annotate('', xy=points[-1], xytext=points[-2],
                    arrowprops=dict(arrowstyle='->', color=color, lw=lw))

    # ================================================================
    # 区域背景色块 (分四大区域)
    # ================================================================
    # 特征提取区 (左上) — 包裹 Backbone, Neck, Decoupled Head
    rect1 = FancyBboxPatch((0.2, 7.5), 11.8, 3.2, boxstyle="round,pad=0.15",
                           facecolor='#F0F7FF', edgecolor='#B0C4DE', linewidth=0.8, alpha=0.5)
    ax.add_patch(rect1)
    ax.text(6.1, 10.5, 'Feature Extraction (Shared Weights)', ha='center', fontsize=8,
            color='#4A7FB5', style='italic')

    # 原型构造区 (左下) — 包裹 Visual Prototype 和 Fusion Block
    rect2 = FancyBboxPatch((0.2, 2.3), 8.0, 5.0, boxstyle="round,pad=0.15",
                           facecolor='#FFF5F0', edgecolor='#E8C4A8', linewidth=0.8, alpha=0.5)
    ax.add_patch(rect2)
    ax.text(4.2, 7.15, 'Visual Prototype Construction', ha='center', fontsize=8,
            color='#CC6600', style='italic')

    # VLM语义区 (右侧)
    rect3 = FancyBboxPatch((12.0, 3.0), 5.8, 7.8, boxstyle="round,pad=0.15",
                           facecolor='#F8F0FF', edgecolor='#D0C0E8', linewidth=0.8, alpha=0.5)
    ax.add_patch(rect3)
    ax.text(14.9, 10.6, 'VLM Semantic Injection', ha='center', fontsize=8,
            color='#7B2D8E', style='italic')

    # 分类与输出区 (中下) — 包裹 Cosine Classifier 和 Detection Output
    rect4 = FancyBboxPatch((7.5, 0.3), 4.5, 6.7, boxstyle="round,pad=0.15",
                           facecolor='#FFFFF0', edgecolor='#D4C898', linewidth=0.8, alpha=0.5)
    ax.add_patch(rect4)
    ax.text(9.75, 6.8, 'Classification & Output', ha='center', fontsize=8,
            color='#B8860B', style='italic')

    # ================================================================
    # 第一层: 输入 + 特征提取 (y=8.0-10.5)
    # ================================================================
    # 查询图像 & 支持集 (左侧输入)
    box(ax, 0.4, 9.2, 1.6, 0.9, 'Query Image\n$I_q \\in \\mathbb{R}^{640\\times640}$',
        '#D5E8D4', 8, True)
    box(ax, 0.4, 8.0, 1.6, 0.9, 'Support Set\n$\\{I_s^k\\}_{k=1}^{K}$',
        '#F8CECC', 8, True)

    # Backbone 大框(含内部结构)
    box(ax, 2.6, 8.5, 2.4, 1.8, '', '#E8F0FE', 9, True, lw=1.3)
    ax.text(3.8, 10.1, 'Backbone', ha='center', fontsize=10, fontweight='bold')
    # Backbone内部子模块
    small_box(ax, 2.85, 9.2, 0.65, 0.6, 'CBS', '#D0E0F0', 6.5)
    small_box(ax, 3.65, 9.2, 0.65, 0.6, 'C3k=2', '#D0E0F0', 6.5)
    small_box(ax, 4.3, 9.2, 0.55, 0.6, 'C3\nk=2', '#D0E0F0', 6)
    small_box(ax, 2.85, 8.6, 0.65, 0.5, 'SPPF', '#D0E0F0', 6.5)
    ax.text(3.8, 8.55, 'CSPDarknet (YOLO11s)', ha='center', fontsize=7, color='#888888')

    # Neck 大框
    box(ax, 5.6, 8.5, 2.4, 1.8, '', '#E8F0FE', 9, True, lw=1.3)
    ax.text(6.8, 10.1, 'Neck (FPN + PAN)', ha='center', fontsize=10, fontweight='bold')
    small_box(ax, 5.8, 9.3, 0.7, 0.5, 'P5/32', '#D0E0F0', 6.5)
    small_box(ax, 6.6, 9.3, 0.7, 0.5, 'P4/16', '#D0E0F0', 6.5)
    small_box(ax, 7.3, 9.3, 0.55, 0.5, 'P3/8', '#D0E0F0', 6)
    ax.text(5.8, 8.85, '↑', ha='center', fontsize=9, color='#888888')
    ax.text(6.6, 8.85, '↑↓', ha='center', fontsize=9, color='#888888')
    ax.text(7.3, 8.85, '↓', ha='center', fontsize=9, color='#888888')
    ax.text(6.8, 8.55, 'Top-down + Bottom-up pathways', ha='center', fontsize=6.5, color='#888888')

    # 输入 → Backbone 箭头
    arrow(ax, 2.0, 9.65, 2.6, 9.65)
    arrow(ax, 2.0, 8.45, 2.6, 8.65, color='#CC4444', lw=1.3)  # 支持集共享

    # Backbone → Neck
    arrow(ax, 5.0, 9.4, 5.6, 9.4)

    # ================================================================
    # 第二层: Decoupled Head (双分支) + 特征分流 (y=6.0-8.0)
    # ================================================================
    # Decoupled Head 大框
    box(ax, 8.5, 8.5, 3.0, 1.8, '', '#E6EEF8', 9, True, lw=1.3)
    ax.text(10.0, 10.1, 'Decoupled Head', ha='center', fontsize=10, fontweight='bold')

    # 分类分支
    small_box(ax, 8.7, 9.2, 1.3, 0.6, 'Cls Branch\ncv3 → Conv', '#D0E0F0', 7)
    # 回归分支
    small_box(ax, 10.1, 9.2, 1.3, 0.6, 'Reg Branch\ncv2 → Conv', '#D0E0F0', 7)
    # 特征输入
    ax.text(10.0, 8.65, 'Feature vectors $f \\in \\mathbb{R}^d$', ha='center', fontsize=7.5,
            color='#555555')

    # Neck → Decoupled Head
    arrow(ax, 8.0, 9.4, 8.5, 9.4)

    # Neck → RoIAlign 分支 (FPN特征分流, 虚线)
    ax.plot([7.0, 7.0], [8.5, 7.6], color='#CC6600', lw=1.3, linestyle='dashed')
    ax.plot([7.0, 3.5], [7.6, 7.6], color='#CC6600', lw=1.3, linestyle='dashed')
    ax.annotate('', xy=(3.5, 6.9), xytext=(3.5, 7.6),
                arrowprops=dict(arrowstyle='->', color='#CC6600', lw=1.3,
                                linestyle='dashed'))
    ax.text(5.0, 7.75, 'P3–P5 FPN features (shared)', fontsize=7, color='#CC6600',
            style='italic')

    # ================================================================
    # 第三层: 视觉原型构造 (左下, y=3.5-6.5)
    # ================================================================
    box(ax, 2.5, 5.8, 2.2, 1.0, 'RoIAlign\n(extract GT-box features)', '#F8CECC', 8.5, True)
    box(ax, 2.5, 4.6, 2.2, 0.8, 'Feature Averaging\n$\\frac{1}{K}\\sum_k f_k$', '#F8CECC', 8)
    box(ax, 2.5, 3.5, 2.2, 0.8, 'L$_2$ Normalization\n$p = \\tilde{p}\\,/\\,\\|\\tilde{p}\\|_2$',
        '#F8CECC', 8)

    arrow(ax, 3.6, 5.8, 3.6, 5.4)
    arrow(ax, 3.6, 4.6, 3.6, 4.3)

    # 视觉原型输出
    box(ax, 5.2, 4.0, 2.2, 0.9, 'Visual Prototype\n$p_c^{\\text{vis}} \\in \\mathbb{R}^d$',
        '#FFE6CC', 9, True)

    arrow(ax, 4.7, 3.9, 5.2, 4.45)  # L2 Norm → Visual Proto

    # ================================================================
    # 第四层: VLM语义原型 (右侧, y=3.5-10.0)
    # ================================================================
    # Florence-2 输入 (向下移动避开VLM区域标签)
    ax.text(14.75, 9.85, 'Region Crop\n+ "<DETAILED_CAPTION>" prompt',
            ha='center', fontsize=8, color='#7B2D8E',
            bbox=dict(boxstyle='round', facecolor='#F5EDF8', edgecolor='#C0A0D0', alpha=0.8))

    box(ax, 12.8, 8.4, 2.2, 1.0, 'Florence-2\n(Multi-Task VLM)', '#E1D5E7', 9, True)
    arrow(ax, 14.75, 9.85, 14.75, 9.4, color='#7B2D8E', lw=1.3)

    box(ax, 12.8, 7.1, 2.2, 1.0, 'BART Text Encoder\n$E_{\\text{text}}(\\text{description})$',
        '#E1D5E7', 8)
    arrow(ax, 13.9, 8.4, 13.9, 8.1)

    box(ax, 12.8, 5.9, 2.2, 0.9, 'Text Embedding\n$e_c \\in \\mathbb{R}^{d_t}$', '#E1D5E7', 8)
    arrow(ax, 13.9, 7.1, 13.9, 6.8)

    box(ax, 12.8, 4.8, 2.2, 0.8, 'Projection $W_f$\n$\\mathbb{R}^{d_t} \\to \\mathbb{R}^{d}$',
        '#E1D5E7', 8)
    arrow(ax, 13.9, 5.9, 13.9, 5.6)

    box(ax, 12.8, 3.7, 2.2, 0.8, 'Text Prototype\n$p_c^{\\text{text}} \\in \\mathbb{R}^{d}$',
        '#E1D5E7', 8.5, True)
    arrow(ax, 13.9, 4.8, 13.9, 4.5)

    # ================================================================
    # 第五层: 原型融合 (中下, y=1.5-3.5)
    # ================================================================
    # 融合模块
    box(ax, 5.2, 2.6, 2.2, 1.0,
        'Fusion Block\n$p_c = \\alpha p_c^{\\text{vis}} + (1-\\alpha) p_c^{\\text{text}}$\n$\\alpha = 0.7$',
        '#D9D2E9', 8, True)

    # 视觉原型 → Fusion (Proto Mode: 直接去分类器, Fused Mode: 过融合)
    arrow(ax, 6.3, 4.0, 6.3, 3.6, color='#CC6600', lw=1.3)
    # 文本原型 → Fusion
    draw_path(ax, [(13.9, 3.7), (13.9, 3.1), (7.4, 3.1)], color='#7B2D8E', lw=1.3)

    # ================================================================
    # 第六层: 余弦分类器 (中右, y=4.0-6.0)
    # ================================================================
    box(ax, 8.5, 5.0, 3.5, 1.5, '', '#FFF2CC', 9, True, '#B8860B', 1.3)
    ax.text(10.25, 6.3, 'Cosine Classifier', ha='center', fontsize=10, fontweight='bold')
    ax.text(10.25, 5.8, '$s_c = \\frac{\\cos(f,\\, p_c)}{\\tau} = \\frac{f \\cdot p_c}{\\tau\\,\\|f\\|\\,\\|p_c\\|}$',
            ha='center', fontsize=8.5, style='italic', color='#8B6914')
    ax.text(10.25, 5.2, '$\\tau$: learnable temperature (init = 5.0)', ha='center',
            fontsize=7, color='#888888')

    # Decoupled Head → Cosine Classifier (特征向量输入)
    arrow(ax, 10.0, 8.5, 10.0, 6.5, color='#333333', lw=1.5)

    # ================================================================
    # 原型到分类器的权重初始化路径
    # ================================================================
    # Path A: Visual Prototype → Cosine Classifier (Proto Mode, 虚线)
    dashed_arrow(ax, 7.4, 4.45, 8.5, 5.7, color='#CC6600', lw=1.3)
    # Path B: Fusion → Cosine Classifier (Fused Mode, 虚线)
    dashed_arrow(ax, 7.4, 3.1, 8.5, 5.3, color='#7B2D8E', lw=1.3)

    # 权重初始化标注
    ax.text(7.8, 5.5, 'weight\ninit', fontsize=6.5, color='#CC6600', style='italic',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
    ax.text(7.8, 4.0, 'weight\ninit', fontsize=6.5, color='#7B2D8E', style='italic',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))

    # ================================================================
    # 第七层: 输出 (底部, y=0.5-2.5)
    # ================================================================
    box(ax, 8.5, 1.2, 3.5, 0.9, '', '#D5E8D4', 9, True, '#5A8A5A', 1.2)
    ax.text(10.25, 2.0, 'Detection Output', ha='center', fontsize=10, fontweight='bold')
    ax.text(10.25, 1.5, 'Class Scores ($s_c$) + Bbox Regression $\\to$ NMS',
            ha='center', fontsize=7.5)

    # Cosine Classifier → Detection Output (实线)
    arrow(ax, 10.25, 5.0, 10.25, 2.1, color='#333333', lw=1.5)

    # 回归分支也到输出 (从Decoupled Head的Reg Branch右端绕行，避开Cosine Classifier)
    draw_path(ax, [(11.4, 9.5), (12.5, 9.5), (12.5, 1.65), (12.0, 1.65)],
              color='#999999', lw=1.0)
    ax.text(12.5, 5.5, 'bbox', fontsize=7, color='#999999', rotation=90, va='center')

    # ================================================================
    # 训练阶段标注
    # ================================================================
    ax.text(0.5, 10.8, 'Stage 1: Base-Class Pretraining (15 base classes, 100 epochs)',
            fontsize=7.5, color='#2E7D32',
            bbox=dict(boxstyle='round', facecolor='#E8F5E9', alpha=0.8))
    ax.text(0.5, 0.5, 'Stage 2: Novel-Class Finetuning (5 novel classes, 200 epochs, backbone frozen first 10 epochs)',
            fontsize=7.5, color='#C62828',
            bbox=dict(boxstyle='round', facecolor='#FFEBEE', alpha=0.8))

    # ================================================================
    # 图例 (右下角)
    # ================================================================
    legend_items = [
        ('Feature Extraction', '#E8F0FE'),
        ('Visual Prototype Path (Proto Mode)', '#F8CECC'),
        ('VLM Semantic Path (Fused Mode)', '#E1D5E7'),
        ('Classification & Output', '#FFF2CC'),
    ]
    ax.text(13.5, 2.9, 'Legend', fontsize=7.5, fontweight='bold')
    for i, (label, color) in enumerate(legend_items):
        ly = 2.5 - i * 0.4
        rect = FancyBboxPatch((13.5, ly), 0.35, 0.25, boxstyle="round,pad=0.02",
                              facecolor=color, edgecolor='#333333', linewidth=0.6)
        ax.add_patch(rect)
        ax.text(14.0, ly + 0.12, label, fontsize=7, va='center')

    # ================================================================
    # 标题
    # ================================================================
    ax.text(9.0, 11.6, 'VCP (VLM-Cosine-Prototype) Architecture for Few-Shot Object Detection',
            ha='center', fontsize=17, fontweight='bold')

    fig.tight_layout(pad=0.8)
    fig.savefig('fig1_framework.jpg', dpi=200, format='jpg', pil_kwargs={'quality': 95})
    plt.close(fig)
    print('fig1_framework.jpg 已生成')


# ============================================================
# 图2: VOC shot-mAP50 折线图
# ============================================================
def draw_fig2():
    shots = [1, 3, 5, 10]

    data = {
        'Finetune':       [10.9, 20.5, 32.9, 51.3],
        'Cosine':         [12.8, 51.3, 66.9, 70.7],
        'Cosine+Proto':   [21.5, 52.3, 68.8, 74.4],
        'Cosine+Fused':   [21.3, 53.9, 69.0, 74.7],
    }

    colors = {'Finetune': '#999999', 'Cosine': '#2196F3',
              'Cosine+Proto': '#FF9800', 'Cosine+Fused': '#E63946'}
    markers = {'Finetune': 's', 'Cosine': 'o',
               'Cosine+Proto': '^', 'Cosine+Fused': 'D'}

    fig, ax = plt.subplots(1, 1, figsize=(8, 5.5))

    for method, values in data.items():
        ax.plot(shots, values, color=colors[method], marker=markers[method],
                linewidth=2.5, markersize=10, label=method, zorder=3)

    # 标注最佳值
    ax.annotate('69.0', xy=(5, 69.0), xytext=(5.5, 72),
                fontsize=10, fontweight='bold', color='#E63946',
                arrowprops=dict(arrowstyle='->', color='#E63946'))
    ax.annotate('74.7', xy=(10, 74.7), xytext=(9, 77),
                fontsize=10, fontweight='bold', color='#E63946',
                arrowprops=dict(arrowstyle='->', color='#E63946'))

    ax.set_xlabel('Shot', fontsize=13)
    ax.set_ylabel('mAP50 (%)', fontsize=13)
    ax.set_title('PASCAL VOC Few-Shot Detection Performance', fontsize=14, fontweight='bold')
    ax.legend(loc='lower right', fontsize=10, framealpha=0.9)
    ax.set_xticks(shots)
    ax.set_xticklabels(['1-shot', '3-shot', '5-shot', '10-shot'])
    ax.set_ylim(0, 85)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig('fig2_shot_map50.jpg', dpi=200, format='jpg', pil_kwargs={'quality': 95})
    plt.close(fig)
    print('fig2_shot_map50.jpg 已生成')


# ============================================================
# 图３: VOC 5-shot  vs COCO 30-shot  P-R 对比
# ============================================================
def draw_fig3():
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.5))

    # --- 左图: VOC 5-shot ---
    ax = axes[0]
    # Finetune
    ax.scatter(0.430, 0.284, s=250, c='#999999', marker='s', edgecolors='#333333',
               linewidths=1.5, zorder=5, label='Finetune')
    # Cosine+Fused
    ax.scatter(0.875, 0.061, s=250, c='#E63946', marker='D', edgecolors='#333333',
               linewidths=1.5, zorder=5, label='Cosine+Fused')
    # 连线箭头
    ax.annotate('', xy=(0.875, 0.061), xytext=(0.430, 0.284),
                arrowprops=dict(arrowstyle='->', color='#E63946', lw=2.5,
                                connectionstyle='arc3,rad=-0.15'))
    # 标签
    ax.text(0.43, 0.37, 'Finetune\nR=43%  P=28%\nmAP=33.1%',
            ha='center', fontsize=10, fontweight='bold',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    ax.text(0.88, 0.13, 'Cosine+Fused\nR=88%  P=6%\nmAP=69.0%',
            ha='center', fontsize=10, fontweight='bold', color='#E63946',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    # 效果标注
    ax.text(0.65, 0.55, 'RECALL BOOSTER\n(+102% recall → mAP↑)',
            ha='center', fontsize=12, fontweight='bold', color='#2E7D32',
            bbox=dict(boxstyle='round', facecolor='#C8E6C9', alpha=0.9))

    ax.set_xlabel('Recall', fontsize=12)
    ax.set_ylabel('Precision', fontsize=12)
    ax.set_title('VOC 5-shot', fontsize=14, fontweight='bold')
    ax.set_xlim(0, 1.05)
    ax.set_ylim(0, 0.7)
    ax.legend(loc='upper right', fontsize=9)
    ax.grid(True, alpha=0.3)

    # --- 右图: COCO 30-shot ---
    ax = axes[1]
    ax.scatter(0.562, 0.675, s=250, c='#999999', marker='s', edgecolors='#333333',
               linewidths=1.5, zorder=5, label='Finetune')
    ax.scatter(0.702, 0.120, s=250, c='#E63946', marker='D', edgecolors='#333333',
               linewidths=1.5, zorder=5, label='Cosine+Fused')
    ax.annotate('', xy=(0.702, 0.120), xytext=(0.562, 0.675),
                arrowprops=dict(arrowstyle='->', color='#E63946', lw=2.5,
                                connectionstyle='arc3,rad=0.15'))
    ax.text(0.56, 0.73, 'Finetune\nR=56%  P=68%\nmAP=60.3%',
            ha='center', fontsize=10, fontweight='bold',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    ax.text(0.70, 0.04, 'Cosine+Fused\nR=70%  P=12%\nmAP=52.3%',
            ha='center', fontsize=10, fontweight='bold', color='#E63946',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    ax.text(0.63, 0.42, 'CEILING EFFECT\n(+25% recall → mAP↓)',
            ha='center', fontsize=12, fontweight='bold', color='#C62828',
            bbox=dict(boxstyle='round', facecolor='#FFCDD2', alpha=0.9))

    ax.set_xlabel('Recall', fontsize=12)
    ax.set_ylabel('Precision', fontsize=12)
    ax.set_title('COCO 30-shot', fontsize=14, fontweight='bold')
    ax.set_xlim(0, 1.05)
    ax.set_ylim(0, 0.85)
    ax.legend(loc='upper right', fontsize=9)
    ax.grid(True, alpha=0.3)

    fig.suptitle('Cosine Classifier: Consistent P↓ R↑ Behavior', fontsize=15, fontweight='bold', y=1.01)
    fig.tight_layout()
    fig.savefig('fig3_pr_comparison.jpg', dpi=200, format='jpg', pil_kwargs={'quality': 95})
    plt.close(fig)
    print('fig3_pr_comparison.jpg 已生成')


# ============================================================
# 执行
# ============================================================
if __name__ == '__main__':
    draw_fig1()
    draw_fig2()
    draw_fig3()
    print('三张图全部生成完毕（JPG格式）。')
