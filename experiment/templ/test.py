import matplotlib.pyplot as plt
import numpy as np

colors = ['#576fa0', '#a7b9d7']  # PGP, Ours
# colors = ['#b57979', '#dea3a2']  # PGP, Ours

# 示例数据（均为0–100）
first1 = [
    [[30, 69], [10, 38], [36, 64], [47, 70]],  # 第一行 4组
    [[30, 39], [10, 16], [36, 44], [47, 52]]   # 第二行 4组
]
first2 = [
    [[35, 52], [31, 47]],  # 第一行 2组
    [[35, 44], [31, 39]]   # 第二行 2组
]
second1 = [
    [[30, 62], [10, 26], [36, 50], [47, 61]],  # 第一行 4组
    [[30, 56], [10, 32], [36, 58], [47, 59]]   # 第二行 4组
]
second2 = [
    [[35, 41], [31, 35]],  # 第一行 2组
    [[35, 42], [31, 36]]   # 第二行 2组
]
x_label1 = ['Cube', 'Bimanual',
          'Can', 'Square']
x_label2 = ['RealMan', 'Agibot']
# 创建 2×4 子图
fig, axs = plt.subplots(
    2, 4,
    figsize=(12, 6),
    dpi=300,
    gridspec_kw={'width_ratios': [3, 2, 3, 2], 'wspace': 0.2, 'hspace': 0.2}
)

def plot_grouped(ax, vals, x_label, narrow=False):
    n = len(vals)
    width = 0.4 if not narrow else 0.24
    x = np.arange(n)
    for i, (pgp, ours) in enumerate(vals):
        # PGP
        rect1 = ax.bar(x[i] - width/2, pgp, width=width, color=colors[0], label='ACT' if i == 0 else "")
        # Ours
        rect2 = ax.bar(x[i] + width/2, ours, width=width, color=colors[1], label='RSAM-ACT' if i == 0 else "")
        # 显示数值
        for rect in [rect1, rect2]:
            for bar in rect:
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2,
                        f'{bar.get_height():.0f}', ha='center', va='bottom', fontsize=8)

    # ax.set_title(title, fontsize=12)
    ax.set_xticks(x)
    ax.set_xticklabels(x_label, fontsize=10)
    ax.set_ylim(0, 100)
    ax.grid(axis='y', linestyle='--', alpha=0.6)
    ax.tick_params(labelsize=10)

# 第一行
plot_grouped(axs[0, 0], first1[0], x_label1, narrow=False)
plot_grouped(axs[0, 1], first2[0], x_label2, narrow=True)
plot_grouped(axs[0, 2], first1[1], x_label1, narrow=False)
plot_grouped(axs[0, 3], first2[1], x_label2, narrow=True)

# 第二行
plot_grouped(axs[1, 0], second1[0], x_label1, narrow=False)
plot_grouped(axs[1, 1], second2[0], x_label2, narrow=True)
plot_grouped(axs[1, 2], second1[1], x_label1, narrow=False)
plot_grouped(axs[1, 3], second2[1], x_label2, narrow=True)

# 图例
axs[0, 0].legend(fontsize=10, loc='upper center', frameon=False)

# plt.tight_layout()
plt.savefig("grouped_subplots_with_values.pdf", bbox_inches='tight')
# plt.show()
