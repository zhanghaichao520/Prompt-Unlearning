import matplotlib.pyplot as plt
import numpy as np

# 数据
forget_ratio = np.array([0.01, 0.03, 0.05, 0.07, 0.09, 0.10])
# ml1m prompt len = 20
# zrf_score = np.array([0.8897, 0.8801, 0.8722, 0.8746, 0.8656, 0.8701])
# r_recall = np.array([0.2334, 0.2216, 0.2041, 0.1835, 0.1728, 0.165])
# r_ndcg = np.array([0.3566, 0.3402, 0.3166, 0.2858, 0.2695, 0.253])

# prompt len = 100
ml1m_zrf_score = np.array([0.9349, 0.8941, 0.896, 0.8922, 0.8876, 0.8824])
ml1m_r_recall = np.array([0.2228, 0.2152, 0.1886, 0.1716, 0.1579, 0.1616])
ml1m_r_ndcg = np.array([0.3419, 0.3297, 0.2942, 0.2712, 0.2513, 0.2555])

amazon_zrf_score = np.array([0.978, 0.9573, 0.9361, 0.9188, 0.9085, 0.8995])
amazon_r_recall = np.array([0.5085, 0.5003, 0.4938, 0.4877, 0.4835, 0.4803])
amazon_r_ndcg = np.array([0.3668, 0.3601, 0.3544, 0.3502, 0.3478, 0.3455])


# 配色
color_zrf = '#1f77b4'
color_recall = '#ff7f0e'
color_ndcg = '#2ca02c'

# 绘图
fig, axs = plt.subplots(1, 2, figsize=(9.8, 2.7), dpi=300, gridspec_kw={'wspace': 0.32})

def plot_dual_axis(ax, y1,y2, zrf_score, r_recall, r_ndcg, ylim1, ylim2):
    ax1 = ax
    ax2 = ax1.twinx()

    # 左轴：ZRF
    ax1.plot(forget_ratio, zrf_score, marker='o', color=color_zrf, linewidth=2, label='ZRF Score')
    if y1:
        ax1.set_ylabel('ZRF Score (↑)', fontsize=15)
    ax1.tick_params(axis='y',  labelsize=15)
    ax1.tick_params(axis='x',  labelsize=15)

    ax1.set_ylim(ylim1)
    ax1.set_xlabel('Unlearning Ratio', fontsize=15)
    ax1.grid(True)

    # 右轴：Retention
    ax2.plot(forget_ratio, r_recall, marker='s', color=color_recall, linewidth=2, label='R-Recall@20')
    ax2.plot(forget_ratio, r_ndcg, marker='^', color=color_ndcg, linewidth=2, label='R-NDCG@20')
    if y2:
        ax2.set_ylabel('Retention Metrics (↑)', fontsize=15)
    ax2.tick_params(axis='y', labelsize=15)
    ax2.set_ylim(ylim2)

    # 合并图例
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax2.legend(lines1 + lines2, labels1 + labels2, loc='lower left', fontsize=10)


plot_dual_axis(axs[0], True, False, ml1m_zrf_score, ml1m_r_recall, ml1m_r_ndcg, (0.8, 1), (0, 0.4))
plot_dual_axis(axs[1], False, True, amazon_zrf_score, amazon_r_recall, amazon_r_ndcg, (0.8, 1), (0.1, 0.65))

# 设置标题
axs[0].set_title('ML-1M', fontsize=15)
axs[1].set_title('Amazon', fontsize=15)
plt.tight_layout()
plt.savefig('/Users/hebert/Desktop/unlearning_ratio_dualaxis_1x2.pdf', bbox_inches='tight')
# plt.show()
