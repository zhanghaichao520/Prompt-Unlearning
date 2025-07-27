import matplotlib.pyplot as plt
import numpy as np

# 数据
prompt_len = np.array([10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 120, 140, 150, 160, 180, 200])
ml1m_zrf_score = np.array([0.8827, 0.8948, 0.902, 0.9004, 0.9156, 0.9172, 0.928, 0.9262, 0.934, 0.9326, 0.9422, 0.9421, 0.9419, 0.9345, 0.9475, 0.9412])
ml1m_recall_20 = np.array([0.2344, 0.2306, 0.2269, 0.2274, 0.2266, 0.2233, 0.2245, 0.2225, 0.2224, 0.2213, 0.2217, 0.2207, 0.222, 0.2233, 0.221, 0.2228])
ml1m_ndcg_20 = np.array([0.3575, 0.3527, 0.3467, 0.347, 0.3446, 0.3424, 0.3438, 0.3413, 0.3402, 0.3394, 0.3404, 0.3383, 0.3388, 0.3401, 0.3392, 0.341])

amazon_zrf_score = np.array([0.9401, 0.9505, 0.9568, 0.9612, 0.9658, 0.9701, 0.9735, 0.9753, 0.9769, 0.978, 0.9801, 0.9815, 0.9821, 0.9828, 0.9838, 0.9845])
amazon_r_recall_20 = np.array([0.5201, 0.5173, 0.5155, 0.5148, 0.5131, 0.5117, 0.5109, 0.5101, 0.5093, 0.5085, 0.5071, 0.506, 0.5055, 0.5049, 0.504, 0.5031])
amazon_r_ndcg_20 = np.array([0.3763, 0.3738, 0.372, 0.3715, 0.3701, 0.3692, 0.3688, 0.3681, 0.3675, 0.3668, 0.3659, 0.3651, 0.3647, 0.3643, 0.3635, 0.3628])


# 配色
color_zrf = '#1f77b4'
color_rrecall = '#ff7f0e'
color_rndcg = '#2ca02c'

# 创建 1x2 子图
fig, axs = plt.subplots(1, 2, figsize=(9.8, 2.7), dpi=300, gridspec_kw={'wspace': 0.37})

def plot_dual_axis(ax, y1, y2, zrf_score, r_recall_20, r_ndcg_20, ylim1, ylim2):
    ax1 = ax
    ax2 = ax1.twinx()

    # 左轴：ZRF
    ax1.plot(prompt_len, zrf_score, marker='o', linewidth=2, color=color_zrf, label='ZRF Score')
    if y1:
        ax1.set_ylabel('ZRF Score (↑)', fontsize=15)
    ax1.tick_params(axis='y', labelsize=15)
    ax1.tick_params(axis='x',  labelsize=15)

    ax1.set_ylim(ylim1)
    ax1.set_xlabel('Prompt Length', fontsize=15)
    ax1.grid(True)

    # 右轴：Retain
    ax2.plot(prompt_len, r_recall_20, marker='s', linewidth=2, color=color_rrecall, label='R-Recall@20')
    ax2.plot(prompt_len, r_ndcg_20, marker='^', linewidth=2, color=color_rndcg, label='R-NDCG@20')
    if y2:
        ax2.set_ylabel('Retention Metrics (↑)', fontsize=15)
    ax2.tick_params(axis='y', labelsize=15)
    ax2.set_ylim(ylim2)

    # 合并图例
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax2.legend(lines1 + lines2, labels1 + labels2, loc='upper left', fontsize=10)

# 画两个一模一样的图
plot_dual_axis(axs[0], True, False, ml1m_zrf_score, ml1m_recall_20, ml1m_ndcg_20, (0.87, 0.97), (0.1, 0.5))
plot_dual_axis(axs[1], False, True, amazon_zrf_score, amazon_r_recall_20, amazon_r_ndcg_20, (0.92, 1), (0.3, 0.8)) 

# 设置标题
axs[0].set_title('ML-1M', fontsize=15)
axs[1].set_title('Amazon', fontsize=15)

plt.tight_layout()
plt.savefig("/Users/hebert/Desktop/prompt_length_1x2.pdf", bbox_inches='tight')
# plt.show()
