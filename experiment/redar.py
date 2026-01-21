import numpy as np
import matplotlib.pyplot as plt

# ===================== 配置 =====================
labels = ['Recall@20', 'NDCG@20', 'ZRF', 'Time', 'Params']
angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist()
angles += angles[:1]  # 闭合

# 三个数据集的原始数值 (示例数据，直接替换即可)
datasets = {
    'ML-1M': {
        # 'Retrain':   [0.2084 ,	0.3398 , 0.9091 , 52, 9992],
        'SISA':      [0.0570 ,	0.0549  , 0.8724 ,	47, 5994],
        'RecEraser': [0.0578 ,	0.2088 , 0.8921 ,	44, 999],
        'SCIF':      [0.1192 ,	0.1304 , 0.9167 ,	19, 100],
        'P2F':       [0.1886 ,  0.2942 ,  0.9154, 16, 70]
    },
    'Netflix': {
        # 'Retrain':   [0.4156, 0.1841, 0.8865,66, 29428],
        'SISA':      [0.0564, 0.0309, 0.8858, 51, 17657],
        'RecEraser': [0.1416, 0.0653, 0.8973, 30, 2943],
        'SCIF':      [0.1539, 0.1636, 0.9173, 32, 294],
        'P2F':       [0.1655, 0.1748, 0.9365, 26, 100]
    }
    # 'Yelp': {
    #     'Retrain':   [0.28, 0.22, 0.40, 180, 500000],
    #     'SISA':      [0.27, 0.21, 0.38, 130, 420000],
    #     'RecEraser': [0.29, 0.23, 0.43, 95, 300000],
    #     'SCIF':      [0.30, 0.24, 0.45, 65, 220000],
    #     'P2F':       [0.32, 0.26, 0.50, 18, 20000]
    # }
}

# 指标方向：True=正向(越大越好)，False=反向(越小越好)
direction = {'Recall@20': True, 'NDCG@20': True, 'ZRF': True, 'Time': False, 'Params': False}

# ===================== 自动归一化函数 =====================
def normalize_dataset(dataset, labels, direction):
    norm_values = {}
    for i, label in enumerate(labels):
        values = [dataset[baseline][i] for baseline in dataset]
        min_v, max_v = min(values), max(values)

        for baseline, baseline_values in dataset.items():
            val = baseline_values[i]
            if direction[label]:  # 正向指标
                norm = (val - min_v) / (max_v - min_v + 1e-9)
            else:  # 反向指标
                norm = (max_v - val) / (max_v - min_v + 1e-9)
            norm_values.setdefault(baseline, []).append(norm)
    return norm_values

# ===================== 绘图 =====================
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#9467bd', '#d62728']
fig, axes = plt.subplots(1, 2, figsize=(16, 5), subplot_kw=dict(polar=True))

for ax, (dataset_name, raw_data) in zip(axes, datasets.items()):
    data = normalize_dataset(raw_data, labels, direction)

    ax.set_facecolor('white')
    ax.spines['polar'].set_color('#444444')
    ax.spines['polar'].set_linewidth(1.2)
    ax.grid(color='#AAAAAA', linestyle='--', linewidth=0.8, alpha=0.7)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, fontsize=11, fontweight='bold')

    ax.set_yticks([0.2, 0.4, 0.6, 0.8])
    ax.set_yticklabels(['0.2','0.4','0.6','0.8'], fontsize=9)
    ax.set_ylim(0, 1)

    for i, (name, values) in enumerate(data.items()):
        values += values[:1]
        ax.plot(angles, values, color=colors[i], linewidth=2, label=name)
        ax.fill(angles, values, color=colors[i], alpha=0.15)

    ax.set_title(dataset_name, fontsize=14, fontweight='bold', pad=15)

axes[-1].legend(loc='upper left', bbox_to_anchor=(1.15, 1.05), fontsize=11, frameon=False)

plt.tight_layout()
# plt.savefig("radar_1x2.pdf", dpi=300, bbox_inches='tight')  # 保存论文级PDF
plt.show()
