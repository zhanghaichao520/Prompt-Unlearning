import matplotlib.pyplot as plt
import numpy as np

# X轴
K_vals = np.array([5, 10, 15, 20, 30, 50])

# 指标数据（Recall@10 / NDCG@10）
forget_recall = {
    'Full Model': np.array([0.0916, 0.1498, 0.1942, 0.2314, 0.2924, 0.3803]),
    'w/o GKR':    np.array([0.0943, 0.1513, 0.1959, 0.2341, 0.2947, 0.3827]),
    'w/o SFD':    np.array([0.0906, 0.1487, 0.1929, 0.2295, 0.2917, 0.3817]),
    'w/o PRD':    np.array([0.0609, 0.1019, 0.1382, 0.1694, 0.2203, 0.2984]),
}

forget_ndcg = {
    'Full Model': np.array([0.3975, 0.3685, 0.3577, 0.3543, 0.3570, 0.3721]),
    'w/o GKR':    np.array([0.3995, 0.3699, 0.3599, 0.3568, 0.3590, 0.3742]),
    'w/o SFD':    np.array([0.3883, 0.3631, 0.3537, 0.3501, 0.3537, 0.3700]),
    'w/o PRD':    np.array([0.2893, 0.2681, 0.2622, 0.2612, 0.2654, 0.2810]),
}

retain_recall = {
    'Full Model': np.array([0.0274, 0.0359, 0.0505, 0.0584, 0.0809, 0.1092]),
    'w/o GKR':    np.array([0.0309, 0.0522, 0.0654, 0.0771, 0.1017, 0.1404]),
    'w/o SFD':    np.array([0.0585, 0.0810, 0.1055, 0.1278, 0.1617, 0.1994]),
    'w/o PRD':    np.array([0.0106, 0.0271, 0.0400, 0.0476, 0.0657, 0.0922]),
}

retain_ndcg = {
    'Full Model': np.array([0.0973, 0.0802, 0.0789, 0.0768, 0.0855, 0.0971]),
    'w/o GKR':    np.array([0.1313, 0.1171, 0.1079, 0.1056, 0.1098, 0.1197]),
    'w/o SFD':    np.array([0.1425, 0.1308, 0.1312, 0.1352, 0.1471, 0.1631]),
    'w/o PRD':    np.array([0.0505, 0.0524, 0.0547, 0.0548, 0.0607, 0.0697]),
}

# AAAI配色
colors = {
    'Full Model': '#1f77b4',   # 深蓝
    'w/o GKR': '#ff7f0e',      # 橙色
    'w/o SFD': '#2ca02c',      # 绿色
    'w/o PRD': '#d62728'       # 红色
}

# 绘图
fig, axs = plt.subplots(1, 4, figsize=(18, 3.2), dpi=300, gridspec_kw={'wspace': 0.35})

# 子图1：Forget Recall@10
for name in forget_recall:
    axs[0].plot(K_vals, forget_recall[name], marker='o', label=name, linewidth=2, color=colors[name])
axs[0].set_title('Retain Set Performance (Recall)', fontsize=16)
axs[0].set_xlabel('topK', fontsize=16)
# axs[0].set_ylabel('Recall@10', fontsize=16)
axs[0].grid(True)
axs[0].tick_params(labelsize=16)

# 子图2：Forget NDCG@10
for name in forget_ndcg:
    axs[1].plot(K_vals, forget_ndcg[name], marker='s', label=name, linewidth=2, color=colors[name])
axs[1].set_title('Retain Set Performance (NDCG)', fontsize=16)
axs[1].set_xlabel('topK', fontsize=16)
# axs[1].set_ylabel('NDCG@10', fontsize=16)
axs[1].grid(True)
axs[1].tick_params(labelsize=16)

# 子图3：Retain Recall@10
for name in retain_recall:
    axs[2].plot(K_vals, retain_recall[name], marker='o', label=name, linewidth=2, color=colors[name])
axs[2].set_title('Forget Set Performance (Recall)', fontsize=16)
axs[2].set_xlabel('topK', fontsize=16)
# axs[2].set_ylabel('Recall@10', fontsize=16)
axs[2].grid(True)
axs[2].tick_params(labelsize=16)

# 子图4：Retain NDCG@10
for name in retain_ndcg:
    axs[3].plot(K_vals, retain_ndcg[name], marker='s', label=name, linewidth=2, color=colors[name])
axs[3].set_title('Forget Set Performance (NDCG)', fontsize=16)
axs[3].set_xlabel('topK', fontsize=16)
# axs[3].set_ylabel('NDCG@10', fontsize=16)
axs[3].grid(True)
axs[3].tick_params(labelsize=16)

# 图例仅放在第一张图中
axs[0].legend(loc='best', fontsize=14)

plt.tight_layout()
plt.savefig("ablation.pdf", bbox_inches='tight', dpi=300)