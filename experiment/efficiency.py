import matplotlib.pyplot as plt
import numpy as np

# ====== 配置区域 ======
titles = ['ML-1M', 'Amazon', 'Netflix']

# 每组 6 个值（真实数据时直接替换这里）
data = [
    [53, 47, 44, 19, 16],
    [42, 35, 21, 20, 13],
    # [33, 50, 58, 52, 60]
]

x_labels = ['Retrain', 'SISA',  'RecEraser', 'SCIF', 'P2F']
colors = ['#576fa0', '#a7b9d7', '#2ca02c', '#ff6347']  # 使用最后一个颜色 (#ff6347) 作为不同的颜色

model_param = [
    [9992.0, 5994.0, 999.0, 100.0, 70.0],
    [744551.0, 446730.59375, 74455.09375, 7444.0, 100.0],
    [497959.0, 298775.390625, 49795.890625, 578.0, 100.0]
]


# ====== 绘图区域 ======
fig, axs = plt.subplots(1, 2, figsize=(12, 3.2), dpi=300, gridspec_kw={'wspace': 0.15}, sharey=True)

for i, ax in enumerate(axs):
    vals = data[i]
    bars = ax.bar(np.arange(5), vals, color=colors[1], width=0.6)  # 给前四个柱子分配颜色
    bars[-1].set_color(colors[0])  # 改变最后一个柱子的颜色，使用一个不同的颜色
    ax.set_xticks(np.arange(5))
    ax.tick_params(axis='y', labelsize=15)
    ax.tick_params(axis='x', labelsize=15)
    ax.set_xticklabels(x_labels, fontsize=15, rotation=30, ha='right')
    ax.set_ylim(0, 60)
    if i == 0:
        ax.set_ylabel('Time (Min)', fontsize=15)
    ax.set_title(titles[i], fontsize=15)
    ax.grid(axis='y', linestyle='--', alpha=0.6)
    
    # 在柱状图上显示每个柱子上的数值
    for bar in bars:
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 2,
                f'{bar.get_height():.0f}', ha='center', va='bottom', fontsize=14)
    
    # 创建第二个 y 轴，用于折线
    ax2 = ax.twinx()  # 创建双 y 轴
    ax2.plot(np.arange(5), model_param[i], color=colors[3], marker='o', linestyle='-', linewidth=2, markersize=6, label="Model Parameter")
    
    # 添加图例
    ax2.legend(loc='upper right', fontsize=15)
    
    if i == 0:
        # 使用symlog坐标轴，在小值区域线性，大值区域对数
        ax2.set_yscale('symlog', linthresh=200)  # 1000以下线性，以上对数
        ax2.set_ylim(0, 15000)
        # 手动设置刻度位置
        custom_ticks = [50, 100, 1000, 2000, 5000, 10000]
        ax2.set_yticks(custom_ticks)
        tick_labels = [f'{int(tick)}' for tick in custom_ticks]
        tick_labels[-1] = '10k'  # 最后一个刻度标签改为 '10k'
        tick_labels[-2] = '5k'  # 倒数第二个刻度标签改为 '5k'
        tick_labels[-3] = '2k'  # 倒数第三个刻度标签改为 '2k'
        tick_labels[-4] = '1k'  # 倒数第四个刻度标签改为 '1k'
        ax2.set_yticklabels(tick_labels)
        ax2.tick_params(axis='y', labelsize=15)

    if i == 1:
        # 使用symlog坐标轴
        ax2.set_yscale('symlog', linthresh=1000)  # 10000以下线性，以上对数
        ax2.set_ylim(0, 1000000)
        # 手动设置刻度位置
        custom_ticks = [100, 500, 2000, 10000, 50000, 100000, 500000]
        ax2.set_yticks(custom_ticks)
        # 对于大于1000的值，转换为"w"单位显示
        tick_labels = []
        for tick in custom_ticks:
            if tick < 1000:
                tick_labels.append(f'{int(tick)}')
            else:
                tick_labels.append(f'{int(tick / 1000)}k')
        ax2.set_yticklabels(tick_labels)
        ax2.tick_params(axis='y', labelsize=15)

    # if i == 2:
    #     ax2.set_ylim(0, 600000)  # 这里设置折线 y 轴的刻度范围
    #     # 设置非等比例刻度：1000以内密集，1000以上稀疏
    #     custom_ticks = [0, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 
    #                    5000, 10000, 20000, 50000, 100000, 200000, 400000, 600000]
    #     ax2.set_yticks(custom_ticks)
    #     # 对于大于1000的值，转换为"w"单位显示
    #     tick_labels = []
    #     for tick in custom_ticks:
    #         if tick < 1000:
    #             tick_labels.append(f'{int(tick)}')
    #         else:
    #             tick_labels.append(f'{int(tick / 10000)}w')
    #     ax2.set_yticklabels(tick_labels)
    #     ax2.tick_params(axis='y', labelsize=15)

plt.tight_layout()
plt.savefig("/Users/hebert/Desktop/efficiency.pdf", bbox_inches='tight')
# plt.show()