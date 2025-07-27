import matplotlib.pyplot as plt
import numpy as np

# 数据
import matplotlib.pyplot as plt
import numpy as np

# 数据
categories = ['overall', 'cold', 'warm']  # x轴的类别
models = ['ColdLLM', 'GPT-2', 'llama3-1B', 'llama3-3B', 'llama2-7B', 'llama2-13B']  # 模型名称
citeulike_ndcg = {
    'ColdLLM': [0.1559, 0.2026, 0.1726],
    'GPT-2': [0.1538, 0.1739, 0.1774],
    'llama3-1B': [0.175, 0.2133, 0.1808],
    'llama3-3B': [0.1878, 0.2302, 0.1853],
    'llama2-7B': [0.1864, 0.235, 0.2006],
    'llama2-13B': [0.124, 0.2069, 0.1325]
}
ml10m_ndcg = {
    'ColdLLM': [0.1504, 0.0728, 0.2317],
    'GPT-2': [0.1, 0.1, 0.1],
    'llama3-1B': [0.1505, 0.119, 0.2323],
    'llama3-3B': [0.1697, 0.1187, 0.2609],
    'llama2-7B': [0.1765, 0.1221, 0.2718],
    'llama2-13B': [0.1, 0.1, 0.1]
}

# 绘制两个柱状图在一张大图中
colors = ['#D3D3D3', '#808000', '#FF8C00', '#1E90FF', '#00CED1', '#B22222']

# 绘制两个柱状图在一张大图中
fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharey=False)  # 一行两列

# 通用绘图函数
def plot_bar(ax, data, categories, title, ylabel, colors):
    x = np.arange(len(categories))  # x轴的位置
    models = list(data.keys())  # 动态获取模型名称
    width = 0.08  # 每个柱子的宽度

    for i, (model, ndcg) in enumerate(data.items()):
        ax.bar(
            x + i * width - (len(models) / 2 - 0.5) * width,
            ndcg,
            width * 0.9,
            label=model,
            color=colors[i],
            edgecolor='black'
        )

    # 添加标题、标签和刻度
    ax.set_title(title, fontsize=12)
    ax.set_xlabel('Categories', fontsize=10)
    ax.set_ylabel(ylabel, fontsize=10)
    ax.set_xticks(x)
    ax.set_xticklabels(categories, fontsize=10)
    ax.grid(axis='y', linestyle='--', alpha=0.7)

# 左侧子图 - CiteULike 数据集
plot_bar(
    axes[0],
    citeulike_ndcg,
    categories,
    'CiteULike',
    'NDCG',
    colors
)

# 右侧子图 - ML-10M 数据集
plot_bar(
    axes[1],
    ml10m_ndcg,
    categories,
    'ML-10M',
    'NDCG',
    colors
)

# 添加图例
fig.legend(
    labels=list(citeulike_ndcg.keys()),
    loc='upper center',
    ncol=6,
    fontsize=10,
    bbox_to_anchor=(0.5, 1.05)
)

# 自动调整布局
plt.tight_layout(rect=[0, 0, 1, 0.9])
plt.show()

import matplotlib.pyplot as plt
import numpy as np

# 数据
categories = ['Overall', 'Cold', 'Warm']  # x轴的类别
models = ['ColdLLM(7B)', 'LLM-CF(1B)', 'LLM-CF(3B)', 'LLM-CF(7B)', 'LLM-CF(13B)']  # 模型名称
citeulike_ndcg = {
    'ColdLLM (7B)': [0.1559, 0.2026, 0.1726],
    #'GPT-2': [0.1538, 0.1739, 0.1774],
    'FilterLLM (1B)': [0.175, 0.2133, 0.1808],
    'FilterLLM (3B)': [0.1868, 0.2302, 0.1853],
    'FilterLLM (7B)': [0.1874, 0.235, 0.2006],
    'FilterLLM (13B)': [0.2030, 0.2473, 0.2180]
}
ml10m_ndcg = {
    'ColdLLM (7B)': [0.1504, 0.0728, 0.2317],
    #'GPT-2': [0.1, 0.1, 0.1],
    'FilterLLM (1B)': [0.1505, 0.119, 0.2323],
    'FilterLLM (3B)': [0.1697, 0.1187, 0.2609],
    'FilterLLM (7B)': [0.1765, 0.1221, 0.2718],
    'FilterLLM (13B)': [0.1734, 0.131, 0.2666]
}

# 绘制两个柱状图在一张大图中
#colors = ['#D3D3D3', '#808000', '#ca6a0d', '#4e82b7', '#2a8a8a', '#a9271d']
colors = ['#D3D3D3', '#ca6a0d', '#4e82b7', '#2a8a8a', '#a9271d']
# 169，169，169
# 128，128，0
# 202，106，13
# 78，130，183
# 绘制两个柱状图在一张大图中
fig, axes = plt.subplots(1, 2, figsize=(8, 3), sharey=False)  # 一行两列

# 通用绘图函数
def plot_bar(ax, data, categories, title, ylabel, colors):
    x = np.arange(len(categories))  # x轴的位置
    models = list(data.keys())  # 动态获取模型名称
    width = 0.7 / len(models)  # 每个柱子的宽度，根据模型数量调整

    for i, (model, ndcg) in enumerate(data.items()):
        ax.bar(
            x + i * width,
            ndcg,
            width,
            label=model,
            color=colors[i],
            edgecolor='black'
        )

    # 添加标题、标签和刻度
    ax.set_title(title, fontsize=12)
#     ax.set_xlabel('Categories', fontsize=10)
    ax.set_ylabel(ylabel, fontsize=10)
    ax.set_xticks(x + 0.4)  # 调整刻度位置以居中显示柱子组
    ax.set_xticklabels(categories, fontsize=10)
    ax.grid(axis='y', linestyle='--', alpha=0.7)

# 左侧子图 - CiteULike 数据集
plot_bar(
    axes[0],
    citeulike_ndcg,
    categories,
    'CiteULike',
    'NDCG',
    colors
)

# 右侧子图 - ML-10M 数据集
plot_bar(
    axes[1],
    ml10m_ndcg,
    categories,
    'ML-10M',
    '',
    colors
)

# 添加图例
fig.legend(
    labels=list(citeulike_ndcg.keys()),
    loc='upper center',
    ncol=6,
    fontsize=10,
    bbox_to_anchor=(0.5, 1.05)
)

# 自动调整布局
plt.tight_layout(rect=[0, 0, 1, 0.9])
plt.savefig('zhuzhuangtu.pdf', format='pdf', dpi=800, bbox_inches='tight')
plt.show()
