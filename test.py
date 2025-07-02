import pandas as pd
import matplotlib.pyplot as plt

# 1. 读取数据（添加表头）
file_path = 'dataset/ml-1m.inter'
df = pd.read_csv(file_path, sep='\t', names=['user_id', 'item_id', 'rating', 'timestamp'])

# 2. 计算物品流行度（每个 item 被评分的次数）
item_popularity = df['item_id'].value_counts().rename_axis('item_id').reset_index(name='popularity')

# 3. 定义热门物品（前10%为热门）
pop_threshold = item_popularity['popularity'].quantile(0.9)
top_items = set(item_popularity[item_popularity['popularity'] >= pop_threshold]['item_id'])

# 4. 给交互打上是否为热门物品的标记
df['is_top'] = df['item_id'].isin(top_items).astype(int)

# 5. 计算每个用户的“从众得分”= 该用户评分中热门物品的占比
user_conformity = df.groupby('user_id')['is_top'].mean().rename('conformity_score').reset_index()

# 6. 把物品流行度和用户从众得分加入原始交互中
df = df.merge(item_popularity, on='item_id')
df = df.merge(user_conformity, on='user_id')

# 7. 把物品按流行度分成10组，统计每组中交互用户的平均从众得分
df['pop_bin'] = pd.qcut(df['popularity'], 10, labels=False)
pop_vs_conform = df.groupby('pop_bin')['conformity_score'].mean()

# 8. 绘图（放大所有字体）
plt.figure(figsize=(10, 6))
plt.plot(pop_vs_conform.index, pop_vs_conform.values, marker='o')

plt.xlabel("Item Popularity Quantile (Low → High)", fontsize=20)
plt.ylabel("Mean Popularity Ratio in User Interactions", fontsize=20)
plt.title("Popularity Drives Conformity", fontsize=22)
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
plt.grid(True)
plt.tight_layout()
plt.show()
