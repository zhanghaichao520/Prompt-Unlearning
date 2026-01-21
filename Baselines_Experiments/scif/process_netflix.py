import os
import pandas as pd

# 创建保存目录
DATASET = "netflix"

output_dir = f"./{DATASET}/processed/{DATASET}"

# 加载数据
inter_df = pd.read_csv(f'{output_dir}/{DATASET}-ori.inter', delimiter='\t')


print(len(inter_df))
import pandas as pd

# 假设你已经加载了inter_df
# Step 1: 统计每个用户的交互记录数
user_interaction_count = inter_df.groupby('user_id:token').size()

# Step 2: 筛选交互记录数在200到500之间的用户
valid_users = user_interaction_count[(user_interaction_count >= 160) & (user_interaction_count <= 180)].index

# Step 3: 根据筛选出的用户ID，提取对应的记录
filtered_inter_df = inter_df[inter_df['user_id:token'].isin(valid_users)]


# Step 4: 保存数据到文件
def save_dataframe(df, filename):
    df.to_csv(os.path.join(output_dir, filename), sep='\t', index=False)

os.makedirs(f"{output_dir}", exist_ok=True)

save_dataframe(filtered_inter_df, f'{DATASET}.inter')

# print(f"item_df len: {len(item_df)}")
print(f"filtered_inter_df len: {len(filtered_inter_df)}")
print(f"filtered_inter_df user len: {len(filtered_inter_df['user_id:token'].unique())}")
print(f"filtered_inter_df item len: {len(filtered_inter_df['item_id:token'].unique())}")