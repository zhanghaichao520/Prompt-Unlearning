import pandas as pd

# 读取数据文件（假设数据存储在一个CSV文件中）
input_file = 'scif_data/amazon-all-beauty-18/amazon-all-beauty-18.inter'
output_file = 'scif_data/amazon-all-beauty-18/amazon-all-beauty-182.inter'

# 加载数据
df = pd.read_csv(input_file, sep='\t', header=None, names=["user_id:token", "item_id:token", "rating:float", "timestamp:float"])

# 创建映射字典
user_id_mapping = {user: idx + 1 for idx, user in enumerate(df['user_id:token'].unique())}
item_id_mapping = {item: idx + 1 for idx, item in enumerate(df['item_id:token'].unique())}

# 使用映射字典替换id
df['user_id:token'] = df['user_id:token'].map(user_id_mapping)
df['item_id:token'] = df['item_id:token'].map(item_id_mapping)

# 保存结果到文件
df.to_csv(output_file, sep='\t', index=False, header=False)
