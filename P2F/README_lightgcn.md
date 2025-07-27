# LightGCN推荐系统实现

## 项目简介

本项目实现了基于LightGCN（Light Graph Convolutional Network）的推荐系统，包括模型训练和评估部分。LightGCN是一种简化的图卷积网络，专为推荐系统设计，通过用户-物品交互图进行协同过滤。

## 数据集

项目使用MovieLens-100K数据集，该数据集包含：
- 约10万条用户对电影的评分记录
- 格式为：user_id, item_id, rating, timestamp

数据集位置：`/dataset/ml-100k.inter`

## 模型实现

模型实现参考了`/recobole/lightgcn.py`，主要特点：

1. **模型架构**：
   - 用户和物品嵌入层
   - 图卷积层（多层）
   - 嵌入聚合

2. **损失函数**：
   - BPR损失（贝叶斯个性化排序）
   - L2正则化

3. **评估指标**：
   - NDCG@k（归一化折损累积增益）
   - Recall@k（召回率）
   - HitRate@k（命中率）
   - 评估时k分别取10和20

## 文件说明

- `train_lightgcn.py`：主脚本，包含数据处理、模型定义、训练和评估

## 使用方法

### 环境要求

- Python 3.6+
- PyTorch 1.0+
- NumPy
- Pandas
- SciPy
- scikit-learn

### 运行方式

```bash
python train_lightgcn.py
```

### 参数设置

主要参数在`main`函数中设置：

```python
# 参数设置
data_path = "dataset/ml-100k.inter"  # 数据集路径
embedding_size = 64                 # 嵌入维度
n_layers = 3                        # 图卷积层数
reg_weight = 1e-4                   # 正则化权重
batch_size = 2048                   # 批次大小
lr = 0.001                          # 学习率
epochs = 100                        # 训练轮数
eval_freq = 5                       # 评估频率
k_list = [10, 20]                   # 评估的k值
```

## 实现细节

1. **数据处理**：
   - 将原始数据转换为用户-物品交互矩阵
   - 按用户划分训练集和测试集
   - 为每个正样本生成负样本

2. **模型训练**：
   - 使用Adam优化器
   - 采用BPR损失函数
   - 定期评估模型性能

3. **模型评估**：
   - 对每个测试用户，预测所有物品的评分
   - 排除训练集中已交互的物品
   - 计算TopK推荐的各项指标

## 参考文献

Xiangnan He et al. "LightGCN: Simplifying and Powering Graph Convolution Network for Recommendation." in SIGIR 2020.