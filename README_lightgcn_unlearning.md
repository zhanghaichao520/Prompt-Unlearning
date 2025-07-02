# LightGCN推荐系统遗忘学习框架

## 项目概述

本项目实现了一个基于LightGCN的推荐系统遗忘学习框架，通过蒸馏学习方法和图提示（Graph Prompt）技术，使模型能够有选择地"遗忘"特定的交互数据，同时保持对其他数据的推荐性能。

### 遗忘学习的目标

- 将数据集分为遗忘集（forget set）和剩余集（remain set），比例为1:9
- 使模型在不影响剩余集性能的情况下，降低对遗忘集数据的推荐准确率
- 通过添加可训练的提示（prompt）到原始模型中实现遗忘功能

## 技术方法

### 图提示学习

本项目基于图提示学习（Graph Prompt Learning）技术，为LightGCN模型添加可训练的提示向量，主要实现了两种提示类型：

1. **简单提示（SimplePrompt）**：为用户和物品嵌入添加全局可学习的向量
2. **注意力提示（GPFplusAtt）**：使用注意力机制选择性地为不同用户和物品添加不同的提示向量

### 蒸馏学习遗忘方法

采用"好老师-坏老师"蒸馏学习框架：

1. **好老师**：初始化为预训练的LightGCN模型，负责指导学生模型学习剩余集数据
2. **坏老师**：初始化为随机参数的LightGCN模型，负责指导学生模型学习遗忘集数据
3. **学生模型**：带有提示的LightGCN模型，通过蒸馏学习同时从好老师和坏老师学习

通过这种方式，学生模型就能够保持对剩余集的推荐性能，同时"忘记"遗忘集的数据。

## 文件结构

- `lightgcn_prompt.py`: 实现图提示模块和带提示的LightGCN模型
- `lightgcn_unlearning.py`: 实现遗忘学习的主要逻辑，包括数据集分割、蒸馏学习和评估
- `evaluate_unlearning.py`: 评估不同提示类型的遗忘效果并可视化结果

## 使用方法

### 1. 训练基础LightGCN模型

```bash
python train_lightgcn.py
```

### 2. 运行遗忘学习

```bash
python lightgcn_unlearning.py
```

### 3. 评估不同提示类型的遗忘效果

```bash
python evaluate_unlearning.py
```

## 核心代码解析

### 提示模块的实现

```python
def get_ego_embeddings(self):
    # 获取原始嵌入并添加提示
    user_embeddings = self.base_model.user_embedding.weight
    item_embeddings = self.base_model.item_embedding.weight
    
    # 应用提示
    user_embeddings = self.user_prompt.add(user_embeddings)
    item_embeddings = self.item_prompt.add(item_embeddings)
    
    ego_embeddings = torch.cat([user_embeddings, item_embeddings], dim=0)
    return ego_embeddings
```