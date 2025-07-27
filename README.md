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
python P2F/train_lightgcn.py
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

---

## 遗忘学习（Unlearning）功能说明

### 1. 预训练LightGCN模型

在进行unlearning之前，需先训练好基础LightGCN模型。  
请运行如下命令：

```bash
python P2F/train_lightgcn.py
```

训练完成后，会在`LIGHTGCN_CONFIG['save_path']`指定的路径（如`./saved/lightgcn.pth`）保存预训练模型权重。

#### 主要参数设置（见`config.py`）：

- `data_path`：数据集路径（如`dataset/ml-100k.inter`）
- `embedding_size`：嵌入维度
- `n_layers`：GCN层数
- `reg_weight`：L2正则化权重
- `batch_size`：训练批次大小
- `lr`：学习率
- `epochs`：训练轮数
- `save_path`：模型保存路径

### 2. 配置Unlearning参数

在`config.py`中，设置unlearning相关参数（`UNLEARNING_CONFIG`），如：

```python
UNLEARNING_CONFIG = {
    'embedding_size': 64,
    'n_layers': 3,
    'reg_weight': 1e-4,
    'batch_size': 2048,
    'lr': 0.001,
    'epochs': 30,
    'forget_ratio': 0.1,         # 遗忘集比例
    'remain_ratio': 1.0,         # 保留集采样比例
    'prompt_type': 'attention',  # 提示类型
    'p_num': 50,                 # prompt数量
    'KL_temperature': 1.0,
    'loss_type': 'WRD',          # 损失类型（如'KL', 'WRD', 'DAD'等）
    'alpha': 0.5,
    'lamda': 10.0,
    'mu': 5.0,
    'K': 5,
    'patience': 5,
    'validation_interval': 1,
    'prompt_save_path': './saved/prompt.pth'
}
```

### 3. 运行Unlearning流程

确保`train_lightgcn.py`已训练并保存了基础模型，然后运行：

```bash
python P2F/unlearning.py
```

该脚本会自动：
- 加载数据和预训练模型
- 按`forget_ratio`划分遗忘集和保留集
- 只训练提示（prompt）参数，基础模型参数保持不变
- 训练过程中自动评估遗忘集和保留集性能
- 保存训练好的prompt参数到`prompt_save_path`

### 4. 推理与评估

推理时，只需加载基础模型和训练好的prompt参数：

```python
from unlearning import load_prompt_for_inference
prompted_model = load_prompt_for_inference(
    base_model, prompt_path, dataset, n_layers, reg_weight, prompt_type, embedding_size, p_num
)
```

可用`evaluate_unlearning`函数分别评估遗忘集和保留集的推荐性能。

---

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

---

## 常见问题

- **预训练模型未找到？**  
  请先运行`train_lightgcn.py`，确保`LIGHTGCN_CONFIG['save_path']`路径下有模型权重文件。

- **如何调整遗忘比例？**  
  修改`UNLEARNING_CONFIG['forget_ratio']`，如`0.1`表示10%样本为遗忘集。

- **如何切换损失函数？**  
  修改`UNLEARNING_CONFIG['loss_type']`，支持`KL`、`WRD`、`DAD`等。

---

## 参考文献

Xiangnan He et al. "LightGCN: Simplifying and Powering Graph Convolution Network for Recommendation." in SIGIR 2020.
