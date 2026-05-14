<div align="center">
  <h1>DPU: Distilled Prompt Tuning for Efficient GNN-based Recommendation Unlearning</h1>
</div>

---

## Usage

### Environment Requirements

* Python 3.6+
* PyTorch 1.0+
* NumPy
* Pandas
* SciPy
* scikit-learn

### Running Method

```bash
python P2F/train_lightgcn.py
```

### Parameter Settings

The main parameters are set in the `main` function:

```python
# Parameter settings
data_path = "dataset/ml-100k.inter"  # Dataset path
embedding_size = 64                 # Embedding dimension
n_layers = 3                        # Number of graph convolution layers
reg_weight = 1e-4                   # Regularization weight
batch_size = 2048                   # Batch size
lr = 0.001                          # Learning rate
epochs = 100                        # Number of training epochs
eval_freq = 5                       # Evaluation frequency
k_list = [10, 20]                   # k values for evaluation
```

---

## Unlearning Function Description

### 1. Pretrain the LightGCN Model

Before performing unlearning, the base LightGCN model needs to be trained first.
Please run the following command:

```bash
python P2F/train_lightgcn.py
```

After training is complete, the pretrained model weights will be saved to the path specified by `LIGHTGCN_CONFIG['save_path']`, such as `./saved/lightgcn.pth`.

#### Main Parameter Settings See `config.py`:

* `data_path`: Dataset path, such as `dataset/ml-100k.inter`
* `embedding_size`: Embedding dimension
* `n_layers`: Number of GCN layers
* `reg_weight`: L2 regularization weight
* `batch_size`: Training batch size
* `lr`: Learning rate
* `epochs`: Number of training epochs
* `save_path`: Model save path

### 2. Configure Unlearning Parameters

In `config.py`, set the parameters related to unlearning `UNLEARNING_CONFIG`, such as:

```python
UNLEARNING_CONFIG = {
    'embedding_size': 64,
    'n_layers': 3,
    'reg_weight': 1e-4,
    'batch_size': 2048,
    'lr': 0.001,
    'epochs': 30,
    'forget_ratio': 0.1,         # Ratio of the forget set
    'remain_ratio': 1.0,         # Sampling ratio of the retain set
    'prompt_type': 'attention',  # Prompt type
    'p_num': 50,                 # Number of prompts
    'KL_temperature': 1.0,
    'loss_type': 'WRD',          # Loss type, such as 'KL', 'WRD', 'DAD', etc.
    'alpha': 0.5,
    'lamda': 10.0,
    'mu': 5.0,
    'K': 5,
    'patience': 5,
    'validation_interval': 1,
    'prompt_save_path': './saved/prompt.pth'
}
```

### 3. Run the Unlearning Process

Make sure that `train_lightgcn.py` has trained and saved the base model, then run:

```bash
python P2F/unlearning.py
```

This script will automatically:

* Load the data and pretrained model
* Split the forget set and retain set according to `forget_ratio`
* Train only the prompt parameters, while keeping the base model parameters unchanged
* Automatically evaluate the performance on the forget set and retain set during training
* Save the trained prompt parameters to `prompt_save_path`

### 4. Inference and Evaluation

During inference, simply load the base model and the trained prompt parameters:

```python
from unlearning import load_prompt_for_inference
prompted_model = load_prompt_for_inference(
    base_model, prompt_path, dataset, n_layers, reg_weight, prompt_type, embedding_size, p_num
)
```

The `evaluate_unlearning` function can be used to evaluate the recommendation performance on the forget set and retain set separately.

---

## Implementation Details

1. **Data Processing**:

   * Convert the raw data into a user-item interaction matrix
   * Split the training set and test set by user
   * Generate negative samples for each positive sample

2. **Model Training**:

   * Use the Adam optimizer
   * Use the BPR loss function
   * Periodically evaluate model performance

3. **Model Evaluation**:

   * For each test user, predict scores for all items
   * Exclude items already interacted with in the training set
   * Calculate various metrics for TopK recommendations

---

## FAQ

* **Pretrained model not found?**
  Please run `train_lightgcn.py` first and make sure that a model weight file exists under the path specified by `LIGHTGCN_CONFIG['save_path']`.

* **How to adjust the forget ratio?**
  Modify `UNLEARNING_CONFIG['forget_ratio']`, for example, `0.1` means that 10% of samples are used as the forget set.

* **How to switch the loss function?**
  Modify `UNLEARNING_CONFIG['loss_type']`, which supports `KL`, `WRD`, `DAD`, etc.

---

## References

Xiangnan He et al. "LightGCN: Simplifying and Powering Graph Convolution Network for Recommendation." in SIGIR 2020.
