# -*- coding: utf-8 -*-
import sys
import os
import time
import torch
import torch.optim as optim
import numpy as np
import scipy.sparse as sp
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm, trange

# Add project root to path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, "../../"))
if project_root not in sys.path:
    sys.path.append(project_root)

from config import COMMON_CONFIG, UNLEARNING_CONFIG, LIGHTGCN_CONFIG
from data import ML100KDataset
from models import LightGCN
from utils import evaluate, print_metrics, set_seed, split_forget_retain
from unlearning import evaluate_unlearning

# Set device
device = COMMON_CONFIG['device']
print(f"Using device: {device}")

# LightGCN Yelp Config
DATASET_NAME = "yelp"
LIGHTGCN_RETRAIN_CONFIG = {
    'embedding_size': LIGHTGCN_CONFIG['embedding_size'],
    'n_layers': LIGHTGCN_CONFIG['n_layers'],         
    'reg_weight': LIGHTGCN_CONFIG['reg_weight'],
    'lr': LIGHTGCN_CONFIG['lr'],
    'epochs': 1000, 
    'batch_size': 2048,
    'eval_freq': 10,
    'patience': 20
}

class RetainDatasetWrapper(Dataset):
    def __init__(self, retain_samples, num_users, num_items, original_test_user_items, original_get_test_samples_func):
        self.retain_samples = retain_samples
        self.n_users = num_users
        self.n_items = num_items
        self.test_user_items = original_test_user_items
        self._get_test_samples_func = original_get_test_samples_func
        
        # Rebuild train_user_items for masking in evaluation
        # This ensures we mask only the items present in the Retain Set (Train Set for Retraining)
        self.train_user_items = {}
        for sample in retain_samples:
            u, i = sample[0], sample[1]
            if u not in self.train_user_items:
                self.train_user_items[u] = set()
            self.train_user_items[u].add(i)
            
    def get_test_samples(self):
        return self._get_test_samples_func()
        
    def __len__(self): 
        return len(self.retain_samples)
        
    def __getitem__(self, idx):
        sample = self.retain_samples[idx]
        user, pos = sample[0], sample[1]
        
        # Negative Sampling
        neg = np.random.randint(0, self.n_items)
        while neg in self.train_user_items.get(user, set()):
             neg = np.random.randint(0, self.n_items)
             
        return user, pos, neg

def train_lightgcn(model, dataset, epochs, norm_adj_matrix, save_path=None):
    optimizer = optim.Adam(model.parameters(), lr=LIGHTGCN_RETRAIN_CONFIG['lr'])
    train_loader = DataLoader(dataset, batch_size=LIGHTGCN_RETRAIN_CONFIG['batch_size'], shuffle=True, num_workers=4)
    
    best_recall = 0.0
    patience_counter = 0
    k_list = COMMON_CONFIG['k_list']
    
    print(f"Starting LightGCN RETRAINING for {epochs} epochs...")
    epoch_iter = trange(1, epochs + 1, desc="Retraining")
    
    for epoch in epoch_iter:
        model.train()
        total_loss = 0
        
        for batch_idx, (users, pos_items, neg_items) in enumerate(train_loader):
            users = users.to(device)
            pos_items = pos_items.to(device)
            neg_items = neg_items.to(device)
            
            optimizer.zero_grad()
            loss = model.calculate_loss(users, pos_items, neg_items, norm_adj_matrix)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
        avg_loss = total_loss / len(train_loader)
        epoch_iter.set_postfix(loss=f"{avg_loss:.4f}")
        
        if epoch % LIGHTGCN_RETRAIN_CONFIG['eval_freq'] == 0:
            eval_results = evaluate(model, dataset, norm_adj_matrix, k_list, LIGHTGCN_RETRAIN_CONFIG['batch_size'])
            curr_recall = eval_results[20]['recall']
            
            if curr_recall > best_recall:
                best_recall = curr_recall
                patience_counter = 0
                if save_path:
                    torch.save(model.state_dict(), save_path)
            else:
                patience_counter += 1
                if patience_counter >= LIGHTGCN_RETRAIN_CONFIG['patience']:
                    print(f"Early stopping at epoch {epoch}")
                    break
    
    if save_path and os.path.exists(save_path):
        print(f"Loading best model from {save_path}")
        model.load_state_dict(torch.load(save_path))
    return model

def calculate_zrf(unlearned_model, incompetent_teacher, forget_samples, norm_adj_matrix, batch_size, device):
    """
    根据论文 'Can Bad Teaching Induce Forgetting?' 计算 ZRF (Zero Retrain Forgetting) 分数。
    """
    print("正在计算 ZRF 分数...")
    unlearned_model.eval()
    incompetent_teacher.eval()

    # ZRF 针对遗忘集 D_f 计算
    class ForgetDataset(Dataset):
        def __init__(self, samples):
            self.samples = samples
        def __len__(self):
            return len(self.samples)
        def __getitem__(self, idx):
            return self.samples[idx]

    forget_dataset = ForgetDataset(forget_samples)
    forget_loader = DataLoader(forget_dataset, batch_size=batch_size, shuffle=False)
    
    js_divergences = []
    
    with torch.no_grad():
        pbar = tqdm(forget_loader, desc="计算ZRF", leave=False)
        for users, items in pbar:
            users = users.to(device)
            items = items.to(device)
            
            # 预测得分
            scores_unlearned = unlearned_model.predict(users, items, norm_adj_matrix)
            prob_unlearned = torch.sigmoid(scores_unlearned)
            
            # 无能教师得分 (Random Model)
            scores_teacher = incompetent_teacher.predict(users, items, norm_adj_matrix)
            prob_teacher = torch.sigmoid(scores_teacher)
            
            p = prob_unlearned.cpu().numpy()
            q = prob_teacher.cpu().numpy()
            
            # JS Divergence
            m = 0.5 * (p + q)
            epsilon = 1e-10
            kl_pm = p * np.log((p + epsilon) / (m + epsilon)) + (1 - p) * np.log((1 - p + epsilon) / (1 - m + epsilon))
            kl_qm = q * np.log((q + epsilon) / (m + epsilon)) + (1 - q) * np.log((1 - q + epsilon) / (1 - m + epsilon))
            
            js = 0.5 * kl_pm + 0.5 * kl_qm
            js_divergences.extend(js)

    if not js_divergences:
        return 0.0
        
    mean_js_divergence = np.mean(js_divergences)
    zrf_score = 1.0 - mean_js_divergence
    
    return zrf_score

if __name__ == "__main__":
    set_seed(COMMON_CONFIG['seed'])
    
    # 1. Load Full Data
    absolute_data_path = os.path.join(project_root, "dataset", f"{DATASET_NAME}.inter")
    print(f"Loading dataset from {absolute_data_path}")
    full_dataset = ML100KDataset(absolute_data_path, test_size=0.1, min_item_count=5)
    
    # 2. Split Data (Forget/Retain)
    forget_ratio = UNLEARNING_CONFIG.get('forget_ratio', 0.01)
    print(f"Splitting data with forget_ratio={forget_ratio}...")
    forget_samples, retain_samples = split_forget_retain(full_dataset, forget_ratio)
    
    # 3. Create Retain Dataset Wrapper
    print("Creating Retain Set Wrapper...")
    retain_dataset = RetainDatasetWrapper(
        retain_samples=retain_samples,
        num_users=full_dataset.n_users,
        num_items=full_dataset.n_items,
        original_test_user_items=full_dataset.test_user_items,
        original_get_test_samples_func=full_dataset.get_test_samples
    )
    
    # 4. Compute Adj Matrix for Retain Set ONLY
    print("Computing Adj Matrix for Retain Set...")
    rows = [s[0] for s in retain_samples]
    cols = [s[1] for s in retain_samples]
    vals = np.ones(len(rows))
    
    retain_interaction_matrix = sp.coo_matrix(
        (vals, (rows, cols)), 
        shape=(full_dataset.n_users, full_dataset.n_items), 
        dtype=np.float32
    )
    
    # Use helper model to build adj matrix
    temp_model = LightGCN(full_dataset.n_users, full_dataset.n_items, 
                          embedding_size=LIGHTGCN_RETRAIN_CONFIG['embedding_size'], 
                          n_layers=LIGHTGCN_RETRAIN_CONFIG['n_layers'])
    retain_norm_adj_matrix = temp_model.get_norm_adj_mat(retain_interaction_matrix).to(device)
    del temp_model

    # 5. Initialize Random Model (Reference for ZRF metric calculation)
    print("\n=== Initializing Random Baseline Model (For ZRF Metric Calculation) ===")
    incompetent = LightGCN(
        full_dataset.n_users, full_dataset.n_items, 
        embedding_size=LIGHTGCN_RETRAIN_CONFIG['embedding_size'],
        n_layers=LIGHTGCN_RETRAIN_CONFIG['n_layers'],
        reg_weight=LIGHTGCN_RETRAIN_CONFIG['reg_weight']
    ).to(device)

    # 6. Initialize Fresh LightGCN Model (Retrained Model)
    print("\n=== Retraining LightGCN on Retain Set ===")
    model = LightGCN(
        full_dataset.n_users, full_dataset.n_items, 
        embedding_size=LIGHTGCN_RETRAIN_CONFIG['embedding_size'],
        n_layers=LIGHTGCN_RETRAIN_CONFIG['n_layers'],
        reg_weight=LIGHTGCN_RETRAIN_CONFIG['reg_weight']
    ).to(device)
    
    # 7. Train on Retain Set
    save_path = f"best_lightgcn_retrained_{DATASET_NAME.replace('-', '')}.pth"
    model = train_lightgcn(model, retain_dataset, LIGHTGCN_RETRAIN_CONFIG['epochs'], retain_norm_adj_matrix, save_path=save_path)
    
    # 8. Final Evaluation
    print("\n=== Final Evaluation of Retrained Model ===")
    
    forget_results, retain_results = evaluate_unlearning(
        model=model, 
        dataset=retain_dataset, 
        forget_samples=forget_samples, 
        retain_samples=retain_samples, 
        norm_adj_matrix=retain_norm_adj_matrix, 
        k_list=COMMON_CONFIG['k_list']
    )
    
    print("\nRetrained模型在遗忘集上的性能:")
    print_metrics(forget_results, prefix="  ")
    
    print("\nRetrained模型在保留集上的性能:")
    print_metrics(retain_results, prefix="  ")
    
    # 9. Calculate ZRF
    zrf = calculate_zrf(model, incompetent, forget_samples, retain_norm_adj_matrix, 256, device)
    print(f"\nZRF Score: {zrf:.4f}")
    
    with open(f"lightgcn_retrain_{DATASET_NAME.replace('-', '')}_results.txt", "w") as f:
        f.write(f"ZRF: {zrf:.4f}\n")
