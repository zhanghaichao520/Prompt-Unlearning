# -*- coding: utf-8 -*-
import sys
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, "../../"))
if project_root not in sys.path:
    sys.path.append(project_root)

import time
import torch
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import scipy.sparse as sp
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm, trange

from config import COMMON_CONFIG, UNLEARNING_CONFIG
from data import ML100KDataset
from ngcf_models import NGCF
from utils import evaluate, print_metrics, set_seed, split_forget_retain, calculate_performance_change
from unlearning import evaluate_unlearning

# Set device
device = COMMON_CONFIG['device']
print(f"Using device: {device}")

# NGCF specific config (Matches run_p2f_ngcf_netflix.py)
NGCF_CONFIG = {
    'embedding_size': 64,
    'n_layers': 3,
    'reg_weight': 1e-5,
    'node_dropout': 0.1,
    'message_dropout': 0.1,
    'lr': 0.001,
    'epochs': 800, # Initial training
    'batch_size': 4096, # Full Batch Training (One step per epoch) as per netflix script
    'eval_freq': 5,
    'patience': 10,
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
        while neg in self.train_user_items[user]:
             neg = np.random.randint(0, self.n_items)
             
        return user, pos, neg

def train_ngcf(model, dataset, epochs, norm_adj_matrix, save_path=None):
    optimizer = optim.Adam(model.parameters(), lr=NGCF_CONFIG['lr'])
    # Netflix script uses num_workers=8
    train_loader = DataLoader(dataset, batch_size=NGCF_CONFIG['batch_size'], shuffle=True, num_workers=8)
    
    best_recall = 0.0
    patience_counter = 0
    k_list = COMMON_CONFIG['k_list']
    
    print(f"Starting NGCF RETRAINING for {epochs} epochs...")
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
        
        if epoch % NGCF_CONFIG['eval_freq'] == 0:
            # Note: evaluate will use dataset.train_user_items to mask
            eval_results = evaluate(model, dataset, norm_adj_matrix, k_list, NGCF_CONFIG['batch_size'])
            # Using Recall@10 for simple early stopping monitoring
            curr_recall = eval_results[10]['recall']
            
            # Print metrics
            if epoch % (NGCF_CONFIG['eval_freq'] * 2) == 0:
                 pass # Avoid too much output

            if curr_recall > best_recall:
                best_recall = curr_recall
                patience_counter = 0
                if save_path:
                    torch.save(model.state_dict(), save_path)
            else:
                patience_counter += 1
                if patience_counter >= NGCF_CONFIG['patience']:
                    print(f"Early stopping triggered at epoch {epoch}")
                    break
    
    if save_path and os.path.exists(save_path):
        print(f"Loading best model from {save_path}")
        model.load_state_dict(torch.load(save_path))
    return model

def calculate_zrf(unlearned_model, incompetent_teacher, forget_samples, norm_adj_matrix, batch_size, device):
    """
    根据论文 'Can Bad Teaching Induce Forgetting?' 计算 ZRF (Zero Retrain Forgetting) 分数 [cite: 15]。
    ZRF 通过比较“遗忘后模型”和“无能教师”在遗忘集上的预测分布来衡量遗忘程度 [cite: 144]。
    """
    print("正在计算 ZRF 分数...")
    unlearned_model.eval()
    incompetent_teacher.eval()

    # ZRF 针对遗忘集 D_f 计算 [cite: 150]
    class ForgetDataset(Dataset):
        def __init__(self, samples):
            self.users = torch.LongTensor([s[0] for s in samples])
            self.items = torch.LongTensor([s[1] for s in samples]) # 使用正样本
        def __len__(self):
            return len(self.users)
        def __getitem__(self, idx):
            return self.users[idx], self.items[idx]

    forget_dataset = ForgetDataset(forget_samples)
    forget_loader = DataLoader(forget_dataset, batch_size=batch_size, shuffle=False)
    
    js_divergences = []
    
    with torch.no_grad():
        pbar = tqdm(forget_loader, desc="计算ZRF", leave=False)
        for users, items in pbar:
            users, items = users.to(device), items.to(device)
            
            # 1. 获取模型对 (user, item) 的预测分数 (logits)
            unlearned_logits = unlearned_model.predict(users, items, norm_adj_matrix)
            incompetent_logits = incompetent_teacher.predict(users, items, norm_adj_matrix)
            
            # 2. 将分数通过 sigmoid 转换为概率 p，构建二元概率分布 [p, 1-p]
            p_unlearned = torch.sigmoid(unlearned_logits)
            p_incompetent = torch.sigmoid(incompetent_logits)
            
            dist_unlearned = torch.stack([p_unlearned, 1 - p_unlearned], dim=1)
            dist_incompetent = torch.stack([p_incompetent, 1 - p_incompetent], dim=1)

            # 3. 根据论文公式 (4) 和 (5) 计算 JS 散度 [cite: 146, 148]
            m = 0.5 * (dist_unlearned + dist_incompetent)
            
            # 使用 clamp 避免 log(0) 导致 NaN
            m_log = torch.log(m.clamp(min=1e-7))

            kl_p_m = F.kl_div(m_log, dist_unlearned, reduction='none').sum(dim=1)
            kl_q_m = F.kl_div(m_log, dist_incompetent, reduction='none').sum(dim=1)
            
            js_div_batch = 0.5 * (kl_p_m + kl_q_m)
            js_divergences.extend(js_div_batch.cpu().numpy())

    # 4. 计算 ZRF 分数: ZRF = 1 - mean(JS) [cite: 148]
    if not js_divergences:
        return 0.0
        
    mean_js_divergence = np.mean(js_divergences)
    zrf_score = 1.0 - mean_js_divergence
    
    return zrf_score

if __name__ == "__main__":
    set_seed(COMMON_CONFIG['seed'])
    
    # 1. Load Full Data
    absolute_data_path = os.path.join(project_root, "dataset", "netflix.inter")
    print(f"Loading dataset from {absolute_data_path}")
    # Use test_size=0.1 and min_item_count=5 to match RecBole standards and speed up evaluation
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
    # This matrix should be used for Retained Model Training AND Evaluation (to be strict)
    print("Computing Adj Matrix for Retain Set...")
    rows = [s[0] for s in retain_samples]
    cols = [s[1] for s in retain_samples]
    vals = np.ones(len(rows))
    
    retain_interaction_matrix = sp.coo_matrix(
        (vals, (rows, cols)), 
        shape=(full_dataset.n_users, full_dataset.n_items), 
        dtype=np.float32
    )
    
    temp_model = NGCF(full_dataset.n_users, full_dataset.n_items)
    retain_norm_adj_matrix = temp_model.get_norm_adj_mat(retain_interaction_matrix).to(device)
    del temp_model

    # 5. Initialize Random Model (Reference for ZRF metric calculation)
    # This is NOT part of the training, only used as a baseline to calculate ZRF score at the end
    print("\n=== Initializing Random Baseline Model (For ZRF Metric Calculation) ===")
    incompetent = NGCF(
        full_dataset.n_users, full_dataset.n_items, 
        embedding_size=NGCF_CONFIG['embedding_size'],
        n_layers=NGCF_CONFIG['n_layers'],
        reg_weight=NGCF_CONFIG['reg_weight']
    ).to(device)

    # 6. Initialize Fresh NGCF Model (Retrained Model)
    # Important: This model sees ONLY the retain set
    print("\n=== Retraining NGCF on Retain Set ===")
    model = NGCF(
        full_dataset.n_users, full_dataset.n_items, 
        embedding_size=NGCF_CONFIG['embedding_size'],
        n_layers=NGCF_CONFIG['n_layers'],
        reg_weight=NGCF_CONFIG['reg_weight'],
        node_dropout=NGCF_CONFIG['node_dropout'],
        message_dropout=NGCF_CONFIG['message_dropout']
    ).to(device)
    
    # 7. Train on Retain Set
    save_path = "best_ngcf_retrained_netflix.pth"
    model = train_ngcf(model, retain_dataset, NGCF_CONFIG['epochs'], retain_norm_adj_matrix, save_path=save_path)
    
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
    # Compare Retrained Model (Unlearned) vs Incompetent Teacher (Random)
    # Using Retain Norm Adj Matrix strictly
    zrf = calculate_zrf(model, incompetent, forget_samples, retain_norm_adj_matrix, 256, device)
    print(f"\nZRF Score: {zrf:.4f}")
    
    # Save results to file
    with open("ngcf_retrain_netflix_results.txt", "w") as f:
        f.write(f"ZRF: {zrf:.4f}\n")
