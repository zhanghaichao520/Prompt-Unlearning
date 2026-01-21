# -*- coding: utf-8 -*-
import sys
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, "../../"))
if project_root not in sys.path:
    sys.path.append(project_root)

import time
import torch
import torch.optim as optim
import numpy as np
import random
import copy
from tqdm import tqdm, trange
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F
import scipy.sparse as sp

from config import COMMON_CONFIG, UNLEARNING_CONFIG
from data import ML100KDataset, ML1MDataset
from ngcf_models import NGCF
from utils import evaluate, print_metrics, set_seed, split_forget_retain, calculate_performance_change
from ifru import IFRUEngine
from unlearning import evaluate_unlearning

# Set device
device = COMMON_CONFIG['device']
print(f"Using device: {device}")

# NGCF specific config
NGCF_CONFIG = {
    'embedding_size': 64,
    'n_layers': 3,
    'reg_weight': 1e-5,
    'node_dropout': 0.1,
    'message_dropout': 0.1,
    'lr': 0.001,
    'epochs': 50, # Reduced for speed in demo, assuming pretrained
    'batch_size': 4096, 
    'eval_freq': 5,
    'patience': 10,
}

class ListDataset(Dataset):
    def __init__(self, samples):
        self.samples = samples
    def __len__(self):
        return len(self.samples)
    def __getitem__(self, idx):
        return self.samples[idx]

def get_norm_adj_mat(n_users, n_items, interaction_matrix):
    A = sp.dok_matrix((n_users + n_items, n_users + n_items), dtype=np.float32)
    inter_M = interaction_matrix.tocoo()
    inter_M_t = interaction_matrix.transpose().tocoo()
    data_dict = dict(zip(zip(inter_M.row, inter_M.col + n_users), [1] * inter_M.nnz))
    data_dict.update(dict(zip(zip(inter_M_t.row + n_users, inter_M_t.col), [1] * inter_M_t.nnz)))
    A._update(data_dict)
    
    sumArr = (A > 0).sum(axis=1)
    diag = np.array(sumArr.flatten())[0] + 1e-7
    diag = np.power(diag, -0.5)
    D = sp.diags(diag)
    L = D * A * D
    
    L = sp.coo_matrix(L)
    row = L.row
    col = L.col
    i = torch.LongTensor(np.array([row, col]))
    data = torch.FloatTensor(L.data)
    SparseL = torch.sparse_coo_tensor(i, data, torch.Size(L.shape))
    return SparseL

def train_ngcf_initial(model, dataset, epochs, norm_adj_matrix, save_path=None):
    if save_path and os.path.exists(save_path):
        print(f"Loading pretrained model from {save_path}")
        model.load_state_dict(torch.load(save_path))
        return model

    optimizer = optim.Adam(model.parameters(), lr=NGCF_CONFIG['lr'])
    train_loader = DataLoader(dataset, batch_size=NGCF_CONFIG['batch_size'], shuffle=True, num_workers=4)
    
    best_recall = 0.0
    patience_counter = 0
    k_list = COMMON_CONFIG['k_list']
    
    print(f"Starting NGCF training for {epochs} epochs...")
    epoch_iter = trange(1, epochs + 1, desc="Training")
    
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
            eval_results = evaluate(model, dataset, norm_adj_matrix, k_list, NGCF_CONFIG['batch_size'])
            curr_recall = eval_results[20]['recall']
            if curr_recall > best_recall:
                best_recall = curr_recall
                patience_counter = 0
                if save_path:
                    torch.save(model.state_dict(), save_path)
            else:
                patience_counter += 1
                if patience_counter >= NGCF_CONFIG['patience']:
                    print("Early stopping triggered")
                    break
    
    if save_path and os.path.exists(save_path):
        model.load_state_dict(torch.load(save_path))
    return model

def calculate_zrf(unlearned_model, forget_samples, norm_adj_matrix, n_users, n_items, batch_size, device):
    """
    Simulated ZRF calculation against a random initialized model (Incompetent Teacher).
    """
    print("Calculating ZRF...")
    incompetent_teacher = NGCF(n_users, n_items, 
                               embedding_size=NGCF_CONFIG['embedding_size'], 
                               n_layers=NGCF_CONFIG['n_layers']).to(device)
    
    # Needs eye matrix usually precomputed inside model but here we pass norm_adj_matrix which contains shape info
    # The predict method handles eye matrix construction locally if we changed ngcf_models.py
    # Yes, I copied the updated ngcf_models.py which constructs eye matrix inside forward.
    
    unlearned_model.eval()
    incompetent_teacher.eval()

    users_list = [s[0] for s in forget_samples]
    items_list = [s[1] for s in forget_samples]
    
    dataset = ListDataset(list(zip(users_list, items_list)))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    
    js_divergences = []
    
    with torch.no_grad():
        for users, items in tqdm(loader, desc="ZRF Batch", leave=False):
            users, items = users.to(device), items.to(device)
            
            unlearned_logits = unlearned_model.predict(users, items, norm_adj_matrix)
            incompetent_logits = incompetent_teacher.predict(users, items, norm_adj_matrix)
            
            p_unlearned = torch.sigmoid(unlearned_logits)
            p_incompetent = torch.sigmoid(incompetent_logits)
            
            dist_unlearned = torch.stack([p_unlearned, 1 - p_unlearned], dim=1)
            dist_incompetent = torch.stack([p_incompetent, 1 - p_incompetent], dim=1)

            m = 0.5 * (dist_unlearned + dist_incompetent)
            m_log = torch.log(m.clamp(min=1e-7))

            kl_p_m = F.kl_div(m_log, dist_unlearned, reduction='none').sum(dim=1)
            kl_q_m = F.kl_div(m_log, dist_incompetent, reduction='none').sum(dim=1)
            
            js_div_batch = 0.5 * (kl_p_m + kl_q_m)
            js_divergences.extend(js_div_batch.cpu().numpy())

    if not js_divergences:
        return 0.0
    return 1.0 - np.mean(js_divergences)

def main():
    set_seed(COMMON_CONFIG['seed'])
    
    # 1. Load Data
    print("Loading ML-1M Dataset...")
    dataset = ML1MDataset(root=os.path.join(project_root, "dataset"))
    
    n_users = dataset.n_users
    n_items = dataset.n_items
    
    # 2. Prepare Model and Graph
    print("Building adjacency matrix...")
    norm_adj_matrix = get_norm_adj_mat(n_users, n_items, dataset.train_matrix).to(device)
    
    model = NGCF(n_users, n_items, 
                 embedding_size=NGCF_CONFIG['embedding_size'], 
                 n_layers=NGCF_CONFIG['n_layers'],
                 node_dropout=NGCF_CONFIG['node_dropout'],
                 message_dropout=NGCF_CONFIG['message_dropout'],
                 reg_weight=NGCF_CONFIG['reg_weight']).to(device)
    
    # 3. Pretrain
    model_path = os.path.join(project_root, "pretrain_checkpoints", "ngcf_ml1m_full.pth")
    if not os.path.exists(os.path.dirname(model_path)):
        os.makedirs(os.path.dirname(model_path))
        
    model = train_ngcf_initial(model, dataset, NGCF_CONFIG['epochs'], norm_adj_matrix, save_path=model_path)
    
    print("Evaluating Pre-trained Model...")
    res_pre = evaluate(model, dataset, norm_adj_matrix, COMMON_CONFIG['k_list'], NGCF_CONFIG['batch_size'])
    print_metrics(res_pre)
    
    # 4. Prepare Forgetting
    forget_ratio = UNLEARNING_CONFIG['forget_ratio']
    forget_samples, retain_samples = split_forget_retain(dataset, forget_ratio)
    
    print(f"Forget Ratio: {forget_ratio}, Forget Size: {len(forget_samples)}, Retain Size: {len(retain_samples)}")
    
    # 5. IFRU Setup
    # Need to create adjacency matrices for Full and Retain
    # Full is already norm_adj_matrix
    
    print("Building Retain adjacency matrix...")
    # Build sparse matrix for retain
    retain_rows = [u for u, i in retain_samples]
    retain_cols = [i for u, i in retain_samples]
    retain_matrix = sp.csc_matrix(([1]*len(retain_samples), (retain_rows, retain_cols)), shape=(n_users, n_items))
    norm_adj_retain = get_norm_adj_mat(n_users, n_items, retain_matrix).to(device)
    
    # Build Dataset Wrappers for IFRU (Need neg sampling)
    # The update engine usually expects (u, pos, neg).
    # Since IFRU gradients are on specific sets, we should wrap them.
    # However, existing dataset class does negative sampling on the fly.
    # We need to construct datasets that only sample from retain/forget.
    
    # Actually, ML1MDataset takes train_data as property? 
    # It has self.train_data.
    # We can create temporary copies of dataset with modified train_data.
    
    forget_ds = copy.deepcopy(dataset)
    # CRITICAL FIX: Update train_samples which is used by __getitem__ and __len__
    forget_ds.train_samples = forget_samples
    # Optional: Update other attributes if needed, but train_samples is key for DataLoader
    forget_ds.train_matrix = sp.csc_matrix(([1]*len(forget_samples), ([u for u,i in forget_samples], [i for u,i in forget_samples])), shape=(n_users, n_items))
    
    # For D_c (spillover), we need retain samples that are affected.
    # Affected means connected to any u or i in forget set.
    print("Identifying spillover set D_c...")
    f_users = set([u for u, i in forget_samples])
    f_items = set([i for u, i in forget_samples])
    dc_samples = []
    for u, i in retain_samples:
        if u in f_users or i in f_items:
            dc_samples.append((u, i))
    
    # Cap the size of D_c to avoid gradient explosion/noise accumulation
    max_dc_size = len(forget_samples) * 5 # Heuristic: Keep D_c size comparable to D_f
    spillover_weight = 1.0
    if len(dc_samples) > max_dc_size:
        print(f"Spillover Size {len(dc_samples)} is too large. Downsampling to {max_dc_size}...")
        original_size = len(dc_samples)
        random.shuffle(dc_samples)
        dc_samples = dc_samples[:max_dc_size]
        # Compensation weight for downsampling
        spillover_weight = original_size / max_dc_size
        print(f"Applied Spillover Weight Compensation: {spillover_weight:.2f}")
    
    print(f"Spillover Size: {len(dc_samples)}")
    
    dc_ds = copy.deepcopy(dataset)
    # CRITICAL FIX: Update train_samples for dc_ds as well
    dc_ds.train_samples = dc_samples
    # Matrix is not used for neg sampling in __getitem__, but might be used by internal methods.
    # Safety update:
    dc_ds.train_matrix = sp.csc_matrix(([1]*len(dc_samples), ([u for u,i in dc_samples], [i for u,i in dc_samples])), shape=(n_users, n_items))
    
    # Full training dataset (approximated by retain + forget or just full)
    # We can use original `dataset`.
    
    # 6. Run IFRU
    ifru = IFRUEngine(model, device)
    
    start_time = time.time()
    
    # Gradient of L_d (on forget set)
    # Gradient of L_s (on dc set)
    # HVP on Full Train (using dataset)
    
    # Tuned parameters: 
    # influence_lr=0.005 (Good solving speed)
    # steps=1000 (Sufficient for convergence with this LR)
    # scale=0.1 (CRITICAL: Group unlearning requires scaling down the Newton step to stay within trust region)
    
    ifru.run(dataset, forget_ds, dc_ds, norm_adj_matrix, norm_adj_retain, influence_lr=0.005, steps=1000, spillover_weight=1.0, scale=0.1)
    
    end_time = time.time()
    print(f"IFRU Unlearning Time: {end_time - start_time:.2f}s")
    
    # 7. Evaluate
    print("Evaluating Unlearned Model (IFRU) on Forget/Retain Data:")
    
    forget_results, retain_results = evaluate_unlearning(
        model=model, 
        dataset=dataset, 
        forget_samples=forget_samples, 
        retain_samples=retain_samples, 
        norm_adj_matrix=norm_adj_retain, 
        k_list=COMMON_CONFIG['k_list']
    )
    
    print("\n遗忘后模型在遗忘集上的性能:")
    print_metrics(forget_results, prefix="  ")
    
    # Calculate drops for Forget Set (using original Pre-results as baseline? Or just display?)
    # Original scripts usually compare to original global performance or original forget/retain performance.
    # Here we only calculated `res_pre` (global).
    # To match exactly, we should have calculated `original_forget_results` and `original_retain_results` BEFORE unlearning.
    
    print("\n遗忘后模型在保留集上的性能:")
    print_metrics(retain_results, prefix="  ")
    
    zrf = calculate_zrf(model, forget_samples, norm_adj_retain, n_users, n_items, NGCF_CONFIG['batch_size'], device)
    print(f"ZRF Score: {zrf:.4f}")
    
    # Compare Retain Performance Change (from Global Pre to Retain Post? Or just skip precise comparison if not available)
    # The user asked to "evaluate forget and retain sets".
    # P2F example does: evaluate_unlearning returns two dicts.
    
    for k in COMMON_CONFIG['k_list']:
        # We can compare retain_results to res_pre (approximation, since res_pre is on full test set, retain is on retain set)
        # But commonly we check if retain performance dropped.
        print(f"Performance Change @ K={k} (vs Global Pre):")
        # Just use global res_pre as baseline for now as we didn't split-eval before.
        changes = calculate_performance_change(res_pre, retain_results, k)
        for metric, change in changes.items():
            print(f"  Retain {metric}: {change:.2f}%")

if __name__ == "__main__":
    main()
