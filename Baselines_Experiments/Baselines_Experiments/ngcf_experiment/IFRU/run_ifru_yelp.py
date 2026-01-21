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
from data import YelpDataset
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
    'epochs': 50, 
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
    print("Loading Yelp Dataset...")
    dataset = YelpDataset(root=os.path.join(project_root, "dataset"))
    
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
    model_path = os.path.join(project_root, "pretrain_checkpoints", "ngcf_yelp_full.pth")
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
    
    print("Building Retain adjacency matrix...")
    retain_rows = [u for u, i in retain_samples]
    retain_cols = [i for u, i in retain_samples]
    retain_matrix = sp.csc_matrix(([1]*len(retain_samples), (retain_rows, retain_cols)), shape=(n_users, n_items))
    norm_adj_retain = get_norm_adj_mat(n_users, n_items, retain_matrix).to(device)
    
    forget_ds = copy.deepcopy(dataset)
    forget_ds.train_samples = forget_samples # Critical fix for DataLoader
    forget_ds.train_data = forget_samples
    forget_ds.train_matrix = sp.csc_matrix(([1]*len(forget_samples), ([u for u,i in forget_samples], [i for u,i in forget_samples])), shape=(n_users, n_items))
    
    print("Identifying spillover set D_c...")
    f_users = set([u for u, i in forget_samples])
    f_items = set([i for u, i in forget_samples])
    dc_samples = []
    for u, i in retain_samples:
        if u in f_users or i in f_items:
            dc_samples.append((u, i))
    
    # Cap the size of D_c to avoid gradient explosion/noise accumulation
    max_dc_size = len(forget_samples) * 5 
    if len(dc_samples) > max_dc_size:
        print(f"Spillover Size {len(dc_samples)} is too large. Downsampling to {max_dc_size}...")
        random.shuffle(dc_samples)
        dc_samples = dc_samples[:max_dc_size]
            
    print(f"Spillover Size: {len(dc_samples)}")
    
    dc_ds = copy.deepcopy(dataset)
    dc_ds.train_samples = dc_samples # Critical fix for DataLoader
    dc_ds.train_data = dc_samples
    dc_ds.train_matrix = sp.csc_matrix(([1]*len(dc_samples), ([u for u,i in dc_samples], [i for u,i in dc_samples])), shape=(n_users, n_items))
    
    # 6. Run IFRU
    ifru = IFRUEngine(model, device)
    
    start_time = time.time()
    
    # Tuned parameters: influence_lr=0.005, steps=1000, scale=1.0
    # Increased scale for Yelp (likely similar to Netflix in stiffness)
    ifru.run(dataset, forget_ds, dc_ds, norm_adj_matrix, norm_adj_retain, influence_lr=0.005, steps=1000, spillover_weight=1.0, scale=1.0)
    
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
    
    print("\n遗忘后模型在保留集上的性能:")
    print_metrics(retain_results, prefix="  ")
    
    zrf = calculate_zrf(model, forget_samples, norm_adj_retain, n_users, n_items, NGCF_CONFIG['batch_size'], device)
    print(f"ZRF Score: {zrf:.4f}")
    
    for k in COMMON_CONFIG['k_list']:
        print(f"Performance Change @ K={k} (vs Global Pre):")
        changes = calculate_performance_change(res_pre, retain_results, k)
        for metric, change in changes.items():
            print(f"  Retain {metric}: {change:.2f}%")

if __name__ == "__main__":
    main()
