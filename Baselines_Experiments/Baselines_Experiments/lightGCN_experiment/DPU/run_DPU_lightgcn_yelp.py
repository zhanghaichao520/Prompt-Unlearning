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

from config import COMMON_CONFIG, UNLEARNING_CONFIG, LIGHTGCN_CONFIG
from data import ML100KDataset
from models import LightGCN
from utils import evaluate, print_metrics, set_seed, split_forget_retain, unlearner_loss_2, UnLearningDataset, calculate_performance_change
from unlearning import blindspot_unlearner, evaluate_unlearning

# Set device
device = COMMON_CONFIG['device']
print(f"Using device: {device}")

# LightGCN specific config
MODEL_CONFIG = {
    'embedding_size': LIGHTGCN_CONFIG['embedding_size'],
    'n_layers': LIGHTGCN_CONFIG['n_layers'],
    'reg_weight': LIGHTGCN_CONFIG['reg_weight'],
    'lr': LIGHTGCN_CONFIG['lr'],
    'epochs': 1000,
    'batch_size': 2048,
    'eval_freq': 10,
    'patience': 10,
}

def train_lightgcn(model, dataset, epochs, norm_adj_matrix, save_path=None):
    optimizer = optim.Adam(model.parameters(), lr=MODEL_CONFIG['lr'])
    train_loader = DataLoader(dataset, batch_size=MODEL_CONFIG['batch_size'], shuffle=True, num_workers=8)
    
    best_recall = 0.0
    patience_counter = 0
    k_list = COMMON_CONFIG['k_list']
    
    print(f"Starting LightGCN training for {epochs} epochs...")
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
        
        if epoch % MODEL_CONFIG['eval_freq'] == 0:
            eval_results = evaluate(model, dataset, norm_adj_matrix, k_list, MODEL_CONFIG['batch_size'])
            curr_recall = eval_results[10]['recall'] # Using k=10 as in NGCF script
            if curr_recall > best_recall:
                best_recall = curr_recall
                patience_counter = 0
                if save_path:
                    # Check if directory exists
                    save_dir = os.path.dirname(save_path)
                    if not os.path.exists(save_dir) and save_dir != '':
                        os.makedirs(save_dir)
                    torch.save(model.state_dict(), save_path)
            else:
                patience_counter += 1
                if patience_counter >= MODEL_CONFIG['patience']:
                    print("Early stopping triggered")
                    break
    
    if save_path and os.path.exists(save_path):
        model.load_state_dict(torch.load(save_path))
    return model

# ZRF function is identical
def calculate_zrf(unlearned_model, incompetent_teacher, forget_samples, norm_adj_matrix, batch_size, device):
    """
    根据论文 'Can Bad Teaching Induce Forgetting?' 计算 ZRF (Zero Retrain Forgetting) 分数 [cite: 15]。
    """
    print("正在计算 ZRF 分数...")
    unlearned_model.eval()
    incompetent_teacher.eval()

    class ForgetDataset(Dataset):
        def __init__(self, samples):
            self.users = torch.LongTensor([s[0] for s in samples])
            self.items = torch.LongTensor([s[1] for s in samples]) 
        def __len__(self):
            return hasattr(self.users, "__len__") and len(self.users) or 0
        def __getitem__(self, idx):
            return self.users[idx], self.items[idx]

    forget_dataset = ForgetDataset(forget_samples)
    forget_loader = DataLoader(forget_dataset, batch_size=batch_size, shuffle=False)
    
    js_divergences = []
    
    with torch.no_grad():
        pbar = tqdm(forget_loader, desc="计算ZRF", leave=False)
        for users, items in pbar:
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
        
    mean_js_divergence = np.mean(js_divergences)
    zrf_score = 1.0 - mean_js_divergence
    
    return zrf_score

if __name__ == "__main__":
    set_seed(COMMON_CONFIG['seed'])
    
    # 1. Load Data (Yelp)
    data_path = COMMON_CONFIG['data_path']
    absolute_data_path = os.path.join(project_root, "dataset", "yelp.inter")
    
    print(f"Loading dataset from {absolute_data_path}")
    dataset = ML100KDataset(absolute_data_path, test_size=0.1, min_item_count=5)
    
    # Compute Norm Adj Matrix
    print("Computing Adj Matrix...")
    interaction_matrix = dataset.interaction_matrix.tocoo()
    temp_model = LightGCN(dataset.n_users, dataset.n_items)
    norm_adj_matrix = temp_model.get_norm_adj_mat(interaction_matrix).to(device)
    del temp_model
    
    # 2. Train Backbone (Full Data)
    print("\n=== Training Backbone (LightGCN) ===")
    backbone = LightGCN(
        dataset.n_users, dataset.n_items, 
        embedding_size=MODEL_CONFIG['embedding_size'],
        n_layers=MODEL_CONFIG['n_layers'],
        reg_weight=MODEL_CONFIG['reg_weight']
    ).to(device)
    
    # Load backbone - USER SPECIFIED PATH
    backbone_path = "/data/P2F/pretrain_checkpoints/lightgcn_yelp_full.pth"
    if os.path.exists(backbone_path):
        print(f"Loading backbone from {backbone_path}...")
        backbone.load_state_dict(torch.load(backbone_path))
    else:
        print(f"Backbone not found at {backbone_path}, training from scratch...") 
        backbone = train_lightgcn(backbone, dataset, MODEL_CONFIG['epochs'], norm_adj_matrix, backbone_path)
        
    print("Evaluating Backbone:")
    eval_res = evaluate(backbone, dataset, norm_adj_matrix, COMMON_CONFIG['k_list'], MODEL_CONFIG['batch_size'])
    print_metrics(eval_res)
    
    # 3. Split Data
    forget_ratio = UNLEARNING_CONFIG.get('forget_ratio', 0.01)
    forget_samples, retain_samples = split_forget_retain(dataset, forget_ratio)
    
    # 4. Incompetent Teacher (Random)
    print("\n=== Initializing Incompetent Teacher (Random) ===")
    incompetent = LightGCN(
        dataset.n_users, dataset.n_items, 
        embedding_size=MODEL_CONFIG['embedding_size'],
        n_layers=MODEL_CONFIG['n_layers'],
        reg_weight=MODEL_CONFIG['reg_weight']
    ).to(device)
    
    # 5. Unlearning
    print("\n=== Unlearning (Blindspot) ===")
    student = LightGCN(
        dataset.n_users, dataset.n_items,
        embedding_size=MODEL_CONFIG['embedding_size'],
        n_layers=MODEL_CONFIG['n_layers'],
        reg_weight=MODEL_CONFIG['reg_weight'],
        prompt_type='attention', 
        p_num=UNLEARNING_CONFIG.get('p_num', 20)
    ).to(device)
    
    state = backbone.state_dict()
    student.load_state_dict(state, strict=False)
    
    # Pre-evaluation
    print("\n评估原始模型:")
    original_forget_results, original_retain_results = evaluate_unlearning(
        model=backbone, 
        dataset=dataset, 
        forget_samples=forget_samples, 
        retain_samples=retain_samples, 
        norm_adj_matrix=norm_adj_matrix, 
        k_list=COMMON_CONFIG['k_list']
    )
    print("\n原始模型在遗忘集上的性能:")
    print_metrics(original_forget_results, prefix="  ")
    print("\n原始模型在保留集上的性能:")
    print_metrics(original_retain_results, prefix="  ")
    
    UNLEARNING_CONFIG['remain_ratio'] = 0.01
    UNLEARNING_CONFIG['batch_size'] = 2000000 # Full Batch
    UNLEARNING_CONFIG['lr'] = 0.005
    UNLEARNING_CONFIG['alpha'] = 0.5
    UNLEARNING_CONFIG['patience'] = 15
    UNLEARNING_CONFIG['epochs'] = 400
    
    student = blindspot_unlearner(
        model=student,
        unlearning_teacher=incompetent,
        full_trained_teacher=backbone,
        retain_data=retain_samples,
        forget_data=forget_samples,
        norm_adj_matrix=norm_adj_matrix,
        dataset=dataset,
        epochs=UNLEARNING_CONFIG['epochs'],
        lr=UNLEARNING_CONFIG['lr'],
        batch_size=UNLEARNING_CONFIG['batch_size'],
        validation_interval=5,
        patience=UNLEARNING_CONFIG['patience']
    )
    
    # 6. Evaluate
    print("\n=== Final Evaluation ===")
    print("Evaluating Unlearned Model (Student) on Forget/Retain Data:")
    
    forget_results, retain_results = evaluate_unlearning(
        model=student, 
        dataset=dataset, 
        forget_samples=forget_samples, 
        retain_samples=retain_samples, 
        norm_adj_matrix=norm_adj_matrix, 
        k_list=COMMON_CONFIG['k_list']
    )
    
    print("\n遗忘后模型在遗忘集上的性能:")
    print_metrics(forget_results, prefix="  ")
    
    for k in COMMON_CONFIG['k_list']:
        changes = calculate_performance_change(original_forget_results, forget_results, k)
        print(f"  性能下降: Recall: {changes['recall']:.2f}%, NDCG: {changes['ndcg']:.2f}%, HitRate: {changes['hit_rate']:.2f}%")
    
    print("\n遗忘后模型在保留集上的性能:")
    print_metrics(retain_results, prefix="  ")
    
    for k in COMMON_CONFIG['k_list']:
        changes = calculate_performance_change(original_retain_results, retain_results, k)
        print(f"  性能变化: Recall: {changes['recall']:.2f}%, NDCG: {changes['ndcg']:.2f}%, HitRate: {changes['hit_rate']:.2f}%")
    
    zrf = calculate_zrf(student, incompetent, forget_samples, norm_adj_matrix, 256, device)
    print(f"\nZRF Score: {zrf:.4f}")
    
    save_path = os.path.join(project_root, "unlearning_checkpoints", "lightgcn_unlearned_p2f_yelp.pth")
    if not os.path.exists(os.path.dirname(save_path)):
        os.makedirs(os.path.dirname(save_path))
    torch.save(student.state_dict(), save_path)
    print(f"Saved unlearned model to {save_path}")

    with open(os.path.join(project_root, "lightGCN_experiment/P2F/lightgcn_experiment_results_yelp.txt"), "w") as f:
        f.write(f"ZRF: {zrf:.4f}\n")
