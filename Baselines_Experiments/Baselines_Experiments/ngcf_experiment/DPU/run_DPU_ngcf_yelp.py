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
from data import ML100KDataset
from ngcf_models import NGCF
from utils import evaluate, print_metrics, set_seed, split_forget_retain, unlearner_loss_2, UnLearningDataset, calculate_performance_change
from unlearning import blindspot_unlearner, evaluate_unlearning

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
    'epochs': 400, # Switched to 200 for SGD
    'batch_size': 4096, # Switched to Mini-Batch SGD for better convergence
    'eval_freq': 10,
    'patience': 10,
}

def train_ngcf(model, dataset, epochs, norm_adj_matrix, save_path=None):
    optimizer = optim.Adam(model.parameters(), lr=NGCF_CONFIG['lr'])
    train_loader = DataLoader(dataset, batch_size=NGCF_CONFIG['batch_size'], shuffle=True, num_workers=8)
    
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
            # Using Recall@10 for simple early stopping monitoring
            curr_recall = eval_results[10]['recall']
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
    
    # 1. Load Data
    data_path = COMMON_CONFIG['data_path']
    absolute_data_path = os.path.join(project_root, "dataset", "yelp.inter")
    
    print(f"Loading dataset from {absolute_data_path}")
    # User requested to match RecBole metrics which likely uses 10% test split and 5-core filtering
    dataset = ML100KDataset(absolute_data_path, test_size=0.1, min_item_count=5)
    
    # Compute Norm Adj Matrix
    print("Computing Adj Matrix...")
    interaction_matrix = dataset.interaction_matrix.tocoo()
    # Need a temporary model to compute adj matrix (cleaner way: static method or helper)
    # But NGCF class method works.
    temp_model = NGCF(dataset.n_users, dataset.n_items)
    norm_adj_matrix = temp_model.get_norm_adj_mat(interaction_matrix).to(device)
    del temp_model
    
    # 2. Train Backbone (Full Data)
    print("\n=== Training Backbone (NGCF) ===")
    backbone = NGCF(
        dataset.n_users, dataset.n_items, 
        embedding_size=NGCF_CONFIG['embedding_size'],
        n_layers=NGCF_CONFIG['n_layers'],
        reg_weight=NGCF_CONFIG['reg_weight'],
        node_dropout=NGCF_CONFIG['node_dropout'],
        message_dropout=NGCF_CONFIG['message_dropout']
    ).to(device)
    
    # Try to load if exists, else train
    backbone_path = os.path.join(project_root, "pretrain_checkpoints", "best_yelp_ngcf_backbone.pth")
    if os.path.exists(backbone_path):
        print("Loading backbone...")
        backbone.load_state_dict(torch.load(backbone_path))
    else:
        backbone = train_ngcf(backbone, dataset, NGCF_CONFIG['epochs'], norm_adj_matrix, backbone_path)
        
    print("Evaluating Backbone:")
    eval_res = evaluate(backbone, dataset, norm_adj_matrix, COMMON_CONFIG['k_list'], NGCF_CONFIG['batch_size'])
    print_metrics(eval_res)
    
    # 3. Split Data
    forget_ratio = UNLEARNING_CONFIG.get('forget_ratio', 0.01)
    forget_samples, retain_samples = split_forget_retain(dataset, forget_ratio)
    
    # 4. Train Incompetent Teacher (Retain Set)
    # Theoretically "Incompetent Teacher" is trained on Retain Set.
    # We need a new Dataset object for Retain Set to use standard train_ngcf or just modify dataloader.
    # ML100KDataset stores strict interaction_matrix. 
    # To train properly on Retain Set, we should ideally construct a new interaction matrix.
    # This might be complicated without refactoring ML100KDataset to accept samples.
    # For now, I'll assume Incompetent Teacher is needed for ZRF.
    # I'll create a simple adapter or train using samples directly if I modify train function?
    # No, train_ngcf takes 'dataset' which implements __getitem__.
    # I can define RetainDataset wrapping retain_samples.
    
    class RetainDataset(Dataset):
        def __init__(self, samples, n_users, n_items, original_dataset):
            self.samples = samples
            self.n_users = n_users
            self.n_items = n_items
            self.original_dataset = original_dataset # Store original dataset to access get_test_samples

        def __len__(self):
            return len(self.samples)

        def __getitem__(self, idx):
            sample = self.samples[idx]
            if len(sample) == 3:
                user, pos_item, neg_item = sample
            else:
                user, pos_item = sample
                neg_item = np.random.randint(0, self.n_items)
                try:
                    while neg_item in self.original_dataset.all_user_items[user]:
                        neg_item = np.random.randint(0, self.n_items)
                except KeyError:
                    pass
            
            # Return scalars to be consistent with ML100KDataset
            return user, pos_item, neg_item

        def get_test_samples(self):
             return self.original_dataset.get_test_samples()
        
        @property
        def all_user_items(self):
             return self.original_dataset.all_user_items
        
        @property
        def test_user_items(self):
             return self.original_dataset.test_user_items

        @property
        def train_user_items(self):
             return self.original_dataset.train_user_items

    print("\n=== Initializing Incompetent Teacher (Random) ===")
    incompetent = NGCF(
        dataset.n_users, dataset.n_items, 
        embedding_size=NGCF_CONFIG['embedding_size'],
        n_layers=NGCF_CONFIG['n_layers'],
        reg_weight=NGCF_CONFIG['reg_weight']
    ).to(device)
    
    # NOTE: Matched user's unlearning.py logic where Incompetent Teacher is NOT trained on retain set, 
    # but is just a randomly initialized model.
    
    # 5. Unlearning
    print("\n=== Unlearning (Blindspot) ===")
    # Initialize Student with Backbone weights + Prompts
    student = NGCF(
        dataset.n_users, dataset.n_items,
        embedding_size=NGCF_CONFIG['embedding_size'],
        n_layers=NGCF_CONFIG['n_layers'],
        prompt_type='attention', 
        p_num=UNLEARNING_CONFIG.get('p_num', 20)
    ).to(device)
    
    # Load backbone weights
    state = backbone.state_dict()
    student.load_state_dict(state, strict=False) # Skip missing prompt weights
    
    # --- Pre-evaluation (Original Model) ---
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
    # ---------------------------------------

    # Unlearning Training
    # Use exact parameters from your successful script
    UNLEARNING_CONFIG['remain_ratio'] = 0.01  # Keep same ratio
    UNLEARNING_CONFIG['batch_size'] = 2000000  # Full Batch
    UNLEARNING_CONFIG['lr'] = 0.005           # Keep same LR
    UNLEARNING_CONFIG['alpha'] = 0.5          # Restore original alpha
    UNLEARNING_CONFIG['patience'] = 15        # Keep increased patience for stability
    UNLEARNING_CONFIG['epochs'] = 400         # Explicitly set epochs to 400
    
    # Use the Blindspot Unlearner from USER's unlearning.py to ensure identical logic
    student = blindspot_unlearner(
        model=student,
        unlearning_teacher=incompetent,
        full_trained_teacher=backbone,
        retain_data=retain_samples,
        forget_data=forget_samples,
        norm_adj_matrix=norm_adj_matrix, # Prompt unlearning typically uses original graph structure
        dataset=dataset,
        epochs=UNLEARNING_CONFIG['epochs'],
        lr=UNLEARNING_CONFIG['lr'],
        batch_size=UNLEARNING_CONFIG['batch_size'],
        validation_interval=5, # Validate every 5 epochs to save time
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
    
    # ZRF
    zrf = calculate_zrf(student, incompetent, forget_samples, norm_adj_matrix, 256, device)
    print(f"\nZRF Score: {zrf:.4f}")
    
    # Save Model
    save_path = os.path.join(project_root, "unlearning_checkpoints", "ngcf_unlearned_p2f_yelp.pth")
    if not os.path.exists(os.path.dirname(save_path)):
        os.makedirs(os.path.dirname(save_path))
    torch.save(student.state_dict(), save_path)
    print(f"Saved unlearned model to {save_path}")

    # Save results to file if needed
    with open("ngcf_experiment_results_yelp.txt", "w") as f:
        f.write(f"ZRF: {zrf:.4f}\n")
