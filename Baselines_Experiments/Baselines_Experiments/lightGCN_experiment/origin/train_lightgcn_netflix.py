# -*- coding: utf-8 -*-
# LightGCN Netflix Training Script

import sys
import os
import time
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm, trange

# Add project root to path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, "../"))
if project_root not in sys.path:
    sys.path.append(project_root)

# Import custom modules
from config import COMMON_CONFIG, LIGHTGCN_CONFIG
from data import ML100KDataset
from models import LightGCN
from utils import evaluate, print_metrics, set_seed

# Set Seed
set_seed(COMMON_CONFIG['seed'])

# Device Config
device = COMMON_CONFIG['device']
print(f"Using device: {device}")

# Dataset & Config - Explicitly set for Netflix
DATASET_NAME = "netflix"
SAVE_PATH = os.path.join(project_root, "pretrain_checkpoints", f"lightgcn_{DATASET_NAME}_full.pth")

PRETRAIN_CONFIG = {
    'embedding_size': LIGHTGCN_CONFIG['embedding_size'],
    'n_layers': LIGHTGCN_CONFIG['n_layers'],
    'reg_weight': LIGHTGCN_CONFIG['reg_weight'],
    'lr': LIGHTGCN_CONFIG['lr'],
    'epochs': 1000,
    'batch_size': 2048,
    'eval_freq': 10,
    'patience': 20
}

def train(model, dataset, optimizer, batch_size, epochs, norm_adj_matrix, k_list, eval_freq, patience, save_path):
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    best_recall = 0.0
    patience_counter = 0
    best_model_state = None
    
    print(f"Starting training on {DATASET_NAME}...")
    
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0
        start_time = time.time()
        
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
        
        if epoch % eval_freq == 0:
            train_time = time.time() - start_time
            eval_results = evaluate(model, dataset, norm_adj_matrix, k_list, batch_size)
            curr_recall = eval_results[20]['recall']
            curr_ndcg = eval_results[20]['ndcg']
            
            print(f"Epoch {epoch}: Loss={avg_loss:.4f}, Recall@20={curr_recall:.4f}, NDCG@20={curr_ndcg:.4f}, Time={train_time:.1f}s")
            
            if curr_recall > best_recall:
                best_recall = curr_recall
                patience_counter = 0
                best_model_state = model.state_dict().copy()
                torch.save(model.state_dict(), save_path)
                print(f"  Saved best model to {save_path}")
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"Early stopping at epoch {epoch}")
                    break
    
    if best_model_state:
        model.load_state_dict(best_model_state)
    return model

if __name__ == "__main__":
    data_path = os.path.join(project_root, "dataset", f"{DATASET_NAME}.inter")
    print(f"Loading {DATASET_NAME} from {data_path}")
    dataset = ML100KDataset(data_path, test_size=0.1, min_item_count=5)
    
    model = LightGCN(
        n_users=dataset.n_users,
        n_items=dataset.n_items,
        embedding_size=PRETRAIN_CONFIG['embedding_size'],
        n_layers=PRETRAIN_CONFIG['n_layers'],
        reg_weight=PRETRAIN_CONFIG['reg_weight']
    ).to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=PRETRAIN_CONFIG['lr'])
    
    print("Computing Adjacency Matrix...")
    norm_adj = model.get_norm_adj_mat(dataset.interaction_matrix).to(device)
    
    model = train(
        model, dataset, optimizer, 
        PRETRAIN_CONFIG['batch_size'], 
        PRETRAIN_CONFIG['epochs'], 
        norm_adj, 
        COMMON_CONFIG['k_list'],
        PRETRAIN_CONFIG['eval_freq'],
        PRETRAIN_CONFIG['patience'],
        SAVE_PATH
    )
    
    print("\nFinal Evaluation:")
    metrics = evaluate(model, dataset, norm_adj, COMMON_CONFIG['k_list'])
    print_metrics(metrics)
