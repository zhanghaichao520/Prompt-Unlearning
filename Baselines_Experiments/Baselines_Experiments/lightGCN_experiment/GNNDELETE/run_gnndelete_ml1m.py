# -*- coding: utf-8 -*-
import sys
import os
import torch
import numpy as np

# Add project root to path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, "../../"))
if project_root not in sys.path:
    sys.path.append(project_root)

from config import COMMON_CONFIG, UNLEARNING_CONFIG, LIGHTGCN_CONFIG
from data import ML100KDataset
from utils import split_forget_retain, print_metrics, set_seed
from unlearning import evaluate_unlearning
from gnndelete import GNNDELETEManager
from lightgcn_models import LightGCNGNNDELETE
from utils import evaluate # Use standard evaluate from root utils

# Set device
device = COMMON_CONFIG['device']
print(f"Using device: {device}")

# Config for GNNDELETE (specific to LightGCN params)
GNNDELETE_CONFIG = {
    'embedding_size': LIGHTGCN_CONFIG['embedding_size'],
    'n_layers': LIGHTGCN_CONFIG['n_layers'],
    'reg_weight': LIGHTGCN_CONFIG['reg_weight'],
    'lr': LIGHTGCN_CONFIG['lr'],
    'epochs': 400, # Same as full training to get good base
    'unlearn_epochs': 20, 
    'batch_size': COMMON_CONFIG.get('batch_size', 2048),
    'seed': COMMON_CONFIG['seed'],
    'deletion_lambda': 0.5
}

if __name__ == "__main__":
    set_seed(GNNDELETE_CONFIG['seed'])
    
    # 1. Load Data
    absolute_data_path = os.path.join(project_root, "dataset", "ml-1m.inter")
    print(f"Loading ML-1M from {absolute_data_path}")
    dataset = ML100KDataset(absolute_data_path, test_size=0.1, min_item_count=5)
    
    # 2. Split Data
    forget_ratio = UNLEARNING_CONFIG.get('forget_ratio', 0.01)
    print(f"Splitting data with forget_ratio={forget_ratio}...")
    forget_samples, retain_samples = split_forget_retain(dataset, forget_ratio)
    
    deleted_edges = [(s[0], s[1]) for s in forget_samples]
    print(f"Number of deleted edges: {len(deleted_edges)}")
    
    # 3. Init GNNDELETE Manager
    save_path = "gnndelete_lightgcn_ml1m.pth"
    gnndelete = GNNDELETEManager(dataset, device, GNNDELETE_CONFIG, save_path)
    
    # 4. Train Base Model
    # Try finding pre-trained LightGCN
    pretrain_path = os.path.join(project_root, "pretrain_checkpoints", "lightgcn_ml1m_full.pth")
    # Check if 'best_model.pth' from train_lightgcn.py exists/is usable? 
    # Usually better to have a dedicated pretrain path or train from scratch.
    
    if os.path.exists(save_path):
        print(f"Loading from {save_path}")
        gnndelete.load_model()
    elif os.path.exists(pretrain_path):
        print(f"Loading Pretrained from {pretrain_path}")
        gnndelete.load_pretrained(pretrain_path)
        torch.save(gnndelete.model.state_dict(), save_path)
    else:
        print("Training Base Model from scratch...")
        gnndelete.train(epochs=GNNDELETE_CONFIG['epochs'], batch_size=GNNDELETE_CONFIG['batch_size'])
        
    # 5. Apply Unlearning
    print("\n=== Applying GNNDELETE Unlearning ===")
    gnndelete.unlearn(deleted_edges, epochs=GNNDELETE_CONFIG['unlearn_epochs'], lr=0.001)
    
    if os.path.exists(save_path + '_unlearned'):
         gnndelete.model.load_state_dict(torch.load(save_path + '_unlearned'))
         
    # 6. Init Teacher (Using LightGCNGNNDELETE structure for random init)
    print("\n=== Initializing Random Teacher (For ZRF) ===")
    incompetent = LightGCNGNNDELETE(
        dataset.n_users, dataset.n_items, 
        embedding_size=GNNDELETE_CONFIG['embedding_size'],
        n_layers=GNNDELETE_CONFIG['n_layers']
    ).to(device)
    
    # 7. Final Evaluation
    print("\n=== Final Evaluation (Unlearned Model) ===")
    
    # A. Test Set
    print("Evaluating on Test Set...")
    metrics = evaluate(gnndelete.model, dataset, gnndelete.norm_adj_matrix, COMMON_CONFIG['k_list'])
    print_metrics(metrics, prefix="Test Set ")
    
    # B. Forget/Retain
    print("\nEvaluating on Forget/Retain...")
    forget_results, retain_results = evaluate_unlearning(
        model=gnndelete.model,
        dataset=dataset,
        forget_samples=forget_samples,
        retain_samples=retain_samples,
        norm_adj_matrix=gnndelete.norm_adj_matrix,
        k_list=COMMON_CONFIG['k_list']
    )
    print_metrics(forget_results, prefix="Forget Set ")
    print_metrics(retain_results, prefix="Retain Set ")
    
    # D. ZRF
    print("\nCalculating ZRF Score...")
    zrf = gnndelete.calculate_zrf(incompetent, forget_samples, batch_size=2048)
    print(f"ZRF Score: {zrf:.4f}")
    
    # Save Results
    with open("gnndelete_lightgcn_ml1m_results.txt", "w") as f:
        f.write(f"ZRF: {zrf:.4f}\n")
        f.write(f"Test_NDCG_20: {metrics[20]['ndcg']:.4f}\n")
        f.write(f"Test_Recall_20: {metrics[20]['recall']:.4f}\n")
        f.write(f"Forget_NDCG_20: {forget_results[20]['ndcg']:.4f}\n")
        f.write(f"Retain_NDCG_20: {retain_results[20]['ndcg']:.4f}\n")
        f.write(f"Full_Test_Metrics: {str(metrics)}\n")

    print("\nExperiment Finished.")