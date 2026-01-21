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

from config import COMMON_CONFIG, UNLEARNING_CONFIG
from data import ML100KDataset
from utils import split_forget_retain, print_metrics, set_seed
from gnndelete import GNNDELETEManager
from ngcf_models import NGCFGNNDELETE

# Set device
device = COMMON_CONFIG['device']
print(f"Using device: {device}")

# Config (Optimized for performance)
GNNDELETE_CONFIG = {
    'embedding_size': 64,
    'n_layers': 3,
    'reg_weight': 1e-5,
    'node_dropout': 0.1,
    'message_dropout': 0.1,
    'lr': 0.001,
    'epochs': 10, 
    'batch_size': 4096, 
    'seed': COMMON_CONFIG['seed']
}

if __name__ == "__main__":
    set_seed(GNNDELETE_CONFIG['seed'])
    
    # 1. Load Data
    absolute_data_path = os.path.join(project_root, "dataset", "yelp.inter")
    print(f"Loading Yelp from {absolute_data_path}")
    dataset = ML100KDataset(absolute_data_path, test_size=0.1, min_item_count=5)
    
    # 2. Split Data (Exactly as Retrain/SISA)
    forget_ratio = UNLEARNING_CONFIG.get('forget_ratio', 0.01)
    print(f"Splitting data with forget_ratio={forget_ratio}...")
    forget_samples, retain_samples = split_forget_retain(dataset, forget_ratio)
    
    # Extract edges (u, i) from samples
    deleted_edges = [(s[0], s[1]) for s in forget_samples]
    print(f"Number of deleted edges: {len(deleted_edges)}")

    # 3. Init GNNDELETE Manager
    save_path = "gnndelete_yelp.pth"
    GNNDELETE_CONFIG['deletion_lambda'] = 0.5
    gnndelete = GNNDELETEManager(dataset, device, GNNDELETE_CONFIG, save_path)
    
    # 4. Train Base Model (on FULL dataset)
    # Check for pre-trained checkpoint first
    pretrain_path = os.path.join(project_root, "pretrain_checkpoints", "ngcf_yelp_full.pth")
    if os.path.exists(save_path):
        print("\n=== Loading Base Model ===")
        gnndelete.load_model()
    elif os.path.exists(pretrain_path):
        # Try loading pre-trained weights
        print(f"\n=== Loading Pre-trained Base Model from {pretrain_path} ===")
        gnndelete.load_pretrained(pretrain_path)
        # Save explicitly to save_path
        torch.save(gnndelete.model.state_dict(), save_path)
    else:
        print("\n=== Training Base Model (Full Data) ===")
        gnndelete.train(epochs=200, batch_size=GNNDELETE_CONFIG['batch_size'])
    
    # 5. Apply Unlearning
    print("\n=== Applying GNNDELETE Unlearning ===")
    gnndelete.unlearn(deleted_edges, epochs=20, lr=0.001)
    
    # Load unlearned model
    if os.path.exists(save_path + '_unlearned'):
         gnndelete.model.load_state_dict(torch.load(save_path + '_unlearned'))
    
    # 6. Init Random Teacher for ZRF
    print("\n=== Initializing Random Teacher (For ZRF) ===")
    incompetent = NGCFGNNDELETE(
        dataset.n_users, dataset.n_items, 
        embedding_size=GNNDELETE_CONFIG['embedding_size'],
        n_layers=GNNDELETE_CONFIG['n_layers']
    ).to(device)
    
    # 7. Final Evaluation
    print("\n=== Final Evaluation (Unlearned Model) ===")
    
    from utils import evaluate, print_metrics
    from unlearning import evaluate_unlearning
    
    # A. Test Set Performance (Overall)
    print("Evaluating on Test Set...")
    metrics = evaluate(gnndelete.model, dataset, gnndelete.norm_adj_matrix, COMMON_CONFIG['k_list'])
    print_metrics(metrics, prefix="Test Set ")

    # B. Forget and Retain Performance using standard evaluation
    print("\nEvaluating Unlearned Model (Student) on Forget/Retain Data:")
    forget_results, retain_results = evaluate_unlearning(
        model=gnndelete.model, 
        dataset=dataset, 
        forget_samples=forget_samples, 
        retain_samples=retain_samples, 
        norm_adj_matrix=gnndelete.norm_adj_matrix, 
        k_list=COMMON_CONFIG['k_list']
    )
    
    print("\nUnlearned model on Forget Set:")
    print_metrics(forget_results, prefix="Forget Set ")
    
    print("\nUnlearned model on Retain Set:")
    print_metrics(retain_results, prefix="Retain Set ")

    # D. ZRF Score
    print("\nCalculating ZRF Score...")
    zrf = gnndelete.calculate_zrf(incompetent, forget_samples, batch_size=2048)
    print(f"ZRF Score: {zrf:.4f}")

    # Save Results
    with open("gnndelete_yelp_results.txt", "w") as f:
        f.write(f"ZRF: {zrf:.4f}\n")
        f.write(f"Test_NDCG_20: {metrics[20]['ndcg']:.4f}\n")
        f.write(f"Test_Recall_20: {metrics[20]['recall']:.4f}\n")
        f.write(f"Forget_NDCG_20: {forget_results[20]['ndcg']:.4f}\n")
        f.write(f"Retain_NDCG_20: {retain_results[20]['ndcg']:.4f}\n")
        f.write(f"Full_Test_Metrics: {str(metrics)}\n")

    print("\nExperiment Finished.")
