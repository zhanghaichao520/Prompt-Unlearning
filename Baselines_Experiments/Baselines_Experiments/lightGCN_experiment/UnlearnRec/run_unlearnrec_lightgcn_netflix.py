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

# Add lightgcn path
lightgcn_path = os.path.join(project_root, "lightGCN_experiment", "GNNDELETE")
if lightgcn_path not in sys.path:
    sys.path.append(lightgcn_path)

from config import COMMON_CONFIG, UNLEARNING_CONFIG, LIGHTGCN_CONFIG
from data import ML100KDataset
from utils import split_forget_retain, print_metrics, set_seed
from unlearnrec import UnlearnRecManager
from unlearnrec_models import LightGCNUnlearnRec
from unlearning import evaluate_unlearning
from utils import evaluate 

# Set device
device = COMMON_CONFIG['device']
print(f"Using device: {device}")

# UnlearnRec Specific Config for LightGCN
UNLEARNREC_LGCN_CONFIG = {
    'embedding_size': LIGHTGCN_CONFIG['embedding_size'],
    'n_layers': LIGHTGCN_CONFIG['n_layers'],
    'reg_weight': LIGHTGCN_CONFIG['reg_weight'],
    'lr_pre': 0.001,
    'lr_finetune': 0.01,
    'lambda_u': 1.0,
    'lambda_p': 100.0, 
    'pre_epochs': 50,
    'unlearn_epochs': 10,
    'batch_size': 2048,
    'seed': COMMON_CONFIG['seed']
}

if __name__ == "__main__":
    set_seed(UNLEARNREC_LGCN_CONFIG['seed'])
    
    # 1. Load Data (Netflix)
    dataset_name = "netflix"
    absolute_data_path = os.path.join(project_root, "dataset", f"{dataset_name}.inter")
    print(f"Loading {dataset_name} from {absolute_data_path}")
    dataset = ML100KDataset(absolute_data_path, test_size=0.1, min_item_count=5)
    
    # 2. Split Data
    forget_ratio = UNLEARNING_CONFIG.get('forget_ratio', 0.01)
    print(f"Splitting data with forget_ratio={forget_ratio}...")
    forget_samples, retain_samples = split_forget_retain(dataset, forget_ratio)
    
    # 3. Initialize Base Model (LightGCN)
    print("\n=== Initializing & Loading Base Model (LightGCN) ===")
    model = LightGCNUnlearnRec(
        dataset.n_users, dataset.n_items, 
        embedding_size=UNLEARNREC_LGCN_CONFIG['embedding_size'],
        n_layers=UNLEARNREC_LGCN_CONFIG['n_layers']
    ).to(device)
    
    # Load Pretrained Weights
    pretrain_path = os.path.join(project_root, "pretrain_checkpoints", "best_lightgcn_retrained_netflix.pth")
    # Alternate names not common for netflix/yelp based on ls, but keeping consistent structure
    
    if os.path.exists(pretrain_path):
        print(f"Loading pretrained weights from {pretrain_path}")
        model.load_state_dict(torch.load(pretrain_path, map_location=device), strict=False)
    else:
         print(f"Warning: Pretrained model not found at {pretrain_path}. Using random init.")

    # 4. Initialize Manager
    manager = UnlearnRecManager(dataset, model, device, UNLEARNREC_LGCN_CONFIG)
    
    # Set Graph Data
    norm_adj = model.get_norm_adj_mat(dataset.interaction_matrix).to(device)
    manager.set_graph_data(norm_adj)
    
    # 5. Pretrain Influence Encoder
    ie_checkpoint = f"ie_lightgcn_{dataset_name}.pth"
    if os.path.exists(ie_checkpoint):
        print(f"Loading IE from {ie_checkpoint}")
        manager.ie.load_state_dict(torch.load(ie_checkpoint))
    else:
        print("\n=== Pretraining Influence Encoder ===")
        manager.pretrain(epochs=UNLEARNREC_LGCN_CONFIG['pre_epochs'])
        torch.save(manager.ie.state_dict(), ie_checkpoint)
        
    # 6. Unlearning
    print("\n=== Performing Unlearning (Fine-tuning E0) ===")
    new_E0 = manager.finetune(forget_samples, epochs=UNLEARNREC_LGCN_CONFIG['unlearn_epochs'])
    
    # Apply new E0
    with torch.no_grad():
        u_emb, i_emb = torch.split(new_E0, [dataset.n_users, dataset.n_items])
        model.user_embedding.weight.data.copy_(u_emb)
        model.item_embedding.weight.data.copy_(i_emb)

    # 7. Final Evaluation
    print("\n=== Final Evaluation ===")
    
    metrics = evaluate(model, dataset, norm_adj, COMMON_CONFIG['k_list'])
    print_metrics(metrics, prefix="Test Set ")
    
    incompetent = LightGCNUnlearnRec(
        dataset.n_users, dataset.n_items, 
        embedding_size=UNLEARNREC_LGCN_CONFIG['embedding_size'],
        n_layers=UNLEARNREC_LGCN_CONFIG['n_layers']
    ).to(device)
    
    forget_results, retain_results = evaluate_unlearning(
        model, 
        dataset, 
        forget_samples, 
        retain_samples, 
        norm_adj, 
        k_list=COMMON_CONFIG['k_list']
    )
    
    print("\nForget Set Performance:")
    print_metrics(forget_results)
    print("\nRetain Set Performance:")
    print_metrics(retain_results)
    
    # ZRF Score
    print("\nCalculating ZRF Score...")
    zrf = manager.calculate_zrf(incompetent, forget_samples, batch_size=2048)
    print(f"ZRF Score: {zrf:.4f}")
    
    with open(f"unlearnrec_lightgcn_{dataset_name}_results.txt", "w") as f:
        f.write(f"ZRF: {zrf:.4f}\n")
        f.write(f"Test_NDCG_20: {metrics[20]['ndcg']:.4f}\n")
        f.write(f"Test_Recall_20: {metrics[20]['recall']:.4f}\n")
        f.write(f"Forget_NDCG_20: {forget_results[20]['ndcg']:.4f}\n")
        f.write(f"Retain_NDCG_20: {retain_results[20]['ndcg']:.4f}\n")
        f.write(f"Full_Test_Metrics: {str(metrics)}\n")

