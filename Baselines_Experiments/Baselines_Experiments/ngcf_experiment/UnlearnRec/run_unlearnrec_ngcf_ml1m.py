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
from unlearnrec import UnlearnRecManager
from unlearnrec_models import NGCFUnlearnRec
from unlearning import evaluate_unlearning
from utils import evaluate, get_norm_adj_mat

# Set device
device = COMMON_CONFIG['device']
print(f"Using device: {device}")

# UnlearnRec Specific Config
UNLEARNREC_CONFIG = {
    'embedding_size': 64,
    'n_layers': 3,
    'lr_pre': 0.001,
    'lr_finetune': 0.005, # Decreased from 0.01
    'lambda_u': 0.1,      # Decreased from 1.0
    'lambda_p': 1000.0,   # Increased from 100.0
    'pre_epochs': 50,
    'unlearn_epochs': 10,
    'batch_size': 2048,
    'seed': COMMON_CONFIG['seed']
}

if __name__ == "__main__":
    set_seed(UNLEARNREC_CONFIG['seed'])
    
    # 1. Load Data (ML-1M)
    dataset_name = "ml-1m"
    absolute_data_path = os.path.join(project_root, "dataset", f"{dataset_name}.inter")
    print(f"Loading {dataset_name} from {absolute_data_path}")
    dataset = ML100KDataset(absolute_data_path, test_size=0.1, min_item_count=0)
    
    # 2. Split Data
    forget_ratio = UNLEARNING_CONFIG.get('forget_ratio', 0.01)
    print(f"Splitting data with forget_ratio={forget_ratio}...")
    forget_samples, retain_samples = split_forget_retain(dataset, forget_ratio)
    
    # 3. Initialize Base Model (NGCF)
    print("\n=== Initializing & Loading Base Model ===")
    model = NGCFUnlearnRec(
        dataset.n_users, dataset.n_items, 
        embedding_size=UNLEARNREC_CONFIG['embedding_size'],
        n_layers=UNLEARNREC_CONFIG['n_layers']
    ).to(device)
    
    # Load Pretrained Weights
    # If explicit pretrained weights exist, use them. Otherwise, assume they've been trained.
    # The workspace has `pretrain_checkpoints/ngcf_ml1m_full.pth`
    pretrain_path = os.path.join(project_root, "pretrain_checkpoints", "best_ml_1m_ngcf_backbone.pth")

    if os.path.exists(pretrain_path):
        print(f"Loading pretrained weights from {pretrain_path}")
        model.load_state_dict(torch.load(pretrain_path, map_location=device), strict=False)
    else:
        # Fallback: check if we can locate any full model
        # Check standard name: best_ml_1m_ngcf_backbone.pth
        alt_path = os.path.join(project_root, "pretrain_checkpoints", f"best_{dataset_name.replace('-', '_')}_ngcf_backbone.pth")
        if os.path.exists(alt_path):
            print(f"Loading pretrained weights from {alt_path}")
            model.load_state_dict(torch.load(alt_path, map_location=device), strict=False)
        else:
            print(f"Warning: Pretrained model not found at {pretrain_path} or {alt_path}. Using random init (Not Recommended for Unlearning baselines).")

    # Evaluate Baseline Immediately
    # Set Graph Data (Full Adjacency)
    norm_adj = get_norm_adj_mat(dataset.interaction_matrix, dataset.n_users, dataset.n_items).to(device)
    
    print("\n=== Evaluating Baseline Model (Before Unlearning) ===")
    base_metrics = evaluate(model, dataset, norm_adj, COMMON_CONFIG['k_list'])
    print_metrics(base_metrics, prefix="Base Model ")
    
    # 4. Initialize Manager
    manager = UnlearnRecManager(dataset, model, device, UNLEARNREC_CONFIG)
    manager.set_graph_data(norm_adj)
    
    # 5. Pretrain Influence Encoder (IE)
    # Check if IE checkpoint exists to save time? 
    ie_checkpoint = f"ie_ngcf_{dataset_name}.pth"
    if os.path.exists(ie_checkpoint):
        print(f"Loading IE from {ie_checkpoint}")
        manager.ie.load_state_dict(torch.load(ie_checkpoint))
    else:
        print("\n=== Pretraining Influence Encoder ===")
        manager.pretrain(epochs=UNLEARNREC_CONFIG['pre_epochs'])
        torch.save(manager.ie.state_dict(), ie_checkpoint)
        
    # 6. Unlearning (Fine-tuning E0)
    print("\n=== Performing Unlearning (Fine-tuning E0) ===")
    new_E0 = manager.finetune(forget_samples, epochs=UNLEARNREC_CONFIG['unlearn_epochs'])
    
    # Evaluate specific samples
    # We need to construct a model that uses this new_E0 PERMANENTLY or Inject it?
    # The models.py implementation of NGCFUnlearnRec uses `override_E0` in forward.
    # For evaluation functions that stick to standard signature `evaluate(model, ...)`, 
    # we might need to update the model's internal embeddings if possible, 
    # OR we need to make sure the model knows to use the new embeddings.
    
    # The NGCFUnlearnRec class inherits from NGCF. 
    # If we update `model.embedding_dict['user_emb']` and `model.embedding_dict['item_emb']`, it persists.
    # `new_E0` is concatenated [users, items].
    
    with torch.no_grad():
        u_emb, i_emb = torch.split(new_E0, [dataset.n_users, dataset.n_items])
        model.user_embedding.weight.data.copy_(u_emb)
        model.item_embedding.weight.data.copy_(i_emb)
        
    # 7. Final Evaluation
    print("\n=== Final Evaluation ===")
    
    # Test Set
    metrics = evaluate(model, dataset, norm_adj, COMMON_CONFIG['k_list'])
    print_metrics(metrics, prefix="Test Set ")
    
    # Forget/Retain
    # For ZRF, we need an incompetent teacher (Untrained model)
    incompetent = NGCFUnlearnRec(
        dataset.n_users, dataset.n_items, 
        embedding_size=UNLEARNREC_CONFIG['embedding_size'],
        n_layers=UNLEARNREC_CONFIG['n_layers']
    ).to(device) # Random init
    
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
    
    # Save Results
    with open("unlearnrec_ngcf_ml1m_results.txt", "w") as f:
        f.write(f"ZRF: {zrf:.4f}\n")
        f.write(f"Test_NDCG_20: {metrics[20]['ndcg']:.4f}\n")
        f.write(f"Test_Recall_20: {metrics[20]['recall']:.4f}\n")
        f.write(f"Forget_NDCG_20: {forget_results[20]['ndcg']:.4f}\n")
        f.write(f"Retain_NDCG_20: {retain_results[20]['ndcg']:.4f}\n")
        f.write(f"Full_Test_Metrics: {str(metrics)}\n")

