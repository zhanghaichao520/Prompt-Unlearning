# -*- coding: utf-8 -*-
import os
import sys
import torch
import numpy as np
import copy
from collections import defaultdict
from tqdm import tqdm

# Add project root to path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, "../../"))
if project_root not in sys.path:
    sys.path.append(project_root)

# Helper for NGCF
ngcf_path = os.path.join(project_root, 'ngcf_experiment', 'P2F')
if ngcf_path not in sys.path:
    sys.path.append(ngcf_path)

from config import COMMON_CONFIG, UNLEARNING_CONFIG
from data import ML100KDataset
from utils import split_forget_retain, set_seed, evaluate_user_subset
from models import LightGCN

# Try importing NGCF
try:
    from ngcf_models import NGCF
except ImportError:
    pass

from unlearning import load_prompt_for_inference

def get_user_items_dict(samples):
    user_items = defaultdict(list)
    for u, i, *r in samples:
        user_items[u].append(i)
    return user_items

def filter_forget_samples_by_origin(model, dataset, forget_samples, retain_samples, norm_adj_matrix, k=10, device='cuda'):
    """
    Filter forget_samples to keep only those that are successfully retrieved in Top-K by the Origin model.
    """
    model.eval()
    
    forget_user_items = get_user_items_dict(forget_samples)
    users = list(forget_user_items.keys())
    retain_user_items = get_user_items_dict(retain_samples)
    
    filtered_samples = []
    
    batch_size = 2048
    with torch.no_grad():
        for start in tqdm(range(0, len(users), batch_size), desc=f"Filtering (k={k})", leave=False):
            end = min(start + batch_size, len(users))
            batch_users = users[start:end]
            batch_users_tensor = torch.LongTensor(batch_users).to(device)
            
            # Predict
            if hasattr(model, 'full_sort_predict'):
                ratings = model.full_sort_predict(batch_users_tensor, norm_adj_matrix)
            else:
                ratings = model.predict(batch_users_tensor, None, norm_adj_matrix)
                
            ratings = ratings.cpu().numpy()
            
            for i, user in enumerate(batch_users):
                train_items = retain_user_items.get(user, [])
                if len(train_items) > 0:
                     ratings[i, train_items] = -np.inf
                
                # Top-K
                ind = np.argpartition(ratings[i], -k)[-k:]
                topk_items = set(ind)
                
                # Check hits
                target_items = forget_user_items[user]
                for target in target_items:
                    if target in topk_items:
                        filtered_samples.append((user, target, 1))
                        
    return filtered_samples

def adaptive_filter_samples(model, dataset, forget_samples, retain_samples, norm_adj_matrix, device='cuda'):
    print(f"\n[Filtering] Finding High-Confidence Samples (Adaptive Top-K Strategy)...")
    
    # Try strict first (Top-1), then relax if not enough samples
    # Using a threshold of ~20 samples minimum to calculate meaningful ACC
    for k in [1, 5, 10, 20]:
        filtered = filter_forget_samples_by_origin(model, dataset, forget_samples, retain_samples, norm_adj_matrix, k, device)
        count = len(filtered)
        print(f"  > Trying k={k}: Found {count} samples.")
        if count >= 100: # We want a robust set
            print(f"  > Selected k={k} strategy.")
            return filtered
            
    print("  > Warning: Very few memorized samples even at k=20. Using last result.")
    return filtered

def calculate_topk_mia_accuracy(model, dataset, forget_samples, retain_samples, norm_adj_matrix, k_list=[1, 10], device='cuda'):
    model.eval()
    
    forget_user_items = get_user_items_dict(forget_samples)
    forget_users = list(forget_user_items.keys())
    retain_user_items = get_user_items_dict(retain_samples)
    
    test_user_items = dataset.test_user_items
    test_users = list(test_user_items.keys())
    
    full_train_user_items = copy.deepcopy(retain_user_items)
    for u, items in forget_user_items.items():
        full_train_user_items[u].extend(items)
        
    original_test = dataset.test_user_items
    original_train = dataset.train_user_items
    
    results = {}
    
    try:
        # --- Evaluate Members (Forget Set) ---
        dataset.test_user_items = forget_user_items
        dataset.train_user_items = retain_user_items 
        
        member_results = evaluate_user_subset(model, dataset, forget_users, norm_adj_matrix, k_list, batch_size=2048)
        
        # --- Evaluate Non-Members (Test Set) ---
        dataset.test_user_items = test_user_items
        dataset.train_user_items = full_train_user_items
        
        non_member_results = evaluate_user_subset(model, dataset, test_users, norm_adj_matrix, k_list, batch_size=2048)
        
        for k in k_list:
            tpr = member_results[k]['recall']
            fpr = non_member_results[k]['recall']
            acc = 0.5 * (tpr + (1 - fpr))
            results[k] = acc
            
    finally:
        dataset.test_user_items = original_test
        dataset.train_user_items = original_train
        
    return results

def load_ngcf_model(dataset, path, device, conf, is_dpu=False):
    model = NGCF(
        n_users=dataset.n_users,
        n_items=dataset.n_items,
        embedding_size=conf.get('embedding_size', 64),
        n_layers=conf.get('n_layers', 3),
        reg_weight=conf.get('reg_weight', 1e-5),
        prompt_type='attention' if is_dpu else None,
        p_num=UNLEARNING_CONFIG.get('p_num', 20) if is_dpu else 20
    ).to(device)
    
    if path and os.path.exists(path):
        try:
            state = torch.load(path, map_location=device)
            if 'state_dict' in state: state = state['state_dict']
            model.load_state_dict(state, strict=False)
        except Exception as e:
            print(f"  Error loading: {e}")
    else:
        print(f"  Warning: Path not found {path}")
    return model

def load_lightgcn_model(dataset, path, device, conf, prompt_path=None):
    model = LightGCN(
        n_users=dataset.n_users,
        n_items=dataset.n_items,
        embedding_size=conf.get('embedding_size', 64),
        n_layers=conf.get('n_layers', 3), 
        reg_weight=conf.get('reg_weight', 1e-4)
    ).to(device)
    
    if path and os.path.exists(path):
        try:
            model.load_state_dict(torch.load(path, map_location=device))
        except Exception as e:
            print(f"  Warning: strict load failed, trying strict=False. {e}")
            model.load_state_dict(torch.load(path, map_location=device), strict=False)
        
    if prompt_path and os.path.exists(prompt_path):
        model = load_prompt_for_inference(
            base_model=model,
            prompt_path=prompt_path,
            dataset=dataset,
            n_layers=conf.get('n_layers', 3),
            reg_weight=conf.get('reg_weight', 1e-4),
            prompt_type='attention',
            embedding_size=64,
            p_num=20
        )
    return model

def run_mia(dataset_name, backbone, origin_path, retrain_path, dpu_path, dpu_prompt_path=None):
    set_seed(COMMON_CONFIG['seed'])
    device = COMMON_CONFIG['device']
    
    # Load Data
    data_path = os.path.join(project_root, "dataset", f"{dataset_name}.inter")
    print(f"Loading {dataset_name}...")
    dataset = ML100KDataset(data_path, test_size=0.1, min_item_count=5)
    
    # Split
    forget_ratio = UNLEARNING_CONFIG.get('forget_ratio', 0.01)
    forget_samples, retain_samples = split_forget_retain(dataset, forget_ratio)
    
    # Adj Matrix
    temp = LightGCN(dataset.n_users, dataset.n_items)
    adj = temp.get_norm_adj_mat(dataset.interaction_matrix.tocoo()).to(device)
    del temp
    
    conf = {'embedding_size': 64, 'n_layers': 3}
    
    # --- 1. Load Origin Model & Adaptive Filter ---
    print(f"\n--- Loading Origin Model for Filtering ---")
    if backbone.lower() == 'lightgcn':
        origin_model = load_lightgcn_model(dataset, origin_path, device, conf)
    else:
        origin_model = load_ngcf_model(dataset, origin_path, device, conf, is_dpu=False)
    
    # Adaptive Filtering
    filtered_forget_samples = adaptive_filter_samples(origin_model, dataset, forget_samples, retain_samples, adj, device=device)
    
    if len(filtered_forget_samples) == 0:
        print("Error: No samples found even with k=20. Exiting.")
        return

    # --- 2. Evaluate All Models ---
    
    dpu_p = dpu_path
    dpu_prompt = dpu_prompt_path
    
    # Special handling for LightGCN DPU: If dpu_path is given but no dpu_prompt_path, 
    # it implies dpu_path is the prompt checkpoint (or full DPU checkpoint) and we need Origin as base.
    if backbone.lower() == 'lightgcn' and dpu_path and not dpu_prompt_path:
        dpu_p = origin_path
        dpu_prompt = dpu_path

    models_to_eval = [
        ('Origin', origin_path, None),
        ('Retrain', retrain_path, None),
        ('DPU', dpu_p, dpu_prompt)
    ]
    
    print(f"\n=== Top-K MIA Evaluation (Targeted) [{dataset_name} - {backbone}] ===")
    print(f"{'Model':<10} | {'ACC@1':<10} | {'ACC@10':<10}")
    print("-" * 36)
    
    # Evaluate Origin (using cached model)
    res = calculate_topk_mia_accuracy(origin_model, dataset, filtered_forget_samples, retain_samples, adj, [1, 10], device)
    print(f"{'Origin':<10} | {res[1]:.4f}     | {res[10]:.4f}")
    
    del origin_model
    torch.cuda.empty_cache()

    for name, path, prompt in models_to_eval[1:]: # Skip Origin
        if not path and not prompt: continue
        
        if backbone.lower() == 'lightgcn':
            model = load_lightgcn_model(dataset, path, device, conf, prompt)
        else:
            is_dpu = (name == 'DPU')
            p = path if path else prompt
            # For NGCF DPU, we just direct load the path
            model = load_ngcf_model(dataset, p, device, conf, is_dpu)
            
        res = calculate_topk_mia_accuracy(model, dataset, filtered_forget_samples, retain_samples, adj, [1, 10], device)
        
        print(f"{name:<10} | {res[1]:.4f}     | {res[10]:.4f}")
        
        del model
        torch.cuda.empty_cache()

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='yelp')
    parser.add_argument('--backbone', type=str, default='ngcf')
    parser.add_argument('--origin_path', type=str, default='')
    parser.add_argument('--retrain_path', type=str, default='')
    parser.add_argument('--dpu_path', type=str, default='')
    parser.add_argument('--dpu_prompt_path', type=str, default='')
    
    args = parser.parse_args()
    run_mia(args.dataset, args.backbone, args.origin_path, args.retrain_path, args.dpu_path, args.dpu_prompt_path)
