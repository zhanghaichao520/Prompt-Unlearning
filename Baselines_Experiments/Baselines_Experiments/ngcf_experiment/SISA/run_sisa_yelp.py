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
from sisa import SISAManager
from ngcf_models import NGCF

# Set device
device = COMMON_CONFIG['device']
print(f"Using device: {device}")

# Config (Optimized for performance)
SISA_CONFIG = {
    'embedding_size': 64,
    'n_layers': 3,
    'reg_weight': 1e-5,
    'node_dropout': 0.1,
    'message_dropout': 0.1,
    'lr': 0.001,
    'epochs': 50, 
    'batch_size': 4096, 
    'seed': COMMON_CONFIG['seed']
}

NUM_SHARDS = 10

if __name__ == "__main__":
    set_seed(SISA_CONFIG['seed'])
    
    # 1. Load Data
    absolute_data_path = os.path.join(project_root, "dataset", "yelp.inter")
    print(f"Loading Yelp from {absolute_data_path}")
    dataset = ML100KDataset(absolute_data_path, test_size=0.1, min_item_count=5)
    
    # 2. Init SISA Manager
    save_dir = "checkpoints_sisa_yelp"
    sisa = SISAManager(dataset, NUM_SHARDS, device, SISA_CONFIG, save_dir)
    

    # 2. Init SISA Manager
    save_dir = "checkpoints_sisa_yelp"
    sisa = SISAManager(dataset, NUM_SHARDS, device, SISA_CONFIG, save_dir)
    
    # 3. Simulate "Original" Training - SKIPPING if checkpoints exist
    # If the user says "models are trained", we assume the checkpoints are the UNLEARNED models.
    # To properly calculate ZRF, we need to reproduce the unlearning state (data distribution).
    
    full_train_samples = dataset.train_samples
    print(f"Total Training Samples: {len(full_train_samples)}")
    
    # Init Random Teacher for ZRF
    print("\n=== Initializing Random Teacher (For ZRF) ===")
    incompetent = NGCF(
        dataset.n_users, dataset.n_items, 
        embedding_size=SISA_CONFIG['embedding_size'],
        n_layers=SISA_CONFIG['n_layers']
    ).to(device)
    
    # 4. Define Forget Set (Must match previous run!)
    forget_ratio = UNLEARNING_CONFIG.get('forget_ratio', 0.01)
    print(f"\nDefining Forget Set (Ratio={forget_ratio})...")
    
    # Important: Use same logic/seed as original run
    all_users = list(dataset.train_user_items.keys())
    # Sort first to ensure shuffle is deterministic across runs provided seed is set
    all_users.sort() 
    
    rs_forget = np.random.RandomState(SISA_CONFIG['seed'])
    rs_forget.shuffle(all_users)
    n_forget = int(len(all_users) * forget_ratio)
    forget_users = all_users[:n_forget]
    print(f"Forgetting {len(forget_users)} users.")
    
    # Get Forget Samples for ZRF
    forget_samples = []
    forget_users_set = set(forget_users)
    for u in forget_users:
        # Reconstruct interactions
        if u in dataset.train_user_items:
            for i in dataset.train_user_items[u]:
                forget_samples.append((u, i))
    
    # 5. Load State (Simulate Unlearning without Training)
    print("\n=== Loading / Simulating SISA Unlearning State ===")
    
    # First, must distribute FULL data to establish baseline mapping and valid NormAdj logic
    sisa.distribute_data(full_train_samples)
    
    # Now remove forget users from shard_data to match the state of the saved checkpoints
    # This updates sisa.shard_data so _create_norm_adj works correctly for the unlearned models
    affected_shards = set()
    for u in forget_users:
        if u in sisa.user_shard_map:
            affected_shards.add(sisa.user_shard_map[u])
            
    print(f"Applying data filtering for {len(affected_shards)} affected shards...")
    forget_set_ids = set(forget_users)
    for sid in affected_shards:
        new_data = [s for s in sisa.shard_data[sid] if s[0] not in forget_set_ids]
        sisa.shard_data[sid] = new_data
        
    # Now load models (Checkpoints should exist)
    # We Iterate all shards to ensure they are loaded
    for i in range(NUM_SHARDS):
        try:
            sisa.load_shard(i)
        except Exception as e:
            print(f"Failed to load shard {i}: {e}. Retraining...")
            sisa.train_shard(i)
            
    # 6. Final Evaluation
    print("\n=== Final Evaluation (Unlearned Model) ===")
    
    # A. Performance on Forget Set
    print("Performance on Forget Set (Should be low/random):")
    forget_res = sisa.evaluate(forget_users, COMMON_CONFIG['k_list'])
    print_metrics(forget_res, prefix="  ")
    
    # B. Retain Performance
    retain_users = [u for u in dataset.get_test_samples() if u not in forget_users_set]
    print(f"Performance on Retain Set ({len(retain_users)} users):")
    retain_res = sisa.evaluate(retain_users, COMMON_CONFIG['k_list'])
    print_metrics(retain_res, prefix="  ")

    # C. ZRF Score
    zrf = sisa.calculate_zrf_sisa(incompetent, forget_samples, batch_size=256)
    print(f"\nZRF Score: {zrf:.4f}")
    
    # Save Results
    with open("sisa_yelp_results.txt", "w") as f:
        f.write(f"ZRF: {zrf:.4f}\n")
