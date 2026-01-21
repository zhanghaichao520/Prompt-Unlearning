# -*- coding: utf-8 -*-
import sys
import os
import torch
import numpy as np
import random
import time

# Add root to path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, "../../"))
sys.path.append(project_root)

from data import ML100KDataset
from ngcf_experiment.RecEraser.rec_eraser import RecEraser
from ngcf_experiment.RecEraser.ngcf_models import NGCF
from utils import recall, ndcg, hit_rate, print_metrics, calculate_performance_change
from config import COMMON_CONFIG
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F
from tqdm import tqdm

def calculate_zrf(rec_eraser, incompetent_teacher, forget_data, device):
    print("Calculating ZRF...")
    incompetent_teacher.eval() # Teacher is also RecEraser (untrained) or NGCF
    
    # Forget Dataset
    class ListDataset(Dataset):
        def __init__(self, samples):
            self.samples = samples
        def __len__(self):
            return len(self.samples)
        def __getitem__(self, idx):
            return self.samples[idx] # (u, i)
            
    loader = DataLoader(ListDataset(forget_data), batch_size=4096, shuffle=False)
    
    js_divergences = []
    
    with torch.no_grad():
        for batch in tqdm(loader, desc="ZRF"):
            users = batch[0].to(device)
            items = batch[1].to(device)
            
            # RecEraser Prediction
            unlearned_logits = rec_eraser.predict(users, items)
            
            # Incompetent Prediction
            if hasattr(incompetent_teacher, 'partition'): # Is RecEraser
                 incompetent_logits = incompetent_teacher.predict(users, items)
            else:
                 incompetent_logits = incompetent_teacher.predict(users, items)

            # JS Div
            p_un = torch.sigmoid(unlearned_logits)
            p_in = torch.sigmoid(incompetent_logits)
            
            # Stack [p, 1-p]
            dist_un = torch.stack([p_un, 1-p_un], dim=1)
            dist_in = torch.stack([p_in, 1-p_in], dim=1)
            
            m = 0.5 * (dist_un + dist_in)
            m_log = torch.log(m.clamp(min=1e-7))
            
            kl_1 = F.kl_div(m_log, dist_un, reduction='none').sum(dim=1)
            kl_2 = F.kl_div(m_log, dist_in, reduction='none').sum(dim=1)
            
            js = 0.5 * (kl_1 + kl_2)
            js_divergences.extend(js.cpu().numpy())
            
    if not js_divergences:
        return 0.0
    return 1.0 - np.mean(js_divergences)


def evaluate_model(model, test_user_items, train_user_items, k_list, batch_size, device):
    results = {k: {'recall': 0.0, 'ndcg': 0.0, 'hit_rate': 0.0} for k in k_list}
    test_users = list(test_user_items.keys())
    n_test_users = len(test_users)
    
    with torch.no_grad():
        for start in range(0, n_test_users, batch_size):
            end = min(start + batch_size, n_test_users)
            batch_users = test_users[start:end]
            batch_users_tensor = torch.LongTensor(batch_users).to(device)
            
            # Predict
            rating_pred = model.predict(batch_users_tensor)
            rating_pred = rating_pred.cpu().numpy()
            
            # Evaluate
            for idx, user in enumerate(batch_users):
                user_pos_items = test_user_items[user]
                if len(user_pos_items) == 0:
                    continue
                
                # Filter train items
                train_items = train_user_items.get(user, set())
                user_pred = rating_pred[idx]
                user_pred[list(train_items)] = -np.inf
                
                rank_indices = np.argsort(-user_pred)
                
                for k in k_list:
                    results[k]['recall'] += recall(rank_indices, user_pos_items, k)
                    results[k]['ndcg'] += ndcg(rank_indices, user_pos_items, k)
                    results[k]['hit_rate'] += hit_rate(rank_indices, user_pos_items, k)
                    
    for k in k_list:
        for metric in results[k]:
            results[k][metric] /= n_test_users
            
    return results

def evaluate(rec_eraser, dataset, k=[10, 20]):
    test_user_items = dataset.test_user_items
    train_user_items = dataset.train_user_items 
    return evaluate_model(rec_eraser, test_user_items, train_user_items, k, 4096, rec_eraser.device)

def main():
    # Config
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    config = {
        'root': os.path.join(project_root, 'ngcf_experiment', 'RecEraser', 'Yelp_Experiment'),
        'num_shards': 10, 
        'embedding_size': 64,
        'n_layers': 3,
        'lr': 0.001,
        'shard_epochs': 100, 
        'seed': 2023
    }
    
    # Set Seed
    random.seed(config['seed'])
    np.random.seed(config['seed'])
    torch.manual_seed(config['seed'])
    
    # Load Dataset
    print("Loading Yelp Dataset...")
    absolute_data_path = os.path.join(project_root, "dataset", "yelp.inter")
    dataset = ML100KDataset(absolute_data_path, test_size=0.1, min_item_count=5)
    
    print(f"Users: {dataset.n_users}, Items: {dataset.n_items}, Interactions: {len(dataset.train_samples)}")
    
    # Init RecEraser
    rec_eraser = RecEraser(dataset, config, device)
    
    # Try Load Checkpoints
    if rec_eraser.load_checkpoints():
        print("Loaded existing checkpoints.")
    else:
        # Partition
        pretrained_path = os.path.join(project_root, 'pretrain_checkpoints', 'ngcf_yelp_full.pth')
        if not os.path.exists(pretrained_path):
            print(f"Warning: Pretrained model not found at {pretrained_path}. Partitioning might be random.")
        
        start_time = time.time()
        rec_eraser.partition(pretrained_path)
        print(f"Partitioning took {time.time() - start_time:.2f}s")
        
        # Train Shards
        print("Training Shards...")
        start_time = time.time()
        rec_eraser.train_all_shards()
        print(f"Shard Training took {time.time() - start_time:.2f}s")
        
        # Train Aggregator
        print("Training Aggregator...")
        start_time = time.time()
        rec_eraser.train_aggregator(epochs=10) 
        print(f"Aggregator Training took {time.time() - start_time:.2f}s")
    
    # ---------------------------------------------------------
    # Unlearning Task Setup
    # ---------------------------------------------------------
    
    # Select target users to unlearn.
    target_users = np.random.choice(list(dataset.train_user_items.keys()), 2, replace=False)
    print(f"Unlearning Users: {target_users}")
    
    forget_data = []
    for u in target_users:
        items = dataset.train_user_items[u]
        for i in items:
            forget_data.append((u, i))
            
    print(f"Total interactions to forget: {len(forget_data)}")
    
    # Format for evaluation
    forget_user_items = {}
    for u, i in forget_data:
        if u not in forget_user_items: forget_user_items[u] = []
        forget_user_items[u].append(i)

    # Evaluate Original Performance
    print("\nEvaluating Original Performance...")
    
    # Retain (Test Set)
    print("Evaluating on Retain Set (Test Data)...")
    original_retain = evaluate(rec_eraser, dataset)
    print("Original Retain:")
    print_metrics(original_retain)
    
    # Forget
    print("Evaluating on Forget Set...")
    original_forget = evaluate_model(rec_eraser, forget_user_items, {}, [10, 20], 4096, rec_eraser.device)
    print("Original Forget:")
    print_metrics(original_forget)
    
    
    # Update global dataset before calling unlearn
    original_len = len(dataset.train_samples)
    forget_set = set([(x[0], x[1]) for x in forget_data])
    new_samples = [x for x in dataset.train_samples if (x[0], x[1]) not in forget_set]
    dataset.train_samples = new_samples
    
    for u in target_users:
        if u in dataset.train_user_items:
             del dataset.train_user_items[u]

    print(f"Dataset samples reduced from {original_len} to {len(dataset.train_samples)}")
    
    # Run Unlearning
    start_time = time.time()
    rec_eraser.unlearn(forget_data)
    print(f"Unlearning took {time.time() - start_time:.2f}s")
    
    # Evaluate Unlearned
    print("\nEvaluating Unlearned Performance...")
    
    print("Evaluating on Retain Set...")
    unlearned_retain = evaluate(rec_eraser, dataset)
    print("Unlearned Retain:")
    print_metrics(unlearned_retain)
    
    print("Evaluating on Forget Set...")
    unlearned_forget = evaluate_model(rec_eraser, forget_user_items, {}, [10, 20], 4096, rec_eraser.device)
    print("Unlearned Forget:")
    print_metrics(unlearned_forget)
    
    print("\n=== Performance Changes ===")
    for k in [10, 20]:
        print(f"K={k}")
        changes_ret = calculate_performance_change(original_retain, unlearned_retain, k)
        print(f"  Retain Set: Recall: {changes_ret['recall']:.2f}%, NDCG: {changes_ret['ndcg']:.2f}%, HitRate: {changes_ret['hit_rate']:.2f}%")
        
        changes_for = calculate_performance_change(original_forget, unlearned_forget, k)
        print(f"  Forget Set: Recall: {changes_for['recall']:.2f}%, NDCG: {changes_for['ndcg']:.2f}%, HitRate: {changes_for['hit_rate']:.2f}%")

    # Calculate ZRF
    print("Calculating ZRF...")
    incompetent_teacher = RecEraser(dataset, config, device)
    incompetent_teacher.init_random()
    
    zrf = calculate_zrf(rec_eraser, incompetent_teacher, forget_data, device)
    print(f"ZRF: {zrf:.4f}")
    
    # Save Results
    with open("receraser_yelp_results.txt", "w") as f:
         f.write(f"ZRF: {zrf:.4f}\n")

if __name__ == "__main__":
    main()
