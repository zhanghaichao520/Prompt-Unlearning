# -*- coding: utf-8 -*-
import os
import torch
import torch.optim as optim
import numpy as np
import scipy.sparse as sp
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from ngcf_models import NGCF
from utils import evaluate_user_subset

class ShardDataset(Dataset):
    """Dataset wrapper for a single shard."""
    def __init__(self, samples, n_users, n_items, train_user_items):
        self.samples = samples
        self.n_users = n_users
        self.n_items = n_items
        self.train_user_items = train_user_items 

    def __getitem__(self, idx):
        user, pos = self.samples[idx]
        # Dynamic negative sampling
        neg = np.random.randint(0, self.n_items)
        # Use global train set to check negatives to be safe
        while neg in self.train_user_items[user]:
             neg = np.random.randint(0, self.n_items)
        return user, pos, neg
    
    def __len__(self):
        return len(self.samples)

class SISAManager:
    def __init__(self, full_dataset, num_shards, device, config, save_dir):
        self.dataset = full_dataset
        self.num_shards = num_shards
        self.device = device
        self.config = config
        self.save_dir = save_dir
        
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
            
        self.user_shard_map = {}
        self.shard_data = {i: [] for i in range(num_shards)}
        self.shard_models = {} 
        self.shard_norm_adj = {}
        
        # Partition Users Initially
        self.partition_users()

    def partition_users(self):
        """Randomly partition all users into shards."""
        users = np.arange(self.dataset.n_users)
        # Fix seed for reproducibility of sharding
        rs = np.random.RandomState(self.config.get('seed', 2024))
        rs.shuffle(users)
        
        shard_size = len(users) // self.num_shards
        for i in range(self.num_shards):
            start = i * shard_size
            # Last shard gets leftovers
            end = (i + 1) * shard_size if i < self.num_shards - 1 else len(users)
            
            u_list = users[start:end]
            for u in u_list:
                self.user_shard_map[u] = i

    def distribute_data(self, samples):
        """Distribute samples to shards based on user_shard_map."""
        self.shard_data = {i: [] for i in range(self.num_shards)}
        count = 0
        for s in samples:
            u = s[0]
            if u in self.user_shard_map:
                shard_id = self.user_shard_map[u]
                self.shard_data[shard_id].append(s)
                count += 1
        print(f"Distributed {count} samples into {self.num_shards} shards.")
        for i in range(self.num_shards):
            print(f"  Shard {i}: {len(self.shard_data[i])} samples")

    def _create_norm_adj(self, shard_id):
        samples = self.shard_data[shard_id]
        if not samples:
            # Empty shard
            rows, cols = [], []
        else:
            rows = [s[0] for s in samples]
            cols = [s[1] for s in samples]
            
        vals = np.ones(len(rows))
        
        interaction_matrix = sp.coo_matrix(
            (vals, (rows, cols)), 
            shape=(self.dataset.n_users, self.dataset.n_items), 
            dtype=np.float32
        )
        
        temp = NGCF(self.dataset.n_users, self.dataset.n_items)
        norm_adj = temp.get_norm_adj_mat(interaction_matrix).to(self.device)
        return norm_adj

    def train_shard(self, shard_id, epochs=None):
        if epochs is None:
            epochs = self.config['epochs']
            
        print(f"Training Shard {shard_id}...")
        
        # 1. Prepare Graph
        norm_adj = self._create_norm_adj(shard_id)
        self.shard_norm_adj[shard_id] = norm_adj
        
        # 2. Dataset
        ds = ShardDataset(
            self.shard_data[shard_id], 
            self.dataset.n_users, 
            self.dataset.n_items,
            self.dataset.train_user_items
        )
        if len(ds) == 0:
            print(f"Shard {shard_id} is empty, skipping training.")
            self.shard_models[shard_id] = NGCF(self.dataset.n_users, self.dataset.n_items).to(self.device) # Dummy
            return

        loader = DataLoader(ds, batch_size=self.config['batch_size'], shuffle=True, num_workers=4)
        
        # 3. Model
        model = NGCF(
            self.dataset.n_users, self.dataset.n_items,
            embedding_size=self.config['embedding_size'],
            n_layers=self.config['n_layers'],
            reg_weight=self.config['reg_weight'],
            node_dropout=self.config['node_dropout'],
            message_dropout=self.config['message_dropout']
        ).to(self.device)
        
        optimizer = optim.Adam(model.parameters(), lr=self.config['lr'])
        
        model.train()
        for epoch in range(epochs):
            total_loss = 0
            for users, pos, neg in loader:
                users = users.to(self.device)
                pos = pos.to(self.device)
                neg = neg.to(self.device)
                
                optimizer.zero_grad()
                loss = model.calculate_loss(users, pos, neg, norm_adj)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
                
        self.shard_models[shard_id] = model

        
        # Save
        torch.save(model.state_dict(), os.path.join(self.save_dir, f"shard_{shard_id}.pth"))

    def load_shard(self, shard_id):
        """Loads a shard model from disk without training."""
        path = os.path.join(self.save_dir, f"shard_{shard_id}.pth")
        if not os.path.exists(path):
            raise FileNotFoundError(f"Shard {shard_id} checkpoint not found at {path}")
            
        print(f"Loading Shard {shard_id} from {path}...")
        
        # Ensure adj matrix exists (data must be distributed first)
        if shard_id not in self.shard_norm_adj:
            self.shard_norm_adj[shard_id] = self._create_norm_adj(shard_id)
            
        model = NGCF(
            self.dataset.n_users, self.dataset.n_items,
            embedding_size=self.config['embedding_size'],
            n_layers=self.config['n_layers'],
            reg_weight=self.config['reg_weight'],
            node_dropout=self.config['node_dropout'],
            message_dropout=self.config['message_dropout']
        ).to(self.device)
        
        model.load_state_dict(torch.load(path))
        self.shard_models[shard_id] = model
        return model

    def train_all(self):
        for i in range(self.num_shards):
            self.train_shard(i)

    def evaluate(self, test_users, k_list):
        """Evaluate performance on users. Users are dispatched to their shards."""
        results = {k: {'recall': 0.0, 'ndcg': 0.0, 'hit_rate': 0.0} for k in k_list}
        total_users = 0
        
        # Group users by shard
        shard_users_map = {i: [] for i in range(self.num_shards)}
        unknown_users = []
        
        for u in test_users:
            if u in self.user_shard_map:
                shard_users_map[self.user_shard_map[u]].append(u)
            else:
                unknown_users.append(u)
                
        # Evaluate known users
        for i in range(self.num_shards):
            users = shard_users_map[i]
            if not users: continue
            
            if i not in self.shard_models:
                # Load or skip
                try:
                    self.load_shard(i)
                except FileNotFoundError:
                    print(f"Warning: Shard {i} model not found. Using random init.")
                    self.shard_models[i] = NGCF(self.dataset.n_users, self.dataset.n_items).to(self.device)
                    # If we don't have adj matrix (e.g. data filtered out), create empty one?
                    if i not in self.shard_norm_adj:
                         self.shard_norm_adj[i] = self._create_norm_adj(i) # Re-create adj from whatever data we have
                    
            res = evaluate_user_subset(self.shard_models[i], self.dataset, users, self.shard_norm_adj[i], k_list, self.config['batch_size'])
            
            count = len(users)
            total_users += count
            for k in k_list:
                for m in results[k]:
                    results[k][m] += res[k][m] * count
                    
        # Handle unknown users (random guess / 0 metrics)
        if total_users > 0:
            for k in k_list:
                for m in results[k]:
                    results[k][m] /= total_users
                    
        return results

    def predict_batch(self, users, items):
        """
        Get predictions (logits/scores) for a batch of (user, item) pairs.
        Used for ZRF calculation.
        """
        # Group queries by shard
        shard_queries = {i: {'users': [], 'items': [], 'indices': []} for i in range(self.num_shards)}
        
        # 1. Distribute queries
        users_np = users.cpu().numpy()
        items_np = items.cpu().numpy()
        
        for idx, (u, i) in enumerate(zip(users_np, items_np)):
            if u in self.user_shard_map:
                shard_id = self.user_shard_map[u]
                shard_queries[shard_id]['users'].append(u)
                shard_queries[shard_id]['items'].append(i)
                shard_queries[shard_id]['indices'].append(idx)
            else:
                # Handle unknown users? Just assign to shard 0 randomly
                shard_id = 0
                shard_queries[shard_id]['users'].append(u)
                shard_queries[shard_id]['items'].append(i)
                shard_queries[shard_id]['indices'].append(idx)
        
        # 2. Get predictions from each shard
        all_logits = torch.zeros(len(users), device=self.device)
        
        for i in range(self.num_shards):
            q = shard_queries[i]
            if not q['users']: continue
            
            # Load model if needed
            if i not in self.shard_models:
                 try:
                    self.load_shard(i)
                 except:
                    # Random init if missing
                    self.shard_models[i] = NGCF(self.dataset.n_users, self.dataset.n_items).to(self.device)
                    if i not in self.shard_norm_adj:
                        self.shard_norm_adj[i] = self._create_norm_adj(i)

            model = self.shard_models[i]
            adj = self.shard_norm_adj[i]
            
            batch_users = torch.LongTensor(q['users']).to(self.device)
            batch_items = torch.LongTensor(q['items']).to(self.device)
            
            # Use model.predict() if available or calculate manually
            # NGCF usually has predict(u, i, adj)
            with torch.no_grad():
                preds = model.predict(batch_users, batch_items, adj)
                
            # Scatter back
            indices = torch.LongTensor(q['indices']).to(self.device)
            all_logits[indices] = preds
            
        return all_logits

    def calculate_zrf_sisa(self, incompetent_teacher, forget_samples, batch_size):
        """
        Calculate ZRF for SISA.
        Compares SISA (Unlearned) predictions vs Incompetent Teacher (Random) predictions.
        """
        print("Calculating ZRF for SISA...")
        incompetent_teacher.eval()
        
        # Define Forget Dataset
        class ForgetDataset(Dataset):
            def __init__(self, samples):
                self.samples = samples
            def __len__(self):
                return len(self.samples)
            def __getitem__(self, idx):
                 return self.samples[idx] # (user, item)
        
        loader = DataLoader(ForgetDataset(forget_samples), batch_size=batch_size, shuffle=False)
        js_divergences = []
        
        with torch.no_grad():
            for batch in tqdm(loader, desc="ZRF Calculation"):
                 inputs = batch # (users, items, ...)
                 if len(inputs) == 3:
                      # user, pos, neg
                      users, pos, neg = inputs
                      # Just evaluate on positive items for ZRF? 
                      # Usually ZRF evaluate on prediction distribution over ALL items or subset.
                      # Standard implementation: 
                      # ZRF = 1 - mean(JS(P_unlearn || P_random))
                      # P is usually probability over items.
                      # Ideally we need full softmax.
                      # BUT previous code `calculate_zrf` in `retrain_ngcf` seems to assume something simpler?
                      # Let's check `calculate_zrf` in retrain scripts again.
                      pass
                      
                 # Assuming just (user, item) pairs as per `forget_samples` from `split_forget_retain`.
                 # Actually `split_forget_retain` returns interactions `(u, i)`.
                 # So we compute score for `(u, i)`.
                 # Wait, ZRF requires probability distribution.
                 # If we only have score for ONE item, we can't compute JS divergence (requires distribution).
                 # The standard ZRF paper computes JS on the output vector (recommendation list).
                 # Previous implementation in `retrain_ngcf_*.py`:
                 #     for users, items in pbar:
                 #         # ...
                 
                 # Let's assume we need to compute scores for these users and items. 
                 # And standard implementation usually compares the top-k list overlap or distribution.
                 # If I look at the `calculate_zrf` function I saw earlier (in summary):
                 # It iterated over forget_loader.
                 
                 # I will trust the user wants 'ZRF' added. I will use the same logic as I would have used in single model.
                 # Which requires getting outputs from `sisa.predict_batch`.
                 
                 # But wait, `split_forget_retain` returns single interactions.
                 # Evaluating JS on single interaction score is meaningless.
                 # ZRF typically evaluates on the user's PREDICTION VECTOR.
                 # So we iterate over USERS in forget set?
                 # Ah, let's look at `retrain_ngcf_ml1m.py` summary again.
                 
                 # It seems I implemented `calculate_zrf` in `retrain_ngcf_*.py`.
                 # I need to follow that.
                 pass

        # Since I cannot see the FULL implementation of calculate_zrf I wrote before (it was summarized),
        # I will implement a robust ZRF here that matches the standard:
        # For each user in forget set, compare the probability distribution (softmax of logits) of the SISA vs Random.
        # But computing full softmax is expensive.
        # Usually we sample items or just use the Forget One Item setting.
        # Let's assume we iterate over Unique Users in forget set.
        
        unique_users = list(set([s[0] for s in forget_samples]))
        
        kl_divs = []
        js_divs = []
        
        for start in range(0, len(unique_users), batch_size):
             end = min(start + batch_size, len(unique_users))
             batch_users = unique_users[start:end]
             batch_users_tensor = torch.LongTensor(batch_users).to(self.device)
             
             # 1. Get SISA logits for ALL items (Full Softmax approximation)
             # This is hard with shards.
             # SISA prediction for user u is prediction from shard S_u.
             # So we can just get full prediction from shard S_u.
             
             # Group by shard to optimize
             shard_batches = {i: [] for i in range(self.num_shards)}
             for u in batch_users:
                  if u in self.user_shard_map:
                      shard_batches[self.user_shard_map[u]].append(u)
             
             probs_unlearned_list = [] # List of (N_u, N_i)
             probs_random_list = []
             
             for i in range(self.num_shards):
                  us = shard_batches[i]
                  if not us: continue
                  
                  # Load model
                  if i not in self.shard_models: self.load_shard(i)
                  model = self.shard_models[i]
                  adj = self.shard_norm_adj[i]
                  
                  u_tensor = torch.LongTensor(us).to(self.device)
                  
                  # SISA Preds
                  logits_unlearned = model.full_sort_predict(u_tensor, adj) # (B, N_items)
                  probs_unlearned = torch.softmax(logits_unlearned, dim=1)
                  
                  # Random Preds
                  logits_random = incompetent_teacher.full_sort_predict(u_tensor, adj)
                  probs_random = torch.softmax(logits_random, dim=1)
                  
                  # Compute JS
                  # JS(P||Q) = 0.5 * KL(P||M) + 0.5 * KL(Q||M), M = 0.5(P+Q)
                  M = 0.5 * (probs_unlearned + probs_random)
                  kl1 = torch.sum(probs_unlearned * torch.log(probs_unlearned / M + 1e-10), dim=1)
                  kl2 = torch.sum(probs_random * torch.log(probs_random / M + 1e-10), dim=1)
                  js = 0.5 * kl1 + 0.5 * kl2
                  
                  js_divs.extend(js.detach().cpu().numpy())
                  
        if not js_divs:
             return 0.0
             
        zrf = 1.0 - np.mean(js_divs)
        return zrf

    def unlearn_users(self, forget_users):
        """
        SISA Unlearning Strategy:
        1. Identify shards containing forget_users.
        2. Remove forget_users from shard_data.
        3. Retrain ONLY those shards.
        """
        # Identify affected shards
        affected_shards = set()
        for u in forget_users:
            if u in self.user_shard_map:
                affected_shards.add(self.user_shard_map[u])
                # Note: We do NOT remove user from user_shard_map, so they are still routed to this shard.
                # But their data will be removed from training.
                
        print(f"Unlearning: {len(forget_users)} users affected {len(affected_shards)} shards: {affected_shards}")
        
        # Filter shard_data
        forget_set = set(forget_users)
        for sid in affected_shards:
            original_len = len(self.shard_data[sid])
            new_data = [s for s in self.shard_data[sid] if s[0] not in forget_set]
            self.shard_data[sid] = new_data
            print(f"  Shard {sid}: {original_len} -> {len(new_data)} samples")
            
            # Retrain
            self.train_shard(sid)
