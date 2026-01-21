import numpy as np
import scipy.sparse as sp
import torch
from sklearn.cluster import KMeans

def get_pretrained_embeddings(model, dataset, device):
    """
    Extracts embeddings from a trained NGCF model.
    """
    model.eval()
    with torch.no_grad():
        # Get ego embeddings (before GNN propagation, or after? Paper uses WMF embeddings)
        # "The embedding vectors are computed by a pretrained model (WMF...)"
        # NGCF user_embedding is the base embedding E^0. GNN outputs E^L.
        # Usually E^L contains better structural info. Let's use get_norm_adj_mat + forward
        # But for partitioning, simple embeddings might be enough. 
        # Let's use the final embeddings from model.
        
        # Construct adj for full sort predict (full graph)
        # Note: calling internal helper if available or we need to pass adj
        # We assume model has access to data or we pass it? 
        # The model class in ngcf_models.py needs adj passed to forward.
        pass 
        
    # Simplified: Just grab the embedding tables directly. 
    # For partition purposes, the learned latent factors are what matters.
    u_emb = model.user_embedding.weight.detach().cpu().numpy()
    i_emb = model.item_embedding.weight.detach().cpu().numpy()
    return u_emb, i_emb

def balanced_interaction_partition(train_data, u_emb, i_emb, num_shards, max_shard_size):
    """
    Interaction-based Balanced Partition (InBP).
    algo 2 in RecEraser paper.
    
    Args:
        train_data: List of [u, i] or [u, i, rating]
        u_emb: User embeddings [n_users, dim]
        i_emb: Item embeddings [n_items, dim]
        num_shards: K
        max_shard_size: t
    Returns:
        shard_indices: Dict {shard_id: [data_indices]} or List of List of data
    """
    print(f"Starting InBP Partitioning into {num_shards} shards...")
    
    num_data = len(train_data)
    # Extract interactions in embedding space
    # To save memory, we don't construct the full N x 2D matrix.
    # We iterate.
    
    # 1. Randomly select K anchors from Y
    # An anchor is a pair (u, i), represented by (p_u, q_i).
    anchor_indices = np.random.choice(num_data, num_shards, replace=False)
    anchors = []
    for idx in anchor_indices:
        u, i = train_data[idx][0], train_data[idx][1]
        anchors.append((u_emb[u], i_emb[i]))
        
    # Assign data to shards
    # Since we need to balance, we sort by distance to anchors?
    # Paper: "Calculate dist... Sort E in ascending order... if |Si| < t... assign"
    
    # This loop in paper "while Stopping criteria..." is K-Means like.
    # But with capacity constraint.
    
    # Optimization: processing 1M interactions in python loop is slow.
    # We use numpy broadcasting if possible.
    
    # Interaction embeddings: Concatenation? Paper says dist = ||pi - pu||^2 * ||qi - qv||^2
    # Eq 3: dist(ai, yuv) = ||pi - pu||^2 + ||qi - qv||^2 ???
    # Paper Eq 3: "dist = || ... ||^2 * || ... ||^2" (multiplication)
    # Wait, usually distance is additive. Let's re-read carefully if I could.
    # The prompt text says: "dist(ai, yuv) = ||pi - pu||^2 * ||qi - qv||^2"
    # Wait, text says multiplication sign nearby linear summation formula?
    # "Sum ... * Sum ..."
    # Yes, it looks like element-wise dist product or similar.
    # Let's assume standard Euclidean distance in composed space for stability if product is weird. 
    # But I will follow the text: "Calculated as ... * ..."
    
    # Actually, for standard clustering of edges, usually we concat embeddings.
    # Let's implement a simplified version: Concat(u, i) and run Balanced K-Means.
    
    # Construct feature matrix for interactions
    # users = [x[0] for x in train_data]
    # items = [x[1] for x in train_data]
    
    # feat_u = u_emb[users]
    # feat_i = i_emb[items]
    # feats = np.concatenate([feat_u, feat_i], axis=1) # [N, 2D]
    
    # Run Balanced KMeans on feats
    # Since writing custom balanced kmeans is error prone, we can use a library or simple heuristic.
    # The paper's algorithm 2 is explicit constraints based assignment.
    
    assignments = [-1] * num_data
    shards = {k: [] for k in range(num_shards)}
    
    # Iterating K-Means is slow for custom balanced logic in Python.
    # We will run 1 iteration or a few.
    
    # Anchor update: Average of p in shard, Average of q in shard.
    
    max_iter = 5
    for iteration in range(max_iter):
        print(f"  Iteration {iteration+1}/{max_iter}")
        # Calculate distances to all anchors
        # P_anchors: [K, D], Q_anchors: [K, D]
        P_anchors = np.array([a[0] for a in anchors])
        Q_anchors = np.array([a[1] for a in anchors])
        
        # We need distance from every data point j to every anchor k.
        # User dist: ||P_data - P_anchors_k||^2
        # Item dist: ||Q_data - Q_anchors_k||^2
        
        # We can implement this batch-wise to save memory
        batch_size = 10000
        all_dists = [] # Tuples (dist, data_idx, anchor_idx) -> Actually we need to sort ALL pairs.
        # Sorting N*K pairs is expensive (1M * 10 = 10M, feasible).
        
        # Pre-calculate data vector arrays if not too large
        # len(train_data) ~ 1M for ML-1M. 1M * 64 float = 64MB. X2 = 128MB. Feasible in memory.
        users_idx = np.array([x[0] for x in train_data])
        items_idx = np.array([x[1] for x in train_data])
        
        P_data = u_emb[users_idx]
        Q_data = i_emb[items_idx]
        
        # Compute distances matrix [N, K]
        # |x - y|^2 = |x|^2 + |y|^2 - 2xy
        
        # User component
        # P_data_sq = (P_data**2).sum(axis=1, keepdims=True)
        # P_anch_sq = (P_anchors**2).sum(axis=1)
        # P_dist = P_data_sq + P_anch_sq - 2 * np.dot(P_data, P_anchors.T)
        
        # Metric from paper: product of L2 norms? 
        # "dist = ||pi - pu||^2 * ||qi - qv||^2"
        # Let's use this EXACTLY.
        
        # Helper for L2 squared
        def dist_sq(X, Y):
            # X: [N, D], Y: [K, D] -> [N, K]
            return (X**2).sum(1, keepdims=True) + (Y**2).sum(1) - 2 * X @ Y.T
            
        d_p = dist_sq(P_data, P_anchors)
        d_q = dist_sq(Q_data, Q_anchors)
        
        d_p = np.maximum(d_p, 1e-9) # Avoid negatives due to precision
        d_q = np.maximum(d_q, 1e-9)
        
        D_total = d_p * d_q
        
        # Assignment with constraint
        # Sort all (data_idx, anchor_idx) by distance?
        # That's N*K elements. 10M.
        # Flatten D_total
        flat_indices = np.argsort(D_total, axis=None) # Returns indices into flattened array
        
        # We need to unravel indices
        # row = idx // K, col = idx % K
        
        shards = {k: [] for k in range(num_shards)}
        assigned_mask = np.zeros(num_data, dtype=bool)
        shard_counts = np.zeros(num_shards, dtype=int)
        
        assigned_count = 0
        
        print("  Assigning points...")
        # Iterating 10M sorted items is slow in python.
        # But we must strictly follow capacity constraint.
        # Is there a faster way? 
        # Stable marriage problem? Or greedy.
        # The paper algorithm is greedy on sorted list.
        
        rows = flat_indices // num_shards
        cols = flat_indices % num_shards
        
        for r, c in zip(rows, cols):
            if assigned_count >= num_data:
                break
            
            if not assigned_mask[r]:
                if shard_counts[c] < max_shard_size:
                    shards[c].append(train_data[r])
                    assigned_mask[r] = True
                    shard_counts[c] += 1
                    assigned_count += 1
        
        # Update Anchors
        new_anchors = []
        diff = 0
        for k in range(num_shards):
            data_indices = shards[k] # This is data list, we need indices or embeddings
            if not data_indices:
                # Keep old anchor or re-init
                new_anchors.append(anchors[k])
                continue
                
            # Re-gather embeddings
            # We stored tuples in shards[k], need to parse
            k_users = [x[0] for x in data_indices]
            k_items = [x[1] for x in data_indices]
            
            mean_p = np.mean(u_emb[k_users], axis=0)
            mean_q = np.mean(i_emb[k_items], axis=0)
            
            new_anchors.append((mean_p, mean_q))
            diff += np.sum((mean_p - anchors[k][0])**2) + np.sum((mean_q - anchors[k][1])**2)
            
        anchors = new_anchors
        print(f"  Anchor diff: {diff}")
        if diff < 1e-4:
            break
            
    return shards
