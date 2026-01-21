# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
import scipy.sparse as sp
import numpy as np
from config import LIGHTGCN_CONFIG

# Copied from GNNDELETE/ngcf_models.py
class DeletionLayer(nn.Module):
    def __init__(self, in_dim):
        super(DeletionLayer, self).__init__()
        # Initialized to identity-like
        self.deletion_weight = nn.Parameter(torch.eye(in_dim) + 1e-4 * torch.randn(in_dim, in_dim))

    def forward(self, embeddings, mask=None):
        if mask is None:
            return embeddings
        
        # Apply only to masked nodes
        # mask is a boolean tensor or indices
        modified_embeddings = embeddings.clone()
        
        if mask.dtype == torch.bool:
            # If boolean mask
            if mask.sum() == 0:
                return embeddings
            modified_embeddings[mask] = torch.matmul(modified_embeddings[mask], self.deletion_weight)
        else:
            # If indices
            if len(mask) == 0:
                return embeddings
            # Ensure mask is on the same device as embeddings
            mask = mask.to(embeddings.device)
            modified_embeddings[mask] = torch.matmul(modified_embeddings[mask], self.deletion_weight)
            
        return modified_embeddings

class LightGCNGNNDELETE(nn.Module):
    """LightGCN with GNNDELETE unlearning operator."""
    
    def __init__(self, n_users, n_items, embedding_size=64, n_layers=3, 
                 reg_weight=1e-4, deletion_lambda=0.5):
        super(LightGCNGNNDELETE, self).__init__()
        self.n_users = n_users
        self.n_items = n_items
        self.latent_dim = embedding_size
        self.n_layers = n_layers
        self.reg_weight = reg_weight
        self.deletion_lambda = deletion_lambda
        self.inference_masks = None
        self.current_masks = None

        # Define embeddings
        self.user_embedding = nn.Embedding(n_users, embedding_size)
        self.item_embedding = nn.Embedding(n_items, embedding_size)
        
        # Init embeddings (Xavier)
        nn.init.xavier_uniform_(self.user_embedding.weight)
        nn.init.xavier_uniform_(self.item_embedding.weight)
        
        # Deletion operators for each layer (excluding layer 0 which is just lookup)
        # LightGCN has K layers of propagation. We'll add a deletion op after each propagation.
        self.deletion_ops = torch.nn.ModuleList()
        for i in range(self.n_layers):
            self.deletion_ops.append(DeletionLayer(embedding_size))
            
        # Storage for full sort accel
        self.restore_user_e = None
        self.restore_item_e = None

    def get_norm_adj_mat(self, interaction_matrix):
        """Construct normalized adjacency matrix"""
        # Exactly as in models.py LightGCN
        A = sp.dok_matrix((self.n_users + self.n_items, self.n_users + self.n_items), dtype=np.float32)
        inter_M = interaction_matrix
        inter_M_t = interaction_matrix.transpose()
        
        # Manual update since ._update is deprecated in newer scipy
        # But we can use dictionary update logic similar to previously fixed code if dok_matrix doesn't support .update well
        # Or just use row/col construction which is faster
        
        # Using dict update method for dok_matrix
        data_dict = dict(zip(zip(inter_M.row, inter_M.col + self.n_users), [1] * inter_M.nnz))
        data_dict.update(dict(zip(zip(inter_M_t.row + self.n_users, inter_M_t.col), [1] * inter_M_t.nnz)))
        dict.update(A, data_dict)
        
        sumArr = (A > 0).sum(axis=1)
        diag = np.array(sumArr.flatten())[0] + 1e-7
        diag = np.power(diag, -0.5)
        D = sp.diags(diag)
        L = D * A * D
        
        L = sp.coo_matrix(L)
        row = L.row
        col = L.col
        i = torch.LongTensor(np.array([row, col]))
        data = torch.FloatTensor(L.data)
        SparseL = torch.sparse_coo_tensor(i, data, torch.Size(L.shape))
        return SparseL

    def forward(self, norm_adj_matrix, masks=None, training=True):
        # Check masks
        current_masks = masks
        if current_masks is None and not training and getattr(self, 'current_masks', None) is not None:
            current_masks = self.current_masks
            
        # 1. Get initial embeddings
        u_emb = self.user_embedding.weight
        i_emb = self.item_embedding.weight
        all_emb = torch.cat([u_emb, i_emb])
        
        embs_list = [all_emb]
        
        # 2. Propagation
        for layer in range(self.n_layers):
            all_emb = torch.sparse.mm(norm_adj_matrix, all_emb)
            
            # Apply Deletion Operator if needed
            if current_masks is not None and layer < len(current_masks) and current_masks[layer] is not None:
                all_emb = self.deletion_ops[layer](all_emb, current_masks[layer])
                
            embs_list.append(all_emb)
            
        # 3. Aggregation (Average)
        final_emb = torch.stack(embs_list, dim=1)
        final_emb = torch.mean(final_emb, dim=1)
        
        users_emb, items_emb = torch.split(final_emb, [self.n_users, self.n_items])
        return users_emb, items_emb

    def calculate_loss(self, users, pos_items, neg_items, norm_adj_matrix):
        """BPR Loss + Reg Loss"""
        batch_users_emb, batch_items_emb = self.forward(norm_adj_matrix, masks=None, training=True)
        
        u_emb = batch_users_emb[users]
        pos_emb = batch_items_emb[pos_items]
        neg_emb = batch_items_emb[neg_items]
        
        # Score
        pos_scores = torch.mul(u_emb, pos_emb).sum(dim=1)
        neg_scores = torch.mul(u_emb, neg_emb).sum(dim=1)
        
        # BPR Loss
        loss = -torch.mean(torch.log(torch.sigmoid(pos_scores - neg_scores) + 1e-8))
        
        # Reg Loss
        # In LightGCN, usually regularize the initial embeddings (layer 0)
        u_emb_0 = self.user_embedding(users)
        pos_emb_0 = self.item_embedding(pos_items)
        neg_emb_0 = self.item_embedding(neg_items)
        
        reg_loss = (1/2) * (u_emb_0.norm(2).pow(2) + 
                            pos_emb_0.norm(2).pow(2) + 
                            neg_emb_0.norm(2).pow(2)) / float(len(users))
                            
        return loss + self.reg_weight * reg_loss

    def calculate_deletion_loss(self, norm_adj_matrix, batch_deleted_edges, masks, subgraph_nodes=None):
        """
        Calculate GNNDELETE loss.
        """
        device = norm_adj_matrix.device
        
        # 1. Forward with deletion operator (Student)
        user_emb_del, item_emb_del = self.forward(norm_adj_matrix, masks=masks, training=True)
        
        # 2. Forward without deletion (Teacher)
        with torch.no_grad():
            user_emb_base, item_emb_base = self.forward(norm_adj_matrix, masks=None, training=True)
            
        dec_loss = 0.0
        ni_loss = 0.0
        
        # --- DEC Loss: Deleted Edge Consistency ---
        # "Predicted probability for deleted edges should be random"
        if batch_deleted_edges:
            u_indices = [u for u, v in batch_deleted_edges]
            v_indices = [v for u, v in batch_deleted_edges]
            
            u_emb_del_batch = user_emb_del[u_indices]
            v_emb_del_batch = item_emb_del[v_indices]
            
            # Score of deleted edges
            score_del = torch.sum(u_emb_del_batch * v_emb_del_batch, dim=1)
            
            # Random score construction directly from embeddings
            # We want score_del to be close to score_random
            
            # Sample random users/items to simulate "unconnected" behavior
            rand_u = torch.randint(0, self.n_users, (len(batch_deleted_edges),)).to(device)
            rand_v = torch.randint(0, self.n_items, (len(batch_deleted_edges),)).to(device)
            
            rand_u_emb = user_emb_del[rand_u]
            rand_v_emb = item_emb_del[rand_v]
            
            # Use Random edge score as target
            score_rand = torch.sum(rand_u_emb * rand_v_emb, dim=1).detach() 
            
            dec_loss = F.mse_loss(score_del, score_rand)
            
        # --- NI Loss: Neighborhood Influence ---
        if subgraph_nodes is not None and len(subgraph_nodes) > 0:
            all_emb_del = torch.cat([user_emb_del, item_emb_del], dim=0)
            all_emb_base = torch.cat([user_emb_base, item_emb_base], dim=0)
            
            # Compare embeddings in the subgraph
            ni_loss = F.mse_loss(all_emb_del[subgraph_nodes], all_emb_base[subgraph_nodes])
            
        total_loss = self.deletion_lambda * dec_loss + (1 - self.deletion_lambda) * ni_loss
        return total_loss
        
    def predict(self, users, items, norm_adj_matrix, masks=None, deleted_edges=None):
        user_all_embeddings, item_all_embeddings = self.forward(norm_adj_matrix, masks=masks, training=False)
        u_embeddings = user_all_embeddings[users]
        i_embeddings = item_all_embeddings[items]
        scores = torch.sum(u_embeddings * i_embeddings, dim=1)
        return scores
    
    def full_sort_predict(self, users, norm_adj_matrix, masks=None, deleted_edges=None):
        user_all_embeddings, item_all_embeddings = self.forward(norm_adj_matrix, masks=masks, training=False)
        u_embeddings = user_all_embeddings[users]
        scores = torch.matmul(u_embeddings, item_all_embeddings.transpose(0, 1))
        return scores
