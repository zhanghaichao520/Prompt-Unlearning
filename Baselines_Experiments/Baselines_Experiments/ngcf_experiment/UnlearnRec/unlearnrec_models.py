# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import scipy.sparse as sp
import sys
# Add path for lightgcn models
sys.path.append("/data/P2F/lightGCN_experiment/GNNDELETE")
# Add path for ngcf models (GNNDELETE)
sys.path.append("/data/P2F/ngcf_experiment/GNNDELETE")
from lightgcn_models import LightGCNGNNDELETE
from ngcf_models import NGCFGNNDELETE, BiGNNLayer, SparseDropout # Reuse basic components

class InfluenceEncoder(nn.Module):
    """
    Influence Encoder (IE) as described in the UnlearnRec paper.
    Takes IDM (Influence Dependency Matrix) and original Embeddings as input.
    """
    def __init__(self, n_users, n_items, embedding_size, n_layers_ie=3, n_layers_mlp=2):
        super(InfluenceEncoder, self).__init__()
        self.n_nodes = n_users + n_items
        self.embedding_size = embedding_size
        self.n_layers_ie = n_layers_ie
        
        # Trainable parameters
        self.H0 = nn.Parameter(torch.zeros(self.n_nodes, embedding_size)) # Initialize around 0
        self.W_eta = nn.Parameter(torch.zeros(self.n_nodes, 1)) # Initialize around 0
        
        # MLP for Eq. 17 & 18
        # Input dim is embedding_size (from Delta E0)
        # Output dim is embedding_size (Delta E0 bar)
        layers = []
        input_dim = embedding_size
        for _ in range(n_layers_mlp - 1):
            layers.append(nn.Linear(input_dim, embedding_size))
            layers.append(nn.ReLU()) # Assuming ReLU activation
            input_dim = embedding_size
        layers.append(nn.Linear(input_dim, embedding_size))
        
        self.mlp = nn.Sequential(*layers)
        
        # Initialize MLP to identity/zero as per paper (Algorithm 1 line 1)
        # "Initialize Wl with identity matrix and bl with 0"
        # Since we use nn.Linear, we can init weights to identity and bias to 0.
        for m in self.mlp:
            if isinstance(m, nn.Linear):
                nn.init.eye_(m.weight)
                nn.init.constant_(m.bias, 0)
                
    def get_idm(self, edge_index, n_nodes, device):
        """Constructs IDM (Influence Dependency Matrix) A_delta from edges."""
        # edge_index: [2, num_edges]
        # Construct symmetric adjacency matrix
        # This mirrors get_norm_adj_mat logic but for dynamic edges
        
        # This needs to be efficient. 
        # Since we are often handling batches or simulated edges, constructing sparse tensor directly is best.
        
        rows = torch.cat([edge_index[0], edge_index[1]])
        cols = torch.cat([edge_index[1], edge_index[0]])
        
        # Self-loops usually not added to IDM in unlearning context unless specified?
        # Eq 15 uses D_delta^-0.5 * A_delta * D_delta^-0.5
        
        indices = torch.stack([rows, cols], dim=0)
        values = torch.ones(indices.shape[1], device=device)
        
        A = torch.sparse_coo_tensor(indices, values, (n_nodes, n_nodes))
        
        # Normalize: D^-0.5 * A * D^-0.5
        # Calculate degrees
        D_vals = torch.sparse.sum(A, dim=1).to_dense()
        D_vals[D_vals == 0] = 1 # Avoid div by zero
        D_pow = D_vals.pow(-0.5)
        
        # Improving efficiency: (D^-0.5 * A) * D^-0.5
        # Element-wise mult for sparse is tricky in pytorch.
        # Alternative: Scale values based on indices
        
        row_d = D_pow[indices[0]]
        col_d = D_pow[indices[1]]
        new_values = values * row_d * col_d
        
        norm_A_delta = torch.sparse_coo_tensor(indices, new_values, (n_nodes, n_nodes))
        return norm_A_delta

    def forward(self, edge_index, original_embeddings):
        """
        Args:
            edge_index: Edges to unlearn (or simulated)
            original_embeddings: E0 (fixed usually). Shape [n_nodes, dim]
        Returns:
            delta_E0_bar: The estimated shift.
            masks: None (placeholder)
        """
        device = original_embeddings.device
        norm_A_delta = self.get_idm(edge_index, self.n_nodes, device)
        
        # 1. Propagate H (Eq. 15)
        H_l = self.H0
        for _ in range(self.n_layers_ie):
             H_l = torch.sparse.mm(norm_A_delta, H_l)
        bar_H = H_l # Final H (Readout IEM)
        
        # 2. Propagate E_w (Eq. 16)
        # E_w,0 = bar_E * W_eta
        # bar_E is "readout embeddings of model M before unlearning"
        # Paper says: "E_w,0 = bar_E * W_eta" where bar_E is FIXED.
        # Wait, the input to this function is `original_embeddings`, which is usually E0.
        # The paper distinguishes E0 (0-layer) and bar_E (readout/final layer).
        # We need bar_E passed in or E0? 
        # Eq 17: delta_E0 = -E_w + bar_H. 
        # And E_w comes from bar_E. 
        # So we need bar_E (final embeddings of pretrained model) as input too?
        # Yes, Section 3.2.1: "E_w,0 = bar_E * W_eta ... where bar_E is readout embeddings... fixed".
        
        # For now, let's assume original_embeddings passed here is bar_E for E_w calculation??
        # But Eq 17 says "tilde_E0 = delta_bar_E0 + E0".
        # So we likely need both E0 and bar_E.
        # For simplicity in many GNNs, E0 and bar_E have same shape.
        pass

    def calculate_delta(self, edge_index, fixed_bar_E, device):
        delta_E0 = self.calculate_delta_pre_mlp(edge_index, fixed_bar_E, device)
        delta_bar_E0 = self.mlp(delta_E0)
        return delta_bar_E0

    def calculate_delta_pre_mlp(self, edge_index, fixed_bar_E, device):
        norm_A_delta = self.get_idm(edge_index, self.n_nodes, device)
        
        # Eq 15
        H_l = self.H0
        for _ in range(self.n_layers_ie):
             H_l = torch.sparse.mm(norm_A_delta, H_l)
        bar_H = H_l
        
        # Eq 16
        E_w = fixed_bar_E * self.W_eta
        for _ in range(self.n_layers_ie):
            E_w = torch.sparse.mm(norm_A_delta, E_w) # E_w,l
        bar_E_w = E_w # Last layer
        
        # Eq 17 (pre-MLP)
        delta_E0 = -bar_E_w + bar_H
        return delta_E0

class NGCFUnlearnRec(NGCFGNNDELETE):
    """
    Subclass of NGCF to allow injecting modifying E0.
    """
    def __init__(self, *args, **kwargs):
        super(NGCFUnlearnRec, self).__init__(*args, **kwargs)
        
    def forward(self, norm_adj_matrix, override_E0=None, **kwargs):
        """
        Args:
            override_E0: If provided, use this instead of self.user_embedding/item_embedding
        """
        if override_E0 is not None:
             ego_embeddings = override_E0
        else:
             ego_embeddings = self.get_ego_embeddings()

        # ... (Rest of forward logic) ...
        # Can't simply call super().forward() because it calls get_ego_embeddings() internally.
        # I need to duplicate the forward logic or refactor parent.
        # Since I can't edit parent easily without possibly breaking other things, I will duplicate logic 
        # or monkey-patch `get_ego_embeddings` temporarily? 
        # Duplicating logic is safer.
        
        device = norm_adj_matrix.device
        all_embeddings = [ego_embeddings]
        
        # Eye matrix
        num = self.n_users + self.n_items
        i = torch.LongTensor([range(0, num), range(0, num)]).to(device)
        val = torch.FloatTensor([1] * num).to(device)
        eye_matrix = torch.sparse_coo_tensor(i, val, (num, num)).to(device)
        
        # GNN Layers
        for idx, layer in enumerate(self.GNNlayers):
             # NGCF forward logic
             # Check if we need deletion layers? UnlearnRec doesn't use DeletionLayer logic usually,
             # it modifies E0 directly. So we skip deletion ops or use identity.
             # Standard NGCF propagation:
             all_embeddings.append(layer(norm_adj_matrix, eye_matrix, all_embeddings[-1]))
             
        # Stack and average/sum
        all_embeddings = torch.stack(all_embeddings, dim=1)
        all_embeddings = torch.mean(all_embeddings, dim=1)
        
        self.result_embed = all_embeddings
        # Split user/item
        user_all_embeddings, item_all_embeddings = torch.split(all_embeddings, [self.n_users, self.n_items])
        return user_all_embeddings, item_all_embeddings

    def get_loss(self, users, pos_items, neg_items, norm_adj_matrix, override_E0=None):
        # BPR Loss
        ua_emb, ia_emb = self.forward(norm_adj_matrix, override_E0=override_E0)
        
        u_emb = ua_emb[users]
        pos_i_emb = ia_emb[pos_items]
        neg_i_emb = ia_emb[neg_items]
        
        pos_scores = torch.mul(u_emb, pos_i_emb).sum(dim=1)
        neg_scores = torch.mul(u_emb, neg_i_emb).sum(dim=1)
        
        loss = torch.mean(F.softplus(neg_scores - pos_scores))
        
        # Reg loss
        if override_E0 is not None:
            # Regularize the modified embeddings? 
            # Usually reg loss is on initial embeddings (0-th layer)
            reg_loss = (1/2)*(override_E0[users].norm(2).pow(2) + 
                              override_E0[self.n_users+pos_items].norm(2).pow(2) + 
                              override_E0[self.n_users+neg_items].norm(2).pow(2)) / float(len(users))
        else:
            # Fallback
            reg_loss = 0 # Or standard
            
        return loss, self.reg_weight * reg_loss


class LightGCNUnlearnRec(LightGCNGNNDELETE):
    def __init__(self, *args, **kwargs):
        super(LightGCNUnlearnRec, self).__init__(*args, **kwargs)
        
    def forward(self, norm_adj_matrix, override_E0=None, **kwargs):
        if override_E0 is not None:
             ego_embeddings = override_E0
        else:
             ego_embeddings = torch.cat([self.user_embedding.weight, self.item_embedding.weight], dim=0)
             
        all_embeddings = [ego_embeddings]
        
        for i in range(self.n_layers):
            # LightGCN propagation: D^-0.5 A D^-0.5 E
            side_embeddings = torch.sparse.mm(norm_adj_matrix, all_embeddings[-1])
            all_embeddings.append(side_embeddings)
            
        all_embeddings = torch.stack(all_embeddings, dim=1)
        all_embeddings = torch.mean(all_embeddings, dim=1)
        
        user_all_embeddings, item_all_embeddings = torch.split(all_embeddings, [self.n_users, self.n_items])
        return user_all_embeddings, item_all_embeddings

    def get_ego_embeddings(self):
         return torch.cat([self.user_embedding.weight, self.item_embedding.weight], dim=0)

    def get_loss(self, users, pos_items, neg_items, norm_adj_matrix, override_E0=None):
        ua_emb, ia_emb = self.forward(norm_adj_matrix, override_E0=override_E0)
        
        u_emb = ua_emb[users]
        pos_i_emb = ia_emb[pos_items]
        neg_i_emb = ia_emb[neg_items]
        
        pos_scores = torch.mul(u_emb, pos_i_emb).sum(dim=1)
        neg_scores = torch.mul(u_emb, neg_i_emb).sum(dim=1)
        
        loss = torch.mean(F.softplus(neg_scores - pos_scores))
        
        if override_E0 is not None:
            reg_loss = (1/2)*(override_E0[users].norm(2).pow(2) + 
                              override_E0[self.n_users+pos_items].norm(2).pow(2) + 
                              override_E0[self.n_users+neg_items].norm(2).pow(2)) / float(len(users))
        else:
            reg_loss = (1/2)*(self.user_embedding.weight[users].norm(2).pow(2) +
                              self.item_embedding.weight[pos_items].norm(2).pow(2) +
                              self.item_embedding.weight[neg_items].norm(2).pow(2)) / float(len(users))
                              
        return loss, self.reg_weight * reg_loss
