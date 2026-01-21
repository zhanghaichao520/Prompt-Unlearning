# -*- coding: utf-8 -*-
import torch
import torch.optim as optim
import numpy as np
from tqdm import tqdm
import torch.nn.functional as F
import torch.nn as nn
from unlearnrec_models import InfluenceEncoder, NGCFUnlearnRec, LightGCNUnlearnRec

class UnlearnRecManager:
    def __init__(self, dataset, model, device, config):
        self.dataset = dataset
        self.model = model
        self.device = device
        self.config = config
        
        self.ie = InfluenceEncoder(
            dataset.n_users, dataset.n_items, 
            config['embedding_size'], 
            n_layers_ie=3, n_layers_mlp=2
        ).to(device)
        
        self.full_norm_adj = None
        self.bar_E = None
        
        # Store original E0
        with torch.no_grad():
            self.original_E0 = self.model.get_ego_embeddings().clone().detach()

    def set_graph_data(self, full_norm_adj):
        self.full_norm_adj = full_norm_adj
        self.model.eval()
        with torch.no_grad():
            # Get bar_E (final embeddings) from original model with full graph
            u, i = self.model.forward(full_norm_adj)
            self.bar_E = torch.cat([u, i], dim=0).clone().detach()

    def pretrain(self, epochs):
        # Optimizing H0 and W_eta only
        optimizer = optim.Adam([
            {'params': [self.ie.H0, self.ie.W_eta], 'lr': self.config['lr_pre']}
        ])
        
        self.ie.train()
        self.model.eval()
        
        # Edges for sampling
        inter_coo = self.dataset.interaction_matrix.tocoo()
        all_edges = np.vstack((inter_coo.row, inter_coo.col + self.dataset.n_users)).T # [N_edges, 2]
        all_edges_tensor = torch.LongTensor(all_edges.T).to(self.device)
        num_edges = all_edges.shape[0]
        
        pbar = tqdm(range(epochs), desc="UnlearnRec Pretraining")
        for epoch in pbar:
            # 1. Simulate Unlearning Request (5% edges)
            num_unlearn = int(num_edges * 0.05)
            perm = torch.randperm(num_edges)
            idx_unlearn = perm[:num_unlearn]
            idx_remain = perm[num_unlearn:]
            
            edges_unlearn = all_edges_tensor[:, idx_unlearn] # [2, num_unlearn]
            
            # 2. Get Shift
            delta_bar_E0 = self.ie.calculate_delta(edges_unlearn, self.bar_E, self.device)
            # Apply shift to COPY of E0 (since we don't update original E0 in pretrain)
            tilde_E0 = self.original_E0 + delta_bar_E0
            
            # 3. Compute Losses
            # A. Model Loss L_M on Remaining Edges
            # Sample batch from remaining
            batch_size = self.config['batch_size']
            batch_idx = np.random.choice(len(idx_remain), batch_size)
            batch_edges = all_edges_tensor[:, idx_remain][:, batch_idx]
            
            users = batch_edges[0]
            pos_items = batch_edges[1] - self.dataset.n_users
            neg_items = torch.randint(0, self.dataset.n_items, (batch_size,), device=self.device)
            
            loss_m, _ = self.model.get_loss(users, pos_items, neg_items, self.full_norm_adj, override_E0=tilde_E0)
            
            # B. Unlearning Loss L_u
            # Maximize distance (minimize score) for unlearned edges
            # Sample batch from unlearned
            batch_u_idx = np.random.choice(len(idx_unlearn), batch_size)
            batch_u_edges = edges_unlearn[:, batch_u_idx]
            u_users = batch_u_edges[0]
            u_items = batch_u_edges[1] - self.dataset.n_users
            
            ua_emb, ia_emb = self.model.forward(self.full_norm_adj, override_E0=tilde_E0)
            u_emb = ua_emb[u_users]
            i_emb = ia_emb[u_items]
            scores = (u_emb * i_emb).sum(dim=1)
            loss_u = -torch.log(torch.sigmoid(-scores) + 1e-9).mean()
            
            # C. Preserving Loss L_p
            # Align final embeddings with original bar_E
            current_bar_E = torch.cat([ua_emb, ia_emb], dim=0)
            loss_p = F.mse_loss(current_bar_E, self.bar_E)
            
            # Total Loss
            loss = loss_m + self.config['lambda_u'] * loss_u + self.config['lambda_p'] * loss_p
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            pbar.set_postfix({'loss': loss.item(), 'lm': loss_m.item(), 'lu': loss_u.item()})

    def finetune(self, forget_samples, epochs=10):
        # forget_samples: list of (u, i)
        # Convert to tensor
        edges_list = []
        for u, i in forget_samples:
            edges_list.append([u, i + self.dataset.n_users])
        edge_tensor = torch.LongTensor(edges_list).T.to(self.device) # [2, N_forget]
        
        # Setup trainable E0
        # Initialize with original E0
        current_E0 = nn.Parameter(self.original_E0.clone().detach())
        
        # Unlock MLP, Lock IE base
        self.ie.H0.requires_grad_(False)
        self.ie.W_eta.requires_grad_(False)
        self.ie.mlp.requires_grad_(True)
        
        optimizer = optim.Adam([
            {'params': self.ie.mlp.parameters(), 'lr': self.config['lr_finetune']},
            {'params': [current_E0], 'lr': self.config['lr_finetune']}
        ] )
        
        # Calculate Delta Pre-MLP (Fixed part) once
        with torch.no_grad():
            delta_pre_mlp = self.ie.calculate_delta_pre_mlp(edge_tensor, self.bar_E, self.device)
        
        # Identify remaining edges for L_M sampling
        inter_coo = self.dataset.interaction_matrix.tocoo()
        
        # Create a set for fast lookup
        forget_set = set(tuple(x) for x in forget_samples)
        retain_indices = []
        # Re-construct all_edges properly
        rows = inter_coo.row
        cols = inter_coo.col # These are item IDs 0..M-1
        
        all_edges_global = np.vstack((rows, cols + self.dataset.n_users)).T # For sampling
        edges_tensor_global = torch.LongTensor(all_edges_global.T).to(self.device)
        
        for k in range(len(rows)):
            u, i = rows[k], cols[k]
            if (u, i) not in forget_set:
                retain_indices.append(k)
        
        pbar = tqdm(range(epochs), desc="UnlearnRec Fine-tuning")
        for epoch in pbar:
            # Shift
            delta_bar_E0 = self.ie.mlp(delta_pre_mlp)
            tilde_E0 = current_E0 + delta_bar_E0
            
            # L_M: Sample from Retain
            batch_size = self.config['batch_size']
            batch_idx = np.random.choice(retain_indices, batch_size)
            batch_edges = edges_tensor_global[:, batch_idx]
            
            users = batch_edges[0]
            pos_items = batch_edges[1] - self.dataset.n_users
            neg_items = torch.randint(0, self.dataset.n_items, (batch_size,), device=self.device)
            
            loss_m, _ = self.model.get_loss(users, pos_items, neg_items, self.full_norm_adj, override_E0=tilde_E0)
            
            # L_u: Sample from Forget
            u_users = edge_tensor[0]
            u_items = edge_tensor[1] - self.dataset.n_users
            
            # If forget set is large, sample batch? Usually unlearning is small.
            # If small, use all.
            if len(u_users) > batch_size:
                idx = torch.randperm(len(u_users))[:batch_size]
                u_u, u_i = u_users[idx], u_items[idx]
            else:
                u_u, u_i = u_users, u_items
                
            ua_emb, ia_emb = self.model.forward(self.full_norm_adj, override_E0=tilde_E0)
            scores = (ua_emb[u_u] * ia_emb[u_i]).sum(dim=1)
            loss_u = -torch.log(torch.sigmoid(-scores) + 1e-9).mean()
            
            # C. Preserving Loss L_p (Critical for preventing catastrophic forgetting)
            current_bar_E = torch.cat([ua_emb, ia_emb], dim=0)
            loss_p = F.mse_loss(current_bar_E, self.bar_E)
            
            loss = loss_m + self.config['lambda_u'] * loss_u + self.config['lambda_p'] * loss_p
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            pbar.set_postfix({'loss': loss.item(), 'lu': loss_u.item(), 'lp': loss_p.item()})
            
        # Return the modified E0 
        return tilde_E0.detach()

    def calculate_zrf(self, incompetent_teacher, forget_samples, batch_size=2048):
        """
        Calculate Zero Retrain Forgetting (ZRF) score.
        ZRF = 1 - JS(M_unlearned(D_f) || M_random(D_f))
        
        Args:
           incompetent_teacher: Randomly initialized model
           forget_samples: List of (u, i) or (u, i, label) in forget set
           batch_size: Batch size
        """
        # Ensure we use the current E0 (modified) for prediction
        # Since we modified the model weights in the script, self.model.forward() 
        # will use the modified weights if we did model.embedding_dict...copy_().
        # However, in finetune() we returned tilde_E0 but didn't permanently set it in self.model if strict.
        # But the scripts DO copy it back: 
        # model.embedding_dict['user_emb'].weight.data.copy_(u_emb)
        # So standard forward is fine.
        
        self.model.eval()
        incompetent_teacher.eval()
        
        js_divs = []
        
        for start in range(0, len(forget_samples), batch_size):
            end = min(start + batch_size, len(forget_samples))
            batch_samples = forget_samples[start:end]
            
            users = [s[0] for s in batch_samples]
            u_tensor = torch.LongTensor(users).to(self.device)
            
            items = [s[1] for s in batch_samples]
            i_tensor = torch.LongTensor(items).to(self.device)
            
            with torch.no_grad():
                # Unlearned preds
                # We need a predict function. NGCF has predict() ?
                # If not, manual dot product. 
                # NGCFGNNDELETE/LightGCNGNNDELETE usually don't have predict() in base, just forward.
                
                # Manual prediction:
                ua_emb, ia_emb = self.model.forward(self.full_norm_adj)
                u_emb = ua_emb[u_tensor]
                i_emb = ia_emb[i_tensor]
                scores_unlearned = (u_emb * i_emb).sum(dim=1)
                probs_unlearned = torch.sigmoid(scores_unlearned)
                
                # Random preds
                # Note: Incompetent teacher might need full_norm_adj too
                ua_emb_r, ia_emb_r = incompetent_teacher.forward(self.full_norm_adj)
                u_emb_r = ua_emb_r[u_tensor]
                i_emb_r = ia_emb_r[i_tensor]
                scores_random = (u_emb_r * i_emb_r).sum(dim=1)
                probs_random = torch.sigmoid(scores_random)
            
            # JS divergence
            dist_unlearned = torch.stack([probs_unlearned, 1 - probs_unlearned], dim=1)
            dist_random = torch.stack([probs_random, 1 - probs_random], dim=1)
            
            dist_unlearned = torch.clamp(dist_unlearned, 1e-10, 1.0)
            dist_random = torch.clamp(dist_random, 1e-10, 1.0)
            M = 0.5 * (dist_unlearned + dist_random)
            M = torch.clamp(M, 1e-10, 1.0)

            kl1 = torch.sum(dist_unlearned * torch.log(dist_unlearned / M), dim=1)
            kl2 = torch.sum(dist_random * torch.log(dist_random / M), dim=1)
            
            js = 0.5 * (kl1 + kl2)
            
            js_divs.extend(js.detach().cpu().numpy())
        
        zrf = 1.0 - np.mean(js_divs) if js_divs else 0.0
        return zrf
