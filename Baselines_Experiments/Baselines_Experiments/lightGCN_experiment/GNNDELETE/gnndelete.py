# -*- coding: utf-8 -*-
import os
import torch
import torch.optim as optim
import numpy as np
import scipy.sparse as sp
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from lightgcn_models import LightGCNGNNDELETE
from utils import evaluate_user_subset

class GNNDELETEManager:
    def __init__(self, dataset, device, config, save_path):
        self.dataset = dataset
        self.device = device
        self.config = config
        self.save_path = save_path
        
        self.model = LightGCNGNNDELETE(
            n_users=dataset.n_users,
            n_items=dataset.n_items,
            embedding_size=config['embedding_size'],
            n_layers=config['n_layers'],
            reg_weight=config['reg_weight'],
            deletion_lambda=config.get('deletion_lambda', 0.5)
        ).to(device)
        
        self.optimizer = optim.Adam(self.model.parameters(), lr=config['lr'])
        
        # Compute norm_adj_matrix
        interaction_matrix = dataset.interaction_matrix.tocoo()
        self.norm_adj_matrix = self.model.get_norm_adj_mat(interaction_matrix).to(device)
        
        self.deleted_edges = None  # Store deleted edges
        self.masks = None
        
    def train(self, epochs=10, batch_size=4096):
        """Train the base model"""
        train_samples = self.dataset.train_samples
        n_samples = len(train_samples)
        
        # Standard BPR Training Loop
        for epoch in range(epochs):
            np.random.shuffle(train_samples)
            total_loss = 0.0
            
            for start in range(0, n_samples, batch_size):
                end = min(start + batch_size, n_samples)
                batch_samples = train_samples[start:end]
                
                users = torch.LongTensor([s[0] for s in batch_samples]).to(self.device)
                pos_items = torch.LongTensor([s[1] for s in batch_samples]).to(self.device)
                neg_items = torch.LongTensor([np.random.randint(0, self.dataset.n_items) for _ in batch_samples]).to(self.device)
                
                self.optimizer.zero_grad()
                loss = self.model.calculate_loss(users, pos_items, neg_items, self.norm_adj_matrix)
                loss.backward()
                self.optimizer.step()
                
                total_loss += loss.item()
            
            print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss:.4f}")
        
        # Save model
        torch.save(self.model.state_dict(), self.save_path)
    
    def load_model(self, path=None):
        load_path = path if path else self.save_path
        if os.path.exists(load_path):
            self.model.load_state_dict(torch.load(load_path))
            print(f"Loaded model from {load_path}")
        else:
            print(f"Model not found at {load_path}")
            
    def load_pretrained(self, path):
        """Load pretrained Base LightGCN weights from standard dict"""
        if os.path.exists(path):
            print(f"Loading pretrained weights from {path}")
            checkpoint = torch.load(path, map_location=self.device)
            model_dict = self.model.state_dict()
            
            # Simple filtering
            pretrained_dict = {k: v for k, v in checkpoint.items() if k in model_dict and v.shape == model_dict[k].shape}
            model_dict.update(pretrained_dict)
            self.model.load_state_dict(model_dict)
        else:
            print(f"Pretrained model not found at {path}")

    def get_k_hop_neighbors(self, nodes, k=1):
        """Get k-hop neighbors"""
        if not hasattr(self, 'adj_list'):
            # Build full adjacency list
            self.adj_list = [set() for _ in range(self.dataset.n_users + self.dataset.n_items)]
            
            inter = self.dataset.interaction_matrix.tocoo()
            for u, i in zip(inter.row, inter.col):
                item_idx = i + self.dataset.n_users
                self.adj_list[u].add(item_idx)
                self.adj_list[item_idx].add(u)
        
        current_shell = set(nodes)
        visited = set(nodes)
        
        for _ in range(k):
            next_shell = set()
            for node in current_shell:
                neighbors = self.adj_list[node]
                for n in neighbors:
                    if n not in visited:
                        visited.add(n)
                        next_shell.add(n)
            current_shell = next_shell
        return list(visited)

    def unlearn(self, deleted_edges, epochs=10, lr=0.001):
        """Apply GNNDELETE unlearning (Batched)"""
        if not deleted_edges:
            print("No edges to delete!")
            return

        # Freeze base model parameters
        for param in self.model.parameters():
            param.requires_grad = False
        
        # Only train deletion operators
        for del_op in self.model.deletion_ops:
            for param in del_op.parameters():
                param.requires_grad = True
        
        # Set optimizer for deletion ops
        del_params = []
        for del_op in self.model.deletion_ops:
            del_params.extend(del_op.parameters())
        del_optimizer = optim.Adam(del_params, lr=lr)
        
        self.deleted_edges = deleted_edges

        # Start nodes for mask computation: U and I+n_users
        start_nodes = [] 
        for u, i in self.deleted_edges:
            start_nodes.append(u)
            start_nodes.append(i + self.dataset.n_users)
        
        print("Computing k-hop neighborhoods for masks...")
        masks = []
        for l in range(1, self.config['n_layers'] + 1):
             nodes_khop = self.get_k_hop_neighbors(start_nodes, k=l)
             masks.append(torch.tensor(nodes_khop, dtype=torch.long).to(self.device))
             
        self.masks = masks
        print(f"Mask Sizes: {[len(m) for m in masks]}")
        
        subgraph_nodes = masks[-1] # Largest neighborhood for NI
        
        # Batching
        batch_size = 2048
        n_edges = len(deleted_edges)
        n_batches = (n_edges + batch_size - 1) // batch_size
        
        self.model.train()
        pbar = tqdm(range(epochs), desc="Unlearning")
        
        for epoch in pbar:
            np.random.shuffle(deleted_edges)
            epoch_loss = 0.0
            
            for i in range(n_batches):
                start = i * batch_size
                end = min((i + 1) * batch_size, n_edges)
                batch_edges = deleted_edges[start:end]
                
                # Sample subgraph nodes
                if len(subgraph_nodes) > 4096:
                    perm = torch.randperm(len(subgraph_nodes))
                    batch_subgraph = subgraph_nodes[perm[:4096]]
                else:
                    batch_subgraph = subgraph_nodes

                del_optimizer.zero_grad()
                loss = self.model.calculate_deletion_loss(
                    self.norm_adj_matrix, 
                    batch_edges, 
                    masks, 
                    batch_subgraph
                )
                loss.backward()
                del_optimizer.step()
                epoch_loss += loss.item()
                
            pbar.set_postfix({'loss': epoch_loss / n_batches if n_batches > 0 else 0})
        
        # Set masks for inference
        self.model.inference_masks = masks
        self.model.current_masks = masks
        
        torch.save(self.model.state_dict(), self.save_path + '_unlearned')
    
    def calculate_zrf(self, incompetent_teacher, forget_samples, batch_size=2048):
        """Calculate ZRF score"""
        self.model.eval()
        incompetent_teacher.eval()
        masks = getattr(self, 'masks', None)
        js_divs = []
        
        for start in range(0, len(forget_samples), batch_size):
            end = min(start + batch_size, len(forget_samples))
            batch_samples = forget_samples[start:end]
            
            users = torch.LongTensor([s[0] for s in batch_samples]).to(self.device)
            items = torch.LongTensor([s[1] for s in batch_samples]).to(self.device)
            
            with torch.no_grad():
                logits_unl = self.model.predict(users, items, self.norm_adj_matrix, masks=masks)
                probs_unl = torch.sigmoid(logits_unl)
                
                logits_rnd = incompetent_teacher.predict(users, items, self.norm_adj_matrix)
                probs_rnd = torch.sigmoid(logits_rnd)
            
            # Binary JS Div
            dist_unl = torch.stack([probs_unl, 1 - probs_unl], dim=1).clamp(1e-10, 1.0)
            dist_rnd = torch.stack([probs_rnd, 1 - probs_rnd], dim=1).clamp(1e-10, 1.0)
            M = 0.5 * (dist_unl + dist_rnd)
            
            kl1 = (dist_unl * torch.log(dist_unl / M)).sum(dim=1)
            kl2 = (dist_rnd * torch.log(dist_rnd / M)).sum(dim=1)
            js = 0.5 * (kl1 + kl2)
            js_divs.extend(js.cpu().numpy())
            
        return 1.0 - np.mean(js_divs) if js_divs else 0.0
