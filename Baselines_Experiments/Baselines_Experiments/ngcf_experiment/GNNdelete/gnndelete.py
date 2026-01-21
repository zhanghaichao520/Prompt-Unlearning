# -*- coding: utf-8 -*-
import os
import torch
import torch.optim as optim
import numpy as np
import scipy.sparse as sp
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from ngcf_models import NGCFGNNDELETE
from utils import evaluate_user_subset

class GNNDELETEManager:
    def __init__(self, dataset, device, config, save_path):
        self.dataset = dataset
        self.device = device
        self.config = config
        self.save_path = save_path
        
        self.model = NGCFGNNDELETE(
            n_users=dataset.n_users,
            n_items=dataset.n_items,
            embedding_size=config['embedding_size'],
            n_layers=config['n_layers'],
            node_dropout=config['node_dropout'],
            message_dropout=config['message_dropout'],
            reg_weight=config['reg_weight'],
            deletion_lambda=config.get('deletion_lambda', 0.5)
        ).to(device)
        
        self.optimizer = optim.Adam(self.model.parameters(), lr=config['lr'])
        
        # Compute norm_adj_matrix
        interaction_matrix = dataset.interaction_matrix.tocoo()
        self.norm_adj_matrix = self.model.get_norm_adj_mat(interaction_matrix).to(device)
        
        self.deleted_edges = None  # Store deleted edges for unlearned model
        
    def train(self, epochs=10, batch_size=4096):
        """Train the base model"""
        train_samples = self.dataset.train_samples
        n_samples = len(train_samples)
        
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
        """Load pretrained weights from standard NGCF (ignoring deletion ops)"""
        if os.path.exists(path):
            print(f"Loading pretrained weights from {path}")
            checkpoint = torch.load(path, map_location=self.device)
            model_dict = self.model.state_dict()
            
            # Filter out unnecessary keys (like deletion_ops)
            pretrained_dict = {k: v for k, v in checkpoint.items() if k in model_dict and v.shape == model_dict[k].shape}
            
            model_dict.update(pretrained_dict)
            self.model.load_state_dict(model_dict)
        else:
            print(f"Pretrained model not found at {path}")
    
    def get_k_hop_neighbors(self, nodes, k=1):
        """
        Get k-hop neighbors of given nodes using the interaction matrix.
        nodes: list or array of node indices (users or items)
        Returns: set of node indices (users and items remapped to [0, n_users+n_items))
        """
        # Build adjacency list if not exists
        if not hasattr(self, 'adj_list'):
            self.adj_list = [set() for _ in range(self.dataset.n_users + self.dataset.n_items)]
            
            # User-Item interactions
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
                # Neighbors
                neighbors = self.adj_list[node]
                for n in neighbors:
                    if n not in visited:
                        visited.add(n)
                        next_shell.add(n)
            current_shell = next_shell
        
        return list(visited)

    def unlearn(self, deleted_edges, epochs=10, lr=0.001):
        """Apply GNNDELETE unlearning"""
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
        
        if not self.deleted_edges:
            print("No edges to delete!")
            return

        # Start nodes for mask computation: U and I+n_users
        start_nodes = [] 
        for u, i in self.deleted_edges:
            start_nodes.append(u)
            start_nodes.append(i + self.dataset.n_users)
        
        # Pre-compute Masks for each layer
        # Since unlearning edges are usually small (1%), we can compute this once.
        print("Computing k-hop neighborhoods for masks...")
        
        masks = []
        # Layer 1: 1-hop neighbors of deleted edge endpoints
        nodes_1hop = self.get_k_hop_neighbors(start_nodes, k=1)
        mask_1 = torch.tensor(nodes_1hop, dtype=torch.long).to(self.device)
        masks.append(mask_1)
        
        # Layer 2: 2-hop neighbors
        nodes_2hop = self.get_k_hop_neighbors(start_nodes, k=2)
        mask_2 = torch.tensor(nodes_2hop, dtype=torch.long).to(self.device)
        masks.append(mask_2)
        
        # Layer 3: 3-hop neighbors (if n_layers >= 3)
        if self.config['n_layers'] >= 3:
            nodes_3hop = self.get_k_hop_neighbors(start_nodes, k=3)
            mask_3 = torch.tensor(nodes_3hop, dtype=torch.long).to(self.device)
            masks.append(mask_3)
            
        # Subgraph nodes for NI (Use the largest scope, e.g. 2-hop or 3-hop)
        subgraph_nodes = masks[-1]
        
        self.masks = masks # Store masks for inference
        
        print(f"Mask Sizes: {[len(m) for m in masks]}")
        print(f"Total Nodes: {self.dataset.n_users + self.dataset.n_items}")
        
        # Batching setup
        batch_size = 2048
        n_edges = len(deleted_edges)
        n_batches = (n_edges + batch_size - 1) // batch_size
        
        self.model.train()
        
        pbar = tqdm(range(epochs), desc="Unlearning")
        
        for epoch in pbar:
            # Shuffle deleted edges each epoch
            np.random.shuffle(deleted_edges)
            epoch_loss = 0.0
            
            for i in range(n_batches):
                start = i * batch_size
                end = min((i + 1) * batch_size, n_edges)
                batch_edges = deleted_edges[start:end]
                
                # Sample subgraph nodes for NI loss to avoid OOM and balance loss (e.g., 4096)
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
        
        # Set inference masks
        self.model.inference_masks = masks
        self.model.current_masks = masks
        
        # Save unlearned model
        torch.save(self.model.state_dict(), self.save_path + '_unlearned')
    
    def evaluate(self, users, k_list=[10, 20]):
        """Evaluate on specific users"""
        return evaluate_user_subset(self.model, self.dataset, users, self.norm_adj_matrix, k_list, deleted_edges=self.deleted_edges)
    
    def calculate_zrf(self, incompetent_teacher, forget_samples, batch_size=256):
        """Calculate ZRF score"""
        self.model.eval()
        incompetent_teacher.eval()
        
        # Use masks stored in self.masks if available (set during unlearn)
        masks = getattr(self, 'masks', None)

        js_divs = []
        
        for start in range(0, len(forget_samples), batch_size):
            end = min(start + batch_size, len(forget_samples))
            batch_samples = forget_samples[start:end]
            
            users = [s[0] for s in batch_samples]
            u_tensor = torch.LongTensor(users).to(self.device)
            
            items = [s[1] for s in batch_samples]
            i_tensor = torch.LongTensor(items).to(self.device)
            
            with torch.no_grad():
                # Unlearned preds (Specific items)
                logits_unlearned = self.model.predict(u_tensor, i_tensor, self.norm_adj_matrix, masks=masks)
                probs_unlearned = torch.sigmoid(logits_unlearned)
                
                # Random preds
                logits_random = incompetent_teacher.predict(u_tensor, i_tensor, self.norm_adj_matrix)
                probs_random = torch.sigmoid(logits_random)
            
            # JS divergence construction for binary distribution [p, 1-p]
            dist_unlearned = torch.stack([probs_unlearned, 1 - probs_unlearned], dim=1)
            dist_random = torch.stack([probs_random, 1 - probs_random], dim=1)
            
            # Clamp for stability
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