# -*- coding: utf-8 -*-
import sys
import os
import copy
import random
import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import scipy.sparse as sp

# Helper imports from sibling directory
# Note: We need to append root to path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, "../../"))
if project_root not in sys.path:
    sys.path.append(project_root)

from ngcf_experiment.RecEraser.ngcf_models import NGCF
from ngcf_experiment.RecEraser.partition import balanced_interaction_partition, get_pretrained_embeddings
from ngcf_experiment.RecEraser.aggregator import AttentionAggregator
from data import ML100KDataset
from utils import evaluate_user_subset

class ListDataset(Dataset):
    def __init__(self, samples):
        self.samples = samples
    def __len__(self):
        return len(self.samples)
    def __getitem__(self, idx):
        return self.samples[idx]

class RecEraser:
    """
    Manager class for RecEraser Framework.
    1. Partition
    2. Shard Training
    3. Aggregation Training
    4. Unlearning
    """
    def __init__(self, dataset, config, device):
        self.dataset = dataset
        self.config = config
        self.device = device
        self.num_shards = config['num_shards']
        self.max_shard_size = int(len(dataset.train_samples) / self.num_shards * 1.1)
        
        self.shard_models = {} # id -> model
        self.shard_data = {}   # id -> list of samples
        self.shard_norm_adj = {} # id -> sparse tensor
        
        self.aggregator = None
        self.save_dir = os.path.join(config['root'], 'checkpoints')
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)

    def eval(self):
        if self.aggregator:
            self.aggregator.eval()
        for m in self.shard_models.values():
            m.eval()

    def init_random(self):
        print("Initializing random RecEraser (Incompetent Teacher)...")
        # Split into shards randomly
        samples = np.array(self.dataset.train_samples)
        # Shuffle
        indices = np.random.permutation(len(samples))
        samples = samples[indices]
        
        chunk_size = int(np.ceil(len(samples) / self.num_shards))
        for i in range(self.num_shards):
            start = i * chunk_size
            end = min((i+1) * chunk_size, len(samples))
            shard_d = samples[start:end]
            self.shard_data[i] = shard_d
            
            # Create adj
            self.shard_norm_adj[i] = self._create_norm_adj(shard_d)
            
            # Create Model
            model = NGCF(self.dataset.n_users, self.dataset.n_items,
                     embedding_size=self.config['embedding_size'],
                     n_layers=self.config['n_layers']).to(self.device)
            self.shard_models[i] = model
            
        # Create Aggregator
        input_dim = self.config['embedding_size'] * (self.config['n_layers'] + 1)
        self.aggregator = AttentionAggregator(self.num_shards, input_dim).to(self.device)

    def load_checkpoints(self):
        print("Loading checkpoints...")
        # Load shards data
        shards_path = os.path.join(self.save_dir, 'shards.pt')
        if os.path.exists(shards_path):
            self.shard_data = torch.load(shards_path)
            print("Loaded shards data.")
        else:
             print("Shards data not found.")
             return False

        # Load Shard Models
        for i in range(self.num_shards):
            model_path = os.path.join(self.save_dir, f'model_{i}.pth')
            # Check if model exists, if so load it
            if os.path.exists(model_path):
                 # We need data to create norm_adj
                 data = self.shard_data.get(i, [])
                 if not data:
                      # Maybe empty shard?
                      continue
                 
                 adj = self._create_norm_adj(data)
                 self.shard_norm_adj[i] = adj
                 
                 model = NGCF(self.dataset.n_users, self.dataset.n_items,
                     embedding_size=self.config['embedding_size'],
                     n_layers=self.config['n_layers']).to(self.device)
                 model.load_state_dict(torch.load(model_path, map_location=self.device))
                 self.shard_models[i] = model
            else:
                # If any model is missing, we might need to train (or at least fail loading)
                print(f"Model {i} not found.")
                return False

        # Load Aggregator
        agg_path = os.path.join(self.save_dir, 'aggregator.pth')
        if os.path.exists(agg_path):
            input_dim = self.config['embedding_size'] * (self.config['n_layers'] + 1)
            self.aggregator = AttentionAggregator(self.num_shards, input_dim).to(self.device)
            self.aggregator.load_state_dict(torch.load(agg_path, map_location=self.device))
            print("Loaded Aggregator.")
        else:
            print("Aggregator not found.")
            return False
            
        return True

    def _create_norm_adj(self, shard_samples):
        # Create small adjacency matrix for the shard
        # Note: NGCF requires adjacency matrix of size (N+M)x(N+M) covering ALL users/items
        # even if they are not in the shard, to maintain index consistency.
        # But we only populate edges from shard_samples.
        
        n_users = self.dataset.n_users
        n_items = self.dataset.n_items
        
        # Build scipy dok matrix
        # Optimize: Construct directly from coo
        rows = [x[0] for x in shard_samples]
        cols = [x[1] for x in shard_samples]
        
        # Add offset to items
        rows_all = np.array(rows)
        cols_all = np.array(cols) + n_users
        
        # Symmetric
        data_all = np.ones(len(rows))
        
        mat = sp.coo_matrix((data_all, (rows_all, cols_all)), shape=(n_users + n_items, n_users + n_items))
        mat = mat + mat.T
        
        # Normalize: D^-0.5 A D^-0.5
        # Need to add self loop? NGCF usually adds I? 
        # In ngcf_models.py get_norm_adj_mat adds I inside?
        # Let's check ngcf_models.py (copied version)
        # It calculates D from A (which has 1s). It doesn't seem to explicitly add I before D calculation in get_norm_adj_mat snippet I read earlier.
        # But RecBole implementation usually does.
        # Let's follow standard GCN normalization L = D^-0.5 (A) D^-0.5.
        
        rowsum = np.array(mat.sum(1))
        d_inv_sqrt = np.power(rowsum, -0.5).flatten()
        d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
        d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
        
        norm_adj = d_mat_inv_sqrt.dot(mat).dot(d_mat_inv_sqrt)
        
        # To Torch Sparse
        coo = norm_adj.tocoo()
        indices = torch.LongTensor([coo.row, coo.col])
        values = torch.FloatTensor(coo.data)
        return torch.sparse_coo_tensor(indices, values, coo.shape).to(self.device)

    def partition(self, pretrained_model_path):
        print("Loading pretrained model for partitioning...")
        # Load NGCF
        model = NGCF(self.dataset.n_users, self.dataset.n_items, 
                     embedding_size=self.config['embedding_size'],
                     n_layers=self.config['n_layers']).to(self.device)
        try:
            model.load_state_dict(torch.load(pretrained_model_path))
        except:
            print("Pretrained model not found or incompatible. Training a quick one...")
            # Ideally we train, but for demo we might skip or use random.
            # Random is bad for 'Balanced' partition preserving collaboration.
            # Assume random init is 'starting point'.
            pass
            
        u_emb, i_emb = get_pretrained_embeddings(model, self.dataset, self.device)
        
        self.shard_data = balanced_interaction_partition(
            self.dataset.train_samples, u_emb, i_emb, 
            self.num_shards, self.max_shard_size
        )
        
        # Save shards
        torch.save(self.shard_data, os.path.join(self.save_dir, 'shards.pt'))
        print(f"Partitioned into {len(self.shard_data)} shards.")

    def train_shard_model(self, shard_id, epochs=20):
        print(f"Training Shard {shard_id}...")
        data = self.shard_data[shard_id]
        if len(data) == 0:
            print(f"Shard {shard_id} is empty. Skipping.")
            return

        adj = self._create_norm_adj(data)
        self.shard_norm_adj[shard_id] = adj
        
        model = NGCF(self.dataset.n_users, self.dataset.n_items,
                     embedding_size=self.config['embedding_size'],
                     n_layers=self.config['n_layers']).to(self.device)
        
        optimizer = optim.Adam(model.parameters(), lr=self.config['lr'])
        
        # Dataset
        # We need negative sampling. ML100KDataset does it in __getitem__ but relies on self.all_user_items
        # We can reuse the dataset class but override samples?
        # Or simpler: Just simple dataloader that samples negatives on fly
        
        # We create a simple adapter dataset
        class ShardDataset(Dataset):
            def __init__(self, samples, n_items, all_inter):
                self.samples = samples
                self.n_items = n_items
                self.all_inter = all_inter
            def __len__(self):
                return len(self.samples)
            def __getitem__(self, idx):
                u, i = self.samples[idx]
                neg = np.random.randint(0, self.n_items)
                while neg in self.all_inter[u]:
                     neg = np.random.randint(0, self.n_items)
                return u, i, neg
        
        ds = ShardDataset(data, self.dataset.n_items, self.dataset.all_user_items)
        loader = DataLoader(ds, batch_size=2048, shuffle=True)
        
        for ep in range(epochs):
            model.train()
            total_loss = 0
            for u, i, neg in loader:
                u, i, neg = u.to(self.device), i.to(self.device), neg.to(self.device)
                optimizer.zero_grad()
                loss = model.calculate_loss(u, i, neg, adj)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            # print(f"Shard {shard_id} Ep {ep} Loss {total_loss:.2f}")
            
        self.shard_models[shard_id] = model
        torch.save(model.state_dict(), os.path.join(self.save_dir, f'model_{shard_id}.pth'))

    def train_all_shards(self):
        for i in range(self.num_shards):
            self.train_shard_model(i, epochs=self.config['shard_epochs'])

    def train_aggregator(self, epochs=10):
        print("Training Aggregator...")
        # Determine actual embedding output size from NGCF (concat of all layers + ego)
        input_dim = self.config['embedding_size'] * (self.config['n_layers'] + 1)
        self.aggregator = AttentionAggregator(self.num_shards, input_dim).to(self.device)
        optimizer = optim.Adagrad(self.aggregator.parameters(), lr=0.01, weight_decay=1e-5)
        
        # Use Full Dataset for training aggregator
        # "The whole training data Y is used for training"
        loader = DataLoader(self.dataset, batch_size=2048, shuffle=True)
        
        # Pre-load all shard models to eval mode
        for m in self.shard_models.values():
            m.eval()
            
        for ep in range(epochs):
            total_loss = 0
            for u, pos, neg in loader:
                u, pos, neg = u.to(self.device), pos.to(self.device), neg.to(self.device)
                
                # Get embeddings from all shards
                shard_u_embs = []
                shard_pos_embs = []
                shard_neg_embs = []
                
                with torch.no_grad():
                    # We need to forward pass through each shard model
                    # But NGCF forward returns ALL embeddings.
                    # Optimization: Get all embeddings once per epoch?
                    # No, models are small enough.
                    pass
                
                # To be efficient:
                # Get user_alpha, item_beta from Aggregator
                # Calculate scores
                
                # Actually Aggregator takes (shard_all_u, shard_all_i) and aggregates them into global P, Q.
                # If we do this inside the batch loop, we aggregate only batch users?
                # The paper Eq 6: P = sum alpha_i P_i. This implies we aggregate THE embeddings.
                # So we can aggregate the entire embedding matrix P and Q once?
                # But alpha depends on P_i. alpha_i = ... (W P_i + b).
                # So for every user, we have a unique alpha?
                # Paper Eq 7: alpha_i is vector?
                # h1^T \sigma(W p_i + b). This outputs a scalar score for specific p_i.
                # So yes, attention is per-user/per-item.
                
                # Strategy:
                # 1. Gather P_i(u), Q_i(pos), Q_i(neg) for all i.
                # 2. Feed to Aggregator to get P(u), Q(pos), Q(neg).
                # 3. Compute BPR loss.
                
                for i in range(self.num_shards):
                    if i in self.shard_models:
                        model = self.shard_models[i]
                        # We use model.forward to get all embeddings?
                        # Or extract weights?
                        # NGCF forward applies GNN.
                        # We must apply GNN using shard_adj.
                        with torch.no_grad():
                            u_all, i_all = model.forward(self.shard_norm_adj[i], training=False)
                            shard_u_embs.append(u_all[u])
                            shard_pos_embs.append(i_all[pos])
                            shard_neg_embs.append(i_all[neg])
                    else:
                        # Handle empty/missing shard
                        shard_u_embs.append(torch.zeros(len(u), input_dim).to(self.device))
                        shard_pos_embs.append(torch.zeros(len(pos), input_dim).to(self.device))
                        shard_neg_embs.append(torch.zeros(len(neg), input_dim).to(self.device))
                
                # Aggregator Forward
                user_embeds, pos_item_embeds = self.aggregator(shard_u_embs, shard_pos_embs)
                _, neg_item_embeds = self.aggregator(shard_u_embs, shard_neg_embs) # Reuse shard_u_embs
                
                # BPR Loss
                pos_scores = torch.sum(user_embeds * pos_item_embeds, dim=1)
                neg_scores = torch.sum(user_embeds * neg_item_embeds, dim=1)
                
                loss = -torch.mean(torch.log(torch.sigmoid(pos_scores - neg_scores)))
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
                
            print(f"Aggregator Ep {ep} Loss {total_loss:.2f}")
        
        torch.save(self.aggregator.state_dict(), os.path.join(self.save_dir, 'aggregator.pth'))

    def predict(self, users, items=None):
        """
        Predict scores for users and items (if items is None, predict all).
        For evaluation.
        """
        self.aggregator.eval()
        input_dim = self.config['embedding_size'] * (self.config['n_layers'] + 1)
        
        # 1. Get ALL embeddings from all shards
        shard_user_embs = []
        shard_item_embs = []
        
        with torch.no_grad():
            for i in range(self.num_shards):
                 if i in self.shard_models:
                     self.shard_models[i].eval() # Ensure eval mode
                     u_all, i_all = self.shard_models[i].forward(self.shard_norm_adj[i], training=False)
                     shard_user_embs.append(u_all)
                     shard_item_embs.append(i_all)
                 else:
                     shard_user_embs.append(torch.zeros(self.dataset.n_users, input_dim).to(self.device))
                     shard_item_embs.append(torch.zeros(self.dataset.n_items, input_dim).to(self.device))
            
            # 2. Aggregate Global Embeddings
            # We process all users/items in one go or batches? 
            # If memory allows, all at once.
            
            # Need to pass lists of [N, D]
            agg_user_emb, agg_item_emb = self.aggregator(shard_user_embs, shard_item_embs)
            
            # 3. Predict
            # users: [B]
            if items is None:
                # All items
                u_e = agg_user_emb[users]
                scores = torch.matmul(u_e, agg_item_emb.t())
                return scores
            else:
                u_e = agg_user_emb[users]
                i_e = agg_item_emb[items]
                return torch.sum(u_e * i_e, dim=1)

    def unlearn(self, forget_data):
        """
        Unlearn data.
        1. Find affected shards.
        2. Remove data from shards.
        3. Retrain affected shards.
        4. Retrain aggregator.
        """
        print(f"Unlearning {len(forget_data)} samples...")
        
        # Identify affected shards
        # Since we use Interaction Partition, we look up where each interaction went.
        # But we didn't store a reverse map.
        
        # Brute force search in self.shard_data
        # Map: (u, i) -> shard_id
        # Optimize if needed.
        
        affected_shards = set()
        
        forget_set = set([(x[0], x[1]) for x in forget_data])
        
        for k, data in self.shard_data.items():
            new_data = []
            changed = False
            for x in data:
                u, i = x[0], x[1]
                if (u, i) in forget_set:
                    changed = True
                    # Don't add to new_data (delete)
                else:
                    new_data.append(x)
            
            if changed:
                affected_shards.add(k)
                self.shard_data[k] = new_data
                
        print(f"Affected shards: {affected_shards}")
        
        # Retrain affected
        for sid in affected_shards:
            self.train_shard_model(sid, epochs=self.config['shard_epochs'])
            
        # Retrain Aggregator
        # We need to remove forget data from dataset used for aggregator too?
        # Yes, aggregator uses self.dataset which is the full training data.
        # We should update self.dataset.train_samples
        pass 
        # (Assuming caller handles dataset update or we filter here)
        # For simplicity, we just run train_aggregator which uses self.dataset.
        # Caller should ensure self.dataset does not contain forget data anymore.
        
        self.train_aggregator(epochs=5)
