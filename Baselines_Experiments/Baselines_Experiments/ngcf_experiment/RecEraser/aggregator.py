import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class AttentionAggregator(nn.Module):
    """
    Attention-based Adaptive Aggregation for RecEraser.
    Aggregates embeddings from multiple submodels (shards).
    """
    def __init__(self, num_shards, embedding_size, attention_size=32):
        super(AttentionAggregator, self).__init__()
        self.num_shards = num_shards
        self.embedding_size = embedding_size
        self.attention_size = attention_size
        
        # Transformation matrices W_i, b_i for each shard
        # We can implement this as a list of Linear layers or a single batched operation.
        # Since num_shards is small (e.g. 10), ModuleList is fine.
        
        self.transform_layers = nn.ModuleList([
            nn.Linear(embedding_size, embedding_size, bias=True) 
            for _ in range(num_shards)
        ])
        
        # Attention Mechanism
        # alpha_i = h1^T \sigma(W1 p_i^{tr} + b1)
        # Using separate attention for Users and Items as per paper
        
        # User Attention
        self.user_att_layer1 = nn.Linear(embedding_size, attention_size, bias=True) # W1, b1
        self.user_att_layer2 = nn.Linear(attention_size, 1, bias=False) # h1
        
        # Item Attention
        self.item_att_layer1 = nn.Linear(embedding_size, attention_size, bias=True) # W2, b2
        self.item_att_layer2 = nn.Linear(attention_size, 1, bias=False) # h2
        
    def forward(self, shard_user_embeddings, shard_item_embeddings):
        """
        Args:
            shard_user_embeddings: List of tensors [batch_size, embed_dim] from each shard
            shard_item_embeddings: List of tensors [batch_size, embed_dim] from each shard
        Returns:
            agg_user_embed: [batch_size, embed_dim]
            agg_item_embed: [batch_size, embed_dim]
        """
        
        # 1. Transformation
        trans_user_embeds = []
        trans_item_embeds = []
        
        for i in range(self.num_shards):
            # Apply W_i P_i + b_i
            u_emb = self.transform_layers[i](shard_user_embeddings[i])
            i_emb = self.transform_layers[i](shard_item_embeddings[i])
            trans_user_embeds.append(u_emb)
            trans_item_embeds.append(i_emb)
            
        # Stack: [num_shards, batch_size, embed_dim] -> [batch_size, num_shards, embed_dim]
        # P^{tr}
        stack_user = torch.stack(trans_user_embeds, dim=1) 
        stack_item = torch.stack(trans_item_embeds, dim=1)
        
        # 2. Attention Weights
        # Users
        # \sigma(W1 P + b1)
        u_att = torch.relu(self.user_att_layer1(stack_user)) 
        # h1^T ...
        u_scores = self.user_att_layer2(u_att) # [batch_size, num_shards, 1]
        u_alphas = F.softmax(u_scores, dim=1) # Normalize over shards
        
        # Items
        i_att = torch.relu(self.item_att_layer1(stack_item))
        i_scores = self.item_att_layer2(i_att)
        i_betas = F.softmax(i_scores, dim=1)
        
        # 3. Aggregation (Weighted Sum)
        # sum(alpha * P_tr)
        agg_user_embed = torch.sum(u_alphas * stack_user, dim=1)
        agg_item_embed = torch.sum(i_betas * stack_item, dim=1)
        
        return agg_user_embed, agg_item_embed
