# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
import scipy.sparse as sp
import numpy as np

# Copied from RecBole layers
class SparseDropout(nn.Module):
    """
    This is a Module that execute Dropout on Pytorch sparse tensor.
    """

    def __init__(self, p=0.5):
        super(SparseDropout, self).__init__()
        # p is ratio of dropout
        # convert to keep probability
        self.kprob = 1 - p

    def forward(self, x):
        if not self.training:
            return x

        mask = ((torch.rand(x._values().size()) + self.kprob).floor()).type(torch.bool)
        rc = x._indices()[:, mask]
        val = x._values()[mask] * (1.0 / self.kprob)
        return torch.sparse_coo_tensor(rc, val, x.shape)

class BiGNNLayer(nn.Module):
    r"""Propagate a layer of Bi-interaction GNN

    .. math::
        output = (L+I)EW_1 + LE \otimes EW_2
    """

    def __init__(self, in_dim, out_dim):
        super(BiGNNLayer, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.linear = torch.nn.Linear(in_features=in_dim, out_features=out_dim)
        self.interActTransform = torch.nn.Linear(
            in_features=in_dim, out_features=out_dim
        )

    def forward(self, lap_matrix, eye_matrix, features):
        # for GCF ajdMat is a (N+M) by (N+M) mat
        # lap_matrix L = D^-1(A)D^-1 # 拉普拉斯矩阵
        x = torch.sparse.mm(lap_matrix, features)

        inter_part1 = self.linear(features + x)
        inter_feature = torch.mul(x, features)
        inter_part2 = self.interActTransform(inter_feature)

        return inter_part1 + inter_part2

# Prompt modules
def glorot(tensor):
    if tensor is not None:
        stdv = torch.sqrt(torch.tensor(6.0 / (tensor.size(-2) + tensor.size(-1))))
        tensor.data.uniform_(-stdv, stdv)

class SimplePrompt(nn.Module):
    def __init__(self, in_channels: int):
        super(SimplePrompt, self).__init__()
        self.global_emb = nn.Parameter(torch.Tensor(1, in_channels))
        self.reset_parameters()

    def reset_parameters(self):
        glorot(self.global_emb)

    def add(self, x: torch.Tensor):
        return x + self.global_emb

class GPFplusAtt(nn.Module):
    def __init__(self, in_channels: int, p_num: int):
        super(GPFplusAtt, self).__init__()
        self.p_list = nn.Parameter(torch.Tensor(p_num, in_channels))
        self.a = nn.Linear(in_channels, p_num)
        self.reset_parameters()

    def reset_parameters(self):
        glorot(self.p_list)
        self.a.reset_parameters()

    def add(self, x: torch.Tensor):
        score = self.a(x)
        weight = F.softmax(score, dim=1)
        p = weight.mm(self.p_list)
        return x + p

# RecBole NGCF adapted for Unlearning
class NGCF(nn.Module):
    r"""NGCF is a model that incorporate GNN for recommendation.
    We implement the model following the original author with a pairwise training mode.
    """

    def __init__(self, n_users, n_items, embedding_size=64, n_layers=3, 
                 node_dropout=0.1, message_dropout=0.1, reg_weight=1e-5,
                 prompt_type=None, p_num=5):
        super(NGCF, self).__init__()
        
        self.n_users = n_users
        self.n_items = n_items
        self.embedding_size = embedding_size
        self.hidden_size_list = [embedding_size] * n_layers
        self.hidden_size_list = [self.embedding_size] + self.hidden_size_list
        self.node_dropout = node_dropout
        self.message_dropout = message_dropout
        self.reg_weight = reg_weight
        self.prompt_type = prompt_type

        # define layers and loss
        self.sparse_dropout = SparseDropout(self.node_dropout)
        self.user_embedding = nn.Embedding(self.n_users, self.embedding_size)
        self.item_embedding = nn.Embedding(self.n_items, self.embedding_size)
        self.emb_dropout = nn.Dropout(self.message_dropout)
        self.GNNlayers = torch.nn.ModuleList()
        for idx, (input_size, output_size) in enumerate(
            zip(self.hidden_size_list[:-1], self.hidden_size_list[1:])
        ):
            self.GNNlayers.append(BiGNNLayer(input_size, output_size))
        
        # storage variables for full sort evaluation acceleration
        self.restore_user_e = None
        self.restore_item_e = None

        # parameters initialization
        self.apply(self._init_weights)
        
        # Prompt initialization
        if prompt_type is not None:
            if prompt_type == 'simple':
                self.user_prompt = SimplePrompt(embedding_size)
                self.item_prompt = SimplePrompt(embedding_size)
            elif prompt_type == 'attention':
                self.user_prompt = GPFplusAtt(embedding_size, p_num)
                self.item_prompt = GPFplusAtt(embedding_size, p_num)
            else:
                raise ValueError(f"Unsupported prompt type: {prompt_type}")
    
    def _init_weights(self, module):
        if isinstance(module, nn.Embedding):
            nn.init.xavier_normal_(module.weight.data)
        elif isinstance(module, nn.Linear):
            nn.init.xavier_normal_(module.weight.data)
            if module.bias is not None:
                nn.init.constant_(module.bias.data, 0)
    
    def get_norm_adj_mat(self, interaction_matrix):
        # NOTE: In RecBole this is done inside __init__, but here we do it externally or pass it.
        # But for compatibility with existing code, we provide this helper.
        # This code is copied from models.py or adapted to return sparse tensor.
        A = sp.dok_matrix((self.n_users + self.n_items, self.n_users + self.n_items), dtype=np.float32)
        inter_M = interaction_matrix
        inter_M_t = interaction_matrix.transpose()
        data_dict = dict(zip(zip(inter_M.row, inter_M.col + self.n_users), [1] * inter_M.nnz))
        data_dict.update(dict(zip(zip(inter_M_t.row + self.n_users, inter_M_t.col), [1] * inter_M_t.nnz)))
        A._update(data_dict)
        
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
    
    def get_eye_mat(self):
        r"""Construct the identity matrix with the size of  n_items+n_users.

        Returns:
            Sparse tensor of the identity matrix. Shape of (n_items+n_users, n_items+n_users)
        """
        num = self.n_items + self.n_users  # number of column of the square matrix
        i = torch.LongTensor([range(0, num), range(0, num)])
        val = torch.FloatTensor([1] * num)  # identity matrix
        return torch.sparse_coo_tensor(i, val)

    def get_ego_embeddings(self):
        r"""Get the embedding of users and items and combine to an embedding matrix.

        Returns:
            Tensor of the embedding matrix. Shape of (n_items+n_users, embedding_dim)
        """
        user_embeddings = self.user_embedding.weight
        item_embeddings = self.item_embedding.weight
        
        if self.prompt_type is not None:
            user_embeddings = self.user_prompt.add(user_embeddings)
            item_embeddings = self.item_prompt.add(item_embeddings)
            
        ego_embeddings = torch.cat([user_embeddings, item_embeddings], dim=0)
        return ego_embeddings

    def forward(self, norm_adj_matrix, training=True):
        # NOTE: RecBole uses internal self.norm_adj_matrix. 
        # We pass it as argument to support different graphs (e.g. for unlearning).
        
        # We also need eye_matrix. RecBole precomputes it. We can compute it on the fly or require it.
        # For simplicity, we construct it here or assume passed?
        # Let's construct it on device of norm_adj_matrix
        device = norm_adj_matrix.device
        num = self.n_users + self.n_items
        i = torch.LongTensor([range(0, num), range(0, num)]).to(device)
        val = torch.FloatTensor([1] * num).to(device)
        eye_matrix = torch.sparse_coo_tensor(i, val, (num, num)).to(device)
        
        A_hat = (
            self.sparse_dropout(norm_adj_matrix)
            if self.node_dropout != 0 and training
            else norm_adj_matrix
        )
        all_embeddings = self.get_ego_embeddings()
        embeddings_list = [all_embeddings]
        for gnn in self.GNNlayers:
            all_embeddings = gnn(A_hat, eye_matrix, all_embeddings)
            all_embeddings = nn.LeakyReLU(negative_slope=0.2)(all_embeddings)
            all_embeddings = self.emb_dropout(all_embeddings)
            all_embeddings = F.normalize(all_embeddings, p=2, dim=1)
            embeddings_list += [
                all_embeddings
            ]  # storage output embedding of each layer
        ngcf_all_embeddings = torch.cat(embeddings_list, dim=1)

        user_all_embeddings, item_all_embeddings = torch.split(
            ngcf_all_embeddings, [self.n_users, self.n_items]
        )

        return user_all_embeddings, item_all_embeddings

    def calculate_loss(self, users, pos_items, neg_items, norm_adj_matrix):
        user_all_embeddings, item_all_embeddings = self.forward(norm_adj_matrix, training=True)
        
        u_embeddings = user_all_embeddings[users]
        pos_embeddings = item_all_embeddings[pos_items]
        neg_embeddings = item_all_embeddings[neg_items]
        
        pos_scores = torch.mul(u_embeddings, pos_embeddings).sum(dim=1)
        neg_scores = torch.mul(u_embeddings, neg_embeddings).sum(dim=1)
        
        # BPR Loss
        loss = -torch.mean(torch.log(torch.sigmoid(pos_scores - neg_scores)))
        
        # Regularization
        u_ego_embeddings = self.user_embedding(users)
        pos_ego_embeddings = self.item_embedding(pos_items)
        neg_ego_embeddings = self.item_embedding(neg_items)
        
        reg_loss = torch.mean(
            torch.norm(u_ego_embeddings, p=2, dim=1) +
            torch.norm(pos_ego_embeddings, p=2, dim=1) +
            torch.norm(neg_ego_embeddings, p=2, dim=1)
        )
        
        return loss + self.reg_weight * reg_loss

    def predict(self, users, items, norm_adj_matrix):
        user_all_embeddings, item_all_embeddings = self.forward(norm_adj_matrix, training=False)
        u_embeddings = user_all_embeddings[users]
        i_embeddings = item_all_embeddings[items]
        scores = torch.sum(u_embeddings * i_embeddings, dim=1)
        return scores
    
    def full_sort_predict(self, users, norm_adj_matrix, **kwargs):
        self.restore_user_e = None
        self.restore_item_e = None
        user_all_embeddings, item_all_embeddings = self.forward(norm_adj_matrix, training=False)
        self.restore_user_e = user_all_embeddings
        self.restore_item_e = item_all_embeddings
        
        u_embeddings = self.restore_user_e[users]
        scores = torch.matmul(u_embeddings, self.restore_item_e.transpose(0, 1))
        return scores
