# -*- coding: utf-8 -*-
# 模型定义模块 - 包含LightGCN和PromptedLightGCN模型

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
import scipy.sparse as sp
import numpy as np

# 导入配置
from config import COMMON_CONFIG, LIGHTGCN_CONFIG, UNLEARNING_CONFIG

# 设备配置
device = COMMON_CONFIG['device']

# LightGCN模型定义
class LightGCN(nn.Module):
    """LightGCN模型实现
    
    Light Graph Convolution Network模型，一种简化的图卷积网络，
    去除了传统GCN中的非线性激活和特征变换，仅保留邻居聚合操作。
    
    Attributes:
        n_users: 用户数量
        n_items: 物品数量
        latent_dim: 嵌入维度
        n_layers: 图卷积层数
        reg_weight: 正则化权重
        user_embedding: 用户嵌入层
        item_embedding: 物品嵌入层
        restore_user_e: 缓存的用户嵌入，用于加速评估
        restore_item_e: 缓存的物品嵌入，用于加速评估
    """
    def __init__(self, n_users, n_items, embedding_size=64, n_layers=3, reg_weight=1e-4, prompt_type=None, p_num=5):
        super(LightGCN, self).__init__()
        self.n_users = n_users
        self.n_items = n_items
        self.latent_dim = embedding_size
        self.n_layers = n_layers
        self.reg_weight = reg_weight
        self.prompt_type = prompt_type
        
        # 定义嵌入层
        self.user_embedding = nn.Embedding(n_users, embedding_size)
        self.item_embedding = nn.Embedding(n_items, embedding_size)
        
        # 初始化嵌入
        nn.init.xavier_uniform_(self.user_embedding.weight)
        nn.init.xavier_uniform_(self.item_embedding.weight)
        
        # 存储变量用于加速评估
        self.restore_user_e = None
        self.restore_item_e = None
        
        # 如果指定了prompt类型，创建相应的提示模块
        if prompt_type is not None:
            if prompt_type == 'simple':
                self.user_prompt = SimplePrompt(embedding_size)
                self.item_prompt = SimplePrompt(embedding_size)
            elif prompt_type == 'attention':
                self.user_prompt = GPFplusAtt(embedding_size, p_num)
                self.item_prompt = GPFplusAtt(embedding_size, p_num)
            else:
                raise ValueError(f"不支持的提示类型: {prompt_type}")
    
    def get_norm_adj_mat(self, interaction_matrix):
        """构建归一化的邻接矩阵
        
        Args:
            interaction_matrix: 用户-物品交互矩阵
            
        Returns:
            torch.sparse.FloatTensor: 归一化的邻接矩阵
        """
        # 构建归一化的邻接矩阵
        A = sp.lil_matrix((self.n_users + self.n_items, self.n_users + self.n_items), dtype=np.float32)
        inter_M = interaction_matrix
        inter_M_t = interaction_matrix.transpose()
        
        # 填充邻接矩阵 - 用户-物品交互
        for i, j, v in zip(inter_M.row, inter_M.col + self.n_users, [1] * inter_M.nnz):
            A[i, j] = v
        
        # 填充邻接矩阵 - 物品-用户交互
        for i, j, v in zip(inter_M_t.row + self.n_users, inter_M_t.col, [1] * inter_M_t.nnz):
            A[i, j] = v
        
        # 转换为CSR格式以加速计算
        A = A.tocsr()
        
        # 归一化邻接矩阵
        sumArr = (A > 0).sum(axis=1)
        # 添加epsilon避免除零警告
        diag = np.array(sumArr.flatten())[0] + 1e-7
        diag = np.power(diag, -0.5)
        D = sp.diags(diag)
        L = D * A * D
        
        # 转换为稀疏张量
        L = sp.coo_matrix(L)
        row = L.row
        col = L.col
        i = torch.LongTensor(np.array([row, col]))
        data = torch.FloatTensor(L.data)
        SparseL = torch.sparse_coo_tensor(i, data, torch.Size(L.shape))
        return SparseL
    
    def get_ego_embeddings(self):
        """获取用户和物品的初始嵌入，如果有提示模块则应用提示
        
        Returns:
            torch.Tensor: 拼接后的用户和物品嵌入
        """
        user_embeddings = self.user_embedding.weight
        item_embeddings = self.item_embedding.weight
        
        # 如果有提示模块，应用提示
        if self.prompt_type is not None:
            user_embeddings = self.user_prompt.add(user_embeddings)
            item_embeddings = self.item_prompt.add(item_embeddings)
            
        ego_embeddings = torch.cat([user_embeddings, item_embeddings], dim=0)
        return ego_embeddings
    
    def forward(self, norm_adj_matrix):
        """前向传播，执行图卷积
        
        Args:
            norm_adj_matrix: 归一化邻接矩阵
            
        Returns:
            tuple: (用户嵌入, 物品嵌入)
        """
        all_embeddings = self.get_ego_embeddings()
        embeddings_list = [all_embeddings]
        
        for layer_idx in range(self.n_layers):
            all_embeddings = torch.sparse.mm(norm_adj_matrix, all_embeddings)
            embeddings_list.append(all_embeddings)
        
        # 聚合多层嵌入
        lightgcn_all_embeddings = torch.stack(embeddings_list, dim=1)
        lightgcn_all_embeddings = torch.mean(lightgcn_all_embeddings, dim=1)
        
        # 分离用户和物品嵌入
        user_all_embeddings, item_all_embeddings = torch.split(
            lightgcn_all_embeddings, [self.n_users, self.n_items]
        )
        return user_all_embeddings, item_all_embeddings
    
    def calculate_loss(self, users, pos_items, neg_items, norm_adj_matrix):
        """计算BPR损失
        
        Args:
            users: 用户ID张量
            pos_items: 正样本物品ID张量
            neg_items: 负样本物品ID张量
            norm_adj_matrix: 归一化邻接矩阵
            
        Returns:
            torch.Tensor: 损失值
        """
        user_all_embeddings, item_all_embeddings = self.forward(norm_adj_matrix)
        
        u_embeddings = user_all_embeddings[users]
        pos_embeddings = item_all_embeddings[pos_items]
        neg_embeddings = item_all_embeddings[neg_items]
        
        # 计算正样本和负样本的得分
        pos_scores = torch.sum(u_embeddings * pos_embeddings, dim=1)
        neg_scores = torch.sum(u_embeddings * neg_embeddings, dim=1)
        
        # BPR损失
        loss = -torch.mean(torch.log(torch.sigmoid(pos_scores - neg_scores)))
        
        # 正则化损失
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
        """预测用户对物品的评分
        
        Args:
            users: 用户ID张量
            items: 物品ID张量
            norm_adj_matrix: 归一化邻接矩阵
            
        Returns:
            torch.Tensor: 预测评分
        """
        user_all_embeddings, item_all_embeddings = self.forward(norm_adj_matrix)
        
        u_embeddings = user_all_embeddings[users]
        i_embeddings = item_all_embeddings[items]
        
        scores = torch.sum(u_embeddings * i_embeddings, dim=1)
        return scores
    
    def full_sort_predict(self, users, norm_adj_matrix):
        """为所有用户预测所有物品的评分
        
        Args:
            users: 用户ID张量
            norm_adj_matrix: 归一化邻接矩阵
            
        Returns:
            torch.Tensor: 预测评分矩阵
        """
        # 每次预测前重置缓存的嵌入，确保在有prompt的情况下也能正确处理
        self.restore_user_e = None
        self.restore_item_e = None
        
        self.restore_user_e, self.restore_item_e = self.forward(norm_adj_matrix)
        
        u_embeddings = self.restore_user_e[users]
        scores = torch.matmul(u_embeddings, self.restore_item_e.transpose(0, 1))
        return scores

# 自定义glorot初始化函数
def glorot(tensor):
    """Glorot初始化
    
    Args:
        tensor: 要初始化的张量
    """
    if tensor is not None:
        stdv = torch.sqrt(torch.tensor(6.0 / (tensor.size(-2) + tensor.size(-1))))
        tensor.data.uniform_(-stdv, stdv)

# 简单的图提示模块，添加全局嵌入
class SimplePrompt(nn.Module):
    """简单的图提示模块，添加全局嵌入
    
    Attributes:
        global_emb: 全局嵌入参数
    """
    def __init__(self, in_channels: int):
        super(SimplePrompt, self).__init__()
        self.global_emb = nn.Parameter(torch.Tensor(1, in_channels))
        self.reset_parameters()

    def reset_parameters(self):
        glorot(self.global_emb)

    def add(self, x: Tensor):
        """添加全局嵌入到输入张量
        
        Args:
            x: 输入张量
            
        Returns:
            torch.Tensor: 添加全局嵌入后的张量
        """
        return x + self.global_emb

# 带注意力机制的图提示模块
class GPFplusAtt(nn.Module):
    """带注意力机制的图提示模块
    
    Attributes:
        p_list: 提示参数列表
        a: 注意力线性层
    """
    def __init__(self, in_channels: int, p_num: int):
        super(GPFplusAtt, self).__init__()
        self.p_list = nn.Parameter(torch.Tensor(p_num, in_channels))
        self.a = nn.Linear(in_channels, p_num)
        self.reset_parameters()

    def reset_parameters(self):
        glorot(self.p_list)
        self.a.reset_parameters()

    def add(self, x: Tensor):
        """使用注意力机制添加提示到输入张量
        
        Args:
            x: 输入张量
            
        Returns:
            torch.Tensor: 添加提示后的张量
        """
        score = self.a(x)
        weight = F.softmax(score, dim=1)
        p = weight.mm(self.p_list)
        return x + p

# 注意：PromptedLightGCN类已被移除，因为LightGCN类已经集成了prompt功能
# 请直接使用LightGCN类并设置prompt_type参数