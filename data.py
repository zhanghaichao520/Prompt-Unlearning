# -*- coding: utf-8 -*-
# 数据处理模块 - 包含数据集类和数据处理函数

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
import scipy.sparse as sp
from sklearn.model_selection import train_test_split

# 导入配置
from config import COMMON_CONFIG

# ML-100K数据集类
class ML100KDataset(Dataset):
    """MovieLens-100K数据集处理类
    
    处理MovieLens-100K数据集，包括数据加载、ID映射、训练测试集划分、
    负采样等功能。
    
    Attributes:
        data_path: 数据集路径
        test_size: 测试集比例
        negative_sample_size: 每个正样本对应的负样本数量
        data: 原始数据DataFrame
        user_ids: 用户ID列表
        item_ids: 物品ID列表
        n_users: 用户数量
        n_items: 物品数量
        user_id_map: 用户ID到索引的映射
        item_id_map: 物品ID到索引的映射
        interaction_matrix: 用户-物品交互矩阵
        train_data: 训练集数据
        test_data: 测试集数据
        train_user_items: 训练集中每个用户交互的物品集合
        test_user_items: 测试集中每个用户交互的物品集合
        all_user_items: 所有用户交互的物品集合
        train_samples: 训练样本列表
    """
    def __init__(self, data_path, test_size=0.2, negative_sample_size=1):
        self.data_path = data_path
        self.test_size = test_size
        self.negative_sample_size = negative_sample_size
        
        # 加载数据
        self.data = pd.read_csv(data_path, sep='\t')
        print(f"数据集大小: {len(self.data)}")
        
        # 获取用户和物品的唯一ID
        self.user_ids = self.data['user_id'].unique()
        self.item_ids = self.data['item_id'].unique()
        self.n_users = len(self.user_ids)
        self.n_items = len(self.item_ids)
        print(f"用户数量: {self.n_users}, 物品数量: {self.n_items}")
        
        # 创建ID映射
        self.user_id_map = {id: i for i, id in enumerate(self.user_ids)}
        self.item_id_map = {id: i for i, id in enumerate(self.item_ids)}
        
        # 转换数据中的ID
        self.data['user_idx'] = self.data['user_id'].map(lambda x: self.user_id_map[x])
        self.data['item_idx'] = self.data['item_id'].apply(lambda x: self.item_id_map[x])
        
        # 创建交互矩阵
        self.interaction_matrix = self._create_interaction_matrix()
        
        # 划分训练集和测试集
        self.train_data, self.test_data = self._split_train_test()
        
        # 创建用于训练的数据
        self.train_user_items = self._get_user_items_dict(self.train_data)
        self.test_user_items = self._get_user_items_dict(self.test_data)
        self.all_user_items = self._get_user_items_dict(self.data)
        
        # 准备训练样本
        self.train_samples = self._prepare_train_samples()
    
    def _create_interaction_matrix(self):
        """创建用户-物品交互矩阵
        
        Returns:
            scipy.sparse.coo_matrix: 用户-物品交互矩阵
        """
        row = self.data['user_idx'].values
        col = self.data['item_idx'].values
        data = np.ones(len(self.data))
        matrix = sp.coo_matrix((data, (row, col)), shape=(self.n_users, self.n_items), dtype=np.float32)
        return matrix
    
    def _split_train_test(self):
        """按用户划分训练集和测试集
        
        Returns:
            tuple: (训练集DataFrame, 测试集DataFrame)
        """
        train_data_list = []
        test_data_list = []
        
        for user in self.user_ids:
            user_data = self.data[self.data['user_id'] == user]
            if len(user_data) > 1:  # 确保用户至少有两个交互
                user_train, user_test = train_test_split(user_data, test_size=self.test_size, random_state=COMMON_CONFIG['seed'])
                train_data_list.append(user_train)
                test_data_list.append(user_test)
            else:
                train_data_list.append(user_data)  # 如果只有一个交互，放入训练集
        
        train_data = pd.concat(train_data_list)
        test_data = pd.concat(test_data_list) if test_data_list else pd.DataFrame(columns=self.data.columns)
        
        print(f"训练集大小: {len(train_data)}, 测试集大小: {len(test_data)}")
        return train_data, test_data
    
    def _get_user_items_dict(self, data):
        """创建用户-物品字典，记录每个用户交互过的物品
        
        Args:
            data: 数据DataFrame
            
        Returns:
            dict: 用户-物品字典
        """
        user_items = {}
        for user in range(self.n_users):
            user_items[user] = set(data[data['user_idx'] == user]['item_idx'].values)
        return user_items
    
    def _prepare_train_samples(self):
        """准备训练样本 (用户, 正样本物品, 负样本物品)
        
        Returns:
            list: 训练样本列表
        """
        samples = []
        for user, pos_items in self.train_user_items.items():
            for pos_item in pos_items:
                # 为每个正样本生成负样本
                for _ in range(self.negative_sample_size):
                    neg_item = np.random.randint(0, self.n_items)
                    while neg_item in self.all_user_items[user]:  # 确保负样本是用户未交互过的
                        neg_item = np.random.randint(0, self.n_items)
                    samples.append((user, pos_item, neg_item))
        return samples
    
    def __len__(self):
        """返回训练样本数量
        
        Returns:
            int: 训练样本数量
        """
        return len(self.train_samples)
    
    def __getitem__(self, idx):
        """获取指定索引的训练样本
        
        Args:
            idx: 样本索引
            
        Returns:
            tuple: (用户ID, 正样本物品ID, 负样本物品ID)
        """
        user, pos_item, neg_item = self.train_samples[idx]
        return user, pos_item, neg_item
    
    def get_test_samples(self):
        """返回测试集中的所有用户
        
        Returns:
            list: 测试用户ID列表
        """
        return list(self.test_user_items.keys())
    
    def generate_candidates(self, users, batch_size=1024):
        """为指定用户生成候选物品
        
        Args:
            users: 用户ID列表
            batch_size: 批处理大小
            
        Returns:
            list: 候选物品ID列表
        """
        # 默认返回所有物品作为候选
        return list(range(self.n_items))