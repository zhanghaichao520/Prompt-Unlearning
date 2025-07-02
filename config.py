# -*- coding: utf-8 -*-
# 配置文件 - 集中管理所有参数

import torch
dataset_name="ml-1m"
# 通用配置
COMMON_CONFIG = {
    # 数据集路径
    'data_path': f"dataset/{dataset_name}.inter",
    
    # 随机种子
    'seed': 42,
    
    # 设备配置
    'device': torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    
    # 评估指标配置
    'k_list': [10,20],
    'batch_size': 2048,
}

# LightGCN模型配置
LIGHTGCN_CONFIG = {
    # 模型参数
    'embedding_size': 64,
    'n_layers': 4,  # 基础训练层数
    'reg_weight': 1e-4,
    
    # 训练参数
    'lr': 0.0005,
    'epochs': 500,
    'eval_freq': 5,
    'patience': 50,  # 早停耐心值
    
    # 模型保存路径
    'save_path': f'pretrain_checkpoints/best_lightgcn_{dataset_name}.pth',
}

# 遗忘学习配置
UNLEARNING_CONFIG = {
    # 模型参数
    'embedding_size': 64,
    'n_layers': 4,  # 遗忘学习使用更深的网络
    'reg_weight': 1e-4,
    
    # 训练参数
    'lr': 0.005,
    'epochs': 300,
    'batch_size': 2048,
    'forget_ratio': 0.01,  # 遗忘集比例
    'remain_ratio': 0.01,  # 保留集比例
    'KL_temperature': 1.0,
    
    # 蒸馏策略配置
    'loss_type': 'DAD',  # 可选: 'kl', 'bpr', 'combined', 'separate', 'WRD', 'KL', 'DAD'
    'alpha': 0.5,  # 平衡不同损失的权重参数 (降低以增加保留集BPR损失的权重)
    'max_pairs': 1000,  # BPR损失最大采样对数 (增加以提高采样多样性)
    
    # 排序蒸馏策略参数 - 改进版
    'lamda': 10.0,  # 位置重要性权重的锐度参数 (降低以更均匀地关注不同位置)
    'mu': 7.0,  # 排序差异权重的锐度参数 (增加以强化重要样本的权重)
    'K': 5,  # 教师模型的示例排名长度 (增加以捕获更多排序信息)
    
    # 提示配置
    'prompt_type': 'attention',  # 'simple' 或 'attention'
    'p_num': 20,  # 注意力提示的数量
    
    # 早停相关参数
    'patience': 5,  # 早停耐心值
    'validation_interval': 5,  # 验证间隔
    
    # 模型保存路径
    'prompt_save_path': f'unlearning_checkpoints/lightgcn_unlearned_prompt_{dataset_name}.pth',
}