# -*- coding: utf-8 -*-
# LightGCN模型评估脚本 - 重构版

import os
import torch
import numpy as np

# 导入自定义模块
from config import COMMON_CONFIG, LIGHTGCN_CONFIG, UNLEARNING_CONFIG
from data import ML100KDataset
from models import LightGCN
from utils import evaluate, print_metrics, set_seed
from unlearning import load_prompt_for_inference

# 设置随机种子
set_seed(COMMON_CONFIG['seed'])

# 设备配置
device = COMMON_CONFIG['device']
print(f"使用设备: {device}")

# 评估函数
def evaluate_model(model_path=None, prompt_path=None, prompt_type='attention', p_num=50):
    """评估LightGCN模型
    
    Args:
        model_path: 模型路径，默认为None
        prompt_path: 提示参数路径，默认为None
        prompt_type: 提示类型，默认为None
        p_num: 提示数量，默认为None
        
    Returns:
        dict: 评估结果
    """
    # 加载配置参数
    data_path = COMMON_CONFIG['data_path']
    embedding_size = LIGHTGCN_CONFIG['embedding_size']
    n_layers = LIGHTGCN_CONFIG['n_layers']
    reg_weight = LIGHTGCN_CONFIG['reg_weight']
    batch_size = COMMON_CONFIG['batch_size']
    k_list = COMMON_CONFIG['k_list']
    
    # 如果提供了提示路径，使用遗忘学习配置
    if prompt_path is not None:
        embedding_size = UNLEARNING_CONFIG['embedding_size']
        n_layers = UNLEARNING_CONFIG['n_layers']
        reg_weight = UNLEARNING_CONFIG['reg_weight']
        prompt_type = UNLEARNING_CONFIG['prompt_type']
        p_num = UNLEARNING_CONFIG['p_num']
    
    # 加载数据集
    print("加载数据集...")
    dataset = ML100KDataset(data_path)
    
    # 创建模型
    print("创建模型...")
    model = LightGCN(
        n_users=dataset.n_users,
        n_items=dataset.n_items,
        embedding_size=embedding_size,
        n_layers=n_layers,
        reg_weight=reg_weight
    ).to(device)
    
    # 如果有预训练模型，加载模型参数
    if model_path and os.path.exists(model_path):
        print(f"加载预训练模型: {model_path}")
        model.load_state_dict(torch.load(model_path))
    else:
        print("使用随机初始化的模型")
    
    # 如果有提示参数，创建带提示的模型
    if prompt_path and os.path.exists(prompt_path):
        print(f"加载提示参数: {prompt_path}")
        model = load_prompt_for_inference(
            base_model=model,
            prompt_path=prompt_path,
            dataset=dataset, 
            n_layers=n_layers,
            reg_weight=reg_weight,
            prompt_type=prompt_type,
            embedding_size=embedding_size,
            p_num=p_num
        )
    
    # 获取归一化邻接矩阵
    print("计算归一化邻接矩阵...")
    # 获取归一化邻接矩阵
    # 转换为稀疏矩阵格式
    interaction_matrix = dataset.interaction_matrix.tocoo()
    norm_adj_matrix = model.get_norm_adj_mat(interaction_matrix).to(device)
    
    # 评估模型
    print("开始评估...")
    eval_results = evaluate(model, dataset, norm_adj_matrix, k_list, batch_size)
    
    # 打印评估结果
    print("\n评估结果:")
    print_metrics(eval_results, prefix="  ")
    
    return eval_results

# 比较不同模型的性能
def compare_models(base_model_path, prompted_model_path, prompt_type, p_num):
    """比较基础模型和带提示模型的性能
    
    Args:
        base_model_path: 基础模型路径
        prompted_model_path: 提示参数路径
        prompt_type: 提示类型，默认为None
        p_num: 提示数量，默认为None
    """
    print("\n=== 评估基础模型 ===")
    base_results = evaluate_model(model_path=base_model_path)
    
    print("\n=== 评估带提示模型（遗忘后） ===")
    prompted_results = evaluate_model(
        model_path=base_model_path,
        prompt_path=prompted_model_path,
        prompt_type=prompt_type,
        p_num=p_num
    )
    
    # 计算性能差异
    print("\n=== 性能比较 ===")
    for k in COMMON_CONFIG['k_list']:
        print(f"\nk={k}:")
        for metric in ['recall', 'ndcg', 'hit_rate']:
            base_value = base_results[k][metric]
            prompted_value = prompted_results[k][metric]
            diff = prompted_value - base_value
            diff_percent = (diff / base_value * 100) if base_value != 0 else float('inf')
            print(f"  {metric.upper()}: {base_value:.4f} -> {prompted_value:.4f} "  
                  f"(差异: {diff:.4f}, {diff_percent:.2f}%)")

# 主函数
def main():
    """主函数，执行模型评估"""
    # 评估最佳模型
    model_path = LIGHTGCN_CONFIG['save_path']
    prompt_path = UNLEARNING_CONFIG['prompt_save_path']
    
    if os.path.exists(model_path):
        print(f"使用最佳模型进行评估: {model_path}")
        if os.path.exists(prompt_path):
            # 比较基础模型和带提示模型的性能
            compare_models(
                base_model_path=model_path,
                prompted_model_path=prompt_path,
                prompt_type=UNLEARNING_CONFIG['prompt_type'],
                p_num=UNLEARNING_CONFIG['p_num']
            )
        else:
            # 只评估基础模型
            evaluate_model(model_path=model_path)
    else:
        print("未找到保存的模型，使用默认初始化模型进行评估")
        evaluate_model()

if __name__ == "__main__":
    main()