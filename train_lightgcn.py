# -*- coding: utf-8 -*-
# LightGCN模型训练与评估脚本 - 重构版

import os
import time
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm, trange

# 导入自定义模块
from config import COMMON_CONFIG, LIGHTGCN_CONFIG
from data import ML100KDataset
from models import LightGCN
from utils import evaluate, print_metrics, set_seed

# 设置随机种子
set_seed(COMMON_CONFIG['seed'])

# 设备配置
device = COMMON_CONFIG['device']
print(f"使用设备: {device}")

# 训练函数
def train(model, dataset, optimizer, batch_size, epochs, norm_adj_matrix, k_list=None, 
         eval_freq=5, patience=5, save_path='best_model.pth'):
    """训练LightGCN模型
    
    Args:
        model: LightGCN模型
        dataset: 数据集对象
        optimizer: 优化器
        batch_size: 批处理大小
        epochs: 训练轮数
        norm_adj_matrix: 归一化邻接矩阵
        k_list: 推荐列表长度列表，默认为[10, 20]
        eval_freq: 评估频率，默认为5
        patience: 早停耐心值，默认为5
        save_path: 模型保存路径，默认为'best_model.pth'
        
    Returns:
        LightGCN: 训练好的模型
    """
    if k_list is None:
        k_list = COMMON_CONFIG['k_list']
        
    # 创建数据加载器
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    
    # 记录最佳性能
    best_recall = {k: 0.0 for k in k_list}
    best_ndcg = {k: 0.0 for k in k_list}
    best_epoch = {k: 0 for k in k_list}
    
    # 早停相关变量
    early_stop_metric = 'ndcg'  # 使用NDCG作为早停指标
    early_stop_k = 10  # 使用k=10的指标
    best_metric = 0
    patience_counter = 0
    best_model_state = None
    
    print(f"开始训练，共{epochs}轮，每{eval_freq}轮评估一次，早停耐心值为{patience}")
    
    # 使用trange创建带进度条的epoch迭代器
    epoch_iter = trange(1, epochs + 1, desc="训练进度")
    for epoch in epoch_iter:
        model.train()
        total_loss = 0
        start_time = time.time()
        
        # 使用tqdm创建带进度条的batch迭代器
        batch_iter = tqdm(train_loader, desc=f"Epoch {epoch}/{epochs}")
        for batch_idx, (users, pos_items, neg_items) in enumerate(batch_iter):
            users = users.to(device)
            pos_items = pos_items.to(device)
            neg_items = neg_items.to(device)
            
            optimizer.zero_grad()
            loss = model.calculate_loss(users, pos_items, neg_items, norm_adj_matrix)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
            # 更新进度条中显示的损失值
            batch_iter.set_postfix(loss=f"{loss.item():.4f}")
        
        avg_loss = total_loss / len(train_loader)
        train_time = time.time() - start_time
        epoch_iter.set_postfix(loss=f"{avg_loss:.4f}", time=f"{train_time:.2f}s")
        print(f"Epoch {epoch}/{epochs} 完成, 平均损失: {avg_loss:.4f}, 训练时间: {train_time:.2f}s")
        
        # 定期评估
        if epoch % eval_freq == 0 or epoch == epochs:
            eval_results = evaluate(model, dataset, norm_adj_matrix, k_list, batch_size)
            print("评估结果:")
            for k in k_list:
                recall_k = eval_results[k]['recall']
                ndcg_k = eval_results[k]['ndcg']
                hit_rate_k = eval_results[k]['hit_rate']
                print(f"  k={k}: Recall@{k}={recall_k:.4f}, NDCG@{k}={ndcg_k:.4f}, HitRate@{k}={hit_rate_k:.4f}")
                
                # 记录最佳性能
                if recall_k > best_recall[k]:
                    best_recall[k] = recall_k
                    best_epoch[k] = epoch
                
                if ndcg_k > best_ndcg[k]:
                    best_ndcg[k] = ndcg_k
            
            # 早停检查
            current_metric = eval_results[early_stop_k][early_stop_metric]
            if current_metric > best_metric:
                best_metric = current_metric
                patience_counter = 0
                # 保存最佳模型状态
                best_model_state = model.state_dict().copy()
                print(f"新的最佳 {early_stop_metric.upper()}@{early_stop_k}: {best_metric:.4f}")
                # 保存最佳模型
                torch.save(model.state_dict(), save_path)
                print(f"模型已保存到: {save_path}")
            else:
                patience_counter += 1
                print(f"早停计数: {patience_counter}/{patience}")
                if patience_counter >= patience:
                    print(f"\n早停! {patience} 轮评估后 {early_stop_metric.upper()}@{early_stop_k} 没有提升")
                    # 恢复最佳模型
                    if best_model_state is not None:
                        model.load_state_dict(best_model_state)
                    break
    
    print("\n训练完成!")
    for k in k_list:
        print(f"最佳 Recall@{k}: {best_recall[k]:.4f} (Epoch {best_epoch[k]})")
        print(f"最佳 NDCG@{k}: {best_ndcg[k]:.4f}")
    
    # 如果训练完成但没有触发早停，确保使用最佳模型
    if best_model_state is not None and patience_counter < patience:
        model.load_state_dict(best_model_state)
        print(f"已加载最佳模型 (最佳 {early_stop_metric.upper()}@{early_stop_k}: {best_metric:.4f})")
    
    return model

# 主函数
def main():
    """主函数，执行模型训练流程"""
    # 记录总训练开始时间
    total_start_time = time.time()
    
    # 加载配置参数
    data_path = COMMON_CONFIG['data_path']
    embedding_size = LIGHTGCN_CONFIG['embedding_size']
    n_layers = LIGHTGCN_CONFIG['n_layers']
    reg_weight = LIGHTGCN_CONFIG['reg_weight']
    batch_size = COMMON_CONFIG['batch_size']
    lr = LIGHTGCN_CONFIG['lr']
    epochs = LIGHTGCN_CONFIG['epochs']
    eval_freq = LIGHTGCN_CONFIG['eval_freq']
    k_list = COMMON_CONFIG['k_list']
    patience = LIGHTGCN_CONFIG['patience']
    save_path = LIGHTGCN_CONFIG['save_path']
    
    # 加载数据集
    print("加载数据集...")
    dataset = ML100KDataset(data_path)
    
    # 创建模型
    print("创建LightGCN模型...")
    model = LightGCN(
        n_users=dataset.n_users,
        n_items=dataset.n_items,
        embedding_size=embedding_size,
        n_layers=n_layers,
        reg_weight=reg_weight
    ).to(device)
    
    # 创建优化器
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    # 获取归一化邻接矩阵
    print("计算归一化邻接矩阵...")
    norm_adj_matrix = model.get_norm_adj_mat(dataset.interaction_matrix).to(device)
    
    # 训练模型
    print("开始训练...")
    model = train(
        model=model, 
        dataset=dataset, 
        optimizer=optimizer, 
        batch_size=batch_size, 
        epochs=epochs, 
        norm_adj_matrix=norm_adj_matrix, 
        k_list=k_list, 
        eval_freq=eval_freq, 
        patience=patience, 
        save_path=save_path
    )
    
    # 最终评估
    print("\n使用最佳模型进行最终评估...")
    final_results = evaluate(model, dataset, norm_adj_matrix, k_list, batch_size)
    print("最终评估结果:")
    print_metrics(final_results, prefix="  ")
    
    # 计算并输出总训练时间
    total_time = time.time() - total_start_time
    hours, remainder = divmod(total_time, 3600)
    minutes, seconds = divmod(remainder, 60)
    print(f"\n总训练时间: {int(hours)}小时 {int(minutes)}分钟 {seconds:.2f}秒")

if __name__ == "__main__":
    main()