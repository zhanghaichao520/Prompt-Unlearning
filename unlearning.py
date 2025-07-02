# -*- coding: utf-8 -*-
# LightGCN推荐系统遗忘学习模块 - 重构版

import os
import time
import torch
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader
import config
from models import LightGCN
# 导入自定义模块
from config import COMMON_CONFIG, UNLEARNING_CONFIG,LIGHTGCN_CONFIG
from data import ML100KDataset
from models import LightGCN
from utils import (
    evaluate_user_subset, print_metrics, set_seed, UnLearningDataset,
    split_forget_retain, unlearner_loss, unlearner_loss_2, calculate_performance_change
)
import random
from tqdm import tqdm, trange
# 设置随机种子
set_seed(COMMON_CONFIG['seed'])

# 设备配置
device = COMMON_CONFIG['device']

# 遗忘学习训练步骤
def unlearning_step(model, unlearning_teacher, full_trained_teacher, unlearn_data_loader, optimizer, 
                   norm_adj_matrix, device, KL_temperature):
    """执行一轮遗忘学习训练
    
    Args:
        model: 学生模型（带提示的模型）
        unlearning_teacher: 遗忘模型（坏老师）
        full_trained_teacher: 完整训练模型（好老师）
        unlearn_data_loader: 遗忘学习数据加载器
        optimizer: 优化器
        norm_adj_matrix: 归一化邻接矩阵
        device: 计算设备
        KL_temperature: KL散度温度参数
        
    Returns:
        float: 平均损失值
    """
    losses = []
    # 添加进度条
    pbar = tqdm(unlearn_data_loader, desc="训练批次", leave=False)
    for batch in pbar:
        users, pos_items, neg_items, labels = batch
        users = users.to(device)
        pos_items = pos_items.to(device)
        neg_items = neg_items.to(device)
        labels = labels.to(device)
        
        # 计算教师模型的输出
        with torch.no_grad():
            # 全模型（好老师）的输出
            full_teacher_pos = full_trained_teacher.predict(users, pos_items, norm_adj_matrix)
            full_teacher_neg = full_trained_teacher.predict(users, neg_items, norm_adj_matrix)
            full_teacher_logits = torch.stack([full_teacher_pos, full_teacher_neg], dim=1)
            
            # 遗忘模型（坏老师）的输出
            unlearn_teacher_pos = unlearning_teacher.predict(users, pos_items, norm_adj_matrix)
            unlearn_teacher_neg = unlearning_teacher.predict(users, neg_items, norm_adj_matrix)
            unlearn_teacher_logits = torch.stack([unlearn_teacher_pos, unlearn_teacher_neg], dim=1)
        
        # 学生模型的输出
        student_pos = model.predict(users, pos_items, norm_adj_matrix)
        student_neg = model.predict(users, neg_items, norm_adj_matrix)
        student_output = torch.stack([student_pos, student_neg], dim=1)
        
        # 计算损失并更新
        optimizer.zero_grad()
        # 从配置中获取损失类型和其他参数
        loss_type = UNLEARNING_CONFIG['loss_type']
        alpha = UNLEARNING_CONFIG['alpha']
        max_pairs = UNLEARNING_CONFIG['max_pairs']
        lamda = UNLEARNING_CONFIG['lamda']
        mu = UNLEARNING_CONFIG['mu']
        K = UNLEARNING_CONFIG['K']
        
        loss = unlearner_loss_2(
            output=student_output, 
            labels=labels, 
            full_teacher_logits=full_teacher_logits, 
            unlearn_teacher_logits=unlearn_teacher_logits, 
            KL_temperature=KL_temperature,
            loss_type=loss_type,
            alpha=alpha,
            max_pairs=max_pairs,
            lamda=lamda,
            mu=mu,
            K=K
        )
        loss.backward()
        optimizer.step()
        losses.append(loss.detach().cpu().numpy())
        
        # 更新进度条显示当前损失
        pbar.set_postfix({"loss": f"{loss.item():.4f}"})
    
    return np.mean(losses)

# 评估遗忘效果
def evaluate_unlearning(model, dataset, forget_samples, retain_samples, norm_adj_matrix, k_list=None):
    """评估遗忘效果
    
    Args:
        model: 要评估的模型
        dataset: 数据集对象
        forget_samples: 遗忘样本列表
        retain_samples: 保留样本列表
        norm_adj_matrix: 归一化邻接矩阵
        k_list: 推荐列表长度列表，默认为None（使用配置中的值）
        
    Returns:
        tuple: (遗忘集评估结果, 保留集评估结果)
    """
    if k_list is None:
        k_list = COMMON_CONFIG['k_list']
        
    model.eval()
    
    # 创建用户集合
    forget_users = set([sample[0] for sample in forget_samples])
    retain_users = set([sample[0] for sample in retain_samples])
    
    # 确保没有重叠
    common_users = forget_users.intersection(retain_users)
    if common_users:
        print(f"警告: 遗忘集和保留集有{len(common_users)}个重叠用户")
    
    # 评估遗忘集
    print("评估遗忘集...")
    forget_results = evaluate_user_subset(model, dataset, list(forget_users), norm_adj_matrix, k_list)
    
    # 评估保留集
    print("评估保留集...")
    retain_results = evaluate_user_subset(model, dataset, list(retain_users), norm_adj_matrix, k_list)
    
    return forget_results, retain_results

# 遗忘学习主函数
def blindspot_unlearner(model, unlearning_teacher, full_trained_teacher, retain_data, forget_data, 
                      norm_adj_matrix, dataset, epochs=10, lr=0.001, batch_size=256, device='cuda', 
                      KL_temperature=1.0, patience=5, validation_interval=1, k_list=None):
    """遗忘学习主函数
    
    Args:
        model: 学生模型（带提示的模型）
        unlearning_teacher: 遗忘模型（坏老师）
        full_trained_teacher: 完整训练模型（好老师）
        retain_data: 保留样本列表
        forget_data: 遗忘样本列表
        norm_adj_matrix: 归一化邻接矩阵
        dataset: 数据集对象
        epochs: 训练轮数，默认为10
        lr: 学习率，默认为0.001
        batch_size: 批处理大小，默认为256
        device: 计算设备，默认为'cuda'
        KL_temperature: KL散度温度参数，默认为1.0
        patience: 早停耐心值，默认为5
        validation_interval: 验证间隔，默认为1
        k_list: 推荐列表长度列表，默认为None（使用配置中的值）
        
    Returns:
        PromptedLightGCN: 训练好的模型
    """
    if k_list is None:
        k_list = COMMON_CONFIG['k_list']
        


    retain_data = random.sample(retain_data, int(len(retain_data) * UNLEARNING_CONFIG['remain_ratio']))
    print(f"使用 {len(retain_data)} 个保留样本进行蒸馏")
    print(f"使用 {len(forget_data)} 个遗忘样本进行蒸馏")
    # 创建遗忘学习数据集
    unlearning_data = UnLearningDataset(forget_data=forget_data, retain_data=retain_data)
    unlearning_loader = DataLoader(unlearning_data, batch_size=batch_size, shuffle=True, pin_memory=True)

    # 设置模型为评估模式
    unlearning_teacher.eval()
    full_trained_teacher.eval()
    
    # 创建优化器 - 只优化提示部分的参数
    prompt_params = []
    # 收集用户提示和物品提示的参数
    for name, param in model.named_parameters():
        if 'user_prompt' in name or 'item_prompt' in name:
            prompt_params.append(param)
            param.requires_grad = True
        else:
            # 冻结基础模型的参数
            param.requires_grad = False
    
    print(f"可训练的提示参数数量: {len(prompt_params)}")
    optimizer = torch.optim.Adam(prompt_params, lr=lr)
    
    # 早停相关变量
    best_retain_score = 0.0  # 保留集上的最佳性能
    best_forget_score = float('inf')  # 遗忘集上的最低性能（我们希望在遗忘集上性能越低越好）
    best_epoch = 0
    no_improve_count = 0
    best_prompt_state = None
    
    print(f"开始训练，共{epochs}轮，每{validation_interval}轮验证一次，早停耐心值为{patience}")
    
    # 训练循环
    epoch_pbar = trange(epochs, desc="训练轮次")
    for epoch in epoch_pbar:
        model.train()
        start_time = time.time()
        loss = unlearning_step(
            model=model, 
            unlearning_teacher=unlearning_teacher, 
            full_trained_teacher=full_trained_teacher, 
            unlearn_data_loader=unlearning_loader, 
            optimizer=optimizer, 
            norm_adj_matrix=norm_adj_matrix,
            device=device, 
            KL_temperature=KL_temperature
        )
        train_time = time.time() - start_time
        epoch_pbar.set_postfix({"loss": f"{loss:.4f}", "time": f"{train_time:.2f}s"})
        
        # 定期在验证集上评估模型
        if (epoch + 1) % validation_interval == 0:
            print(f"\nEpoch {epoch+1}/{epochs} 验证中...")
            forget_results, retain_results = evaluate_unlearning(
                model=model,
                dataset=dataset,
                forget_samples=forget_data,
                retain_samples=retain_data,
                norm_adj_matrix=norm_adj_matrix,
                k_list=k_list
            )
            
            # 计算评估指标（以NDCG@20为例）
            retain_score = retain_results[20]['ndcg']  # 保留集上的NDCG@20
            forget_score = forget_results[20]['ndcg']  # 遗忘集上的NDCG@20
            
            print(f"验证 - 保留集NDCG@20: {retain_score:.4f}, 遗忘集NDCG@20: {forget_score:.4f}")
            
            # 判断是否有改进 - 使用加权得分
            # 我们希望保留集性能高且遗忘集性能低
            current_score = retain_score - forget_score  # 保留集高分减去遗忘集高分
                        # 只保存提示部分的参数
            best_prompt_state = {}
            if current_score > best_retain_score - best_forget_score:
                best_retain_score = retain_score
                best_forget_score = forget_score
                best_epoch = epoch
                no_improve_count = 0
                

                for name, param in model.named_parameters():
                    if 'user_prompt' in name or 'item_prompt' in name:
                        best_prompt_state[name] = param.data.clone()
                print(f"保存了 {len(best_prompt_state)} 个提示参数")
                print(f"发现更好的模型！保存在epoch {epoch+1}")
            else:
                no_improve_count += 1
                print(f"模型性能未改善，已经 {no_improve_count}/{patience} 个验证周期")
            
            # 检查是否应该早停
            if no_improve_count >= patience:
                print(f"早停触发！在 {patience} 个验证周期内没有改善。")
                print(f"恢复到最佳模型（epoch {best_epoch+1}）")
                # 恢复到最佳提示状态
                for name, param in model.named_parameters():
                    if name in best_prompt_state:
                        param.data.copy_(best_prompt_state[name])
                break
    
    # 如果训练完成但没有触发早停，确保使用最佳提示
    if best_prompt_state is not None and no_improve_count < patience:
        print(f"训练完成，恢复到最佳提示（epoch {best_epoch+1}）")
        for name, param in model.named_parameters():
            if name in best_prompt_state:
                param.data.copy_(best_prompt_state[name])
    
    return model

# 加载训练好的提示并应用到原始模型
def load_prompt_for_inference(base_model, prompt_path, dataset, n_layers, reg_weight, prompt_type='attention', embedding_size=64, p_num=50):
    """加载训练好的提示并应用到原始模型，用于推理
    
    Args:
        base_model: 原始LightGCN模型
        prompt_path: 训练好的提示参数路径
        dataset: 数据集对象，用于获取用户和物品数量
        n_layers: 图卷积层数
        reg_weight: 正则化权重
        prompt_type: 提示类型，'simple'或'attention'，默认为'attention'
        embedding_size: 嵌入维度，默认为64
        p_num: 注意力提示的数量，默认为50
        
    Returns:
        LightGCN: 添加了训练好的提示的模型
    """
    # 创建带提示的模型
    prompted_model = LightGCN(
        n_users=dataset.n_users,
        n_items=dataset.n_items,
        embedding_size=embedding_size,
        n_layers=n_layers,
        reg_weight=reg_weight,
        prompt_type=prompt_type,
        p_num=p_num
    ).to(device)
    
    # 确保基础模型参数与原始模型相同
    prompted_model.load_state_dict(base_model.state_dict(), strict=False)
    
    # 加载训练好的提示
    if os.path.exists(prompt_path):
        prompt_state_dict = torch.load(prompt_path)
        # 加载提示参数
        missing_keys = []
        for name, param in prompted_model.named_parameters():
            if name in prompt_state_dict:
                param.data.copy_(prompt_state_dict[name])
            elif 'user_prompt' in name or 'item_prompt' in name:
                missing_keys.append(name)
        
        if missing_keys:
            print(f"警告: 以下提示参数未在保存的状态中找到: {missing_keys}")
        print(f"成功加载提示参数，共 {len(prompt_state_dict)} 个参数")
    else:
        print(f"错误: 提示参数文件 {prompt_path} 不存在")
    
    return prompted_model

# 主函数
def main():
    """主函数，执行遗忘学习流程"""
    # 记录总运行时间
    total_start_time = time.time()
    
    # 加载配置参数
    data_path = COMMON_CONFIG['data_path']
    embedding_size = UNLEARNING_CONFIG['embedding_size']
    n_layers = UNLEARNING_CONFIG['n_layers']
    reg_weight = UNLEARNING_CONFIG['reg_weight']
    batch_size = UNLEARNING_CONFIG['batch_size']
    lr = UNLEARNING_CONFIG['lr']
    epochs = UNLEARNING_CONFIG['epochs']
    forget_ratio = UNLEARNING_CONFIG['forget_ratio']
    prompt_type = UNLEARNING_CONFIG['prompt_type']
    p_num = UNLEARNING_CONFIG['p_num']
    KL_temperature = UNLEARNING_CONFIG['KL_temperature']
    k_list = COMMON_CONFIG['k_list']
    patience = UNLEARNING_CONFIG['patience']
    validation_interval = UNLEARNING_CONFIG['validation_interval']
    prompt_save_path = UNLEARNING_CONFIG['prompt_save_path']
    
    # 加载数据集
    print("加载数据集...")
    dataset = ML100KDataset(data_path)
    
    # 分割遗忘集和保留集
    print("分割遗忘集和保留集...")
    forget_samples, retain_samples = split_forget_retain(dataset, forget_ratio)
    
    # 加载预训练的LightGCN模型
    print("创建LightGCN模型...")
    base_model = LightGCN(
        n_users=dataset.n_users,
        n_items=dataset.n_items,
        embedding_size=embedding_size,
        n_layers=n_layers,
        reg_weight=reg_weight
    ).to(device)
    
    # 加载预训练模型权重
    model_path = LIGHTGCN_CONFIG['save_path']
    if os.path.exists(model_path):
        print(f"加载预训练模型: {model_path}")
        base_model.load_state_dict(torch.load(model_path))
    else:
        print("未找到预训练模型，请先运行train_lightgcn.py训练模型")
        return
    
    # 获取归一化邻接矩阵
    print("计算归一化邻接矩阵...")
    norm_adj_matrix = base_model.get_norm_adj_mat(dataset.interaction_matrix).to(device)
    
    # 创建好老师模型（完整训练模型）
    full_trained_teacher = base_model
    
    # 创建坏老师模型（随机初始化）
    print("创建遗忘教师模型（随机初始化）...")
    unlearning_teacher = LightGCN(
        n_users=dataset.n_users,
        n_items=dataset.n_items,
        embedding_size=embedding_size,
        n_layers=n_layers,
        reg_weight=reg_weight
    ).to(device)
    
    # 创建学生模型（带提示的模型）
    print(f"创建学生模型（带{prompt_type}提示）...")
    student_model = LightGCN(
        n_users=dataset.n_users,
        n_items=dataset.n_items,
        embedding_size=embedding_size,
        n_layers=n_layers,
        reg_weight=reg_weight,
        prompt_type=prompt_type,
        p_num=p_num
    ).to(device)
        
    base_state_dict = full_trained_teacher.state_dict()
    filtered_state_dict = {k: v for k, v in base_state_dict.items() if 'user_prompt' not in k and 'item_prompt' not in k}
    student_model.load_state_dict(filtered_state_dict, strict=False)

    
    # 评估原始模型
    print("\n评估原始模型:")
    original_forget_results, original_retain_results = evaluate_unlearning(
        model=base_model, 
        dataset=dataset, 
        forget_samples=forget_samples, 
        retain_samples=retain_samples, 
        norm_adj_matrix=norm_adj_matrix, 
        k_list=k_list
    )
    
    print("\n原始模型在遗忘集上的性能:")
    print_metrics(original_forget_results, prefix="  ")
    
    print("\n原始模型在保留集上的性能:")
    print_metrics(original_retain_results, prefix="  ")
    
    # 进行遗忘学习 - 只训练提示部分
    print("\n开始遗忘学习...")
    trained_prompt_model = blindspot_unlearner(
        model=student_model,
        unlearning_teacher=unlearning_teacher,
        full_trained_teacher=full_trained_teacher,
        retain_data=retain_samples,
        forget_data=forget_samples,
        norm_adj_matrix=norm_adj_matrix,
        dataset=dataset,
        epochs=epochs,
        lr=lr,
        batch_size=batch_size,
        device=str(device),
        KL_temperature=KL_temperature,
        patience=patience,
        validation_interval=validation_interval,
        k_list=k_list
    )
    
    # 评估遗忘后的模型 - 使用添加了训练好的提示的模型
    print("\n评估遗忘后的模型:")
    forget_results, retain_results = evaluate_unlearning(
        model=trained_prompt_model, 
        dataset=dataset, 
        forget_samples=forget_samples, 
        retain_samples=retain_samples, 
        norm_adj_matrix=norm_adj_matrix, 
        k_list=k_list
    )
    
    print("\n遗忘后模型在遗忘集上的性能:")
    print_metrics(forget_results, prefix="  ")
    
    # 计算并打印性能变化
    for k in k_list:
        changes = calculate_performance_change(original_forget_results, forget_results, k)
        print(f"  性能下降: Recall: {changes['recall']:.2f}%, NDCG: {changes['ndcg']:.2f}%, HitRate: {changes['hit_rate']:.2f}%")
    
    print("\n遗忘后模型在保留集上的性能:")
    print_metrics(retain_results, prefix="  ")
    
    # 计算并打印性能变化
    for k in k_list:
        changes = calculate_performance_change(original_retain_results, retain_results, k)
        print(f"  性能变化: Recall: {changes['recall']:.2f}%, NDCG: {changes['ndcg']:.2f}%, HitRate: {changes['hit_rate']:.2f}%")
    
    # 只保存提示部分的参数
    prompt_state_dict = {}
    for name, param in trained_prompt_model.named_parameters():
        if 'user_prompt' in name or 'item_prompt' in name:
            prompt_state_dict[name] = param
    
    # 保存遗忘后的提示
    torch.save(prompt_state_dict, prompt_save_path)
    print(f"\n遗忘后的提示已保存到: {prompt_save_path}")
    
    # 打印提示信息
    print("\n注意: 该模型只训练了提示部分的参数，基础模型参数保持不变。")
    print("使用时，只需要加载这些提示参数并添加到原始模型上即可实现遗忘功能。")
    print("可以使用load_prompt_for_inference函数加载训练好的提示并应用到原始模型。")
    
    # 输出总运行时间
    total_time = time.time() - total_start_time
    hours, remainder = divmod(total_time, 3600)
    minutes, seconds = divmod(remainder, 60)
    print(f"\n总运行时间: {int(hours)}小时 {int(minutes)}分钟 {seconds:.2f}秒")

# 示例：如何使用训练好的提示进行推理
def inference_example():
    """示例：如何使用训练好的提示进行推理"""
    # 加载配置参数
    data_path = COMMON_CONFIG['data_path']
    embedding_size = UNLEARNING_CONFIG['embedding_size']
    n_layers = UNLEARNING_CONFIG['n_layers']
    reg_weight = UNLEARNING_CONFIG['reg_weight']
    prompt_type = UNLEARNING_CONFIG['prompt_type']
    p_num = UNLEARNING_CONFIG['p_num']
    prompt_save_path = UNLEARNING_CONFIG['prompt_save_path']
    
    # 加载数据集
    dataset = ML100KDataset(data_path)
    
    # 创建基础模型
    base_model = LightGCN(
        n_users=dataset.n_users,
        n_items=dataset.n_items,
        embedding_size=embedding_size,
        n_layers=n_layers,
        reg_weight=reg_weight
    ).to(device)
    
    # 加载预训练模型权重
    model_path = LIGHTGCN_CONFIG['save_path']
    if os.path.exists(model_path):
        base_model.load_state_dict(torch.load(model_path))
    else:
        print("未找到预训练模型")
        return
    
    # 加载训练好的提示并应用到原始模型
    prompted_model = load_prompt_for_inference(
        base_model=base_model,
        prompt_path=prompt_save_path,
        dataset=dataset,
        n_layers=n_layers,
        reg_weight=reg_weight,
        prompt_type=prompt_type,
        embedding_size=embedding_size,
        p_num=p_num
    )
    
    # 获取归一化邻接矩阵
    norm_adj_matrix = base_model.get_norm_adj_mat(dataset.interaction_matrix).to(device)
    
    # 示例：为特定用户推荐物品
    user_id = 1  # 示例用户ID
    with torch.no_grad():
        user_tensor = torch.LongTensor([user_id]).to(device)
        scores = prompted_model.full_sort_predict(user_tensor, norm_adj_matrix)
        scores = scores.cpu().numpy()
        
        # 排除训练集中的物品
        train_items = dataset.train_user_items[user_id]
        scores[0, list(train_items)] = -np.inf
        
        # 获取推荐物品
        top_k = 10  # 推荐物品数量
        top_items = np.argsort(-scores[0])[:top_k]
        
        print(f"为用户 {user_id} 推荐的物品: {top_items}")
    
    print("可以使用evaluate_unlearning函数评估模型在遗忘集和保留集上的性能。")

if __name__ == "__main__":
    main()
    # 如果要运行推理示例，取消下面的注释
    # inference_example()