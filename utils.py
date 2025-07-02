# -*- coding: utf-8 -*-
# 工具函数模块 - 包含评估指标和通用功能

import os
import math
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm, trange

# 导入配置
from config import COMMON_CONFIG

# 设备配置
device = COMMON_CONFIG['device']

# 评估指标
def recall(rank, ground_truth, k):
    """计算Recall@k
    
    Args:
        rank: 预测的物品排名列表
        ground_truth: 真实交互物品集合
        k: 推荐列表长度
        
    Returns:
        float: Recall@k值
    """
    hits = 0
    for idx, item in enumerate(rank[:k]):
        if item in ground_truth:
            hits += 1
    return hits / len(ground_truth) if len(ground_truth) > 0 else 0

def ndcg(rank, ground_truth, k):
    """计算NDCG@k
    
    Args:
        rank: 预测的物品排名列表
        ground_truth: 真实交互物品集合
        k: 推荐列表长度
        
    Returns:
        float: NDCG@k值
    """
    hits = 0
    dcg = 0
    idcg = sum([1 / math.log2(i + 2) for i in range(min(k, len(ground_truth)))])
    for idx, item in enumerate(rank[:k]):
        if item in ground_truth:
            dcg += 1 / math.log2(idx + 2)
    return dcg / idcg if idcg > 0 else 0

def hit_rate(rank, ground_truth, k):
    """计算HitRate@k
    
    Args:
        rank: 预测的物品排名列表
        ground_truth: 真实交互物品集合
        k: 推荐列表长度
        
    Returns:
        float: HitRate@k值，命中为1，否则为0
    """
    for idx, item in enumerate(rank[:k]):
        if item in ground_truth:
            return 1
    return 0

# 通用评估函数
def evaluate(model, dataset, norm_adj_matrix, k_list=None, batch_size=None):
    """评估模型性能
    
    Args:
        model: 要评估的模型
        dataset: 数据集对象
        norm_adj_matrix: 归一化邻接矩阵
        k_list: 推荐列表长度列表，默认为[10, 20]
        batch_size: 批处理大小，默认为2048
        
    Returns:
        dict: 包含各项评估指标的字典
    """
    if k_list is None:
        k_list = COMMON_CONFIG['k_list']
    if batch_size is None:
        batch_size = COMMON_CONFIG['batch_size']
        
    model.eval()
    test_users = dataset.get_test_samples()
    n_test_users = len(test_users)
    
    # 初始化指标
    results = {k: {'recall': 0.0, 'ndcg': 0.0, 'hit_rate': 0.0} for k in k_list}
    
    with torch.no_grad():
        for start in range(0, n_test_users, batch_size):
            end = min(start + batch_size, n_test_users)
            batch_users = test_users[start:end]
            batch_users_tensor = torch.LongTensor(batch_users).to(device)
            
            # 预测所有物品的评分
            rating_pred = model.full_sort_predict(batch_users_tensor, norm_adj_matrix)
            rating_pred = rating_pred.cpu().numpy()
            
            # 对每个用户进行评估
            for idx, user in enumerate(batch_users):
                user_pos_items = dataset.test_user_items[user]
                if len(user_pos_items) == 0:
                    continue
                
                # 排除训练集中的物品
                train_items = dataset.train_user_items[user]
                user_pred = rating_pred[idx]
                user_pred[list(train_items)] = -np.inf
                
                # 获取预测排名最高的物品
                rank_indices = np.argsort(-user_pred)
                
                # 计算各项指标
                for k in k_list:
                    results[k]['recall'] += recall(rank_indices, user_pos_items, k)
                    results[k]['ndcg'] += ndcg(rank_indices, user_pos_items, k)
                    results[k]['hit_rate'] += hit_rate(rank_indices, user_pos_items, k)
    
    # 计算平均值
    for k in k_list:
        for metric in results[k]:
            results[k][metric] /= n_test_users
    
    return results

# 评估特定用户子集
def evaluate_user_subset(model, dataset, users, norm_adj_matrix, k_list=None, batch_size=None):
    """评估模型在特定用户子集上的性能
    
    Args:
        model: 要评估的模型
        dataset: 数据集对象
        users: 用户ID列表
        norm_adj_matrix: 归一化邻接矩阵
        k_list: 推荐列表长度列表，默认为[10, 20]
        batch_size: 批处理大小，默认为256
        
    Returns:
        dict: 包含各项评估指标的字典
    """
    if k_list is None:
        k_list = COMMON_CONFIG['k_list']
    if batch_size is None:
        batch_size = COMMON_CONFIG['batch_size']
        
    model.eval()
    n_test_users = len(users)
    
    # 初始化指标
    results = {k: {'recall': 0.0, 'ndcg': 0.0, 'hit_rate': 0.0} for k in k_list}
    
    with torch.no_grad():
        # 添加进度条
        batch_progress = tqdm(range(0, n_test_users, batch_size), desc="评估批次", leave=False)
        for start in batch_progress:
            end = min(start + batch_size, n_test_users)
            batch_users = users[start:end]
            batch_users_tensor = torch.LongTensor(batch_users).to(device)
            
            # 预测所有物品的评分
            rating_pred = model.full_sort_predict(batch_users_tensor, norm_adj_matrix)
            rating_pred = rating_pred.cpu().numpy()
            
            # 对每个用户进行评估
            user_progress = tqdm(enumerate(batch_users), total=len(batch_users), desc="评估用户", leave=False)
            for idx, user in user_progress:
                user_pos_items = dataset.test_user_items[user]
                if len(user_pos_items) == 0:
                    continue
                
                # 排除训练集中的物品
                train_items = dataset.train_user_items[user]
                user_pred = rating_pred[idx]
                user_pred[list(train_items)] = -np.inf
                
                # 获取预测排名最高的物品
                rank_indices = np.argsort(-user_pred)
                
                # 计算各项指标
                for k in k_list:
                    results[k]['recall'] += recall(rank_indices, user_pos_items, k)
                    results[k]['ndcg'] += ndcg(rank_indices, user_pos_items, k)
                    results[k]['hit_rate'] += hit_rate(rank_indices, user_pos_items, k)
                
                # 更新用户进度条
                user_progress.set_postfix({"用户": f"{user}"})
            
            # 更新批次进度条
            batch_progress.set_postfix({"进度": f"{end}/{n_test_users}"})
    
    # 计算平均值
    for k in k_list:
        for metric in results[k]:
            results[k][metric] /= n_test_users if n_test_users > 0 else 1
    
    return results

# 遗忘学习数据集类
class UnLearningDataset(Dataset):
    """遗忘学习数据集类，将数据分为遗忘集和保留集
    
    这个数据集用于遗忘学习过程，将样本分为需要遗忘的样本和需要保留的样本。
    每个样本包含用户ID、正向物品ID、负向物品ID和标签（1表示遗忘样本，0表示保留样本）。
    
    Attributes:
        forget_data: 需要遗忘的数据
        retain_data: 需要保留的数据
        forget_len: 遗忘数据的长度
        retain_len: 保留数据的长度
    """
    def __init__(self, forget_data, retain_data):
        super().__init__()
        self.forget_data = forget_data
        self.retain_data = retain_data
        self.forget_len = len(forget_data)
        self.retain_len = len(retain_data)

    def __len__(self):
        return self.retain_len + self.forget_len
    
    def __getitem__(self, index):
        if(index < self.forget_len):
            # 遗忘数据标记为1
            user, pos_item, neg_item = self.forget_data[index]
            return user, pos_item, neg_item, 1
        else:
            # 保留数据标记为0
            user, pos_item, neg_item = self.retain_data[index - self.forget_len]
            return user, pos_item, neg_item, 0

# 将数据集分割为遗忘集和保留集
def split_forget_retain(dataset, forget_ratio=0.1):
    """将数据集分割为遗忘集和保留集
    
    Args:
        dataset: 数据集对象
        forget_ratio: 遗忘集比例，默认为0.1
        
    Returns:
        tuple: (遗忘样本列表, 保留样本列表)
    """
    # 按用户ID对样本进行分组
    user_samples = {}
    for sample in dataset.train_samples:
        user, pos_item, neg_item = sample
        if user not in user_samples:
            user_samples[user] = []
        user_samples[user].append(sample)
    
    # 确定要遗忘的用户
    all_users = list(user_samples.keys())
    np.random.shuffle(all_users)
    forget_user_count = int(len(all_users) * forget_ratio)
    forget_users = set(all_users[:forget_user_count])
    
    forget_samples = []
    retain_samples = []
    
    # 分配样本到遗忘集和保留集
    for user, samples in user_samples.items():
        if user in forget_users:
            # 对于遗忘用户，保留至少一个样本在保留集中
            if len(samples) > 1:
                # 随机选择一个样本保留
                retain_idx = np.random.randint(0, len(samples))
                retain_samples.append(samples[retain_idx])
                
                # 其余样本放入遗忘集
                for i, sample in enumerate(samples):
                    if i != retain_idx:
                        forget_samples.append(sample)
            else:
                # 如果用户只有一个样本，放入保留集
                retain_samples.append(samples[0])
        else:
            # 非遗忘用户的所有样本都放入保留集
            retain_samples.extend(samples)
    
    print(f"遗忘集大小: {len(forget_samples)}, 保留集大小: {len(retain_samples)}")
    print(f"遗忘用户数: {len(forget_users)}, 总用户数: {len(all_users)}")
    return forget_samples, retain_samples

def unlearner_loss(output, labels, full_teacher_logits, unlearn_teacher_logits, KL_temperature, loss_type='kl', alpha=0.5, max_pairs=1000, lamda=10.0, mu=5.0, K=5):
    """
    蒸馏学习损失函数，支持多种蒸馏策略
    """
    labels = torch.unsqueeze(labels, dim=1)

    # 计算KL散度损失
    def compute_kl_loss(only_forget=False):
        f_teacher_out = F.softmax(full_teacher_logits / KL_temperature, dim=1)
        u_teacher_out = F.softmax(unlearn_teacher_logits / KL_temperature, dim=1)

        if only_forget:
            forget_mask = (labels.squeeze() == 1)
            if not torch.any(forget_mask):
                return torch.tensor(0.0, device=output.device)
            forget_student_out = F.log_softmax(output[forget_mask] / KL_temperature, dim=1)
            forget_teacher_out = u_teacher_out[forget_mask]
            return F.kl_div(forget_student_out, forget_teacher_out, reduction='batchmean') * (KL_temperature ** 2)
        else:
            labels_expanded = labels.view(-1, 1)
            overall_teacher_out = labels_expanded * u_teacher_out + (1 - labels_expanded) * f_teacher_out
            student_out = F.log_softmax(output / KL_temperature, dim=1)
            return F.kl_div(student_out, overall_teacher_out, reduction='batchmean') * (KL_temperature ** 2)

    # 计算BPR排序损失
    def compute_bpr_loss():
        retain_mask = (labels.squeeze() == 0)
        if not torch.any(retain_mask):
            return torch.tensor(0.0, device=output.device)
        retain_student_out = output[retain_mask]
        retain_teacher_out = full_teacher_logits[retain_mask]
        batch_size = retain_student_out.size(0)
        if batch_size <= 1:
            return torch.tensor(0.0, device=output.device)

        i_indices = []
        j_indices = []

        if batch_size * (batch_size - 1) > max_pairs:
            pairs = 0
            max_attempts = max_pairs * 2
            attempts = 0
            while pairs < max_pairs and attempts < max_attempts:
                i = torch.randint(0, batch_size, (1,)).item()
                j = torch.randint(0, batch_size, (1,)).item()
                attempts += 1
                if i != j:
                    teacher_i = retain_teacher_out[i, 0] - retain_teacher_out[i, 1]
                    teacher_j = retain_teacher_out[j, 0] - retain_teacher_out[j, 1]
                    if abs(teacher_i - teacher_j) > 0.01:
                        i_indices.append(i)
                        j_indices.append(j)
                        pairs += 1
            if pairs < max_pairs // 2:
                i_indices = []
                j_indices = []
                pairs = 0
                while pairs < max_pairs:
                    i = torch.randint(0, batch_size, (1,)).item()
                    j = torch.randint(0, batch_size, (1,)).item()
                    if i != j:
                        i_indices.append(i)
                        j_indices.append(j)
                        pairs += 1
        else:
            for i in range(batch_size):
                for j in range(batch_size):
                    if i != j:
                        i_indices.append(i)
                        j_indices.append(j)

        if len(i_indices) == 0:
            return torch.tensor(0.0, device=output.device)

        i_indices = torch.tensor(i_indices, device=output.device)
        j_indices = torch.tensor(j_indices, device=output.device)

        student_i = retain_student_out[i_indices, 0]
        student_j = retain_student_out[j_indices, 0]
        teacher_i = retain_teacher_out[i_indices, 0]
        teacher_j = retain_teacher_out[j_indices, 0]

        teacher_diff = teacher_i - teacher_j
        student_diff = student_i - student_j

        teacher_pref = (teacher_diff > 0).float()

        bpr_loss = -torch.log(torch.sigmoid(student_diff) + 1e-10) * teacher_pref

        return bpr_loss.sum() / teacher_pref.sum() if teacher_pref.sum() > 0 else torch.tensor(0.0, device=output.device)
        
    
    # 根据损失类型返回相应的损失
    if loss_type == 'kl':
        # KL散度损失：适用于简单的知识蒸馏场景，对所有样本使用统一的KL散度损失
        # 对遗忘样本使用坏老师，对保留样本使用好老师
        return compute_kl_loss()
    elif loss_type == 'bpr':
        # BPR排序损失：适用于排序敏感的推荐系统，仅对保留样本使用BPR损失
        # 这种方法可能不会对遗忘样本产生足够的遗忘效果
        return compute_bpr_loss()
    elif loss_type == 'combined':
        # 组合损失：结合KL散度和BPR损失的优点
        # 对所有样本使用KL散度损失，同时对保留样本使用BPR排序损失
        kl_loss = compute_kl_loss()
        bpr_loss = compute_bpr_loss()
        
        # 计算每种样本的数量，用于动态调整alpha权重
        forget_count = torch.sum(labels == 1).float()
        retain_count = torch.sum(labels == 0).float()
        total_count = forget_count + retain_count
        
        # 如果某种样本占比过高，适当调整权重
        dynamic_alpha = alpha
        if total_count > 0:
            forget_ratio = forget_count / total_count
            # 当遗忘样本比例过高或过低时，调整alpha值
            if forget_ratio > 0.8:
                dynamic_alpha = min(0.8, alpha * 1.5)  # 增加KL损失权重
            elif forget_ratio < 0.2:
                dynamic_alpha = max(0.2, alpha * 0.5)  # 减少KL损失权重
        
        # 返回加权损失，确保两种损失的数量级相近
        return dynamic_alpha * kl_loss + (1 - dynamic_alpha) * bpr_loss
    elif loss_type == 'separate':
        # 分离损失：对保留集使用BPR排序损失（好老师约束），对遗忘集使用KL散度损失（坏老师约束）
        # 这是最直观的方法，为不同类型的样本使用不同的损失函数
        kl_loss = compute_kl_loss(only_forget=True)
        bpr_loss = compute_bpr_loss()
        
        # 计算每种样本的数量，用于平衡损失权重
        forget_count = torch.sum(labels == 1).float()
        retain_count = torch.sum(labels == 0).float()
        total_count = forget_count + retain_count
        
        # 如果某种样本不存在，则权重为0
        kl_weight = forget_count / total_count if total_count > 0 else 0.0
        bpr_weight = retain_count / total_count if total_count > 0 else 0.0
        
        # 返回加权损失
        return kl_weight * kl_loss + bpr_weight * bpr_loss

    else:
        raise ValueError(f"不支持的损失类型: {loss_type}，请使用 'kl', 'bpr', 'combined', 'separate' 或 'rank_distill'")

# 打印评估结果
def print_metrics(results, prefix=""):
    """打印评估指标
    
    Args:
        results: 评估结果字典
        prefix: 前缀文本，默认为空
    """
    for k in results.keys():
        recall_k = results[k]['recall']
        ndcg_k = results[k]['ndcg']
        hit_rate_k = results[k]['hit_rate']
        print(f"{prefix}k={k}: Recall@{k}={recall_k:.4f}, NDCG@{k}={ndcg_k:.4f}, HitRate@{k}={hit_rate_k:.4f}")

# 计算性能变化百分比
def calculate_performance_change(original_results, new_results, k):
    """计算性能变化百分比
    
    Args:
        original_results: 原始评估结果
        new_results: 新评估结果
        k: 推荐列表长度
        
    Returns:
        dict: 包含各项指标变化百分比的字典
    """
    changes = {}
    for metric in ['recall', 'ndcg', 'hit_rate']:
        original = original_results[k][metric]
        new = new_results[k][metric]
        if original > 0:
            changes[metric] = (new - original) / original * 100
        else:
            changes[metric] = float('inf') if new > 0 else 0.0
    return changes

# 设置随机种子
def set_seed(seed=42):
    """设置随机种子以确保结果可复现
    
    Args:
        seed: 随机种子，默认为42
    """
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


import torch
import torch.nn.functional as F

def loss_strategy_1(output, labels, full_teacher_logits, unlearn_teacher_logits, KL_temperature=1.0):
    """
    策略一：遗忘集使用 KL 散度，保留集使用 BPR 排序损失
    """
    labels = labels.view(-1)
    retain_mask = (labels == 0)
    forget_mask = (labels == 1)

    # 保留集 BPR 损失
    if retain_mask.sum() > 0:
        retain_output = output[retain_mask]
        pos_scores = retain_output[:, 0]
        neg_scores = retain_output[:, 1]
        bpr_loss = -torch.log(torch.sigmoid(pos_scores - neg_scores) + 1e-10).mean()
    else:
        bpr_loss = torch.tensor(0.0, device=output.device)

    # 遗忘集 KL 散度损失
    if forget_mask.sum() > 0:
        forget_output = output[forget_mask]
        forget_teacher = unlearn_teacher_logits[forget_mask]
        student_log_probs = F.log_softmax(forget_output / KL_temperature, dim=1)
        teacher_probs = F.softmax(forget_teacher / KL_temperature, dim=1)
        kl_loss = F.kl_div(student_log_probs, teacher_probs, reduction='batchmean') * (KL_temperature ** 2)
    else:
        kl_loss = torch.tensor(0.0, device=output.device)

    # 动态权重调整
    total = retain_mask.sum() + forget_mask.sum()
    retain_weight = retain_mask.sum().float() / total
    forget_weight = forget_mask.sum().float() / total

    total_loss = retain_weight * bpr_loss + forget_weight * kl_loss
    return total_loss






def loss_strategy_2(output, labels, full_teacher_logits, unlearn_teacher_logits, KL_temperature=1.0, alpha=0.5, lamda=10.0, mu=5.0, K=5):
    """
    改进的策略二：差异化蒸馏策略
    对遗忘集使用强化的KL散度蒸馏，对保留集使用增强的Ranking Distillation蒸馏策略
    
    Args:
        output: 学生模型输出的logits
        labels: 样本标签，1表示遗忘样本，0表示保留样本
        full_teacher_logits: 完整教师模型的logits
        unlearn_teacher_logits: 遗忘教师模型的logits
        KL_temperature: KL散度温度参数
        alpha: 损失权重平衡因子
        lamda: 位置重要性权重的衰减参数
        mu: 排序差异权重的缩放参数
        K: 考虑的排名前K个物品
        
    Returns:
        torch.Tensor: 计算得到的损失值
    """
    labels = labels.view(-1)
    retain_mask = (labels == 0)
    forget_mask = (labels == 1)
    
    # 1. 遗忘集增强KL散度损失 - 使用更低的温度参数增强对遗忘模型的学习
    if forget_mask.sum() > 0:
        forget_output = output[forget_mask]
        forget_teacher = unlearn_teacher_logits[forget_mask]
        
        # 使用更低的温度参数使分布更加尖锐，增强学习效果
        forget_temperature = max(0.5, KL_temperature * 0.7)  # 降低温度参数
        
        # 计算KL散度损失
        student_log_probs = F.log_softmax(forget_output / forget_temperature, dim=1)
        teacher_probs = F.softmax(forget_teacher / forget_temperature, dim=1)
        kl_loss = F.kl_div(student_log_probs, teacher_probs, reduction='batchmean') * (forget_temperature ** 2)
        
        # 添加额外的对比损失，进一步增强遗忘效果
        # 鼓励学生模型对遗忘样本的预测与好老师的预测相反
        with torch.no_grad():
            opposite_teacher = full_teacher_logits[forget_mask]  # 好老师的预测
            opposite_probs = F.softmax(opposite_teacher / forget_temperature, dim=1)
        
        # 计算反向KL散度，鼓励学生模型预测与好老师相反
        reverse_kl = -F.kl_div(student_log_probs, opposite_probs, reduction='batchmean') * 0.3
        
        # 组合KL损失
        kl_loss = kl_loss + reverse_kl
    else:
        kl_loss = torch.tensor(0.0, device=output.device)
    
    # 2. 保留集增强排序蒸馏损失
    if retain_mask.sum() > 0:
        retain_output = output[retain_mask]
        retain_teacher = full_teacher_logits[retain_mask]
        
        batch_size = retain_output.size(0)
        if batch_size <= 1:
            rank_loss = torch.tensor(0.0, device=output.device)
        else:
            # 对每个样本计算排序蒸馏损失
            rank_loss = torch.tensor(0.0, device=output.device)
            
            for i in range(batch_size):
                # 获取当前样本的学生模型和教师模型输出
                student_scores = retain_output[i]
                teacher_scores = retain_teacher[i]
                
                # 获取教师模型的排序
                teacher_ranks = torch.argsort(torch.argsort(teacher_scores, descending=True))
                
                # 改进的位置重要性权重计算 - 使用更陡峭的衰减曲线
                # 使前几个位置的权重更高，后面位置权重衰减更快
                position_weights = torch.exp(-torch.arange(len(teacher_scores), device=output.device).float() / (lamda * 0.8))
                
                # 获取学生模型的排序
                student_ranks = torch.argsort(torch.argsort(student_scores, descending=True))
                
                # 改进的排序差异权重计算 - 使用更敏感的差异函数
                rank_diff = student_ranks - torch.arange(len(student_scores), device=output.device).float()
                rank_diff_weights = torch.tanh(mu * 1.2 * rank_diff)  # 增加敏感度
                
                # 增加考虑的前K个物品数量，确保更多高排名物品被正确排序
                effective_k = min(K + 2, len(teacher_scores))
                top_k_mask = teacher_ranks < effective_k
                
                if top_k_mask.sum() > 0:
                    # 计算排序蒸馏损失
                    sample_loss = (position_weights * rank_diff_weights * top_k_mask.float()).sum() / top_k_mask.sum()
                    rank_loss += sample_loss
            
            # 计算平均损失
            rank_loss = rank_loss / batch_size
            
            # 添加额外的MSE损失，确保保留集上的预测分数接近好老师
            # 这有助于保持在保留集上的性能
            mse_loss = F.mse_loss(retain_output, retain_teacher) * 0.5
            rank_loss = rank_loss + mse_loss
    else:
        rank_loss = torch.tensor(0.0, device=output.device)
    
    # 3. 改进的动态权重调整策略
    total = retain_mask.sum() + forget_mask.sum()
    retain_count = retain_mask.sum().float()
    forget_count = forget_mask.sum().float()
    
    # 基于样本数量的权重计算
    retain_weight = retain_count / total if total > 0 else 0.0
    forget_weight = forget_count / total if total > 0 else 0.0
    
    # 动态调整alpha值 - 使用更激进的调整策略
    # 默认alpha值为0.5，但根据样本比例动态调整
    dynamic_alpha = alpha
    if total > 0:
        forget_ratio = forget_weight
        # 当遗忘样本比例过高或过低时，调整alpha值
        if forget_ratio > 0.7:
            # 遗忘样本比例高，增加遗忘损失权重
            dynamic_alpha = min(0.85, alpha * 1.7)  
        elif forget_ratio < 0.3:
            # 遗忘样本比例低，适当减少遗忘损失权重但保持一定强度
            dynamic_alpha = max(0.3, alpha * 0.6)  
    
    # 4. 返回加权损失 - 使用更强的遗忘损失权重
    # 增加遗忘损失的权重，确保遗忘效果更明显
    forget_loss_weight = dynamic_alpha * 1.2  # 增强遗忘损失权重
    retain_loss_weight = 1 - dynamic_alpha  # 保留集损失权重
    
    # 确保权重和为1
    total_weight = forget_loss_weight + retain_loss_weight
    forget_loss_weight = forget_loss_weight / total_weight
    retain_loss_weight = retain_loss_weight / total_weight
    
    # 最终损失
    total_loss = forget_loss_weight * kl_loss + retain_loss_weight * rank_loss
    return total_loss

def loss_strategy_3_dual_anchor_fixed(output, labels, full_teacher_logits, unlearn_teacher_logits,
                                      KL_temperature=1.0, alpha=0.5, lamda=10.0, mu=5.0,
                                      K=5, tau=0.3, lambda_tau=1.0):
    labels = labels.view(-1)
    device = output.device
    retain_mask = (labels == 0)
    forget_mask = (labels == 1)

    ### 1. Remain ranking distillation
    if retain_mask.sum() > 0:
        retain_output = output[retain_mask]
        retain_teacher = full_teacher_logits[retain_mask]
        rank_loss = 0.0
        for s_out, t_out in zip(retain_output, retain_teacher):
            t_rank = torch.argsort(torch.argsort(t_out, descending=True))
            s_rank = torch.argsort(torch.argsort(s_out, descending=True))
            weight = torch.exp(-torch.arange(len(t_out), device=device).float() / lamda)
            diff = torch.tanh(mu * (s_rank - t_rank).float())
            mask = (t_rank < K).float()
            rank_loss += (weight * diff * mask).sum() / mask.sum().clamp(min=1)
        rank_loss /= retain_output.size(0)
        rank_loss += 0.5 * F.mse_loss(retain_output, retain_teacher)
    else:
        rank_loss = torch.tensor(0.0, device=device)

    ### 2. Forget selective KL divergence
    if forget_mask.sum() > 0:
        forget_output = output[forget_mask]
        forget_teacher = unlearn_teacher_logits[forget_mask]
        diff = (forget_output - forget_teacher).abs()
        selection_mask = (diff > tau).any(dim=1)
        if selection_mask.sum() > 0:
            selected_output = forget_output[selection_mask]
            selected_teacher = forget_teacher[selection_mask]
            s_log_probs = F.log_softmax(selected_output / KL_temperature, dim=1)
            t_probs = F.softmax(selected_teacher / KL_temperature, dim=1)
            kl_loss = F.kl_div(s_log_probs, t_probs, reduction='batchmean') * (KL_temperature ** 2)
        else:
            kl_loss = torch.tensor(0.0, device=device)
    else:
        kl_loss = torch.tensor(0.0, device=device)

    ### 3. Kendall structure regularization (label-specific)
    kendall_loss = 0.0
    for i in range(output.size(0)):
        s, t_good, t_bad = output[i], full_teacher_logits[i], unlearn_teacher_logits[i]
        s_diff = s.unsqueeze(0) - s.unsqueeze(1)
        good_diff = t_good.unsqueeze(0) - t_good.unsqueeze(1)
        bad_diff = t_bad.unsqueeze(0) - t_bad.unsqueeze(1)
        concord_good = ((s_diff * good_diff) > 0).float().mean()
        concord_bad = ((s_diff * bad_diff) > 0).float().mean()
        if labels[i] == 0:
            score = concord_good - concord_bad  # remain: close to good
        else:
            score = concord_bad - concord_good  # forget: close to bad
        kendall_loss += (1 - score)
    kendall_loss /= output.size(0)

    ### 4. Weighted final loss
    total = retain_mask.sum() + forget_mask.sum()
    retain_weight = retain_mask.sum().float() / total if total > 0 else 0.0
    forget_weight = forget_mask.sum().float() / total if total > 0 else 0.0

    distill_loss = alpha * (retain_weight * rank_loss + forget_weight * kl_loss)
    total_loss = distill_loss + lambda_tau * kendall_loss
    return total_loss

def unlearner_loss_2(output, labels, full_teacher_logits, unlearn_teacher_logits, KL_temperature, loss_type, alpha=0.5, max_pairs=1000, lamda=10.0, mu=5.0, K=5):
    """蒸馏学习损失函数，支持多种蒸馏策略
    
    Args:
        output: 学生模型输出的logits
        labels: 样本标签，1表示遗忘样本，0表示保留样本
        full_teacher_logits: 完整教师模型的logits
        unlearn_teacher_logits: 遗忘教师模型的logits
        KL_temperature: KL散度温度参数
        loss_type: 损失类型，支持'kl_bpr'和'rank'
        alpha: 损失权重平衡因子
        max_pairs: BPR损失中最大考虑的样本对数量
        lamda: 位置重要性权重的衰减参数
        mu: 排序差异权重的缩放参数
        K: 考虑的排名前K个物品
        
    Returns:
        torch.Tensor: 计算得到的损失值
    """
    """
    蒸馏学习损失函数，支持多种蒸馏策略
    """
    labels = labels.view(-1)
    retain_mask = (labels == 0)
    # 只对remain集计算BCE损失
    if retain_mask.sum() > 0:
        retain_output = output[retain_mask]
        pos_scores = retain_output[:, 0]
        neg_scores = retain_output[:, 1]
        bce_loss = F.binary_cross_entropy_with_logits(pos_scores, torch.ones_like(pos_scores)) + \
                   F.binary_cross_entropy_with_logits(neg_scores, torch.zeros_like(neg_scores))
        bce_loss = bce_loss / 2.0
    else:
        bce_loss = torch.tensor(0.0, device=output.device)
    # 根据损失类型返回相应的损失
    if loss_type == 'KL':
        # KL散度损失：适用于简单的知识蒸馏场景，对所有样本使用统一的KL散度损失
        # 对遗忘样本使用坏老师，对保留样本使用好老师
        return bce_loss + loss_strategy_1(output, labels, full_teacher_logits, unlearn_teacher_logits, KL_temperature)
    elif loss_type == 'WRD':
        # 排序蒸馏损失：适用于排序敏感的推荐系统
        # 对保留集使用排序蒸馏损失，对遗忘集使用KL散度损失
        return bce_loss + loss_strategy_2(output, labels, full_teacher_logits, unlearn_teacher_logits, KL_temperature, alpha=alpha, lamda=lamda, mu=mu, K=K)
    elif loss_type == 'DAD':
        return bce_loss + loss_strategy_3_dual_anchor_fixed(output, labels, full_teacher_logits, unlearn_teacher_logits,
                                  KL_temperature=1.0, alpha=0.4, lamda=5, mu=3, K=10, tau=0.5, lambda_tau=0.5)


    return bce_loss + unlearner_loss(output, labels, full_teacher_logits, unlearn_teacher_logits, KL_temperature, loss_type, alpha, max_pairs, lamda, mu, K)
