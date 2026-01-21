#!/usr/bin/env python
# coding: utf-8

import os
import time
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import json

from recbole.config import Config
from recbole.data import create_dataset, data_preparation
from recbole.utils import init_seed, init_logger, get_model, get_trainer

# ------------------------------------------------------------
# 1. 计算 ZRF 分数
# ------------------------------------------------------------
def calculate_zrf(unlearned_model, incompetent_teacher,
                  forget_samples, batch_size, device):
    # 把图卷积矩阵都搬到 device
    if hasattr(unlearned_model, 'norm_adj_matrix'):
        unlearned_model.norm_adj_matrix = unlearned_model.norm_adj_matrix.to(device)
    if hasattr(incompetent_teacher, 'norm_adj_matrix'):
        incompetent_teacher.norm_adj_matrix = incompetent_teacher.norm_adj_matrix.to(device)
    unlearned_model.eval()
    incompetent_teacher.eval()

    class ForgetDataset(Dataset):
        def __init__(self, samples):
            self.users = torch.LongTensor([u for u,i in samples])
            self.items = torch.LongTensor([i for u,i in samples])
        def __len__(self):
            return len(self.users)
        def __getitem__(self, idx):
            return self.users[idx], self.items[idx]

    loader = DataLoader(ForgetDataset(forget_samples),
                        batch_size=batch_size, shuffle=False)

    js_vals = []
    with torch.no_grad():
        for users, items in tqdm(loader, desc="计算 ZRF", leave=False):
            users, items = users.to(device), items.to(device)
            inter = {
                unlearned_model.USER_ID: users,
                unlearned_model.ITEM_ID: items
            }
            log_u = unlearned_model.predict(inter)
            log_i = incompetent_teacher.predict(inter)
            p_u = torch.sigmoid(log_u)
            p_i = torch.sigmoid(log_i)
            dist_u = torch.stack([p_u, 1-p_u], dim=1)
            dist_i = torch.stack([p_i, 1-p_i], dim=1)
            m = 0.5 * (dist_u + dist_i)
            m_log = torch.log(m.clamp(min=1e-7))
            kl_um = F.kl_div(m_log, dist_u, reduction='none').sum(dim=1)
            kl_im = F.kl_div(m_log, dist_i, reduction='none').sum(dim=1)
            js_vals.extend((0.5*(kl_um+kl_im)).cpu().numpy())

    mean_js = float(np.mean(js_vals)) if js_vals else 0.0
    return 1.0 - mean_js
# ------------------------------------------------------------
# 2. 数据预处理：Amazon
# ------------------------------------------------------------
def process_data(original_file, output_dir, forget_ratio=0.01):
    df = pd.read_csv(original_file, sep='\t',
                     names=['user','item','rating','timestamp'])

    # 映射 user/item 为整数 ID
    user2id = {u: i for i, u in enumerate(df['user'].unique())}
    item2id = {i: j for j, i in enumerate(df['item'].unique())}
    df['user'] = df['user'].map(user2id)
    df['item'] = df['item'].map(item2id)

    df['label'] = (df['rating'] >= 4).astype(int)
    groups = df.groupby('user')
    users = list(groups.groups.keys())
    forget = set(
        np.random.choice(users, int(len(users) * forget_ratio), replace=False)
    )

    tn, tr, vd, tt = [], [], [], []
    for u, g in groups:
        if len(g) < 3: continue
        g = g.sample(frac=1, random_state=42)
        n1 = int(len(g) * 0.6)
        n2 = int(len(g) * 0.2)
        tgt = tr if u in forget else tn
        for row in g.iloc[:n1].itertuples(index=False):
            tgt.append([row.user, row.item, row.rating, row.timestamp, row.label])
        for row in g.iloc[n1:n1+n2].itertuples(index=False):
            vd.append([row.user, row.item, row.rating, row.timestamp, row.label])
        for row in g.iloc[n1+n2:].itertuples(index=False):
            tt.append([row.user, row.item, row.rating, row.timestamp, row.label])

    os.makedirs(output_dir, exist_ok=True)
    cols = ['user','item','rating','timestamp','label']
    pd.DataFrame(tn, columns=cols).to_csv(f"{output_dir}/train_normal.csv", index=False)
    pd.DataFrame(tr, columns=cols).to_csv(f"{output_dir}/train_random.csv", index=False)
    pd.DataFrame(vd, columns=cols).to_csv(f"{output_dir}/valid.csv",        index=False)
    pd.DataFrame(tt, columns=cols).to_csv(f"{output_dir}/test.csv",         index=False)



# ------------------------------------------------------------
# 3. DataForSCIF
# ------------------------------------------------------------
class DataForSCIF:
    def __init__(self, tn, tr, vd, tt, device):
        self.device     = device
        self.train_norm = tn.reset_index(drop=True)
        self.train_rand = tr.reset_index(drop=True)
        self.valid      = vd.reset_index(drop=True)
        self.test       = tt.reset_index(drop=True)
        self.train      = pd.concat([self.train_norm, self.train_rand],
                                     ignore_index=True)
        try:
            max_u = max(df['user'].max() for df in (self.train, self.valid, self.test) if not df.empty)
            max_i = max(df['item'].max() for df in (self.train, self.valid, self.test) if not df.empty)
        except ValueError:
            raise ValueError("All input DataFrames are empty after mapping. Check raw_to_internal mapping or source CSV files.")

        self.n_users = int(max_u + 1)
        self.n_items = int(max_i + 1)




    def get_train_tensor(self):
        arr = self.train[['user','item','label']].values
        return torch.from_numpy(arr).long().to(self.device)

    def get_unlearn_tensor(self):
        arr = self.train_rand[['user','item','label']].values
        return torch.from_numpy(arr).long().to(self.device)

    def get_valid_tensor(self):
        arr = self.valid[['user','item','label']].values
        return torch.from_numpy(arr).long().to(self.device)

    def get_test_tensor(self):
        arr = self.test[['user','item','label']].values
        return torch.from_numpy(arr).long().to(self.device)


# ------------------------------------------------------------
# 4. SCIF_Unlearner (on CPU)
# ------------------------------------------------------------
class SCIF_Unlearner:
    def __init__(self, model, data_gen, device, if_epoch, if_lr, reg):
        self.model    = model
        self.data_gen = data_gen
        self.device   = device
        self.if_epoch = if_epoch
        self.if_lr    = if_lr
        self.reg      = reg

        un = self.data_gen.get_unlearn_tensor()
        us = un[:,0].unique(); is_ = un[:,1].unique()
        max_u = model.user_embedding.num_embeddings
        max_i = model.item_embedding.num_embeddings
        self.u_idx = us[(us<max_u)&(us>=0)].cpu()
        self.i_idx = is_[(is_<max_i)&(is_>=0)].cpu()

        with torch.no_grad():
            self.orig_u = model.user_embedding.weight[self.u_idx].cpu().clone()
            self.orig_i = model.item_embedding.weight[self.i_idx].cpu().clone()

    def _full_loss(self, u_emb, i_emb):
        t = self.data_gen.get_train_tensor()
        labels = (t[:,2]>0).float()
        logits = (u_emb[t[:,0]] * i_emb[t[:,1]]).sum(dim=-1)
        bce = F.binary_cross_entropy_with_logits(logits, labels, reduction='mean')
        return bce + self.reg*(u_emb.norm()+i_emb.norm())

    def _unlearn_loss(self, u_emb, i_emb):
        t = self.data_gen.get_unlearn_tensor()
        if t.shape[0]==0:
            return torch.tensor(0., device=self.device)
        labels = (t[:,2]>0).float()
        logits = (u_emb[t[:,0]] * i_emb[t[:,1]]).sum(dim=-1)
        return F.binary_cross_entropy_with_logits(logits, labels, reduction='sum')

    def run(self, save_path):
        u_all = self.model.user_embedding.weight.detach().cpu().clone()
        i_all = self.model.item_embedding.weight.detach().cpu().clone()
        u_flat = u_all[self.u_idx].reshape(-1)
        i_flat = i_all[self.i_idx].reshape(-1)
        flat   = torch.cat([u_flat, i_flat], dim=0).clone().requires_grad_(True)

        u_para = u_all.clone(); i_para = i_all.clone()
        d = u_flat.numel()
        u_para[self.u_idx] = flat[:d].reshape(-1, u_all.shape[-1])
        i_para[self.i_idx] = flat[d:].reshape(-1, i_all.shape[-1])

        u_gpu, i_gpu = u_para.to(self.device), i_para.to(self.device)
        L  = self._full_loss(u_gpu, i_gpu)
        gL = torch.autograd.grad(L, flat, create_graph=True, retain_graph=True)[0].reshape(-1,1)
        Lu = self._unlearn_loss(u_gpu, i_gpu)
        gU = torch.autograd.grad(Lu, flat, retain_graph=True)[0].reshape(-1,1)

        def hvp(v): return torch.autograd.grad((gL*v).sum(), flat, retain_graph=True)[0]
        def goal(p): return hvp(p).reshape(-1,1) - gU.detach()

        p = Variable(torch.randn_like(flat).reshape(-1,1)*1e-1, requires_grad=True)
        opt = torch.optim.Adam([p], lr=self.if_lr, weight_decay=0.01)
        for _ in range(self.if_epoch):
            opt.zero_grad()
            p.grad = goal(p)
            torch.nn.utils.clip_grad_norm_([p], 1.0)
            opt.step()

        with torch.no_grad():
            adj = flat + p.squeeze()
            u_all[self.u_idx] = adj[:d].reshape(-1, u_all.shape[-1])
            i_all[self.i_idx] = adj[d:].reshape(-1, i_all.shape[-1])
            self.model.user_embedding.weight.data.copy_(u_all.to(self.device))
            self.model.item_embedding.weight.data.copy_(i_all.to(self.device))

        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        torch.save(self.model.state_dict(), save_path)


# ------------------------------------------------------------
# 5. 修复的 Forget Set 评估函数 - 正确的实现
# ------------------------------------------------------------
def correct_evaluate_on_forget_set(model, forget_data, n_items, config, model_name, device, batch_size=1000):
    """
    正确的forget set评估：
    对每个forget用户，在所有物品中计算topk推荐，
    然后看这些推荐中有多少是forget集中的物品
    
    这才是正确的遗忘效果评估方式
    """
    print(f"\n=== {model_name} Correct Eval on Forget Set ===")
    
    model.eval()
    model = model.to(device)
    
    # 确保图相关组件在正确设备上
    if hasattr(model, 'norm_adj_matrix'):
        model.norm_adj_matrix = model.norm_adj_matrix.to(device)
    
    # 构建forget集合：{user: set(forget_items)}
    forget_dict = {}
    for _, row in forget_data.iterrows():
        user, item, label = int(row['user']), int(row['item']), int(row['label'])
        if label == 1:  # 只考虑正样本
            if user not in forget_dict:
                forget_dict[user] = set()
            forget_dict[user].add(item)
    
    if not forget_dict:
        print("No positive forget samples found!")
        return {f'{metric}@{k}': 0.0 for k in config["topk"] for metric in ['hit', 'ndcg', 'recall']}
    
    results = {}
    
    with torch.no_grad():
        for k in config["topk"]:
            hits, ndcgs, recalls = [], [], []
            
            # 对每个forget用户进行评估
            for user in tqdm(forget_dict.keys(), desc=f"Evaluating @{k}", leave=False):
                forget_items = forget_dict[user]
                
                # 计算该用户对所有物品的分数
                all_items = torch.arange(n_items, dtype=torch.long, device=device)
                users_tensor = torch.full((n_items,), user, dtype=torch.long, device=device)
                
                # 分批处理以避免内存问题
                scores = []
                for i in range(0, n_items, batch_size):
                    end_idx = min(i + batch_size, n_items)
                    batch_users = users_tensor[i:end_idx]
                    batch_items = all_items[i:end_idx]
                    
                    inter = {
                        model.USER_ID: batch_users,
                        model.ITEM_ID: batch_items
                    }
                    batch_scores = model.predict(inter)
                    scores.append(batch_scores)
                
                all_scores = torch.cat(scores, dim=0)
                
                # 获取topk推荐
                _, topk_indices = torch.topk(all_scores, k)
                topk_items = topk_indices.cpu().numpy()
                
                # 计算指标
                # Hit@k: topk中是否有forget物品
                hit_count = sum(1 for item in topk_items if item in forget_items)
                hit = 1.0 if hit_count > 0 else 0.0
                hits.append(hit)
                
                # Recall@k: topk中的forget物品数量 / 总forget物品数量
                recall = hit_count / len(forget_items)
                recalls.append(recall)
                
                # NDCG@k
                dcg = 0.0
                for i, item in enumerate(topk_items):
                    if item in forget_items:
                        dcg += 1.0 / np.log2(i + 2)
                
                # 理想情况下的DCG（所有forget物品都在最前面）
                idcg = 0.0
                for i in range(min(len(forget_items), k)):
                    idcg += 1.0 / np.log2(i + 2)
                
                ndcg = dcg / idcg if idcg > 0 else 0.0
                ndcgs.append(ndcg)
            
            # 计算平均值
            results[f'hit@{k}'] = np.mean(hits) if hits else 0.0
            results[f'ndcg@{k}'] = np.mean(ndcgs) if ndcgs else 0.0
            results[f'recall@{k}'] = np.mean(recalls) if recalls else 0.0
            
            print(f"Hit@{k}:{results[f'hit@{k}']:.6f}, NDCG@{k}:{results[f'ndcg@{k}']:.6f}, Recall@{k}:{results[f'recall@{k}']:.6f}")
    
    return results

# ------------------------------------------------------------
# 5. 主流程：训练 → SCIF → ZRF → 评估
# ------------------------------------------------------------
def train_and_unlearn(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cpu = torch.device("cpu")
    data_p = "./scif_data"
    os.makedirs(data_p, exist_ok=True)

    # 1) 数据预处理
    if not os.path.exists(f"{data_p}/train_normal.csv"):
        process_data("scif/amazon-all-beauty-18.inter", data_p,
                     forget_ratio=args.forget_ratio)

    # 2) 生成 RecBole CSV
    ds = "amazon-all-beauty-18"
    tn = pd.read_csv(f"{data_p}/train_normal.csv")
    tr = pd.read_csv(f"{data_p}/train_random.csv")
    vd = pd.read_csv(f"{data_p}/valid.csv")
    tt = pd.read_csv(f"{data_p}/test.csv")
    pd.concat([tn, tr], ignore_index=True).to_csv(f"{data_p}/{ds}_train.csv", index=False)
    vd.to_csv(f"{data_p}/{ds}_valid.csv", index=False)
    tt.to_csv(f"{data_p}/{ds}_test.csv", index=False)

    # 3) RecBole 训练
    print(f"\n=== Training {args.model} on full dataset ({ds}) ===")
    config = Config(
        model=args.model,
        dataset=ds,
        config_dict={
            "data_path": data_p,
            "epochs": 100,
            "train_batch_size": 2048,
            "eval_batch_size": 2048,
            "learning_rate": 0.0001,
            "topk": [5, 10, 20],
            "metrics": ["Hit", "NDCG", "Recall"],
        }
    )
    init_seed(config["seed"], config["reproducibility"])
    init_logger(config)
    dataset_full = create_dataset(config)
    train_data, valid_data, test_data = data_preparation(config, dataset_full)
    model = get_model(config["model"])(config, train_data._dataset).to(device)
    trainer = get_trainer(config["MODEL_TYPE"], config["model"])(config, model)
    trainer.fit(train_data, verbose=True)
    save_full = f"./saved/{ds}_full.pth"
    os.makedirs(os.path.dirname(save_full), exist_ok=True)
    torch.save(model.state_dict(), save_full)

    # 4) Pre-Unlearn 评估
    print(f"\n=== Pre-Unlearn Evaluation ({args.model}) ===")
    t0 = time.time()
    res1 = trainer.evaluate(test_data, load_best_model=False, show_progress=False)
    for k in config["topk"]:
        print(f"Hit@{k}:{res1[f'hit@{k}']:.6f}, "
              f"NDCG@{k}:{res1[f'ndcg@{k}']:.6f}, "
              f"Recall@{k}:{res1[f'recall@{k}']:.6f}")

    # 5) 获取 item 总数
    n_items = dataset_full.item_num

    # 6) SCIF Unlearning
    print(f"\n=== Running SCIF Unlearning ===")
    scif_model = get_model(config["model"])(config, train_data._dataset).to(cpu)
    scif_model.load_state_dict(torch.load(save_full, map_location=device))
    data_gen = DataForSCIF(tn, tr, vd, tt, cpu)
    unlearner = SCIF_Unlearner(scif_model, data_gen, cpu,
                                args.if_epoch, args.if_lr, reg=0.01)
    save_un = f"./saved/{ds}_unlearned.pth"
    unlearner.run(save_un)
    print(f">> SCIF time: {time.time() - t0:.2f}s")

    # 7) ZRF Score
    print(f"\n=== Calculating ZRF ===")
    incompet = get_model(config["model"])(config, train_data._dataset).to(cpu)
    forget_samples = list(zip(tr['user'], tr['item']))
    zrf_score = calculate_zrf(scif_model, incompet, forget_samples,
                        batch_size=2048, device=cpu)

    print(f"\nZRF score: {zrf_score:.6f}")

    # 8) Post-Unlearn 评估
    model_un = get_model(config["model"])(config, train_data._dataset).to(device)
    model_un.load_state_dict(torch.load(save_un, map_location=device))
    trainer_un = get_trainer(config["MODEL_TYPE"], config["model"])(config, model_un)
    res2 = trainer_un.evaluate(test_data, load_best_model=False, show_progress=False)
    for k in config["topk"]:
        print(f"Hit@{k}:{res2[f'hit@{k}']:.6f}, "
              f"NDCG@{k}:{res2[f'ndcg@{k}']:.6f}, "
              f"Recall@{k}:{res2[f'recall@{k}']:.6f}")

    # 9) Forget Set Evaluation
    print(f"\n=== Forget Set Evaluation (Correct Method) ===")
    original_forget_results = correct_evaluate_on_forget_set(
        model, tr, n_items, config, "Pre-Unlearn (Original)", device
    )
    unlearn_forget_results = correct_evaluate_on_forget_set(
        model_un, tr, n_items, config, "Post-Unlearn (SCIF)", device
    )

    # 10) 总结
    print(f"\n" + "=" * 60)
    print(f"COMPREHENSIVE EVALUATION SUMMARY")
    print(f"=" * 60)
    print(f"ZRF Score: {zrf_score:.6f}")
    print(f"\nTest Set Performance (Utility Preservation):")
    print(f"{'-' * 50}")
    for k in config["topk"]:
        print(f"  Hit@{k}:    {res1[f'hit@{k}']:.6f} -> {res2[f'hit@{k}']:.6f} (Δ: {res2[f'hit@{k}'] - res1[f'hit@{k}']:+.6f})")
        print(f"  NDCG@{k}:   {res1[f'ndcg@{k}']:.6f} -> {res2[f'ndcg@{k}']:.6f} (Δ: {res2[f'ndcg@{k}'] - res1[f'ndcg@{k}']:+.6f})")
        print(f"  Recall@{k}: {res1[f'recall@{k}']:.6f} -> {res2[f'recall@{k}']:.6f} (Δ: {res2[f'recall@{k}'] - res1[f'recall@{k}']:+.6f})")
        if k != config["topk"][-1]:
            print()

    print(f"\nForget Set Performance (Forgetting Effectiveness):")
    print(f"{'-' * 50}")
    for k in config["topk"]:
        orig_hit, unlearn_hit = original_forget_results[f'hit@{k}'], unlearn_forget_results[f'hit@{k}']
        orig_ndcg, unlearn_ndcg = original_forget_results[f'ndcg@{k}'], unlearn_forget_results[f'ndcg@{k}']
        orig_recall, unlearn_recall = original_forget_results[f'recall@{k}'], unlearn_forget_results[f'recall@{k}']
        print(f"  Hit@{k}:    {orig_hit:.6f} -> {unlearn_hit:.6f} (Δ: {unlearn_hit - orig_hit:+.6f})")
        print(f"  NDCG@{k}:   {orig_ndcg:.6f} -> {unlearn_ndcg:.6f} (Δ: {unlearn_ndcg - orig_ndcg:+.6f})")
        print(f"  Recall@{k}: {orig_recall:.6f} -> {unlearn_recall:.6f} (Δ: {unlearn_recall - orig_recall:+.6f})")
        if k != config["topk"][-1]:
            print()

    print(f"\nPerformance Analysis:")
    print(f"{'-' * 50}")
    test_preserve_rates = []
    for k in config["topk"]:
        for metric in ['hit', 'ndcg', 'recall']:
            if res1[f'{metric}@{k}'] > 0:
                preserve_rate = res2[f'{metric}@{k}'] / res1[f'{metric}@{k}']
                test_preserve_rates.append(preserve_rate)
    avg_test_preserve = np.mean(test_preserve_rates) if test_preserve_rates else 0
    print(f"  Average Test Performance Preservation: {avg_test_preserve:.2%}")

    forget_rates = []
    for k in config["topk"]:
        for metric in ['hit', 'ndcg', 'recall']:
            orig_val = original_forget_results[f'{metric}@{k}']
            unlearn_val = unlearn_forget_results[f'{metric}@{k}']
            if orig_val > 0:
                forget_rate = 1 - (unlearn_val / orig_val)
                forget_rates.append(forget_rate)
            elif unlearn_val == 0:
                forget_rates.append(1.0)
    avg_forget_rate = np.mean(forget_rates) if forget_rates else 0
    print(f"  Average Forget Performance Degradation: {avg_forget_rate:.2%}")
    print(f"  ZRF Score (Higher is Better): {zrf_score:.6f}")

    num_u = len(unlearner.u_idx)
    num_i = len(unlearner.i_idx)
    dim = unlearner.model.user_embedding.embedding_dim

    num_params_updated = (num_u + num_i) * dim
    total_params = (unlearner.model.user_embedding.num_embeddings +
                    unlearner.model.item_embedding.num_embeddings) * dim
    ratio = num_params_updated / total_params * 100

    print(f">> SCIF updated {num_params_updated} parameters "
        f"({num_u} users × {dim}, {num_i} items × {dim}) "
        f"占全模型参数 {ratio:.4f}%")


    results_summary = {
        'model': args.model,
        'dataset': ds,
        'num_params_updated': num_params_updated,
        'scif_params': {
            'if_epoch': args.if_epoch,
            'if_lr': args.if_lr
        },
        'zrf_score': float(zrf_score),
        'test_original': {f"{metric}@{k}": float(res1[f'{metric}@{k}']) for k in config["topk"] for metric in ['hit', 'ndcg', 'recall']},
        'test_unlearned': {f"{metric}@{k}": float(res2[f'{metric}@{k}']) for k in config["topk"] for metric in ['hit', 'ndcg', 'recall']},
        'forget_original': {k: float(v) for k, v in original_forget_results.items()},
        'forget_unlearned': {k: float(v) for k, v in unlearn_forget_results.items()},
        'summary_metrics': {
            'avg_test_preserve_rate': float(avg_test_preserve),
            'avg_forget_rate': float(avg_forget_rate),
            'total_time_seconds': float(time.time() - t0)
        }
    }

    results_file = f"./results/{ds}_{args.model}_scif_results.json"
    os.makedirs(os.path.dirname(results_file), exist_ok=True)
    with open(results_file, 'w') as f:
        json.dump(results_summary, f, indent=2)
    print(f"\nDetailed results saved to: {results_file}")
    print(f">> Total SCIF execution time: {time.time() - t0:.2f}s")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",       "-m", default="LightGCN")
    parser.add_argument("--if_epoch",    type=int,   default=1000)
    parser.add_argument("--if_lr",       type=float, default=2e-4)
    parser.add_argument("--forget_ratio",type=float, default=0.05)
    args = parser.parse_args()
    train_and_unlearn(args)