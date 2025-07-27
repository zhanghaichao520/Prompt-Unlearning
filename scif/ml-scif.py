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
# 2. 数据预处理：ML-100K / ML-1M
# ------------------------------------------------------------
def process_ml100k_data(orig_file, out_dir, forget_ratio=0.1):
    df = pd.read_csv(orig_file, sep='\t', names=['user','item','rating','timestamp'])
    df['label'] = (df['rating'] >= 4).astype(int)
    groups = df.groupby('user')
    users = list(groups.groups.keys())
    forget = set(np.random.choice(users, int(len(users)*forget_ratio), replace=False))
    tn, tr, vd, tt = [], [], [], []
    for u, g in groups:
        if len(g)<3: continue
        g = g.sample(frac=1, random_state=42)
        n1, n2 = int(len(g)*0.6), int(len(g)*0.2)
        tgt = tr if u in forget else tn
        for row in g.iloc[:n1].itertuples(index=False):
            tgt.append([row.user,row.item,row.rating,row.timestamp,row.label])
        for row in g.iloc[n1:n1+n2].itertuples(index=False):
            vd.append([row.user,row.item,row.rating,row.timestamp,row.label])
        for row in g.iloc[n1+n2:].itertuples(index=False):
            tt.append([row.user,row.item,row.rating,row.timestamp,row.label])
    os.makedirs(out_dir, exist_ok=True)
    cols = ['user','item','rating','timestamp','label']
    pd.DataFrame(tn, columns=cols).to_csv(f"{out_dir}/train_normal.csv", index=False)
    pd.DataFrame(tr, columns=cols).to_csv(f"{out_dir}/train_random.csv", index=False)
    pd.DataFrame(vd, columns=cols).to_csv(f"{out_dir}/valid.csv",        index=False)
    pd.DataFrame(tt, columns=cols).to_csv(f"{out_dir}/test.csv",         index=False)
    all_df = pd.concat([pd.DataFrame(tn,columns=cols),
                        pd.DataFrame(tr,columns=cols),
                        pd.DataFrame(vd,columns=cols),
                        pd.DataFrame(tt,columns=cols)],
                       ignore_index=True)
    avg = all_df.groupby(['user','item'], sort=False)['label'].mean().reset_index()
    np.save(f"{out_dir}/avg_labels.npy",     avg['label'].to_numpy())
    np.save(f"{out_dir}/user_item_pairs.npy",avg[['user','item']].to_numpy())

def process_ml1m_data(orig_file, out_dir, forget_ratio=0.1):
    df = pd.read_csv(orig_file, sep='\t', names=['user','item','rating','timestamp'])
    df['user'] = df['user'].astype(int)-1
    df['item'] = df['item'].astype(int)-1
    df['label']= (df['rating'] >= 4).astype(int)
    groups = df.groupby('user')
    users = list(groups.groups.keys())
    forget = set(np.random.choice(users, int(len(users)*forget_ratio), replace=False))
    tn, tr, vd, tt = [], [], [], []
    for u, g in groups:
        if len(g)<3: continue
        g = g.sample(frac=1, random_state=42)
        n1, n2 = int(len(g)*0.6), int(len(g)*0.2)
        tgt = tr if u in forget else tn
        for row in g.iloc[:n1].itertuples(index=False):
            tgt.append([row.user,row.item,row.rating,row.timestamp,row.label])
        for row in g.iloc[n1:n1+n2].itertuples(index=False):
            vd.append([row.user,row.item,row.rating,row.timestamp,row.label])
        for row in g.iloc[n1+n2:].itertuples(index=False):
            tt.append([row.user,row.item,row.rating,row.timestamp,row.label])
    os.makedirs(out_dir, exist_ok=True)
    cols=['user','item','rating','timestamp','label']
    pd.DataFrame(tn, columns=cols).to_csv(f"{out_dir}/train_normal.csv", index=False)
    pd.DataFrame(tr, columns=cols).to_csv(f"{out_dir}/train_random.csv", index=False)
    pd.DataFrame(vd, columns=cols).to_csv(f"{out_dir}/valid.csv",        index=False)
    pd.DataFrame(tt, columns=cols).to_csv(f"{out_dir}/test.csv",         index=False)
    all_df = pd.concat([pd.DataFrame(tn,columns=cols),
                        pd.DataFrame(tr,columns=cols),
                        pd.DataFrame(vd,columns=cols),
                        pd.DataFrame(tt,columns=cols)],
                       ignore_index=True)
    avg = all_df.groupby(['user','item'], sort=False)['label'].mean().reset_index()
    np.save(f"{out_dir}/avg_labels.npy",     avg['label'].to_numpy())
    np.save(f"{out_dir}/user_item_pairs.npy",avg[['user','item']].to_numpy())


# ------------------------------------------------------------
# 3. DataForSCIF
# ------------------------------------------------------------
class DataForSCIF:
    def __init__(self, tn_df, tr_df, vd_df, tt_df, device):
        self.device     = device
        self.train_norm = tn_df.reset_index(drop=True)
        self.train_rand = tr_df.reset_index(drop=True)
        self.valid      = vd_df.reset_index(drop=True)
        self.test       = tt_df.reset_index(drop=True)
        self.train      = pd.concat([self.train_norm, self.train_rand], ignore_index=True)
        max_u = max(df['user'].max() for df in (self.train,self.valid,self.test) if not df.empty)
        max_i = max(df['item'].max() for df in (self.train,self.valid,self.test) if not df.empty)
        self.n_users = int(max_u + 1)
        self.n_items = int(max_i + 1)

    def get_train_tensor(self):
        arr = self.train[['user','item','label']].values.astype(np.int64)
        return torch.from_numpy(arr).long().to(self.device)
    def get_unlearn_tensor(self):
        arr = self.train_rand[['user','item','label']].values.astype(np.int64)
        return torch.from_numpy(arr).long().to(self.device)
    def get_valid_tensor(self):
        arr = self.valid[['user','item','label']].values.astype(np.int64)
        return torch.from_numpy(arr).long().to(self.device)
    def get_test_tensor(self):
        arr = self.test[['user','item','label']].values.astype(np.int64)
        return torch.from_numpy(arr).long().to(self.device)


# ------------------------------------------------------------
# 4. SCIF_Unlearner (CPU)
# ------------------------------------------------------------
class SCIF_Unlearner:
    def __init__(self, model, data_gen, device,
                 if_epoch=1000, if_lr=5e-2, reg=0.01):
        self.model    = model
        self.data_gen = data_gen
        self.device   = device
        self.if_epoch = if_epoch
        self.if_lr    = if_lr
        self.reg      = reg

        un = self.data_gen.get_unlearn_tensor()
        us, is_ = un[:,0].unique(), un[:,1].unique()
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
        return F.binary_cross_entropy_with_logits(logits, labels, reduction='mean') \
               + self.reg*(u_emb.norm()+i_emb.norm())

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
        flat   = torch.cat([u_flat,i_flat],dim=0).clone().requires_grad_(True)

        u_para, i_para = u_all.clone(), i_all.clone()
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
            torch.nn.utils.clip_grad_norm_([p],1.0)
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
# 6. 主流程：训练 → SCIF → ZRF → 评估
# ------------------------------------------------------------
def train_and_unlearn(args):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cpu    = torch.device("cpu")
    print(f"Using device: {device}")
    dp = "./scif_data"
    os.makedirs(dp, exist_ok=True)

    # 6.1 数据预处理
    if not os.path.exists(f"{dp}/train_normal.csv"):
        if args.use_ml100k:
            process_ml100k_data("scif/ml-100k.inter", dp, forget_ratio=0.1)
        else:
            process_ml1m_data("scif/ml-1m.inter", dp, forget_ratio=0.1)

    ds = "ml-100k" if args.use_ml100k else "ml-1m"

    # 6.2 生成 RecBole CSV
    tn_raw = pd.read_csv(f"{dp}/train_normal.csv")
    tr_raw = pd.read_csv(f"{dp}/train_random.csv")
    vd_raw = pd.read_csv(f"{dp}/valid.csv")
    tt_raw = pd.read_csv(f"{dp}/test.csv")
    combined = pd.concat([tn_raw, tr_raw], ignore_index=True)
    combined.to_csv(f"{dp}/{ds}_train.csv", index=False)
    vd_raw.to_csv(f"{dp}/{ds}_valid.csv", index=False)
    tt_raw.to_csv(f"{dp}/{ds}_test.csv",  index=False)

    # 6.3 RecBole 训练
    config = Config(
        model=args.model, dataset=ds,
        config_dict={
            "data_path": dp, "epochs":100,
            "train_batch_size":2048, "eval_batch_size":2048,
            "learning_rate":0.0002,
            "topk":[5,10,20], "metrics":["Hit","NDCG","Recall"],
            # "gpu_id": None,   # 禁用GPU
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

    # 6.4 Pre-Unlearn 评估
    print(f"\n=== Pre-Unlearn Eval ({args.model}) ===")
    res1 = trainer.evaluate(test_data, load_best_model=False, show_progress=False)
    for k in config["topk"]:
        print(f"Hit@{k}:{res1[f'hit@{k}']:.6f}, NDCG@{k}:{res1[f'ndcg@{k}']:.6f}, Recall@{k}:{res1[f'recall@{k}']:.6f}")

    # 6.5 raw_to_internal
    raw2u = dataset_full.field2token_id[config['USER_ID_FIELD']]
    raw2i = dataset_full.field2token_id[config['ITEM_ID_FIELD']]
    def raw_to_internal(df_raw, m_u, m_i):
        df = df_raw[['user','item','label']].copy()
        df['user']=df['user'].astype(str).map(m_u)
        df['item']=df['item'].astype(str).map(m_i)
        df = df.dropna(subset=['user','item'])
        df[['user','item','label']] = df[['user','item','label']].astype(int)
        return df
    tn = raw_to_internal(tn_raw, raw2u, raw2i)
    tr = raw_to_internal(tr_raw, raw2u, raw2i)
    vd = raw_to_internal(vd_raw, raw2u, raw2i)
    tt = raw_to_internal(tt_raw, raw2u, raw2i)
    
    # 获取物品总数（用于correct evaluation）
    n_items = dataset_full.item_num

    # 6.6 SCIF Unlearning (CPU)
    print(f"\n=== Running SCIF Unlearning ===")
    t0 = time.time()  # Start timer just before unlearning
    scif_model = get_model(config["model"])(config, train_data._dataset).to(cpu)
    scif_model.load_state_dict(torch.load(save_full, map_location="cpu"))
    data_gen = DataForSCIF(tn, tr, vd, tt, cpu)
    unlearner = SCIF_Unlearner(scif_model, data_gen, cpu,
                                args.if_epoch, args.if_lr, reg=0.01)
    save_un = f"./saved/{ds}_unlearned.pth"
    unlearner.run(save_un)
    print(f">> SCIF time: {time.time()-t0:.2f}s")

    # 6.7 计算 ZRF
    print(f"\n=== Calculating ZRF ===")
    incompet = get_model(config["model"])(config, train_data._dataset).to(cpu)
    forget_samples = list(zip(tr['user'], tr['item']))
    zrf = calculate_zrf(scif_model, incompet, forget_samples,
                        batch_size=2048, device=cpu)
    print(f"ZRF score: {zrf:.6f}")

    # 6.8 Post-Unlearn 评估
    print(f"\n=== Post-Unlearn Eval ({args.model}) ===")
    model_un = get_model(config["model"])(config, train_data._dataset).to(device)
    model_un.load_state_dict(torch.load(save_un, map_location=device))
    trainer_un = get_trainer(config["MODEL_TYPE"], config["model"])(config, model_un)
    res2 = trainer_un.evaluate(test_data, load_best_model=False, show_progress=False)
    for k in config["topk"]:
        print(f"Hit@{k}:{res2[f'hit@{k}']:.6f}, NDCG@{k}:{res2[f'ndcg@{k}']:.6f}, Recall@{k}:{res2[f'recall@{k}']:.6f}")
    
    # 6.9 使用正确的方法在forget集上评估
    print(f"\n=== Forget Set Evaluation (Correct Method) ===")
    
    # 原始模型在forget集上的评估
    original_forget_results = correct_evaluate_on_forget_set(
        model, tr, n_items, config, "Pre-Unlearn (Original)", device
    )
    
    # Unlearn模型在forget集上的评估
    unlearn_forget_results = correct_evaluate_on_forget_set(
        model_un, tr, n_items, config, "Post-Unlearn (SCIF)", device
    )
    
    # 6.10 总结所有关键指标
    print(f"\n" + "="*60)
    print(f"COMPREHENSIVE EVALUATION SUMMARY")
    print(f"="*60)
    print(f"ZRF Score: {zrf:.6f}")
    print(f"\nTest Set Performance (Utility Preservation):")
    print(f"{'-'*50}")
    for k in config["topk"]:
        print(f"  Hit@{k}:    {res1[f'hit@{k}']:.6f} -> {res2[f'hit@{k}']:.6f} (Δ: {res2[f'hit@{k}']-res1[f'hit@{k}']:+.6f})")
        print(f"  NDCG@{k}:   {res1[f'ndcg@{k}']:.6f} -> {res2[f'ndcg@{k}']:.6f} (Δ: {res2[f'ndcg@{k}']-res1[f'ndcg@{k}']:+.6f})")
        print(f"  Recall@{k}: {res1[f'recall@{k}']:.6f} -> {res2[f'recall@{k}']:.6f} (Δ: {res2[f'recall@{k}']-res1[f'recall@{k}']:+.6f})")
        if k != config["topk"][-1]:
            print()
    
    print(f"\nForget Set Performance (Forgetting Effectiveness):")
    print(f"{'-'*50}")
    for k in config["topk"]:
        orig_hit, unlearn_hit = original_forget_results[f'hit@{k}'], unlearn_forget_results[f'hit@{k}']
        orig_ndcg, unlearn_ndcg = original_forget_results[f'ndcg@{k}'], unlearn_forget_results[f'ndcg@{k}']
        orig_recall, unlearn_recall = original_forget_results[f'recall@{k}'], unlearn_forget_results[f'recall@{k}']
        
        print(f"  Hit@{k}:    {orig_hit:.6f} -> {unlearn_hit:.6f} (Δ: {unlearn_hit-orig_hit:+.6f})")
        print(f"  NDCG@{k}:   {orig_ndcg:.6f} -> {unlearn_ndcg:.6f} (Δ: {unlearn_ndcg-orig_ndcg:+.6f})")
        print(f"  Recall@{k}: {orig_recall:.6f} -> {unlearn_recall:.6f} (Δ: {unlearn_recall-orig_recall:+.6f})")
        if k != config["topk"][-1]:
            print()
    
    # 6.11 性能评估总结
    print(f"\nPerformance Analysis:")
    print(f"{'-'*50}")
    
    # 计算平均性能保持率（测试集）
    test_preserve_rates = []
    for k in config["topk"]:
        for metric in ['hit', 'ndcg', 'recall']:
            if res1[f'{metric}@{k}'] > 0:
                preserve_rate = res2[f'{metric}@{k}'] / res1[f'{metric}@{k}']
                test_preserve_rates.append(preserve_rate)
    avg_test_preserve = np.mean(test_preserve_rates) if test_preserve_rates else 0
    print(f"  Average Test Performance Preservation: {avg_test_preserve:.2%}")
    
    # 计算平均遗忘率（forget集）
    forget_rates = []
    for k in config["topk"]:
        for metric in ['hit', 'ndcg', 'recall']:
            orig_val = original_forget_results[f'{metric}@{k}']
            unlearn_val = unlearn_forget_results[f'{metric}@{k}']
            if orig_val > 0:
                forget_rate = 1 - (unlearn_val / orig_val)
                forget_rates.append(forget_rate)
            else:
                # 如果原始值为0，但unlearn后仍为0，这是好事
                if unlearn_val == 0:
                    forget_rates.append(1.0)  # 完全遗忘
    avg_forget_rate = np.mean(forget_rates) if forget_rates else 0
    print(f"  Average Forget Performance Degradation: {avg_forget_rate:.2%}")
    
    print(f"  ZRF Score (Higher is Better): {zrf:.6f}")
    
    # 6.12 保存结果到文件
    results_summary = {
        'model': args.model,
        'dataset': ds,
        'scif_params': {
            'if_epoch': args.if_epoch,
            'if_lr': args.if_lr
        },
        'zrf_score': float(zrf),
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
    
    print(f">> Total SCIF execution time: {time.time()-t0:.2f}s")
    print(f"="*60)

if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--model","-m",default="LightGCN")
    p.add_argument("--if_epoch", type=int,   default=1000)
    p.add_argument("--if_lr",    type=float, default=1e-4)
    p.add_argument("--use_ml100k",action="store_true")
    args=p.parse_args()
    train_and_unlearn(args)