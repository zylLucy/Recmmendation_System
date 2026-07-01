"""
基于 Microsoft Recommenders 评估标准评估 NCF 模型

参考: https://github.com/recommenders-team/recommenders
      examples/03_evaluate/evaluation.ipynb
      recommenders/evaluation/python_evaluation.py

输出指标: Recall@10, MRR@10, NDCG@10, Hit@10, Precision@10

评估方法:
  - 对每个用户，模型对测试集中的所有候选项（1个正样本 + 99个负样本）打分
  - 按预测分数降序排列，取前 K 个
  - 计算各项排序指标
"""

import torch
import torch.nn as nn
import numpy as np
import random
from collections import defaultdict
from sklearn.model_selection import train_test_split


# ==================== 数据加载 ====================

def load_data():
    ratings = []
    with open("ratings.dat", "r") as f:
        for line in f:
            uid, mid, rating, _ = line.strip().split("::")
            ratings.append((int(uid), int(mid), float(rating)))

    users = sorted(set(r[0] for r in ratings))
    items = sorted(set(r[1] for r in ratings))
    user2idx = {u: i for i, u in enumerate(users)}
    item2idx = {m: i for i, m in enumerate(items)}

    num_users = len(users)
    num_items = len(items)

    user_item_map = defaultdict(dict)
    for uid, mid, rating in ratings:
        user_item_map[user2idx[uid]][item2idx[mid]] = rating

    return user_item_map, num_users, num_items


def prepare_test_data(user_item_map, num_users, num_items, test_ratio=0.2, pos_threshold=4):
    """
    准备测试数据: 每个用户留出 20% 的正样本，每个正样本配 99 个随机负样本
    返回: test_data = [(user_idx, item_idx, label), ...]
    """
    all_pos = []
    for u, items in user_item_map.items():
        for i, r in items.items():
            if r >= pos_threshold:
                all_pos.append((u, i))

    user_pos = defaultdict(list)
    for u, i in all_pos:
        user_pos[u].append(i)

    test_pos = []
    for u, items in user_pos.items():
        if len(items) < 2:
            continue
        _, te_items = train_test_split(items, test_size=test_ratio, random_state=42)
        test_pos.extend([(u, i) for i in te_items])

    all_items_set = set(range(num_items))
    test_data = []
    for u, i in test_pos:
        test_data.append((u, i, 1.0))
        interacted = set(user_item_map[u].keys())
        neg_candidates = list(all_items_set - interacted - {i})
        neg_samples = random.sample(neg_candidates, min(99, len(neg_candidates)))
        for ni in neg_samples:
            test_data.append((u, ni, 0.0))

    return test_data


# ==================== 模型定义 ====================

class NCF(nn.Module):
    def __init__(self, num_users, num_items, embedding_dim=32,
                 mlp_layers=[64, 32, 16], dropout=0.2):
        super().__init__()
        self.gmf_user_emb = nn.Embedding(num_users, embedding_dim)
        self.gmf_item_emb = nn.Embedding(num_items, embedding_dim)
        self.mlp_user_emb = nn.Embedding(num_users, embedding_dim)
        self.mlp_item_emb = nn.Embedding(num_items, embedding_dim)

        mlp_modules = []
        input_dim = embedding_dim * 2
        for out_dim in mlp_layers:
            mlp_modules.append(nn.Linear(input_dim, out_dim))
            mlp_modules.append(nn.ReLU())
            mlp_modules.append(nn.Dropout(dropout))
            input_dim = out_dim
        self.mlp = nn.Sequential(*mlp_modules)

        fusion_dim = embedding_dim + mlp_layers[-1]
        self.predict = nn.Sequential(
            nn.Linear(fusion_dim, 1),
            nn.Sigmoid()
        )

    def forward(self, user, item):
        gmf_out = self.gmf_user_emb(user) * self.gmf_item_emb(item)
        mlp_out = self.mlp(torch.cat([self.mlp_user_emb(user), self.mlp_item_emb(item)], dim=-1))
        return self.predict(torch.cat([gmf_out, mlp_out], dim=-1)).squeeze()


# ==================== 评估指标（参考 Recommenders 实现） ====================

def get_predictions_per_user(model, test_data, device):
    """
    对测试集中每个用户的所有候选项打分，返回 {user: [(item, score, label), ...]}
    """
    model.eval()
    user_items = defaultdict(list)
    for u, i, label in test_data:
        user_items[u].append((i, label))

    user_preds = {}
    with torch.no_grad():
        for u, items in user_items.items():
            item_ids = [x[0] for x in items]
            labels = [x[1] for x in items]

            user_tensor = torch.LongTensor([u] * len(item_ids)).to(device)
            item_tensor = torch.LongTensor(item_ids).to(device)
            scores = model(user_tensor, item_tensor).cpu().numpy()

            user_preds[u] = list(zip(item_ids, scores, labels))

    return user_preds


def precision_at_k(user_preds, k=10):
    """
    Precision@K: 前 K 个推荐中正样本的比例，对所有用户取平均
    参考 Spark MLlib RankingMetrics.precisionAt
    """
    precisions = []
    for u, items in user_preds.items():
        ranked = sorted(items, key=lambda x: x[1], reverse=True)[:k]
        hits = sum(1 for _, _, label in ranked if label == 1.0)
        precisions.append(hits / k)
    return np.mean(precisions)


def recall_at_k(user_preds, k=10):
    """
    Recall@K: 前 K 个推荐中命中的正样本数 / 该用户所有正样本数，对所有用户取平均
    """
    recalls = []
    for u, items in user_preds.items():
        total_pos = sum(1 for _, _, label in items if label == 1.0)
        if total_pos == 0:
            continue
        ranked = sorted(items, key=lambda x: x[1], reverse=True)[:k]
        hits = sum(1 for _, _, label in ranked if label == 1.0)
        recalls.append(hits / total_pos)
    return np.mean(recalls)


def hit_at_k(user_preds, k=10):
    """
    Hit@K (HR@K): 前 K 个推荐中至少命中一个正样本的用户比例
    """
    hits = 0
    total = 0
    for u, items in user_preds.items():
        ranked = sorted(items, key=lambda x: x[1], reverse=True)[:k]
        if any(label == 1.0 for _, _, label in ranked):
            hits += 1
        total += 1
    return hits / total if total > 0 else 0.0


def ndcg_at_k(user_preds, k=10):
    """
    NDCG@K: 归一化折损累计增益
    参考: https://en.wikipedia.org/wiki/Discounted_cumulative_gain
    """
    ndcgs = []
    for u, items in user_preds.items():
        ranked = sorted(items, key=lambda x: x[1], reverse=True)[:k]
        labels = [label for _, _, label in ranked]

        # DCG@K
        dcg = sum((2**rel - 1) / np.log2(i + 2) for i, rel in enumerate(labels))

        # IDCG@K: 理想排序（所有正样本排最前面）
        ideal_labels = sorted([label for _, _, label in items], reverse=True)[:k]
        idcg = sum((2**rel - 1) / np.log2(i + 2) for i, rel in enumerate(ideal_labels))

        ndcgs.append(dcg / idcg if idcg > 0 else 0.0)
    return np.mean(ndcgs)


def mrr_at_k(user_preds, k=10):
    """
    MRR@K (Mean Reciprocal Rank): 第一个正样本出现位置的倒数的平均值
    参考: https://en.wikipedia.org/wiki/Mean_reciprocal_rank
    """
    rr_list = []
    for u, items in user_preds.items():
        ranked = sorted(items, key=lambda x: x[1], reverse=True)[:k]
        for rank, (_, _, label) in enumerate(ranked, start=1):
            if label == 1.0:
                rr_list.append(1.0 / rank)
                break
        else:
            rr_list.append(0.0)
    return np.mean(rr_list)


# ==================== 主流程 ====================

def main():
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {DEVICE}")

    # 固定随机种子
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)

    # 加载数据
    print("加载数据...")
    user_item_map, num_users, num_items = load_data()
    print(f"用户数: {num_users}, 电影数: {num_items}")

    # 根据模型版本选择正样本阈值和模型路径
    # V1: rating >= 4, V2: rating >= 3, V3: rating >= 3
    POS_THRESHOLD = 3
    MODEL_PATH = "ncf_model_v3.pth"
    print(f"正样本阈值: rating >= {POS_THRESHOLD}")
    print(f"模型路径: {MODEL_PATH}")

    # 准备测试数据
    test_data = prepare_test_data(user_item_map, num_users, num_items,
                                  pos_threshold=POS_THRESHOLD)
    print(f"测试样本: {len(test_data):,}")

    # 加载模型
    print("\n加载模型...")
    model = NCF(num_users, num_items).to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()

    # 获取每个用户的预测结果
    print("计算预测分数...")
    user_preds = get_predictions_per_user(model, test_data, DEVICE)
    print(f"评估用户数: {len(user_preds)}")

    # 计算各项指标
    print("\n" + "=" * 60)
    print("NCF 模型评估结果（基于 Microsoft Recommenders 标准）")
    print("测试方式: Leave-one-out, 每个正样本配 99 个随机负样本")
    print("=" * 60)

    metrics = {}
    k = 10
    metrics[f"Precision@{k}"] = precision_at_k(user_preds, k=k)
    metrics[f"Recall@{k}"] = recall_at_k(user_preds, k=k)
    metrics[f"Hit@{k}"] = hit_at_k(user_preds, k=k)
    metrics[f"NDCG@{k}"] = ndcg_at_k(user_preds, k=k)
    metrics[f"MRR@{k}"] = mrr_at_k(user_preds, k=k)

    print(f"\n{'指标':<16} {'K=10':<10}")
    print("-" * 26)
    for metric_name in ["Precision", "Recall", "Hit", "NDCG", "MRR"]:
        print(f"{metric_name:<16} {metrics[f'{metric_name}@{k}']:.4f}")

    print("\n" + "=" * 60)
    print("重点指标 (K=10):")
    print(f"  Precision@10: {metrics['Precision@10']:.4f}")
    print(f"  Recall@10:    {metrics['Recall@10']:.4f}")
    print(f"  Hit@10:       {metrics['Hit@10']:.4f}")
    print(f"  NDCG@10:      {metrics['NDCG@10']:.4f}")
    print(f"  MRR@10:       {metrics['MRR@10']:.4f}")
    print("=" * 60)


if __name__ == "__main__":
    main()
