"""
模型评估脚本：加载训练好的 NCF 模型，计算分类指标和排序指标

指标包括:
  - 分类指标: Accuracy, Precision, Recall, F1-Score, Confusion Matrix
  - 排序指标: HR@K, NDCG@K
"""

import torch
import torch.nn as nn
import numpy as np
import random
from collections import defaultdict
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score
)

# ==================== 数据加载（与训练脚本一致） ====================

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


def implicit_split(user_item_map, num_users, num_items, neg_ratio=4, test_ratio=0.2):
    """改进版：正样本 rating>=3，负样本 rating<=2，训练集不用未交互电影"""
    all_pos = []
    all_neg = []
    for u, items in user_item_map.items():
        for i, r in items.items():
            if r >= 3:
                all_pos.append((u, i))
            else:
                all_neg.append((u, i))

    train_pos, test_pos = [], []
    user_pos = defaultdict(list)
    for u, i in all_pos:
        user_pos[u].append(i)

    for u, items in user_pos.items():
        if len(items) < 2:
            train_pos.extend([(u, i) for i in items])
        else:
            t_items, te_items = train_test_split(items, test_size=test_ratio, random_state=42)
            train_pos.extend([(u, i) for i in t_items])
            test_pos.extend([(u, i) for i in te_items])

    user_neg = defaultdict(list)
    for u, i in all_neg:
        user_neg[u].append(i)

    train_neg, test_neg = [], []
    for u, items in user_neg.items():
        if len(items) < 2:
            train_neg.extend([(u, i) for i in items])
        else:
            t_items, te_items = train_test_split(items, test_size=test_ratio, random_state=42)
            train_neg.extend([(u, i) for i in t_items])
            test_neg.extend([(u, i) for i in te_items])

    # 训练集: 正样本 + 强负样本
    train_data = []
    for u, i in train_pos:
        train_data.append((u, i, 1.0))
    for u, i in train_neg:
        train_data.append((u, i, 0.0))

    # 测试集（排序评估用）: 每个正样本配 99 个随机未交互电影
    all_items_set = set(range(num_items))
    test_data = []
    for u, i in test_pos:
        test_data.append((u, i, 1.0))
        interacted = set(user_item_map[u].keys())
        neg_candidates = list(all_items_set - interacted - {i})
        neg_samples = random.sample(neg_candidates, min(99, len(neg_candidates)))
        for ni in neg_samples:
            test_data.append((u, ni, 0.0))

    return train_data, test_data


# ==================== 模型定义（与训练脚本一致） ====================

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


# ==================== 评估函数 ====================

def evaluate_classification(model, test_data, device, threshold=0.5):
    """
    计算二分类指标: Accuracy, Precision, Recall, F1-Score, AUC
    """
    model.eval()
    all_preds, all_labels, all_scores = [], [], []

    # 分批处理避免内存溢出
    batch_size = 4096
    users = np.array([d[0] for d in test_data])
    items = np.array([d[1] for d in test_data])
    labels = np.array([d[2] for d in test_data])

    with torch.no_grad():
        for start in range(0, len(test_data), batch_size):
            end = min(start + batch_size, len(test_data))
            u_tensor = torch.LongTensor(users[start:end]).to(device)
            i_tensor = torch.LongTensor(items[start:end]).to(device)
            scores = model(u_tensor, i_tensor).cpu().numpy()
            all_scores.extend(scores)
            all_preds.extend((scores >= threshold).astype(int))
            all_labels.extend(labels[start:end])

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_scores = np.array(all_scores)

    # 计算各项指标
    acc = accuracy_score(all_labels, all_preds)
    prec = precision_score(all_labels, all_preds, zero_division=0)
    rec = recall_score(all_labels, all_preds, zero_division=0)
    f1 = f1_score(all_labels, all_preds, zero_division=0)
    auc = roc_auc_score(all_labels, all_scores)
    cm = confusion_matrix(all_labels, all_preds)

    return {
        "Accuracy": acc,
        "Precision": prec,
        "Recall": rec,
        "F1-Score": f1,
        "AUC": auc,
        "Confusion_Matrix": cm,
        "all_labels": all_labels,
        "all_preds": all_preds,
    }


def evaluate_ranking(model, test_data, num_items, device, K=10):
    """计算 HR@K 和 NDCG@K"""
    model.eval()
    user_items = defaultdict(list)
    for u, i, label in test_data:
        user_items[u].append((i, label))

    hr_sum, ndcg_sum, user_count = 0, 0, 0

    with torch.no_grad():
        for u, items in user_items.items():
            item_ids = [x[0] for x in items]
            labels = [x[1] for x in items]

            user_tensor = torch.LongTensor([u] * len(item_ids)).to(device)
            item_tensor = torch.LongTensor(item_ids).to(device)
            preds = model(user_tensor, item_tensor).cpu().numpy()

            ranked = sorted(zip(item_ids, preds, labels), key=lambda x: x[1], reverse=True)
            top_k = [x[2] for x in ranked[:K]]

            hr = 1.0 if 1.0 in top_k else 0.0
            hr_sum += hr

            dcg = sum((2**rel - 1) / np.log2(i + 2) for i, rel in enumerate(top_k))
            ideal_labels = sorted(labels, reverse=True)[:K]
            idcg = sum((2**rel - 1) / np.log2(i + 2) for i, rel in enumerate(ideal_labels))
            ndcg = dcg / idcg if idcg > 0 else 0.0
            ndcg_sum += ndcg

            user_count += 1

    return hr_sum / user_count, ndcg_sum / user_count


# ==================== 主流程 ====================

def main():
    DEVICE = torch.device("mps" if torch.mps.is_available() else "cpu")
    print(f"使用设备: {DEVICE}")

    # 固定随机种子
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)

    # 加载数据
    print("加载数据...")
    user_item_map, num_users, num_items = load_data()
    train_data, test_data = implicit_split(user_item_map, num_users, num_items)
    print(f"用户数: {num_users}, 电影数: {num_items}")
    print(f"测试样本(排序评估): {len(test_data)}")

    # 构建分类评估用的测试集（正样本 + 真实负样本 rating<=2，不用未交互电影）
    all_pos = []
    all_neg = []
    for u, items in user_item_map.items():
        for i, r in items.items():
            if r >= 3:
                all_pos.append((u, i))
            else:
                all_neg.append((u, i))

    user_pos = defaultdict(list)
    for u, i in all_pos:
        user_pos[u].append(i)
    test_pos = []
    for u, items in user_pos.items():
        if len(items) < 2:
            continue
        _, te_items = train_test_split(items, test_size=0.2, random_state=42)
        test_pos.extend([(u, i) for i in te_items])

    user_neg = defaultdict(list)
    for u, i in all_neg:
        user_neg[u].append(i)
    test_neg = []
    for u, items in user_neg.items():
        if len(items) < 2:
            continue
        _, te_items = train_test_split(items, test_size=0.2, random_state=42)
        test_neg.extend([(u, i) for i in te_items])

    cls_test_data = []
    for u, i in test_pos:
        cls_test_data.append((u, i, 1.0))
    for u, i in test_neg:
        cls_test_data.append((u, i, 0.0))

    pos_cls = len(test_pos)
    neg_cls = len(test_neg)
    print(f"测试样本(分类评估): {len(cls_test_data)}, 正样本 {pos_cls:,}, 负样本 {neg_cls:,}, 正负比 1:{neg_cls/pos_cls:.1f}")

    # 加载模型
    print("\n加载模型...")
    model = NCF(num_users, num_items).to(DEVICE)
    model.load_state_dict(torch.load("ncf_model.pth", map_location=DEVICE))
    model.eval()

    # ========== 1. 分类指标评估（真实正负样本，非 1:99） ==========
    print("\n" + "=" * 60)
    print("【分类指标评估 — 真实正负样本】(threshold=0.5)")
    print(f"  正样本: rating>=3, 负样本: rating<=2")
    print("=" * 60)

    results = evaluate_classification(model, cls_test_data, DEVICE, threshold=0.5)

    print(f"\n  Accuracy:   {results['Accuracy']:.4f}")
    print(f"  Precision:  {results['Precision']:.4f}")
    print(f"  Recall:     {results['Recall']:.4f}")
    print(f"  F1-Score:   {results['F1-Score']:.4f}")
    print(f"  AUC:        {results['AUC']:.4f}")

    cm = results["Confusion_Matrix"]
    print(f"\n  混淆矩阵:")
    print(f"                预测负样本  预测正样本")
    print(f"  实际负样本       {cm[0][0]:>6d}      {cm[0][1]:>6d}")
    print(f"  实际正样本       {cm[1][0]:>6d}      {cm[1][1]:>6d}")

    print(f"\n  详细分类报告:")
    print(classification_report(results["all_labels"], results["all_preds"],
                                target_names=["负样本(rating<=2)", "正样本(rating>=3)"],
                                digits=4))

    # ========== 2. 不同阈值下的指标 ==========
    print("=" * 60)
    print("【不同阈值下的指标对比（真实正负样本）】")
    print("=" * 60)
    print(f"  {'Threshold':<12} {'Accuracy':<10} {'Precision':<10} {'Recall':<10} {'F1-Score':<10}")
    print("  " + "-" * 52)
    for th in [0.3, 0.4, 0.5, 0.6, 0.7]:
        r = evaluate_classification(model, cls_test_data, DEVICE, threshold=th)
        print(f"  {th:<12.1f} {r['Accuracy']:<10.4f} {r['Precision']:<10.4f} {r['Recall']:<10.4f} {r['F1-Score']:<10.4f}")

    # ========== 3. 排序指标评估（1:99 模拟真实场景） ==========
    print("\n" + "=" * 60)
    print("【排序指标评估 — 1:99 模拟真实推荐场景】")
    print("=" * 60)
    for k in [5, 10, 20]:
        hr, ndcg = evaluate_ranking(model, test_data, num_items, DEVICE, K=k)
        print(f"  HR@{k:<2d}:  {hr:.4f}  |  NDCG@{k:<2d}:  {ndcg:.4f}")


if __name__ == "__main__":
    main()
