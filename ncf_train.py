"""
Neural Collaborative Filtering (NCF) on MovieLens 1M
参考论文: He et al. "Neural Collaborative Filtering" (WWW 2017)

模型结构: GMF + MLP 双分支融合
  - GMF 分支: 逐元素乘积 → 捕捉线性交互
  - MLP 分支: 拼接后多层感知机 → 捕捉非线性交互
  - 最终融合: 两个分支的隐向量拼接后通过全连接层输出预测分数
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import random
from collections import defaultdict
from sklearn.model_selection import train_test_split

# ==================== 1. 数据加载 ====================

def load_data():
    """加载 MovieLens 1M 数据集"""
    # 加载评分数据
    ratings = []
    with open("ratings.dat", "r") as f:
        for line in f:
            uid, mid, rating, _ = line.strip().split("::")
            ratings.append((int(uid), int(mid), float(rating)))

    # 构建 user_id -> 连续索引, item_id -> 连续索引 的映射
    users = sorted(set(r[0] for r in ratings))
    items = sorted(set(r[1] for r in ratings))
    user2idx = {u: i for i, u in enumerate(users)}
    item2idx = {m: i for i, m in enumerate(items)}

    num_users = len(users)
    num_items = len(items)
    print(f"用户数: {num_users}, 电影数: {num_items}, 评分数: {len(ratings)}")

    # 构建 user -> {item: rating} 字典
    user_item_map = defaultdict(dict)
    for uid, mid, rating in ratings:
        user_item_map[user2idx[uid]][item2idx[mid]] = rating

    return user_item_map, num_users, num_items


def implicit_split(user_item_map, num_users, num_items, neg_ratio=4, test_ratio=0.2):
    """
    改进版数据划分:
    - 正样本: rating >= 3（喜欢 + 中性）
    - 负样本: rating <= 2（用户明确不喜欢的，用于训练）
    - 训练集不使用未交互电影，只用真实评分数据
    - 测试集: 每个正样本配 99 个随机未交互电影（模拟真实推荐场景）
    """
    # 提取正样本 (rating >= 3) 和 强负样本 (rating <= 2)
    all_pos = []
    all_neg = []
    for u, items in user_item_map.items():
        for i, r in items.items():
            if r >= 3:
                all_pos.append((u, i))
            else:
                all_neg.append((u, i))

    # 按用户划分正样本训练/测试集
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

    # 按用户划分负样本训练/测试集（用于分类评估）
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

    # 训练集: 正样本(rating>=3) + 强负样本(rating<=2)，不使用未交互电影
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

    pos_count = len(train_pos)
    neg_count = len(train_neg)
    print(f"训练集: 正样本 {pos_count:,} (rating>=3), 负样本 {neg_count:,} (rating<=2), 正负比 1:{neg_count/pos_count:.1f}")
    print(f"测试集: {len(test_data):,} (排序评估用, 正负比约 1:99)")
    return train_data, test_data


class RatingDataset(Dataset):
    def __init__(self, data):
        self.users = torch.LongTensor([d[0] for d in data])
        self.items = torch.LongTensor([d[1] for d in data])
        self.labels = torch.FloatTensor([d[2] for d in data])

    def __len__(self):
        return len(self.users)

    def __getitem__(self, idx):
        return self.users[idx], self.items[idx], self.labels[idx]


# ==================== 2. NCF 模型定义 ====================

class NCF(nn.Module):
    """
    Neural Collaborative Filtering (GMF + MLP)
    """
    def __init__(self, num_users, num_items, embedding_dim=32,
                 mlp_layers=[64, 32, 16], dropout=0.2):
        super().__init__()

        # GMF 分支的 Embedding
        self.gmf_user_emb = nn.Embedding(num_users, embedding_dim)
        self.gmf_item_emb = nn.Embedding(num_items, embedding_dim)

        # MLP 分支的 Embedding
        self.mlp_user_emb = nn.Embedding(num_users, embedding_dim)
        self.mlp_item_emb = nn.Embedding(num_items, embedding_dim)

        # MLP 层
        mlp_modules = []
        input_dim = embedding_dim * 2  # user + item 拼接
        for out_dim in mlp_layers:
            mlp_modules.append(nn.Linear(input_dim, out_dim))
            mlp_modules.append(nn.ReLU())
            mlp_modules.append(nn.Dropout(dropout))
            input_dim = out_dim
        self.mlp = nn.Sequential(*mlp_modules)

        # 融合层: GMF 输出 (embedding_dim) + MLP 输出 (mlp_layers[-1])
        fusion_dim = embedding_dim + mlp_layers[-1]
        self.predict = nn.Sequential(
            nn.Linear(fusion_dim, 1),
            nn.Sigmoid()
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, std=0.01)

    def forward(self, user, item):
        # GMF 分支: 逐元素乘积
        gmf_u = self.gmf_user_emb(user)
        gmf_i = self.gmf_item_emb(item)
        gmf_out = gmf_u * gmf_i

        # MLP 分支: 拼接后过 MLP
        mlp_u = self.mlp_user_emb(user)
        mlp_i = self.mlp_item_emb(item)
        mlp_out = self.mlp(torch.cat([mlp_u, mlp_i], dim=-1))

        # 融合
        fusion = torch.cat([gmf_out, mlp_out], dim=-1)
        return self.predict(fusion).squeeze()


# ==================== 3. 评估指标 ====================

def evaluate(model, test_data, num_items, device, K=10):
    """
    计算 HR@K 和 NDCG@K
    对每个用户，将其所有测试正样本和 99 个负样本一起打分排序
    """
    model.eval()
    # 按用户组织测试数据
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

            # 按预测分数降序排列
            ranked = sorted(zip(item_ids, preds, labels), key=lambda x: x[1], reverse=True)
            top_k = [x[2] for x in ranked[:K]]

            # HR@K: 前 K 个中是否有正样本
            hr = 1.0 if 1.0 in top_k else 0.0
            hr_sum += hr

            # NDCG@K
            dcg = sum((2**rel - 1) / np.log2(i + 2) for i, rel in enumerate(top_k))
            ideal_labels = sorted(labels, reverse=True)[:K]
            idcg = sum((2**rel - 1) / np.log2(i + 2) for i, rel in enumerate(ideal_labels))
            ndcg = dcg / idcg if idcg > 0 else 0.0
            ndcg_sum += ndcg

            user_count += 1

    return hr_sum / user_count, ndcg_sum / user_count


# ==================== 4. 训练 ====================

def train():
    # 超参数
    BATCH_SIZE = 256
    EMBEDDING_DIM = 32
    MLP_LAYERS = [64, 32, 16]
    DROPOUT = 0.2
    LR = 1e-3
    EPOCHS = 20
    DEVICE = torch.device("mps" if torch.mps.is_available() else "cpu")

    print(f"使用设备: {DEVICE}")

    # 加载数据
    user_item_map, num_users, num_items = load_data()
    train_data, test_data = implicit_split(user_item_map, num_users, num_items)

    train_dataset = RatingDataset(train_data)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

    # 模型
    model = NCF(num_users, num_items, EMBEDDING_DIM, MLP_LAYERS, DROPOUT).to(DEVICE)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)

    print(f"\n模型参数量: {sum(p.numel() for p in model.parameters()):,}")
    print("=" * 60)

    best_hr = 0.0
    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_loss = 0

        for users, items, labels in train_loader:
            users = users.to(DEVICE)
            items = items.to(DEVICE)
            labels = labels.to(DEVICE)

            optimizer.zero_grad()
            preds = model(users, items)
            loss = criterion(preds, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        # 评估
        hr10, ndcg10 = evaluate(model, test_data, num_items, DEVICE, K=10)
        avg_loss = total_loss / len(train_loader)

        print(f"Epoch {epoch:2d} | Loss: {avg_loss:.4f} | HR@10: {hr10:.4f} | NDCG@10: {ndcg10:.4f}")

        if hr10 > best_hr:
            best_hr = hr10
            torch.save(model.state_dict(), "ncf_model.pth")
            print(f"  >>> 模型已保存 (HR@10: {best_hr:.4f})")

    print("=" * 60)
    print(f"训练完成! 最佳 HR@10: {best_hr:.4f}")

    # 加载最佳模型做最终评估
    model.load_state_dict(torch.load("ncf_model.pth"))
    hr10, ndcg10 = evaluate(model, test_data, num_items, DEVICE, K=10)
    hr5, ndcg5 = evaluate(model, test_data, num_items, DEVICE, K=5)
    print(f"\n最终结果:")
    print(f"  HR@5:  {hr5:.4f}  | NDCG@5:  {ndcg5:.4f}")
    print(f"  HR@10: {hr10:.4f} | NDCG@10: {ndcg10:.4f}")


if __name__ == "__main__":
    # 固定随机种子
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    train()
