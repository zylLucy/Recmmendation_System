"""
基于 V3 模型生成用户推荐表

输出格式: user_id, recommendations1, recommendations2, ..., recommendations10
每行是一个用户，后面跟着模型推荐的 TOP10 电影 ID（原始 MovieID）
"""

import torch
import torch.nn as nn
import numpy as np
import csv
from collections import defaultdict


# ==================== 模型定义（与 ncf_train_v3.py 一致） ====================

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


# ==================== 主流程 ====================

def main():
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {DEVICE}")

    # 1. 加载原始数据，建立 ID 映射
    print("加载数据...")
    ratings = []
    with open("ratings.dat", "r") as f:
        for line in f:
            uid, mid, rating, _ = line.strip().split("::")
            ratings.append((int(uid), int(mid), float(rating)))

    users = sorted(set(r[0] for r in ratings))
    items = sorted(set(r[1] for r in ratings))
    user2idx = {u: i for i, u in enumerate(users)}
    item2idx = {m: i for i, m in enumerate(items)}
    idx2user = {i: u for u, i in user2idx.items()}
    idx2item = {i: m for m, i in item2idx.items()}

    num_users = len(users)
    num_items = len(items)
    print(f"用户数: {num_users}, 电影数: {num_items}")

    # 构建每个用户已交互的电影集合
    user_interacted = defaultdict(set)
    for uid, mid, _ in ratings:
        user_interacted[user2idx[uid]].add(item2idx[mid])

    # 2. 加载模型
    print("加载 V3 模型...")
    model = NCF(num_users, num_items).to(DEVICE)
    model.load_state_dict(torch.load("ncf_model_v3.pth", map_location=DEVICE))
    model.eval()

    # 3. 为每个用户生成推荐
    print("生成推荐...")
    all_item_indices = list(range(num_items))
    BATCH_SIZE = 512
    TOP_K = 10

    with open("recommender.csv", "w", newline="") as f:
        writer = csv.writer(f)
        # 写表头
        header = ["user_id"] + [f"recommendations{i}" for i in range(1, TOP_K + 1)]
        writer.writerow(header)

        for user_idx in range(num_users):
            original_uid = idx2user[user_idx]
            interacted = user_interacted[user_idx]

            # 只考虑未交互的电影
            candidate_items = [i for i in all_item_indices if i not in interacted]

            if len(candidate_items) == 0:
                # 没有可推荐的电影，填 -1
                row = [original_uid] + [-1] * TOP_K
                writer.writerow(row)
                continue

            # 分批打分
            all_scores = []
            with torch.no_grad():
                for start in range(0, len(candidate_items), BATCH_SIZE):
                    batch_items = candidate_items[start:start + BATCH_SIZE]
                    user_tensor = torch.LongTensor([user_idx] * len(batch_items)).to(DEVICE)
                    item_tensor = torch.LongTensor(batch_items).to(DEVICE)
                    scores = model(user_tensor, item_tensor).cpu().numpy()
                    # 确保 scores 是一维数组
                    scores = np.atleast_1d(scores)
                    all_scores.extend(zip(batch_items, scores))

            # 按分数降序排列，取 TOP_K
            all_scores.sort(key=lambda x: x[1], reverse=True)
            top_items = [idx2item[item_idx] for item_idx, _ in all_scores[:TOP_K]]

            # 如果不够 TOP_K，用 -1 填充
            while len(top_items) < TOP_K:
                top_items.append(-1)

            row = [original_uid] + top_items
            writer.writerow(row)

            if (user_idx + 1) % 500 == 0:
                print(f"  已处理 {user_idx + 1}/{num_users} 用户...")

    print(f"\n推荐表已生成: recommender.csv")
    print(f"共 {num_users} 个用户，每个用户推荐 {TOP_K} 部电影")


if __name__ == "__main__":
    main()
