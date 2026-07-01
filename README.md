# MovieLens 1M 推荐系统

基于 MovieLens 1M 数据集的推荐系统，实现了多种经典与深度推荐算法，包含模型训练、评估、推荐生成与指标可视化。

## 数据集

使用 [MovieLens 1M](https://grouplens.org/datasets/movielens/1m/) 数据集：

| 文件 | 说明 | 规模 |
|------|------|------|
| `ratings.dat` | 用户-电影评分 | 1,000,209 条 |
| `users.dat` | 用户画像 | 6,040 人 |
| `movies.dat` | 电影元数据 | 3,883 部 |

## 项目结构

```
├── ratings.dat / users.dat / movies.dat   # MovieLens 1M 数据集
├── ncf_train.py / ncf_train_v3.py         # NCF 模型训练脚本
├── ncf_model.pth / ncf_model_v3.pth       # 训练好的 NCF 模型权重
├── generate_recommendations.py             # 生成用户推荐列表
├── evaluate.py                            # 离线评估（Recall/MRR/NDCG/Hit）
├── eval_recommenders.py                    # 多模型推荐效果对比评估
├── _analyze.py                            # 数据分析脚本
├── ANALYSIS.md                            # 数据分析报告
├── recommender.csv                        # NCF 模型推荐结果 (Top10)
├── xsimgcl_recommendation.csv             # XSimGCL 推荐结果
├── RecSys/                                # 推荐系统主模块
│   ├── main.py                            # 统一运行入口
│   ├── RecSys/model/                      # 多种推荐模型实现
│   │   ├── ncf/                           # Neural Collaborative Filtering
│   │   ├── user_cf/                       # 基于用户的协同过滤
│   │   ├── item_cf/                       # 基于物品的协同过滤
│   │   ├── dcn/                           # Deep & Cross Network
│   │   ├── fm/                            # Factorization Machines
│   │   ├── pnn/                           # Product-based Neural Network
│   │   └── xsimgcl/                       # XSimGCL (图对比学习)
│   ├── RecSys/data/                       # 数据处理
│   ├── RecSys/outputs/                    # 各模型推荐输出
│   ├── metric_server/                     # FastAPI 测评服务
│   └── manual/                            # 学习笔记与参考资料
└── .gitignore
```

## 模型列表

| 模型 | 类型 | 实现路径 |
|------|------|----------|
| **NCF / NeuMF** | 深度协同过滤 (GMF + MLP) | `ncf_train.py` / `RecSys/RecSys/model/ncf/` |
| **User-CF** | 基于用户的协同过滤 | `RecSys/RecSys/model/user_cf/` |
| **Item-CF** | 基于物品的协同过滤 | `RecSys/RecSys/model/item_cf/` |
| **DCN** | Deep & Cross Network | `RecSys/RecSys/model/dcn/` |
| **FM** | Factorization Machines | `RecSys/RecSys/model/fm/` |
| **PNN** | Product-based Neural Network | `RecSys/RecSys/model/pnn/` |
| **XSimGCL** | 基于图对比学习的推荐 | `RecSys/RecSys/model/xsimgcl/` |

## 快速开始

### 环境要求

- Python 3.10
- PyTorch >= 2.0.0
- NumPy >= 1.24.0
- scikit-learn >= 1.3.0

### 安装

```bash
pip install -r requirements.txt
```

### 训练 NCF 模型

```bash
python ncf_train.py
```

### 生成推荐结果

```bash
python generate_recommendations.py
```

### 评估推荐效果

```bash
python evaluate.py
```

### 运行其他推荐模型

```bash
# 以 DCN 为例
python RecSys/main.py --model dcn --config RecSys/RecSys/config/dcn.yaml
```

### 启动测评服务

```bash
pip install -r RecSys/metric_server/requirements.txt
uvicorn RecSys.metric_server.main:app --reload --host 0.0.0.0 --port 8000
```

打开 http://127.0.0.1:8000 上传推荐结果 CSV 即可查看各项指标。

## 评估指标

支持 Recall@K、MRR@K、NDCG@K、Hit@K、Precision@K，K 默认为 10。测评机按 RecBole-GNN 标准设置：ratio-based 8:1:1 分割，full sort 评估，valid MRR@10 选择最佳 epoch。

## 参考链接

- [MovieLens 1M 数据集](https://grouplens.org/datasets/movielens/1m/)
- [NCF 论文 - He et al. WWW 2017](https://arxiv.org/abs/1708.05031)
- [DCN 论文 - Wang et al. ADKDD 2017](https://arxiv.org/abs/1708.05123)
- [XSimGCL 论文 - Yu et al. SIGIR 2022](https://arxiv.org/abs/2204.10811)
- [学习笔记](https://kaiyuanyokii2n.com/)
