"""数据分析脚本"""
from collections import Counter

# ===== ratings.dat =====
ratings = []
with open("ratings.dat", "r") as f:
    for line in f:
        uid, mid, rating, ts = line.strip().split("::")
        ratings.append((int(uid), int(mid), int(rating), int(ts)))

print("=== ratings.dat ===")
print(f"总行数: {len(ratings)}")
uids = set(r[0] for r in ratings)
mids = set(r[1] for r in ratings)
print(f"唯一用户数: {len(uids)}, 范围: {min(uids)}~{max(uids)}")
print(f"唯一电影数: {len(mids)}, 范围: {min(mids)}~{max(mids)}")
print(f"rating 范围: {min(r[2] for r in ratings)}~{max(r[2] for r in ratings)}")
print(f"timestamp 范围: {min(r[3] for r in ratings)}~{max(r[3] for r in ratings)}")

user_counts = Counter(r[0] for r in ratings)
print(f"用户最少评分数: {min(user_counts.values())}")
print(f"用户最多评分数: {max(user_counts.values())}")
print(f"用户平均评分数: {sum(user_counts.values())/len(user_counts):.1f}")

item_counts = Counter(r[1] for r in ratings)
print(f"电影最少评分数: {min(item_counts.values())}")
print(f"电影最多评分数: {max(item_counts.values())}")
print(f"电影平均评分数: {sum(item_counts.values())/len(item_counts):.1f}")

# 评分分布
print("\n评分分布:")
for r in range(1, 6):
    cnt = sum(1 for x in ratings if x[2] == r)
    print(f"  rating={r}: {cnt:>8,}  ({cnt/len(ratings)*100:.2f}%)")

# 检查是否有缺失行（字段数不对）
print("\n检查数据完整性...")
bad_lines = 0
with open("ratings.dat", "r") as f:
    for line in f:
        parts = line.strip().split("::")
        if len(parts) != 4:
            bad_lines += 1
print(f"ratings.dat 字段数异常行: {bad_lines}")

# ===== users.dat =====
print("\n=== users.dat ===")
users = []
with open("users.dat", "r") as f:
    for line in f:
        parts = line.strip().split("::")
        if len(parts) == 5:
            users.append((int(parts[0]), parts[1], int(parts[2]), int(parts[3]), parts[4]))
        else:
            bad_lines += 1

print(f"总行数: {len(users)}")
print(f"字段数异常行: {bad_lines}")
genders = Counter(u[1] for u in users)
ages = Counter(u[2] for u in users)
occs = Counter(u[3] for u in users)
print(f"性别分布: {dict(genders)}")
print(f"年龄分布: {dict(sorted(ages.items()))}")
print(f"职业分布: {dict(sorted(occs.items()))}")

# ===== movies.dat =====
print("\n=== movies.dat ===")
movies = []
bad_movie_lines = 0
with open("movies.dat", "r", encoding="latin-1") as f:
    for line in f:
        parts = line.strip().split("::")
        if len(parts) == 3:
            movies.append((int(parts[0]), parts[1], parts[2]))
        else:
            bad_movie_lines += 1

print(f"总行数: {len(movies)}")
print(f"字段数异常行: {bad_movie_lines}")

# 类型分布
all_genres = Counter()
for _, _, genres in movies:
    for g in genres.split("|"):
        all_genres[g] += 1
print(f"电影类型分布: {dict(all_genres.most_common())}")

# 检查 movies.dat 中是否有不在 ratings.dat 中的电影
rated_mids = set(r[1] for r in ratings)
movie_mids = set(m[0] for m in movies)
print(f"\nratings 中的电影数: {len(rated_mids)}")
print(f"movies 中的电影数: {len(movie_mids)}")
print(f"ratings 中有但 movies 中没有的电影: {len(rated_mids - movie_mids)}")
print(f"movies 中有但 ratings 中没有的电影: {len(movie_mids - rated_mids)}")

# 检查 users.dat 中是否有不在 ratings.dat 中的用户
rated_uids = set(r[0] for r in ratings)
user_uids = set(u[0] for u in users)
print(f"\nratings 中的用户数: {len(rated_uids)}")
print(f"users 中的用户数: {len(user_uids)}")
print(f"ratings 中有但 users 中没有的用户: {len(rated_uids - user_uids)}")
print(f"users 中有但 ratings 中没有的用户: {len(user_uids - rated_uids)}")
