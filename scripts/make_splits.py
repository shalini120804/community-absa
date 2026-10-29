"""Build leakage-safe splits: train / val / unseen reviews / products / reviewers."""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

p = argparse.ArgumentParser()
p.add_argument("--max_users", type=int, default=30000)
p.add_argument("--min_count", type=int, default=5)
p.add_argument("--seed", type=int, default=42)
args = p.parse_args()

rng = np.random.default_rng(args.seed)
OUT = Path("data/processed/splits")
OUT.mkdir(parents=True, exist_ok=True)

df = pd.read_parquet("data/processed/reviews_5core.parquet")
print("Full 5-core file:", len(df), "reviews")

# 1. Sample users, then re-apply the k-core filter
users = np.sort(df["user_id"].unique())
if len(users) > args.max_users:
    keep = rng.choice(users, args.max_users, replace=False)
    df = df[df["user_id"].isin(keep)]
while True:
    before = len(df)
    df = df[df.groupby("user_id")["item_id"].transform("count") >= args.min_count]
    df = df[df.groupby("item_id")["user_id"].transform("count") >= args.min_count]
    if len(df) == before:
        break
df = df.reset_index(drop=True)
df["review_id"] = df.index
print("After sampling:", len(df), "reviews,",
      df["user_id"].nunique(), "users,", df["item_id"].nunique(), "items")

# 2. Choose held-out users and items (10% each)
users = np.sort(df["user_id"].unique())
items = np.sort(df["item_id"].unique())
test_users = set(rng.choice(users, int(0.1 * len(users)), replace=False))
test_items = set(rng.choice(items, int(0.1 * len(items)), replace=False))
is_tu = df["user_id"].isin(test_users)
is_ti = df["item_id"].isin(test_items)

rev_all = df[is_tu & ~is_ti].copy()      # held-out users, known items
unseen_items = df[~is_tu & is_ti].copy()  # known users, held-out items
known = df[~is_tu & ~is_ti].copy()        # known users, known items
print("Dropped (held-out user AND held-out item):", int((is_tu & is_ti).sum()))

# 3. Temporal split for known users
known = known.sort_values(["user_id", "timestamp", "review_id"])
g = known.groupby("user_id")
rank = g.cumcount(ascending=False)         # 0 = latest review
n = g["item_id"].transform("count")
eligible = n >= 3
test_unseen_reviews = known[(rank == 0) & eligible]
val = known[(rank == 1) & eligible]
train = known[~((rank <= 1) & eligible)]

# 4. Unseen reviewers: latest review = test, earlier reviews = history
rev_all = rev_all.sort_values(["user_id", "timestamp", "review_id"])
g2 = rev_all.groupby("user_id")
rank2 = g2.cumcount(ascending=False)
n2 = g2["item_id"].transform("count")
test_unseen_reviewers = rev_all[(rank2 == 0) & (n2 >= 2)]
reviewer_history = rev_all[rank2 >= 1]

# 5. Leakage checks
sets = {
    "train": train, "val": val,
    "test_unseen_reviews": test_unseen_reviews,
    "test_unseen_items": unseen_items,
    "test_unseen_reviewers": test_unseen_reviewers,
    "reviewer_history": reviewer_history,
}
assert not set(train["user_id"]) & test_users, "test user in train"
assert not set(train["item_id"]) & test_items, "test item in train"
assert not set(train["review_id"]) & set(test_unseen_reviews["review_id"])
assert not set(train["review_id"]) & set(val["review_id"])
assert not set(test_unseen_reviewers["review_id"]) & set(reviewer_history["review_id"])
print("Leakage checks passed.")

# 6. Save and report
for name, d in sets.items():
    d.to_parquet(OUT / f"{name}.parquet", index=False)
    lo = pd.to_datetime(d["timestamp"].min(), unit="ms").date()
    hi = pd.to_datetime(d["timestamp"].max(), unit="ms").date()
    print(f"{name:24s} reviews={len(d):7d} users={d['user_id'].nunique():6d} "
          f"items={d['item_id'].nunique():6d}  dates {lo} to {hi}")
print("Saved to", OUT)