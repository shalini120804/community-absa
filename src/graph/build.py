import os
import numpy as np, pandas as pd, torch
from torch_geometric.data import HeteroData

SPLITS = "data/processed/splits"

def user_profiles(reviews, aspects, aspect_list):
    """Per-user aspect profile: mean sentiment and mention count per aspect.
    Works on any set of reviews (train, or reviewer_history for new reviewers)."""
    g = aspects[aspects.review_id.isin(reviews.review_id)]
    idx = {a: i for i, a in enumerate(aspect_list)}
    users = sorted(reviews.user_id.unique())
    uidx = {u: i for i, u in enumerate(users)}
    mean = np.zeros((len(users), len(aspect_list)), dtype=np.float32)
    cnt = np.zeros_like(mean)
    for (u, a), s in g.groupby(["user_id", "aspect"]).sentiment:
        mean[uidx[u], idx[a]] = s.mean()
        cnt[uidx[u], idx[a]] = len(s)
    return users, mean, cnt

def build_train_graph(out="data/processed/graph_train.pt"):
    tr = pd.read_parquet(f"{SPLITS}/train.parquet")
    asp = pd.read_parquet("data/processed/aspects.parquet")
    g = asp[asp.review_id.isin(tr.review_id)]          # train reviews only: no leakage
    aspect_list = sorted(asp.aspect.unique())

    users = {u: i for i, u in enumerate(sorted(tr.user_id.unique()))}
    items = {v: i for i, v in enumerate(sorted(tr.item_id.unique()))}
    revs = {r: i for i, r in enumerate(tr.review_id.values)}
    aidx = {a: i for i, a in enumerate(aspect_list)}

    d = HeteroData()
    d["user"].num_nodes, d["item"].num_nodes = len(users), len(items)
    d["review"].num_nodes, d["aspect"].num_nodes = len(revs), len(aspect_list)

    r = tr.review_id.map(revs).values
    d["user", "wrote", "review"].edge_index = torch.tensor(
        np.stack([tr.user_id.map(users).values, r]), dtype=torch.long)
    d["review", "about", "item"].edge_index = torch.tensor(
        np.stack([r, tr.item_id.map(items).values]), dtype=torch.long)
    d["review", "mentions", "aspect"].edge_index = torch.tensor(
        np.stack([g.review_id.map(revs).values, g.aspect.map(aidx).values]), dtype=torch.long)
    d["review", "mentions", "aspect"].edge_attr = torch.tensor(
        g[["sentiment", "confidence"]].values, dtype=torch.float)

    ulist, mean, cnt = user_profiles(tr, asp, aspect_list)
    assert ulist == sorted(users, key=users.get)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    torch.save(dict(data=d, users=users, items=items, aspects=aspect_list,
                    profile_mean=mean, profile_cnt=cnt), out)

    print(d)
    print("users with >=1 aspect mention:", int((cnt.sum(1) > 0).sum()), "/", len(users))
    print("users with >=3 train reviews:", int((tr.groupby("user_id").size() >= 3).sum()))
    print("saved", out)

if __name__ == "__main__":
    build_train_graph()