"""Graph-based baseline: a 2-layer GCN over the reviewer-item graph (no text, no communities).
Node features are aspect-mention shares only (never sentiment), so no labels leak.
Unseen reviewers join the graph through their reviewer_history reviews only."""
import json, os, sys
import numpy as np, pandas as pd, torch, torch.nn as nn
from torch_geometric.nn import GCNConv
from sklearn.metrics import accuracy_score, f1_score

SPL = "data/processed/splits"
TESTS = ["test_unseen_reviews", "test_unseen_items", "test_unseen_reviewers"]
dev = "cuda" if torch.cuda.is_available() else "cpu"

ck = torch.load("data/processed/graph_train.pt", weights_only=False)
aspects = ck["aspects"]; A = len(aspects)
aidx = {a: i for i, a in enumerate(aspects)}
items = ck["items"]
asp = pd.read_parquet("data/processed/aspects.parquet")
rd = lambda n: pd.read_parquet(f"{SPL}/{n}.parquet")[["review_id", "user_id", "item_id"]]
tr, hist = rd("train"), rd("reviewer_history")

train_users = sorted(tr.user_id.unique())
hist_users = sorted(set(hist.user_id) - set(train_users))
umap = {u: i for i, u in enumerate(train_users + hist_users)}
UNK_U = len(umap); n_users = UNK_U + 1
UNK_I = len(items); n_items = UNK_I + 1
N = n_users + n_items
uid = lambda s: s.map(umap).fillna(UNK_U).astype(int).values
iid = lambda s: s.map(items).fillna(UNK_I).astype(int).values

# edges: train reviews + history reviews (only for items that exist in the train graph)
h = hist[hist.item_id.isin(items)]
eu = np.concatenate([uid(tr.user_id), uid(h.user_id)])
ei = n_users + np.concatenate([iid(tr.item_id), iid(h.item_id)])
ei_t = torch.tensor(np.stack([np.concatenate([eu, ei]), np.concatenate([ei, eu])]), dtype=torch.long)

# node features: aspect mention shares (+ log count)
def feats(df, idx, n):
    M = np.zeros((n, A), dtype=np.float32)
    np.add.at(M, (idx, df.aspect.map(aidx).values.astype(int)), 1)
    tot = M.sum(1, keepdims=True)
    return np.hstack([M / np.maximum(tot, 1), np.log1p(tot)])
a_tr = asp[asp.review_id.isin(tr.review_id)]
a_hi = asp[asp.review_id.isin(hist.review_id)]
a_u = pd.concat([a_tr, a_hi])
Xu = feats(a_u, uid(a_u.user_id), n_users)
Xi = feats(a_tr, iid(a_tr.item_id), n_items)
X = torch.tensor(np.vstack([Xu, Xi]), dtype=torch.float)

def rows(name):
    d = rd(name).merge(asp[["review_id", "aspect", "sentiment"]], on="review_id")
    return (torch.tensor(uid(d.user_id)), torch.tensor(n_users + iid(d.item_id)),
            torch.tensor(d.aspect.map(aidx).values.astype(int)),
            torch.tensor(d.sentiment.astype(int).values + 1))

class Net(nn.Module):
    def __init__(s, F, H=64):
        super().__init__()
        s.c1, s.c2 = GCNConv(F, H), GCNConv(H, H)
        s.asp = nn.Embedding(A, 16)
        s.head = nn.Sequential(nn.Linear(2 * H + 16, 64), nn.ReLU(), nn.Dropout(0.3), nn.Linear(64, 3))
        s.drop = nn.Dropout(0.3)
    def forward(s, x, ei, u, i, a):
        h = s.drop(torch.relu(s.c1(x, ei)))
        h = s.c2(h, ei)
        return s.head(torch.cat([h[u], h[i], s.asp(a)], 1))

def score(model, x, ei, r):
    model.eval()
    with torch.no_grad():
        p = model(x, ei, *[t.to(dev) for t in r[:3]]).argmax(1).cpu().numpy()
    y = r[3].numpy()
    return dict(acc=float(accuracy_score(y, p)), macro_f1=float(f1_score(y, p, average="macro")), n=len(y))

data = {n: rows(n) for n in ["train", "val"] + TESTS}
x, ei = X.to(dev), ei_t.to(dev)
tu, ti, ta, ty = [t.to(dev) for t in data["train"]]

for seed in [0, 1, 2]:
    torch.manual_seed(seed); np.random.seed(seed)
    model = Net(X.shape[1]).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=3e-3, weight_decay=1e-4)
    best, result = -1, {}
    for ep in range(300):
        model.train()
        loss = nn.functional.cross_entropy(model(x, ei, tu, ti, ta), ty)
        opt.zero_grad(); loss.backward(); opt.step()
        if ep % 10 == 9:
            v = score(model, x, ei, data["val"])
            if v["macro_f1"] > best:
                best = v["macro_f1"]
                result = dict(mode="gcn", seed=seed, epoch=ep, val=v)
                for n in TESTS:
                    result[n] = score(model, x, ei, data[n])
    print(f"seed {seed}: val f1 {best:.4f}", {n: round(result[n]["macro_f1"], 4) for n in TESTS})
    os.makedirs("results", exist_ok=True)
    json.dump(result, open(f"results/gcn_seed{seed}.json", "w"), indent=2)
    