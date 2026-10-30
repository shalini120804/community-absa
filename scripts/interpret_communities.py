import os
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import kruskal

MIN_CELL = 30
os.makedirs("results", exist_ok=True)

g = pd.read_parquet("data/processed/aspects.parquet")
uc = pd.read_parquet("data/processed/user_community.parquet")
tr = pd.read_parquet("data/processed/splits/train.parquet")[["review_id"]]
g = g.merge(tr, on="review_id").merge(uc[["user_id", "community"]], on="user_id")
g = g[g.community >= 0].copy()
g["aspect"] = g.aspect.replace({"durability": "quality"})

piv = g.pivot_table(index="community", columns="aspect", values="sentiment", aggfunc="mean")
cnt = g.pivot_table(index="community", columns="aspect", values="sentiment", aggfunc="size")
piv = piv.where(cnt >= MIN_CELL)
fig, ax = plt.subplots(figsize=(11, 5))
im = ax.imshow(piv.values, cmap="RdYlGn", vmin=-0.2, vmax=0.8, aspect="auto")
ax.set_xticks(range(piv.shape[1])); ax.set_xticklabels(piv.columns, rotation=45, ha="right")
ax.set_yticks(range(piv.shape[0])); ax.set_yticklabels([f"community {c}" for c in piv.index])
for i in range(piv.shape[0]):
    for j in range(piv.shape[1]):
        v = piv.values[i, j]
        if not np.isnan(v):
            ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=8)
fig.colorbar(im, label="mean sentiment (-1 to +1)")
ax.set_title("Mean aspect sentiment by community (train reviews)")
fig.tight_layout()
fig.savefig("results/heatmap.png", dpi=200)
piv.round(3).to_csv("results/heatmap.csv")
print("saved results/heatmap.png and results/heatmap.csv")

share = g.pivot_table(index="community", columns="aspect", values="sentiment", aggfunc="size").fillna(0)
share = share.div(share.sum(axis=1), axis=0)
overall = g.groupby("aspect").sentiment.mean()
print("\nCommunity descriptions:")
for c in piv.index:
    top = share.loc[c].sort_values(ascending=False).head(3).index.tolist()
    diff = (piv.loc[c] - overall).dropna().sort_values()
    users = int((uc[uc.source == "train"].community == c).sum())
    print(f"community {c} ({users} users): talks most about {', '.join(top)}; "
          f"most negative vs average: {diff.index[0]} ({diff.iloc[0]:+.2f}); "
          f"most positive vs average: {diff.index[-1]} ({diff.iloc[-1]:+.2f})")

g["resid"] = g.sentiment - g.groupby(["item_id", "aspect"]).sentiment.transform("mean")
n = g.groupby(["item_id", "aspect"]).sentiment.transform("size")
c2 = g[n >= 2]
rows = []
for a, d in c2.groupby("aspect"):
    groups = [x.resid.values for _, x in d.groupby("community") if len(x) >= MIN_CELL]
    if len(groups) < 2:
        continue
    h, p = kruskal(*groups)
    rows.append(dict(aspect=a, n=len(d), p_item_adjusted=f"{p:.1e}"))
ctrl = pd.DataFrame(rows)
print("\nItem-controlled test (differences left after removing product effects):")
print(ctrl.to_string(index=False))
ctrl.to_csv("results/item_control.csv", index=False)