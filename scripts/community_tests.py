import pandas as pd
from scipy.stats import kruskal

uc = pd.read_parquet("data/processed/user_community.parquet")
g = pd.read_parquet("data/processed/aspects.parquet")
tr = pd.read_parquet("data/processed/splits/train.parquet")[["review_id"]]
g = g.merge(tr, on="review_id").merge(uc[["user_id", "community"]], on="user_id")
g = g[g.community >= 0]

rows = []
for a, d in g.groupby("aspect"):
    h, p = kruskal(*[x.sentiment.values for _, x in d.groupby("community")])
    m = d.groupby("community").sentiment.agg(["mean", "size"])
    rows.append(dict(aspect=a, n=len(d), smallest_cell=int(m["size"].min()),
                     spread=round(m["mean"].max() - m["mean"].min(), 2), p=f"{p:.1e}"))
print(pd.DataFrame(rows).sort_values("spread", ascending=False).to_string(index=False))