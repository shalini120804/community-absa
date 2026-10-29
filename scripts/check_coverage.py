import glob, os
import pandas as pd

g = pd.read_parquet("data/processed/aspects.parquet")
for f in sorted(glob.glob("data/processed/splits/*.parquet")):
    d = pd.read_parquet(f)
    sub = g[g.review_id.isin(d.review_id)]
    n = sub.review_id.nunique()
    print(f"{os.path.basename(f):32s} reviews={len(d):6d}  with_aspect={n:6d} ({n/len(d):.0%})  rows={len(sub)}")