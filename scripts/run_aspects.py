import argparse, glob, os
import numpy as np, pandas as pd
from src.aspects.extract import extract_mentions
from src.aspects.sentiment import Teacher

ID, USER, ITEM, TEXT, RATING = "review_id", "user_id", "item_id", "text", "rating"

ap = argparse.ArgumentParser()
ap.add_argument("--n", type=int, default=0, help="sample size; 0 = all")
ap.add_argument("--out", default="data/processed/aspects.parquet")
a = ap.parse_args()

files = glob.glob("data/processed/splits/*.parquet")
df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
if ID not in df.columns:
    df[ID] = df[USER].astype(str) + "_" + df[ITEM].astype(str) + "_" + df["timestamp"].astype(str)
df = df.drop_duplicates(ID)
if a.n:
    df = df.sample(a.n, random_state=0)
print("reviews:", len(df))

m = pd.DataFrame(extract_mentions(df[ID].tolist(), df[TEXT].tolist()))
print("mentions:", len(m), " reviews with >=1 aspect:", m.review_id.nunique())

t = Teacher()
probs = t.predict(m.sentence.tolist(), m.term.tolist())
for i, l in enumerate(t.labels):
    m[f"p_{l}"] = probs[:, i]

# aggregate to (review, aspect): mean probs over mentions
pc = [f"p_{l}" for l in t.labels]
g = m.groupby([ID, "aspect"])[pc].mean().reset_index()
g["sentiment"] = t.signs[g[pc].values.argmax(1)]
g["confidence"] = g[pc].max(axis=1)
meta_cols = [c for c in [ID, USER, ITEM, RATING] if c in df.columns]
g = g.merge(df[meta_cols], on=ID).drop(columns=pc)

os.makedirs(os.path.dirname(a.out), exist_ok=True)
g.to_parquet(a.out, index=False)
if a.n:
    m.sample(min(40, len(m)), random_state=1).to_csv("data/processed/aspect_mentions_sample.csv", index=False)
print(g.head(), "\n", g.aspect.value_counts(), "\n", g.sentiment.value_counts(normalize=True))
print("saved", a.out)