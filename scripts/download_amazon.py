"""Download Amazon Reviews 2023 (one category) and build a 5-core subset."""
import argparse
import json
from pathlib import Path

import pandas as pd
from huggingface_hub import hf_hub_download

parser = argparse.ArgumentParser()
parser.add_argument("--category", default="Cell_Phones_and_Accessories")
parser.add_argument("--max_lines", type=int, default=4_000_000)
parser.add_argument("--min_count", type=int, default=5)
args = parser.parse_args()

RAW = Path("data/raw")
OUT = Path("data/processed")
RAW.mkdir(parents=True, exist_ok=True)
OUT.mkdir(parents=True, exist_ok=True)

# 1. Download (resumes automatically if interrupted)
print(f"Downloading {args.category} ... this can take a while")
path = hf_hub_download(
    repo_id="McAuley-Lab/Amazon-Reviews-2023",
    filename=f"raw/review_categories/{args.category}.jsonl",
    repo_type="dataset",
    local_dir=str(RAW),
)
print("Saved to:", path)

# 2. Read the first max_lines reviews, keeping only the columns we need
rows = []
with open(path, "r", encoding="utf-8") as f:
    for i, line in enumerate(f):
        if i >= args.max_lines:
            break
        d = json.loads(line)
        rows.append((
            d.get("user_id"), d.get("parent_asin"), d.get("rating"),
            d.get("timestamp"), d.get("title", ""), d.get("text", ""),
        ))
df = pd.DataFrame(rows, columns=["user_id", "item_id", "rating",
                                 "timestamp", "title", "text"])
print("Read reviews:", len(df))

# 3. Keep only reviews with a real text
df = df[df["text"].str.len() >= 20]

# 4. Iterative k-core filter (users and items both need >= min_count reviews)
while True:
    before = len(df)
    df = df[df.groupby("user_id")["item_id"].transform("count") >= args.min_count]
    df = df[df.groupby("item_id")["user_id"].transform("count") >= args.min_count]
    if len(df) == before:
        break

# 5. Save
out_file = OUT / "reviews_5core.parquet"
df.to_parquet(out_file, index=False)
print("\n=== Done ===")
print("Reviews:", len(df))
print("Users:  ", df["user_id"].nunique())
print("Items:  ", df["item_id"].nunique())
print("Saved to:", out_file)