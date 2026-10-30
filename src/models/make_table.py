import glob, json
import pandas as pd

rows = []
for f in sorted(glob.glob("results/*_seed*.json")):
    r = json.load(open(f))
    for t in ["test_unseen_reviews", "test_unseen_items", "test_unseen_reviewers"]:
        rows.append(dict(model=r["mode"], seed=r["seed"], test=t.replace("test_", ""),
                         acc=r[t]["acc"], macro_f1=r[t]["macro_f1"]))
d = pd.DataFrame(rows)
t = d.groupby(["model", "test"])[["acc", "macro_f1"]].mean().round(4).unstack("test")
print(t.to_string())
t.to_csv("results/summary.csv")