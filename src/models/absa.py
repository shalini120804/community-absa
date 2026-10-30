"""Stage 6: text-only / user-aware / community-aware aspect sentiment classifiers."""
import argparse, json, os, random, time
import numpy as np, pandas as pd, torch, torch.nn as nn
from torch.utils.data import DataLoader
from transformers import AutoModel, AutoTokenizer, get_linear_schedule_with_warmup
from sklearn.metrics import accuracy_score, f1_score

SPLITS = "data/processed/splits"
TESTS = ["test_unseen_reviews", "test_unseen_items", "test_unseen_reviewers"]


def load(name, asp, aspects, umap, cmap, K, limit):
    d = pd.read_parquet(f"{SPLITS}/{name}.parquet")[["review_id", "user_id", "text"]]
    df = asp[["review_id", "aspect", "sentiment"]].merge(d, on="review_id")
    df["label"] = df.sentiment.astype(int) + 1
    df["asp"] = df.aspect.map({a: i for i, a in enumerate(aspects)})
    df["user"] = df.user_id.map(umap).fillna(0).astype(int)      # 0 = unseen reviewer
    c = df.user_id.map(cmap).fillna(-1).astype(int)
    df["comm"] = np.where(c < 0, K, c)                            # K = unknown community
    if limit and len(df) > limit:
        df = df.sample(limit, random_state=0)
    return df.reset_index(drop=True)


def make_loader(df, tok, bs, shuffle, max_len):
    def collate(idx):
        b = df.iloc[idx]
        enc = tok(b.aspect.tolist(), b.text.fillna("").tolist(), truncation="only_second",
                  max_length=max_len, padding=True, return_tensors="pt")
        t = lambda col: torch.tensor(b[col].values)
        return enc, t("user"), t("comm"), t("asp"), t("label")
    return DataLoader(range(len(df)), batch_size=bs, shuffle=shuffle, collate_fn=collate)


class Net(nn.Module):
    def __init__(self, enc, mode, n_users, K, n_asp, dim=32):
        super().__init__()
        self.enc = AutoModel.from_pretrained(enc)
        H = self.enc.config.hidden_size
        self.mode = mode
        if mode == "user":
            self.ctx = nn.Embedding(n_users + 1, dim, padding_idx=0)
        elif mode == "community":
            self.ctx = nn.Embedding(K + 1, dim)
        if mode != "text":
            self.asp = nn.Embedding(n_asp, dim)
        self.drop = nn.Dropout(0.1)
        self.head = nn.Linear(H + (2 * dim if mode != "text" else 0), 3)

    def forward(self, enc, u, c, a):
        h = self.enc(**enc).last_hidden_state[:, 0]
        if self.mode != "text":
            e = self.ctx(u if self.mode == "user" else c)
            h = torch.cat([h, e, e * self.asp(a)], 1)
        return self.head(self.drop(h))


@torch.no_grad()
def evaluate(model, loader, dev):
    model.eval()
    P, Y = [], []
    for enc, u, c, a, y in loader:
        with torch.autocast("cuda", dtype=torch.float16, enabled=dev == "cuda"):
            lo = model(enc.to(dev), u.to(dev), c.to(dev), a.to(dev))
        P.append(lo.argmax(1).cpu().numpy())
        Y.append(y.numpy())
    P, Y = np.concatenate(P), np.concatenate(Y)
    return dict(acc=float(accuracy_score(Y, P)), macro_f1=float(f1_score(Y, P, average="macro")), n=len(Y))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["text", "user", "community"], required=True)
    ap.add_argument("--enc", default="distilroberta-base")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--max_len", type=int, default=160)
    ap.add_argument("--lr", type=float, default=3e-5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--limit", type=int, default=0, help="rows per split, for a quick smoke test")
    a = ap.parse_args()

    random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed)
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    ck = torch.load("data/processed/graph_train.pt", weights_only=False)
    aspects = ck["aspects"]
    umap = {u: i + 1 for u, i in ck["users"].items()}
    uc = pd.read_parquet("data/processed/user_community.parquet")
    cmap = dict(zip(uc.user_id, uc.community))
    K = int(uc.community.max()) + 1
    asp = pd.read_parquet("data/processed/aspects.parquet")

    tok = AutoTokenizer.from_pretrained(a.enc)
    sets = {n: load(n, asp, aspects, umap, cmap, K, a.limit) for n in ["train", "val"] + TESTS}
    print({n: len(d) for n, d in sets.items()}, "K =", K)
    tr_loader = make_loader(sets["train"], tok, a.bs, True, a.max_len)
    ev = {n: make_loader(sets[n], tok, 64, False, a.max_len) for n in ["val"] + TESTS}

    model = Net(a.enc, a.mode, len(umap), K, len(aspects)).to(dev)
    enc_p = list(model.enc.parameters())
    other = [p for n, p in model.named_parameters() if not n.startswith("enc.")]
    opt = torch.optim.AdamW([{"params": enc_p, "lr": a.lr}, {"params": other, "lr": 1e-3}], weight_decay=0.01)
    steps = a.epochs * len(tr_loader)
    sch = get_linear_schedule_with_warmup(opt, int(0.06 * steps), steps)
    scaler = torch.amp.GradScaler("cuda", enabled=dev == "cuda")
    lossf = nn.CrossEntropyLoss()

    best, result = -1, {}
    for ep in range(a.epochs):
        model.train()
        t0, tot = time.time(), 0.0
        for i, (enc, u, c, asp_i, y) in enumerate(tr_loader):
            with torch.autocast("cuda", dtype=torch.float16, enabled=dev == "cuda"):
                lo = model(enc.to(dev), u.to(dev), c.to(dev), asp_i.to(dev))
            loss = lossf(lo.float(), y.to(dev))
            opt.zero_grad()
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt); scaler.update(); sch.step()
            tot += loss.item()
            if i % 500 == 0:
                print(f"  ep{ep} step {i}/{len(tr_loader)} loss {tot / (i + 1):.4f}", flush=True)
        v = evaluate(model, ev["val"], dev)
        print(f"epoch {ep}  train_loss {tot / len(tr_loader):.4f}  val acc {v['acc']:.4f} "
              f"f1 {v['macro_f1']:.4f}  ({time.time() - t0:.0f}s)", flush=True)
        if v["macro_f1"] > best:                      # test only at the best-val epoch
            best = v["macro_f1"]
            result = dict(mode=a.mode, seed=a.seed, enc=a.enc, epoch=ep, val=v)
            for n in TESTS:
                result[n] = evaluate(model, ev[n], dev)

    print(json.dumps(result, indent=2))
    if not a.limit:
        os.makedirs("results", exist_ok=True)
        json.dump(result, open(f"results/{a.mode}_seed{a.seed}.json", "w"), indent=2)


if __name__ == "__main__":
    main()