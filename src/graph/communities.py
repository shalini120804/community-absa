"""Stage 5: reviewer communities (Leiden), built from TRAIN data only.
Edges use (a) co-reviewed products and (b) similarity of WHICH aspects reviewers mention.
Sentiment is never used. New reviewers are placed using their reviewer_history only."""
import argparse, os, pickle
import numpy as np, pandas as pd, scipy.sparse as sp, torch
import igraph as ig, leidenalg
from sklearn.neighbors import NearestNeighbors

SPLITS = "data/processed/splits"
MAX_ITEM_DEG = 300   # items reviewed by more users than this link everyone to everyone
KNN = 15
MIN_MENTIONS = 3     # users with fewer aspect mentions get no aspect-similarity edges
ALPHA = 1.0          # weight of aspect similarity relative to shared products


def mention_counts(reviews, asp, aspect_list, users):
    uidx = {u: i for i, u in enumerate(users)}
    aidx = {a: i for i, a in enumerate(aspect_list)}
    g = asp[asp.review_id.isin(reviews.review_id)]
    M = np.zeros((len(users), len(aspect_list)), dtype=np.float32)
    np.add.at(M, (g.user_id.map(uidx).values.astype(int), g.aspect.map(aidx).values.astype(int)), 1)
    return M


def embed(M):
    X = np.sqrt(M)
    return X / np.maximum(np.linalg.norm(X, axis=1, keepdims=True), 1e-9)


def item_matrix(reviews, uidx, items):
    r = reviews[reviews.item_id.isin(items)].drop_duplicates(["user_id", "item_id"])
    return sp.csr_matrix((np.ones(len(r), dtype=np.float32),
                          (r.user_id.map(uidx).values, r.item_id.map(items).values)),
                         shape=(len(uidx), len(items)))


def aspect_sim(Xq, qmask, ref, same):
    """Sparse (n_query, n_train) cosine similarities to the KNN nearest train reviewers."""
    qids = np.where(qmask)[0]
    k = KNN + 1 if same else KNN
    d, j = ref["nn"].kneighbors(Xq[qids], n_neighbors=k)
    rows, cols, vals = np.repeat(qids, k), ref["valid_ids"][j.ravel()], np.clip(1 - d.ravel(), 0, None)
    if same:
        keep = rows != cols
        rows, cols, vals = rows[keep], cols[keep], vals[keep]
    return sp.csr_matrix((vals, (rows, cols)), shape=(len(Xq), ref["n"]))


def build(ck, tr, asp):
    uidx, items, aspects = ck["users"], ck["items"], ck["aspects"]
    users = sorted(uidx, key=uidx.get)
    B = item_matrix(tr, uidx, items)
    deg = np.asarray(B.sum(0)).ravel()
    w = np.where((deg >= 2) & (deg <= MAX_ITEM_DEG), 1 / np.log1p(deg), 0).astype(np.float32)
    Wi = (B @ sp.diags(w) @ B.T).tocsr()
    Wi = Wi - sp.diags(Wi.diagonal())
    Wi.eliminate_zeros()

    M = mention_counts(tr, asp, aspects, users)
    X = embed(M)
    mask = M.sum(1) >= MIN_MENTIONS
    valid = np.where(mask)[0]
    nn = NearestNeighbors(metric="cosine", algorithm="brute").fit(X[valid])
    ref = dict(valid_ids=valid, nn=nn, n=len(users))
    Wa = aspect_sim(X, mask, ref, same=True)
    Wa = Wa.maximum(Wa.T)

    ni, na = Wi.data.mean(), Wa.data.mean()
    W = (Wi / ni + ALPHA * Wa / na).tocsr()
    art = dict(users=users, uidx=uidx, items=items, aspects=aspects, B=B, w=w, ref=ref, ni=ni, na=na)
    return W, art


def leiden(W, gamma, seed=0):
    t = sp.triu(W, k=1).tocoo()
    g = ig.Graph(n=W.shape[0], edges=list(zip(t.row.tolist(), t.col.tolist())))
    g.es["weight"] = t.data.tolist()
    p = leidenalg.find_partition(g, leidenalg.RBConfigurationVertexPartition, weights="weight",
                                 resolution_parameter=gamma, n_iterations=-1, seed=seed)
    return np.array(p.membership), g.modularity(p.membership, weights="weight")


def score(Wrows, member, S, total, gamma, K):
    """Pick the community with the largest modularity gain for each row (-1 if no edges)."""
    ok = np.where(member >= 0)[0]
    C = sp.csr_matrix((np.ones(len(ok)), (ok, member[ok])), shape=(len(member), K))
    wc = (Wrows @ C).toarray()
    k = np.asarray(Wrows.sum(1)).ravel()
    s = wc - gamma * k[:, None] * S[None, :] / total
    out = s.argmax(1)
    out[k == 0] = -1
    return out


def strengths(W, member, K):
    st = np.asarray(W.sum(1)).ravel()
    return np.array([st[member == c].sum() for c in range(K)]), st.sum()


def finalize(W, gamma, min_size):
    mem, mod = leiden(W, gamma)
    sizes = np.bincount(mem)
    big = [c for c in np.argsort(-sizes) if sizes[c] >= min_size]
    remap = {c: i for i, c in enumerate(big)}
    member = np.array([remap.get(c, -1) for c in mem])
    K = len(big)
    S, total = strengths(W, member, K)
    small = np.where(member < 0)[0]
    if len(small):
        member[small] = score(W[small], member, S, total, gamma, K)
    S, total = strengths(W, member, K)
    return member, K, S, total, mod


def assign_new(hist, asp, art):
    """Community for reviewers outside the train graph, using their history reviews only."""
    users = sorted(hist.user_id.unique())
    uidx = {u: i for i, u in enumerate(users)}
    Bn = item_matrix(hist, uidx, art["items"])
    Wi = (Bn @ sp.diags(art["w"]) @ art["B"].T).tocsr() / art["ni"]
    M = mention_counts(hist, asp, art["aspects"], users)
    Wa = aspect_sim(embed(M), M.sum(1) >= MIN_MENTIONS, art["ref"], same=False) / art["na"]
    Wn = (Wi + ALPHA * Wa).tocsr()
    c = score(Wn, art["member"], art["S"], art["total"], art["gamma"], art["K"])
    return pd.DataFrame(dict(user_id=users, community=c))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gamma", type=float, default=0, help="0 = just print a resolution sweep")
    ap.add_argument("--min_size", type=int, default=100)
    a = ap.parse_args()
    pd.set_option("display.width", 250, "display.max_columns", 30)

    ck = torch.load("data/processed/graph_train.pt", weights_only=False)
    tr = pd.read_parquet(f"{SPLITS}/train.parquet")
    asp = pd.read_parquet("data/processed/aspects.parquet")
    W, art = build(ck, tr, asp)
    print(f"user graph: {W.shape[0]} users, {W.nnz // 2} edges, "
          f"isolated users: {int((W.getnnz(axis=1) == 0).sum())}, "
          f"users with >={MIN_MENTIONS} mentions: {len(art['ref']['valid_ids'])}")

    if a.gamma == 0:
        print("\ngamma  communities  big  largest  users_in_big  modularity")
        for g in [0.02, 0.05, 0.1, 0.2, 0.5, 1.0]:
            mem, mod = leiden(W, g)
            sz = np.bincount(mem)
            big = sz[sz >= a.min_size]
            print(f"{g:5.2f} {len(sz):11d} {len(big):4d} {sz.max() / len(mem):8.0%} "
                  f"{big.sum() / len(mem):12.0%} {mod:11.3f}")
        return

    member, K, S, total, mod = finalize(W, a.gamma, a.min_size)
    art.update(member=member, K=K, S=S, total=total, gamma=a.gamma)
    print(f"\ngamma={a.gamma}  communities K={K}  modularity={mod:.3f}  "
          f"unassigned train users: {int((member < 0).sum())}")

    hist = pd.read_parquet(f"{SPLITS}/reviewer_history.parquet")
    tur = pd.read_parquet(f"{SPLITS}/test_unseen_reviewers.parquet")
    print("history users that are also train users (must be 0):", int(hist.user_id.isin(art["uidx"]).sum()))
    print("history users == test_unseen_reviewers users:", set(hist.user_id) == set(tur.user_id))

    train_df = pd.DataFrame(dict(user_id=art["users"], community=member, source="train"))
    hist_df = assign_new(hist, asp, art)
    hist_df["source"] = "history"
    uc = pd.concat([train_df, hist_df], ignore_index=True)
    os.makedirs("data/processed", exist_ok=True)
    uc.to_parquet("data/processed/user_community.parquet", index=False)
    pickle.dump(art, open("data/processed/community_artifacts.pkl", "wb"))

    cmap = uc.set_index("user_id").community
    print("\nshare of each split's reviews whose user has a community:")
    for f in sorted(os.listdir(SPLITS)):
        d = pd.read_parquet(f"{SPLITS}/{f}")
        print(f"  {f:34s} {(d.user_id.map(cmap) >= 0).mean():.0%}")

    M = mention_counts(tr, asp, art["aspects"], art["users"])
    share = pd.DataFrame(M).groupby(member).sum()
    share = share.div(share.sum(1), axis=0)
    share.columns = art["aspects"]
    share.insert(0, "n_users", pd.Series(member).value_counts().sort_index())
    print("\nAspect mention share per community (-1 = unassigned):")
    print(share.round(2))

    g = asp[asp.review_id.isin(tr.review_id)].copy()
    g["community"] = g.user_id.map(dict(zip(art["users"], member)))
    print("\nMean sentiment per community x aspect (train, preview only):")
    print(g.pivot_table(index="community", columns="aspect", values="sentiment", aggfunc="mean").round(2))


if __name__ == "__main__":
    main()