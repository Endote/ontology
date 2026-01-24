# python semantic_cluster_by_doctype.py \
#   --corpus outputs/near_dedup/corpus_canonical.csv.gz \
#   --preview_jsonl_gz manifest_all_with_preview.jsonl.gz \
#   --out_dir outputs/semantic \
#   --min_docs 30


#!/usr/bin/env python3
import argparse, gzip, json, re
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd
from tqdm import tqdm

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler

import umap
import hdbscan

# added (scikit-learn depends on scipy; this should be available)
from scipy import sparse


# ----------------------------
# Preview loader
# ----------------------------
def load_preview_map(path_jsonl_gz: Path) -> dict:
    m = {}
    with gzip.open(path_jsonl_gz, "rt", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            m[obj["doc_id"]] = obj.get("preview", "")
    return m


# ----------------------------
# Cleaning per doc type (simple, extensible)
# ----------------------------
RE_EMAIL_HDR = re.compile(r"^(from|to|cc|bcc|sent|date|subject|attachments|importance)\s*:\s*", re.I)
RE_QP = re.compile(r"=([0-9A-F]{2})", re.I)
RE_TS_LINE = re.compile(r"^\s*\d{1,2}/\d{1,2}/\d{2,4}\s+\d{1,2}:\d{2}(:\d{2})?\s*(AM|PM)?\s*$", re.I)

def clean_text(doc_type: str, text: str) -> str:
    if not isinstance(doc_type, str) or not doc_type.strip():
        doc_type = "unknown"
    if not isinstance(text, str):
        text = ""

    t = text.replace("\r", "\n")
    t = re.sub(r"Non-Responsive\s*-\s*Redacted", " ", t, flags=re.I)
    t = re.sub(r"Privileged\s*-\s*Redacted", " ", t, flags=re.I)

    lines = [ln.strip() for ln in t.splitlines() if ln.strip()]

    if doc_type.startswith("chat_") or "ichat" in doc_type:
        out = []
        for ln in lines:
            if ln.lower().startswith(("source entry:", "service:", "start time:", "end time:", "last message")):
                continue
            if RE_TS_LINE.match(ln):
                continue
            out.append(ln)
        t2 = " ".join(out)

    elif doc_type.startswith("email_") or "email" in doc_type:
        out = []
        for ln in lines:
            if RE_EMAIL_HDR.match(ln):
                continue
            out.append(ln)
        t2 = " ".join(out)
        t2 = RE_QP.sub(" ", t2)

    else:
        t2 = " ".join(lines)

    t2 = re.sub(r"\s+", " ", t2).strip()
    return t2


# ----------------------------
# Representative samples per topic cluster
# ----------------------------
def pick_samples(df_type: pd.DataFrame, cluster_col: str, n_per: int = 10, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for cid, g in df_type.groupby(cluster_col, dropna=False):
        if cid == -1:
            continue
        if len(g) <= n_per:
            rows.append(g)
        else:
            idx = rng.choice(g.index.to_numpy(), size=n_per, replace=False)
            rows.append(g.loc[idx])
    if not rows:
        return df_type.head(0)
    return pd.concat(rows).copy()


# ----------------------------
# Topic keywords (TF-IDF mean vector per cluster)
# ----------------------------
def top_keywords_per_cluster(X, labels, feature_names, topn: int = 15, min_docs: int = 5) -> dict:
    """
    X: csr_matrix [n_docs, n_terms] tf-idf
    labels: np.array [n_docs] topic labels (-1 = noise)
    feature_names: array-like[str] of length n_terms
    Returns dict: cid -> {'topic_cluster', 'topic_size', 'keywords', 'top_terms'}
    """
    labels = np.asarray(labels)
    out = {}

    # ensure CSR for fast row slicing
    if not sparse.isspmatrix_csr(X):
        X = X.tocsr()

    for cid in sorted(set(labels.tolist())):
        if cid == -1:
            continue
        idx = np.where(labels == cid)[0]
        if len(idx) < min_docs:
            continue

        Xc = X[idx]
        mean_vec = Xc.mean(axis=0)  # 1 x n_terms
        mean_arr = np.asarray(mean_vec).ravel()
        if mean_arr.size == 0:
            continue

        top_idx = np.argsort(mean_arr)[::-1][:topn]
        terms = [(str(feature_names[i]), float(mean_arr[i])) for i in top_idx if mean_arr[i] > 0]
        keywords = ", ".join([t for t, _ in terms[:topn]])

        out[int(cid)] = {
            "topic_cluster": int(cid),
            "topic_size": int(len(idx)),
            "keywords": keywords,
            "top_terms": terms,
        }

    return out


# ----------------------------
# Main clustering per doc_type
# ----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default="outputs/near_dedup/corpus_canonical.csv.gz")
    ap.add_argument("--preview_jsonl_gz", default="manifest_all_with_preview.jsonl.gz")
    ap.add_argument("--out_dir", default="outputs/semantic")
    ap.add_argument("--min_docs", type=int, default=30, help="Skip doc_types with fewer docs than this")
    ap.add_argument("--max_features", type=int, default=60000)
    ap.add_argument("--ngram_min", type=int, default=1)
    ap.add_argument("--ngram_max", type=int, default=2)
    ap.add_argument("--umap_dim", type=int, default=5)
    ap.add_argument("--umap_neighbors", type=int, default=25)
    ap.add_argument("--min_cluster_size", type=int, default=25)
    ap.add_argument("--min_samples", type=int, default=10)
    # added: keyword settings
    ap.add_argument("--topn_keywords", type=int, default=15, help="Top N TF-IDF keywords per topic cluster")
    ap.add_argument("--min_docs_keywords", type=int, default=5, help="Minimum docs in cluster to emit keywords")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.corpus, compression="gzip", low_memory=False)
    prev = load_preview_map(Path(args.preview_jsonl_gz))

    # Ensure doc_type present
    if "human_doc_type" not in df.columns:
        df["human_doc_type"] = "unknown"

    # Preview coverage check
    miss = df["doc_id"].map(lambda x: x not in prev).sum()
    print(f"[PREVIEW] missing previews: {miss:,}/{len(df):,}")

    # Attach cleaned text
    texts = []
    for r in tqdm(df.itertuples(index=False), total=len(df), desc="Clean text"):
        dt = getattr(r, "human_doc_type", "")
        if not isinstance(dt, str) or not dt.strip():
            dt = "unknown"
        raw = prev.get(r.doc_id, "")
        texts.append(clean_text(dt, raw))
    df["clean_text"] = texts
    df["clean_len"] = df["clean_text"].str.len().fillna(0).astype(int)

    # Filter out useless rows
    df = df[df["clean_len"] >= 30].copy()
    print(f"[FILTER] usable for semantic clustering: {len(df):,}")

    all_type_stats = []
    all_label_rows = []

    for doc_type, g in df.groupby("human_doc_type"):
        if not isinstance(doc_type, str) or not doc_type.strip():
            doc_type = "unknown"
        if len(g) < args.min_docs:
            continue

        type_dir = out_dir / doc_type
        type_dir.mkdir(parents=True, exist_ok=True)

        # TF-IDF
        vec = TfidfVectorizer(
            max_features=args.max_features,
            ngram_range=(args.ngram_min, args.ngram_max),
            min_df=2,
            max_df=0.95,
            strip_accents="unicode"
        )
        X = vec.fit_transform(g["clean_text"].tolist())
        feature_names = vec.get_feature_names_out()  # added
        vocab_size = len(vec.vocabulary_)
        nnz = X.nnz
        density = nnz / (X.shape[0] * max(1, X.shape[1]))

        # UMAP (on dense-ish reduced)
        # Using UMAP directly on sparse works but can be slow; we standardize via sparse-friendly approach:
        emb = umap.UMAP(
            n_neighbors=args.umap_neighbors,
            n_components=args.umap_dim,
            metric="cosine",
            random_state=42
        ).fit_transform(X)

        # HDBSCAN on embedding
        clusterer = hdbscan.HDBSCAN(
            min_cluster_size=args.min_cluster_size,
            min_samples=args.min_samples,
            metric="euclidean"
        )
        labels = clusterer.fit_predict(emb)

        # --- topic keywords + topic_summary.csv (robust even if no clusters)
        min_docs_kw = max(args.min_docs_keywords, int(0.01 * len(g)))  # tiny clusters aren't meaningful
        kw = top_keywords_per_cluster(
            X, labels, feature_names,
            topn=args.topn_keywords,
            min_docs=min_docs_kw
        )

        topic_summary_rows = []
        for cid in sorted(set(labels.tolist())):
            if cid == -1:
                continue
            topic_summary_rows.append({
                "human_doc_type": doc_type,
                "topic_cluster": int(cid),
                "topic_size": int((labels == cid).sum()),
                "keywords": kw.get(int(cid), {}).get("keywords", ""),
            })

        # IMPORTANT: build with explicit columns so empty DF still has schema
        topic_summary_df = pd.DataFrame(
            topic_summary_rows,
            columns=["human_doc_type", "topic_cluster", "topic_size", "keywords"]
        )

        if len(topic_summary_df):
            topic_summary_df = topic_summary_df.sort_values(["topic_size"], ascending=False)

        topic_summary_df.to_csv(type_dir / "topic_summary.csv", index=False)

        # pd.DataFrame(topic_summary_rows)\
        #   .sort_values(["topic_size"], ascending=False)\
        #   .to_csv(type_dir / "topic_summary.csv", index=False)

        # Save per-doc outputs
        out = g[["doc_id","sha256","rel_path","docform_cluster","human_doc_type","clean_len"]].copy()
        out["topic_cluster"] = labels
        out["topic_prob"] = clusterer.probabilities_

        # added: attach topic keywords + size for convenience
        def _kw_for(x):
            if x == -1:
                return ""
            return kw.get(int(x), {}).get("keywords", "")
        def _sz_for(x):
            if x == -1:
                return 0
            return kw.get(int(x), {}).get("topic_size", int((labels == int(x)).sum()))
        out["topic_keywords"] = out["topic_cluster"].apply(_kw_for)
        out["topic_size"] = out["topic_cluster"].apply(_sz_for)

        out.to_csv(type_dir / "docs_with_topic_clusters.csv.gz", index=False, compression="gzip")

        emb_df = pd.DataFrame(emb, columns=[f"u{i}" for i in range(emb.shape[1])])
        emb_df.insert(0, "doc_id", g["doc_id"].values)
        emb_df["topic_cluster"] = labels
        emb_df.to_csv(type_dir / "umap_embedding.csv", index=False)

        # Stats
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        noise = int((labels == -1).sum())
        all_type_stats.append({
            "human_doc_type": doc_type,
            "n_docs": int(len(g)),
            "vocab_size": int(vocab_size),
            "tfidf_nnz": int(nnz),
            "tfidf_density": float(density),
            "umap_dim": int(args.umap_dim),
            "n_clusters": int(n_clusters),
            "noise_docs": noise,
            # added: how many clusters got keywords (i.e., passed min_docs_kw)
            "clusters_with_keywords": int(len(kw)),
            "min_docs_keywords_used": int(min_docs_kw),
        })

        # Labeling sheet: 10 examples per cluster (excluding noise)
        out_for_samples = out.merge(
            pd.DataFrame({"doc_id": g["doc_id"].values, "preview": [prev.get(x, "")[:400] for x in g["doc_id"].values]}),
            on="doc_id", how="left"
        )
        samples = pick_samples(out_for_samples, "topic_cluster", n_per=10)
        # Add columns for human labeling
        if len(samples):
            samples = samples.sort_values(["topic_cluster","topic_prob"], ascending=[True, False])
            samples["topic_label_human"] = ""
            samples["topic_keep"] = ""
            samples["topic_notes"] = ""

            # keep column order stable and useful for labeling
            preferred_cols = [
                "human_doc_type",
                "topic_cluster",
                "topic_size",
                "topic_keywords",
                "topic_prob",
                "doc_id",
                "sha256",
                "rel_path",
                "docform_cluster",
                "clean_len",
                "preview",
                "topic_label_human",
                "topic_keep",
                "topic_notes",
            ]
            cols = [c for c in preferred_cols if c in samples.columns] + [c for c in samples.columns if c not in preferred_cols]
            samples = samples[cols]

            samples.to_csv(type_dir / "topic_labeling_sheet.csv", index=False)
            all_label_rows.append(samples.assign(_doc_type=doc_type))

        print(f"[TYPE] {doc_type} | docs={len(g):,} vocab={vocab_size:,} clusters={n_clusters} noise={noise}")

    stats_df = pd.DataFrame(all_type_stats).sort_values(["n_docs"], ascending=False)
    stats_df.to_csv(out_dir / "semantic_type_stats.csv", index=False)

    print(f"[WRITE] {out_dir/'semantic_type_stats.csv'}")
    print("[DONE] semantic clustering per doc_type complete.")

if __name__ == "__main__":
    main()
