#!/usr/bin/env python3
"""
Doc-form clustering + labeling sheet generator.

Input:
  - manifest_all.csv.gz
  - manifest_all_with_preview.jsonl.gz  (doc_id, sha256, rel_path, preview, extraction_method)

Output (default under outputs/docform/):
  - manifest_with_docform_clusters.csv.gz
  - docform_labeling_sheet.csv
  - docform_cluster_samples.tsv
  - docform_cluster_feature_means.csv
  - umap_embedding.csv

Install:
  pip install pandas tqdm scikit-learn umap-learn hdbscan scipy
"""

import argparse
import gzip
import json
import os
from pathlib import Path
from typing import Optional, List, Dict

import numpy as np
import pandas as pd
from tqdm import tqdm

from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy import sparse

import umap
import hdbscan


# ----------------------------
# Helpers
# ----------------------------
DEFAULT_NUMERIC_FEATURES = [
    # size-ish
    "n_chars",
    "n_lines",
    "n_tokens",
    "blank_lines",
    "avg_line_len",
    "max_line_len",
    # ratios
    "digit_ratio",
    "upper_ratio",
    "non_ascii_ratio",
    # lightweight patterns
    "email_count",
    "phone_like_count",
    "date_like_count",
    "money_like_count",
]

LOG1P_FEATURES = {"n_chars", "n_lines", "n_tokens", "blank_lines", "max_line_len",
                  "email_count", "phone_like_count", "date_like_count", "money_like_count"}


def read_jsonl_gz_previews(path: Path) -> pd.DataFrame:
    rows = []
    with gzip.open(path, "rt", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)
            rows.append({
                "doc_id": obj.get("doc_id"),
                "sha256": obj.get("sha256"),
                "rel_path": obj.get("rel_path"),
                "preview": obj.get("preview", ""),
                "extraction_method_preview": obj.get("extraction_method", ""),
            })
    return pd.DataFrame(rows)


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def safe_fill_numeric(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    out = df.copy()
    for c in cols:
        if c not in out.columns:
            out[c] = 0.0
        out[c] = pd.to_numeric(out[c], errors="coerce").fillna(0.0)
    return out


def build_numeric_matrix(df: pd.DataFrame, feature_cols: List[str]) -> sparse.csr_matrix:
    """
    Returns standardized numeric matrix as CSR sparse.
    Applies log1p to selected count/size-like columns.
    """
    X = df[feature_cols].astype(float).copy()

    for c in feature_cols:
        if c in LOG1P_FEATURES:
            X[c] = np.log1p(np.clip(X[c].values, 0, None))

    scaler = StandardScaler(with_mean=True, with_std=True)
    Xs = scaler.fit_transform(X.values)
    return sparse.csr_matrix(Xs)


def build_char_ngram_matrix(
    texts: List[str],
    min_df: int,
    max_features: int,
    ngram_min: int,
    ngram_max: int
) -> sparse.csr_matrix:
    """
    Char n-gram TF-IDF to capture formatting artifacts (headers, spacing, punctuation).
    """
    vec = TfidfVectorizer(
        analyzer="char",
        ngram_range=(ngram_min, ngram_max),
        min_df=min_df,
        max_features=max_features,
        lowercase=False,
        strip_accents=None,
    )
    X = vec.fit_transform(texts)
    return X


def compute_umap(
    X: sparse.csr_matrix,
    n_neighbors: int,
    min_dist: float,
    n_components: int,
    metric: str,
    random_state: int
) -> np.ndarray:
    reducer = umap.UMAP(
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        n_components=n_components,
        metric=metric,
        random_state=random_state,
        verbose=False,
    )
    emb = reducer.fit_transform(X)
    return emb


def run_hdbscan(
    emb: np.ndarray,
    min_cluster_size: int,
    min_samples: Optional[int],
    cluster_selection_epsilon: float
) -> Dict[str, np.ndarray]:
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        cluster_selection_epsilon=cluster_selection_epsilon,
        metric="euclidean",
        prediction_data=False
    )
    labels = clusterer.fit_predict(emb)
    probs = getattr(clusterer, "probabilities_", np.full(len(labels), np.nan))
    return {"labels": labels, "probs": probs}


def make_cluster_id(labels: np.ndarray) -> List[str]:
    # HDBSCAN noise is -1. Keep it explicit.
    out = []
    for lb in labels:
        if lb == -1:
            out.append("noise")
        else:
            out.append(f"c{int(lb):04d}")
    return out


def top_stats_by_cluster(df: pd.DataFrame, feature_cols: List[str]) -> pd.DataFrame:
    grp = df.groupby("docform_cluster", dropna=False)
    means = grp[feature_cols].mean().reset_index()
    counts = grp.size().reset_index(name="n_docs")
    return counts.merge(means, on="docform_cluster", how="left").sort_values("n_docs", ascending=False)


def build_labeling_sheet(
    df: pd.DataFrame,
    out_dir: Path,
    sample_per_cluster: int = 12,
    preview_chars: int = 240
) -> None:
    """
    Produces:
      - docform_labeling_sheet.csv : one row per cluster, includes placeholders for human labels
      - docform_cluster_samples.tsv: doc-level samples per cluster for manual inspection
    """
    # Sample docs for each cluster: prefer high probability and longer previews
    df2 = df.copy()
    df2["preview_len"] = df2["preview"].fillna("").str.len()
    df2["cluster_prob"] = pd.to_numeric(df2["docform_cluster_prob"], errors="coerce").fillna(0.0)

    df2 = df2.sort_values(
        ["docform_cluster", "cluster_prob", "preview_len"],
        ascending=[True, False, False]
    )

    samples = (
        df2.groupby("docform_cluster", dropna=False)
           .head(sample_per_cluster)
           .copy()
    )

    # Compact preview for TSV
    samples["preview_snip"] = (
        samples["preview"].fillna("")
        .str.replace("\r", " ", regex=False)
        .str.replace("\n", " ", regex=False)
        .str.slice(0, preview_chars)
    )

    samples_tsv = samples[[
        "docform_cluster",
        "docform_cluster_prob",
        "doc_family",
        "ext",
        "extraction_method",
        "doc_id",
        "sha256",
        "source_root",
        "rel_path",
        "size_bytes",
        "n_chars",
        "n_lines",
        "digit_ratio",
        "upper_ratio",
        "email_count",
        "phone_like_count",
        "date_like_count",
        "money_like_count",
        "preview_snip",
    ]].copy()

    samples_tsv_path = out_dir / "docform_cluster_samples.tsv"
    samples_tsv.to_csv(samples_tsv_path, sep="\t", index=False)

    # Labeling sheet: one row per cluster with summary + human fields
    cluster_counts = (
        df.groupby("docform_cluster", dropna=False)
          .size()
          .reset_index(name="n_docs")
          .sort_values("n_docs", ascending=False)
    )

    # Attach a few example rel_paths and previews
    examples = (
        samples_tsv.groupby("docform_cluster", dropna=False)
        .agg({
            "rel_path": lambda x: " | ".join(list(x)[:5]),
            "preview_snip": lambda x: " || ".join(list(x)[:5]),
        })
        .reset_index()
        .rename(columns={"rel_path": "example_rel_paths", "preview_snip": "example_previews"})
    )

    labeling = cluster_counts.merge(examples, on="docform_cluster", how="left")

    # Human annotation columns (you fill these)
    labeling.insert(0, "human_doc_type", "")
    labeling.insert(1, "human_doc_type_confidence", "")
    labeling.insert(2, "human_notes", "")
    labeling.insert(3, "keep_for_ontology", "")  # yes/no/maybe
    labeling.insert(4, "priority", "")           # high/med/low

    labeling_path = out_dir / "docform_labeling_sheet.csv"
    labeling.to_csv(labeling_path, index=False)


# ----------------------------
# Main
# ----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest_csv", default="manifest_all.csv.gz")
    ap.add_argument("--previews_jsonl", default="manifest_all_with_preview.jsonl.gz")
    ap.add_argument("--out_dir", default="outputs/docform")

    # Filtering
    ap.add_argument("--use_only_text_usable_now", action="store_true", default=True,
                    help="Use df[text_usable_now] if present; otherwise doc_family in {text,pdf} and has_text==True.")

    # Features
    ap.add_argument("--numeric_features", default=",".join(DEFAULT_NUMERIC_FEATURES))
    ap.add_argument("--use_char_ngrams", action="store_true", default=True)
    ap.add_argument("--char_ngram_min", type=int, default=2)
    ap.add_argument("--char_ngram_max", type=int, default=4)
    ap.add_argument("--char_min_df", type=int, default=5)
    ap.add_argument("--char_max_features", type=int, default=50_000)
    ap.add_argument("--char_weight", type=float, default=1.0,
                    help="Multiply char ngram matrix by this weight before concatenation (tune 0.3–2.0).")

    # UMAP
    ap.add_argument("--umap_neighbors", type=int, default=30)
    ap.add_argument("--umap_min_dist", type=float, default=0.05)
    ap.add_argument("--umap_components", type=int, default=5)
    ap.add_argument("--umap_metric", default="cosine")
    ap.add_argument("--random_state", type=int, default=42)

    # HDBSCAN
    ap.add_argument("--min_cluster_size", type=int, default=30)
    ap.add_argument("--min_samples", type=int, default=None)
    ap.add_argument("--cluster_selection_epsilon", type=float, default=0.0)

    # Labeling sheet
    ap.add_argument("--samples_per_cluster", type=int, default=12)
    ap.add_argument("--preview_chars", type=int, default=240)

    # misc
    ap.add_argument("--write_umap", action="store_true", default=True)

    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    ensure_dir(out_dir)

    manifest_csv = Path(args.manifest_csv)
    previews_jsonl = Path(args.previews_jsonl)

    print(f"[LOAD] {manifest_csv}")
    df = pd.read_csv(manifest_csv, compression="gzip")
    print(f"[LOAD] rows={len(df):,} cols={len(df.columns)}")

    print(f"[LOAD] {previews_jsonl}")
    pv = read_jsonl_gz_previews(previews_jsonl)
    print(f"[LOAD] previews rows={len(pv):,}")

    # Merge preview into df (by doc_id; sha256 is also present)
    df = df.merge(pv[["doc_id", "preview"]], on="doc_id", how="left")
    df["preview"] = df["preview"].fillna("")

    # Filter to usable now (text corpus)
    if args.use_only_text_usable_now:
        if "text_usable_now" in df.columns:
            df_use = df[df["text_usable_now"] == True].copy()
        else:
            # fallback
            df_use = df[(df["doc_family"].isin(["text", "pdf"])) & (df["has_text"] == True)].copy()
    else:
        df_use = df.copy()

    print(f"[FILTER] Using rows={len(df_use):,} for doc-form clustering")

    # Numeric features
    feature_cols = [c.strip() for c in args.numeric_features.split(",") if c.strip()]
    df_use = safe_fill_numeric(df_use, feature_cols)
    X_num = build_numeric_matrix(df_use, feature_cols)

    # Optional char n-grams (format signals)
    if args.use_char_ngrams:
        texts = df_use["preview"].astype(str).tolist()
        X_char = build_char_ngram_matrix(
            texts=texts,
            min_df=args.char_min_df,
            max_features=args.char_max_features,
            ngram_min=args.char_ngram_min,
            ngram_max=args.char_ngram_max,
        )
        if args.char_weight != 1.0:
            X_char = X_char.multiply(args.char_weight)

        X = sparse.hstack([X_num, X_char], format="csr")
        print(f"[FEATURES] X_num={X_num.shape} X_char={X_char.shape} -> X={X.shape}")
    else:
        X = X_num
        print(f"[FEATURES] X_num={X_num.shape} -> X={X.shape}")

    # UMAP embedding
    print("[UMAP] computing embedding…")
    emb = compute_umap(
        X=X,
        n_neighbors=args.umap_neighbors,
        min_dist=args.umap_min_dist,
        n_components=args.umap_components,
        metric=args.umap_metric,
        random_state=args.random_state,
    )
    print(f"[UMAP] emb shape={emb.shape}")

    # HDBSCAN
    print("[HDBSCAN] clustering…")
    res = run_hdbscan(
        emb=emb,
        min_cluster_size=args.min_cluster_size,
        min_samples=args.min_samples,
        cluster_selection_epsilon=args.cluster_selection_epsilon,
    )
    labels = res["labels"]
    probs = res["probs"]

    df_use["docform_label_raw"] = labels
    df_use["docform_cluster"] = make_cluster_id(labels)
    df_use["docform_cluster_prob"] = probs

    n_noise = int((labels == -1).sum())
    n_clusters = len(set(labels)) - (1 if -1 in set(labels) else 0)
    print(f"[HDBSCAN] clusters={n_clusters} noise={n_noise} used={len(df_use):,}")

    # Write embedding for inspection
    if args.write_umap:
        umap_df = df_use[["doc_id", "sha256", "rel_path", "docform_cluster"]].copy()
        for k in range(emb.shape[1]):
            umap_df[f"umap_{k}"] = emb[:, k]
        umap_path = out_dir / "umap_embedding.csv"
        umap_df.to_csv(umap_path, index=False)
        print(f"[WRITE] {umap_path}")

    # Merge clusters back to full manifest (left join)
    df_out = df.merge(
        df_use[["doc_id", "docform_cluster", "docform_cluster_prob", "docform_label_raw"]],
        on="doc_id",
        how="left"
    )

    out_manifest = out_dir / "manifest_with_docform_clusters.csv.gz"
    df_out.to_csv(out_manifest, index=False, compression="gzip")
    print(f"[WRITE] {out_manifest}")

    # Feature means per cluster (use df_use)
    means = top_stats_by_cluster(df_use, feature_cols)
    means_path = out_dir / "docform_cluster_feature_means.csv"
    means.to_csv(means_path, index=False)
    print(f"[WRITE] {means_path}")

    # Labeling sheet + samples TSV
    build_labeling_sheet(
        df=df_use,
        out_dir=out_dir,
        sample_per_cluster=args.samples_per_cluster,
        preview_chars=args.preview_chars
    )
    print(f"[WRITE] {out_dir / 'docform_labeling_sheet.csv'}")
    print(f"[WRITE] {out_dir / 'docform_cluster_samples.tsv'}")

    # Quick summary
    vc = df_use["docform_cluster"].value_counts().head(20)
    print("[TOP CLUSTERS]")
    print(vc.to_string())


if __name__ == "__main__":
    main()
