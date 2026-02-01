#!/usr/bin/env python3

#run
# python docform_clusters.py --manifest_csv manifest_all.csv.gz --previews_jsonl manifest_all_with_preview.jsonl.gz  --text_path_col text_path  --text_root . --out_dir outputs/docform --min_cluster_size 35 --min_samples 6 --hdbscan_selection_method eom --umap_components 30 --refine_min_docs 200 --char_max_features 15000 --char_min_df 10

"""
Doc-form clustering + labeling sheet generator (v2.2) — fixes refine IndexError for sparse X.

Root cause fixed:
- df_use kept original manifest index labels (0..26062), but X has only len(df_use) rows.
- groupby(...).groups returned those original labels; indexing X with them caused out-of-range.
Fix:
- reset_index(drop=True) on df_use before building X
- refinement groups use positional indices guaranteed 0..len(df_use)-1

Also:
- pd.read_csv(..., low_memory=False) removes the DtypeWarning chunking weirdness.
"""

import argparse
import gzip
import json
from pathlib import Path
from typing import Optional, List, Dict, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm

from sklearn.preprocessing import StandardScaler, normalize
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy import sparse

import umap
import hdbscan

import re
import string
from collections import Counter


# ----------------------------
# Helpers / constants
# ----------------------------
DEFAULT_NUMERIC_FEATURES = [
    "n_chars", "n_lines", "n_tokens", "blank_lines", "avg_line_len", "max_line_len",
    "digit_ratio", "upper_ratio", "non_ascii_ratio",
    "email_count", "phone_like_count", "date_like_count", "money_like_count",
    "punct_ratio", "colon_ratio", "header_line_ratio", "unique_line_ratio", "repeated_line_max_frac",
]

LOG1P_FEATURES = {
    "n_chars", "n_lines", "n_tokens", "blank_lines", "max_line_len",
    "email_count", "phone_like_count", "date_like_count", "money_like_count"
}

RE_HEADER_LINE = re.compile(r"^[A-Za-z][A-Za-z0-9 _\-/]{0,40}:\s+\S+")
PUNCT_SET = set(string.punctuation)


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


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


def safe_fill_numeric(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    out = df.copy()
    for c in cols:
        if c not in out.columns:
            out[c] = 0.0
        out[c] = pd.to_numeric(out[c], errors="coerce").fillna(0.0)
    return out


def build_numeric_matrix(df: pd.DataFrame, feature_cols: List[str]) -> sparse.csr_matrix:
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
) -> Tuple[sparse.csr_matrix, TfidfVectorizer]:
    vec = TfidfVectorizer(
        analyzer="char",
        ngram_range=(ngram_min, ngram_max),
        min_df=min_df,
        max_features=max_features,
        lowercase=False,
        strip_accents=None,
    )
    X = vec.fit_transform(texts)
    return X, vec


def compute_umap(
    X: sparse.csr_matrix,
    n_neighbors: int,
    min_dist: float,
    n_components: int,
    metric: str,
    random_state: Optional[int]
) -> np.ndarray:
    reducer = umap.UMAP(
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        n_components=n_components,
        metric=metric,
        random_state=random_state,
        verbose=False,
    )
    return reducer.fit_transform(X)


def run_hdbscan(
    emb: np.ndarray,
    min_cluster_size: int,
    min_samples: Optional[int],
    cluster_selection_epsilon: float,
    selection_method: str,
) -> Dict[str, np.ndarray]:
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        cluster_selection_epsilon=cluster_selection_epsilon,
        cluster_selection_method=selection_method,
        metric="euclidean",
        prediction_data=False,
    )
    labels = clusterer.fit_predict(emb)
    probs = getattr(clusterer, "probabilities_", np.full(len(labels), np.nan))
    return {"labels": labels, "probs": probs}


def make_cluster_id(labels: np.ndarray, prefix: str = "c") -> List[str]:
    out = []
    for lb in labels:
        if lb == -1:
            out.append("noise")
        else:
            out.append(f"{prefix}{int(lb):04d}")
    return out


def top_stats_by_cluster(df: pd.DataFrame, feature_cols: List[str]) -> pd.DataFrame:
    grp = df.groupby("docform_cluster", dropna=False)
    means = grp[feature_cols].mean(numeric_only=True).reset_index()
    counts = grp.size().reset_index(name="n_docs")
    return counts.merge(means, on="docform_cluster", how="left").sort_values("n_docs", ascending=False)


def _read_text_file(path: Path) -> str:
    if not path.exists() or not path.is_file():
        return ""
    try:
        if path.suffix.lower() == ".gz":
            with gzip.open(path, "rt", encoding="utf-8", errors="ignore") as f:
                return f.read()
        return path.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return ""


def _unescape_preview(s: str) -> str:
    if not isinstance(s, str):
        return ""
    return s.replace("\\r", "\r").replace("\\n", "\n")


def get_doc_text(
    row: pd.Series,
    *,
    text_col: Optional[str],
    text_path_col: Optional[str],
    text_root: Optional[Path],
) -> str:
    if text_col and text_col in row and isinstance(row[text_col], str) and row[text_col].strip():
        return row[text_col]

    if text_path_col and text_path_col in row and isinstance(row[text_path_col], str) and row[text_path_col].strip():
        p = Path(row[text_path_col])
        if not p.is_absolute() and text_root is not None:
            p = text_root / p
        txt = _read_text_file(p)
        if txt.strip():
            return txt

    if "preview" in row and isinstance(row["preview"], str):
        return _unescape_preview(row["preview"])

    return ""


def make_docform_slice(text: str, head_chars: int = 4000, tail_chars: int = 4000) -> str:
    t = text or ""
    if len(t) <= head_chars + tail_chars + 50:
        return t
    return t[:head_chars] + "\n...\n" + t[-tail_chars:]


def compute_extra_docform_features(text: str) -> dict:
    t = (text or "").replace("\r\n", "\n").replace("\r", "\n")
    lines = [ln.strip() for ln in t.split("\n")]
    lines_nonempty = [ln for ln in lines if ln]

    n_chars = len(t)
    n_lines_nonempty = len(lines_nonempty)

    punct_cnt = sum(1 for ch in t if ch in PUNCT_SET)
    punct_ratio = punct_cnt / max(1, n_chars)

    colon_ratio = t.count(":") / max(1, n_chars)

    header_like = sum(1 for ln in lines_nonempty if RE_HEADER_LINE.match(ln) is not None)
    header_line_ratio = header_like / max(1, n_lines_nonempty)

    if n_lines_nonempty:
        counts = Counter(lines_nonempty)
        unique_line_ratio = len(counts) / n_lines_nonempty
        repeated_line_max_frac = max(counts.values()) / n_lines_nonempty
    else:
        unique_line_ratio = 0.0
        repeated_line_max_frac = 0.0

    return {
        "punct_ratio": float(punct_ratio),
        "colon_ratio": float(colon_ratio),
        "header_line_ratio": float(header_line_ratio),
        "unique_line_ratio": float(unique_line_ratio),
        "repeated_line_max_frac": float(repeated_line_max_frac),
    }


def top_chargrams_by_cluster(
    X_char: sparse.csr_matrix,
    labels: np.ndarray,
    vec: TfidfVectorizer,
    topk: int = 30,
    min_docs: int = 30
) -> pd.DataFrame:
    feats = vec.get_feature_names_out()
    labels = np.asarray(labels)
    Xc = X_char.tocsr()

    rows = []
    for cid in sorted(set(labels.tolist())):
        if cid == -1:
            continue
        idx = np.where(labels == cid)[0]
        if len(idx) < min_docs:
            continue
        mean = Xc[idx].mean(axis=0)
        arr = np.asarray(mean).ravel()
        top_idx = np.argsort(arr)[::-1][:topk]
        grams = [feats[i] for i in top_idx if arr[i] > 0]
        rows.append({
            "docform_coarse_raw": int(cid),
            "docform_coarse": f"c{int(cid):04d}",
            "n_docs": int(len(idx)),
            "top_chargrams": " | ".join(grams[:topk]),
        })
    return pd.DataFrame(rows).sort_values("n_docs", ascending=False)


def adaptive_params(n: int) -> Tuple[int, int, int]:
    mcs = max(10, int(0.05 * n))
    ms = max(5, int(0.02 * n))
    nn = min(40, max(10, n // 3))
    return mcs, ms, nn


def build_labeling_sheet(
    df: pd.DataFrame,
    out_dir: Path,
    sample_per_cluster: int = 12,
    preview_chars: int = 240,
    include_chargrams_col: Optional[str] = None
) -> None:
    df2 = df.copy()
    df2["preview_len"] = df2["preview"].fillna("").astype(str).str.len()
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

    samples["preview_snip"] = (
        samples["preview"].fillna("").astype(str)
        .str.replace("\r", " ", regex=False)
        .str.replace("\n", " ", regex=False)
        .str.slice(0, preview_chars)
    )

    base_cols = [
        "docform_cluster", "docform_cluster_prob",
        "doc_family", "ext", "extraction_method",
        "doc_id", "sha256", "source_root", "rel_path", "text_path",
        "size_bytes", "n_chars", "n_lines",
        "digit_ratio", "upper_ratio",
        "email_count", "phone_like_count", "date_like_count", "money_like_count",
        "preview_snip",
    ]
    extra_cols = [c for c in [
        "punct_ratio", "colon_ratio", "header_line_ratio", "unique_line_ratio", "repeated_line_max_frac"
    ] if c in samples.columns]

    if include_chargrams_col and include_chargrams_col in samples.columns:
        extra_cols.append(include_chargrams_col)

    keep_cols = [c for c in base_cols if c in samples.columns] + extra_cols
    samples_tsv = samples[keep_cols].copy()

    (out_dir / "docform_cluster_samples.tsv").write_text(
        samples_tsv.to_csv(sep="\t", index=False),
        encoding="utf-8"
    )

    cluster_counts = (
        df.groupby("docform_cluster", dropna=False)
          .size()
          .reset_index(name="n_docs")
          .sort_values("n_docs", ascending=False)
    )

    examples = (
        samples_tsv.groupby("docform_cluster", dropna=False)
        .agg({
            "rel_path": lambda x: " | ".join(list(x)[:5]) if "rel_path" in samples_tsv.columns else "",
            "preview_snip": lambda x: " || ".join(list(x)[:5]) if "preview_snip" in samples_tsv.columns else "",
        })
        .reset_index()
        .rename(columns={"rel_path": "example_rel_paths", "preview_snip": "example_previews"})
    )

    labeling = cluster_counts.merge(examples, on="docform_cluster", how="left")

    if include_chargrams_col and include_chargrams_col in df.columns:
        chargrams = (
            df.groupby("docform_cluster", dropna=False)[include_chargrams_col]
              .agg(lambda x: next((v for v in x if isinstance(v, str) and v.strip()), ""))
              .reset_index()
              .rename(columns={include_chargrams_col: "cluster_chargrams_hint"})
        )
        labeling = labeling.merge(chargrams, on="docform_cluster", how="left")

    labeling.insert(0, "human_doc_type", "")
    labeling.insert(1, "human_doc_type_confidence", "")
    labeling.insert(2, "human_notes", "")
    labeling.insert(3, "keep_for_ontology", "")
    labeling.insert(4, "priority", "")

    labeling.to_csv(out_dir / "docform_labeling_sheet.csv", index=False)


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--manifest_csv", default="manifest_all.csv.gz")
    ap.add_argument("--previews_jsonl", default="manifest_all_with_preview.jsonl.gz")
    ap.add_argument("--out_dir", default="outputs/docform")

    ap.add_argument("--use_only_text_usable_now", action=argparse.BooleanOptionalAction, default=True)

    ap.add_argument("--text_col", default=None)
    ap.add_argument("--text_path_col", default="text_path")
    ap.add_argument("--text_root", default=".")
    ap.add_argument("--docform_head_chars", type=int, default=4000)
    ap.add_argument("--docform_tail_chars", type=int, default=4000)

    ap.add_argument("--numeric_features", default=",".join(DEFAULT_NUMERIC_FEATURES))
    ap.add_argument("--use_char_ngrams", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--char_ngram_min", type=int, default=2)
    ap.add_argument("--char_ngram_max", type=int, default=4)
    ap.add_argument("--char_min_df", type=int, default=5)
    ap.add_argument("--char_max_features", type=int, default=50_000)
    ap.add_argument("--char_weight", type=float, default=1.0)

    ap.add_argument("--umap_neighbors", type=int, default=40)
    ap.add_argument("--umap_min_dist", type=float, default=0.05)
    ap.add_argument("--umap_components", type=int, default=7)
    ap.add_argument("--umap_metric", default="euclidean")
    ap.add_argument("--random_state", type=int, default=42)

    ap.add_argument("--min_cluster_size", type=int, default=80)
    ap.add_argument("--min_samples", type=int, default=10)
    ap.add_argument("--cluster_selection_epsilon", type=float, default=0.0)
    ap.add_argument("--hdbscan_selection_method", default="leaf", choices=["eom", "leaf"])

    ap.add_argument("--two_pass", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--refine_min_docs", type=int, default=60)

    ap.add_argument("--samples_per_cluster", type=int, default=12)
    ap.add_argument("--preview_chars", type=int, default=240)

    ap.add_argument("--write_umap", action=argparse.BooleanOptionalAction, default=True)

    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    ensure_dir(out_dir)

    print(f"[LOAD] {args.manifest_csv}")
    df = pd.read_csv(Path(args.manifest_csv), compression="gzip", low_memory=False)
    print(f"[LOAD] rows={len(df):,} cols={len(df.columns)}")

    print(f"[LOAD] {args.previews_jsonl}")
    pv = read_jsonl_gz_previews(Path(args.previews_jsonl))
    print(f"[LOAD] previews rows={len(pv):,}")

    df = df.merge(pv[["doc_id", "preview"]], on="doc_id", how="left")
    df["preview"] = df["preview"].fillna("").astype(str)

    if args.use_only_text_usable_now:
        if "text_usable_now" in df.columns:
            df_use = df[df["text_usable_now"] == True].copy()
        else:
            fam_ok = df["doc_family"].isin(["text", "pdf"]) if "doc_family" in df.columns else True
            has_text_ok = (df["has_text"] == True) if "has_text" in df.columns else True
            df_use = df[fam_ok & has_text_ok].copy()
    else:
        df_use = df.copy()

    # CRITICAL FIX: make df_use index positional 0..N-1 so X rows align
    df_use = df_use.reset_index(drop=True)

    print(f"[FILTER] Using rows={len(df_use):,} for doc-form clustering")

    text_col = args.text_col if args.text_col else None
    text_path_col = args.text_path_col if args.text_path_col else None
    text_root = Path(args.text_root) if args.text_root else None

    # extra numeric features
    extra_rows = []
    for _, row in tqdm(df_use.iterrows(), total=len(df_use), desc="Docform extra features"):
        full_text = get_doc_text(row, text_col=text_col, text_path_col=text_path_col, text_root=text_root)
        extra_rows.append(compute_extra_docform_features(full_text))
    extra_df = pd.DataFrame(extra_rows)
    for c in extra_df.columns:
        df_use[c] = extra_df[c].values

    feature_cols = [c.strip() for c in args.numeric_features.split(",") if c.strip()]
    df_use = safe_fill_numeric(df_use, feature_cols)
    X_num = build_numeric_matrix(df_use, feature_cols)

    # char ngrams
    char_vec = None
    X_char = None
    if args.use_char_ngrams:
        docform_texts = []
        for _, row in tqdm(df_use.iterrows(), total=len(df_use), desc="Docform text slices"):
            full_text = get_doc_text(row, text_col=text_col, text_path_col=text_path_col, text_root=text_root)
            s = make_docform_slice(full_text, head_chars=args.docform_head_chars, tail_chars=args.docform_tail_chars)
            docform_texts.append(s)

        X_char, char_vec = build_char_ngram_matrix(
            texts=docform_texts,
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

    X = normalize(X, norm="l2", axis=1, copy=False)
    rs = None if args.random_state == -1 else args.random_state

    print("[UMAP] computing coarse embedding…")
    emb = compute_umap(
        X=X,
        n_neighbors=args.umap_neighbors,
        min_dist=args.umap_min_dist,
        n_components=args.umap_components,
        metric=args.umap_metric,
        random_state=rs,
    )
    print(f"[UMAP] emb shape={emb.shape}")

    print("[HDBSCAN] coarse clustering…")
    res = run_hdbscan(
        emb=emb,
        min_cluster_size=args.min_cluster_size,
        min_samples=args.min_samples,
        cluster_selection_epsilon=args.cluster_selection_epsilon,
        selection_method=args.hdbscan_selection_method,
    )
    coarse_labels = res["labels"]
    coarse_probs = res["probs"]

    df_use["docform_coarse_raw"] = coarse_labels
    df_use["docform_coarse"] = make_cluster_id(coarse_labels, prefix="c")
    df_use["docform_coarse_prob"] = coarse_probs

    n_noise = int((coarse_labels == -1).sum())
    n_clusters = len(set(coarse_labels)) - (1 if -1 in set(coarse_labels) else 0)
    print(f"[HDBSCAN] coarse clusters={n_clusters} coarse_noise={n_noise} used={len(df_use):,}")

    if args.use_char_ngrams and X_char is not None and char_vec is not None:
        try:
            cg = top_chargrams_by_cluster(X_char, coarse_labels, char_vec, topk=30, min_docs=30)
            cg_path = out_dir / "docform_cluster_chargrams.csv"
            cg.to_csv(cg_path, index=False)
            print(f"[WRITE] {cg_path}")

            cg_map = {int(r["docform_coarse_raw"]): r["top_chargrams"] for _, r in cg.iterrows()}
            df_use["docform_coarse_chargrams"] = df_use["docform_coarse_raw"].apply(
                lambda x: "" if int(x) == -1 else cg_map.get(int(x), "")
            )
        except Exception as e:
            print(f"[WARN] failed to compute coarse chargram hints: {e}")
            df_use["docform_coarse_chargrams"] = ""
    else:
        df_use["docform_coarse_chargrams"] = ""

    # refinement
    if args.two_pass:
        print("[REFINE] two-pass refinement enabled")
        refined = np.full(len(df_use), fill_value="noise", dtype=object)
        refined_prob = np.full(len(df_use), fill_value=np.nan, dtype=float)

        # IMPORTANT: groups now return positional indices 0..N-1 because we reset_index(drop=True)
        groups = df_use.groupby("docform_coarse", dropna=False).groups

        for coarse_id, idx in tqdm(groups.items(), desc="Refining coarse clusters"):
            idx = np.array(list(idx), dtype=int)
            if coarse_id == "noise":
                continue

            n = len(idx)
            if n < args.refine_min_docs:
                refined[idx] = f"{coarse_id}_r0000"
                refined_prob[idx] = df_use.loc[idx, "docform_coarse_prob"].values
                continue

            mcs, ms, nn = adaptive_params(n)

            emb2 = compute_umap(
                X=X[idx],
                n_neighbors=nn,
                min_dist=args.umap_min_dist,
                n_components=args.umap_components,
                metric=args.umap_metric,
                random_state=rs,
            )
            res2 = run_hdbscan(
                emb=emb2,
                min_cluster_size=mcs,
                min_samples=ms,
                cluster_selection_epsilon=args.cluster_selection_epsilon,
                selection_method=args.hdbscan_selection_method,
            )
            lb2 = res2["labels"]
            pr2 = res2["probs"]

            for j, (lb, pr) in enumerate(zip(lb2, pr2)):
                if lb == -1:
                    refined[idx[j]] = f"{coarse_id}_r0000"  # fallback bucket inside family
                    refined_prob[idx[j]] = pr
                else:
                    refined[idx[j]] = f"{coarse_id}_r{int(lb):04d}"
                    refined_prob[idx[j]] = pr

        df_use["docform_label_raw"] = refined
        df_use["docform_cluster"] = refined
        df_use["docform_cluster_prob"] = refined_prob
    else:
        df_use["docform_label_raw"] = df_use["docform_coarse_raw"]
        df_use["docform_cluster"] = df_use["docform_coarse"]
        df_use["docform_cluster_prob"] = df_use["docform_coarse_prob"]

    # writes
    if args.write_umap:
        umap_df = df_use[["doc_id", "sha256", "rel_path", "docform_cluster"]].copy()
        for k in range(emb.shape[1]):
            umap_df[f"umap_{k}"] = emb[:, k]
        umap_path = out_dir / "umap_embedding.csv"
        umap_df.to_csv(umap_path, index=False)
        print(f"[WRITE] {umap_path}")

    # merge back to full manifest
    df_out = df.merge(
        df_use[["doc_id", "docform_cluster", "docform_cluster_prob", "docform_label_raw", "docform_coarse"]],
        on="doc_id",
        how="left"
    )
    out_manifest = out_dir / "manifest_with_docform_clusters.csv.gz"
    df_out.to_csv(out_manifest, index=False, compression="gzip")
    print(f"[WRITE] {out_manifest}")

    means = top_stats_by_cluster(df_use, feature_cols)
    means_path = out_dir / "docform_cluster_feature_means.csv"
    means.to_csv(means_path, index=False)
    print(f"[WRITE] {means_path}")

    build_labeling_sheet(
        df=df_use,
        out_dir=out_dir,
        sample_per_cluster=args.samples_per_cluster,
        preview_chars=args.preview_chars,
        include_chargrams_col="docform_coarse_chargrams"
    )
    print(f"[WRITE] {out_dir / 'docform_labeling_sheet.csv'}")
    print(f"[WRITE] {out_dir / 'docform_cluster_samples.tsv'}")

    print("[TOP CLUSTERS]")
    print(df_use["docform_cluster"].value_counts().head(20).to_string())


if __name__ == "__main__":
    main()
