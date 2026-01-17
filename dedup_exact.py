#!/usr/bin/env python3
"""
Exact deduplication by content hash (sha256) with provenance retention.

Reads:
  outputs/docform/manifest_with_docform_clusters.csv.gz

Writes:
  outputs/dedup/manifest_dedup.csv.gz
  outputs/dedup/duplicates_map.csv.gz
"""

import argparse
from pathlib import Path
import pandas as pd

def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)

def pick_canonical(group: pd.DataFrame) -> pd.Series:
    # Deterministic: prefer usable text, then more text, then stable path
    g = group.copy()

    def fam_rank(x):
        if x == "text": return 0
        if x == "pdf":  return 1
        if x == "image":return 2
        return 3

    g["fam_rank"] = g["doc_family"].map(fam_rank).fillna(9).astype(int)
    g["text_rank"] = (g.get("text_usable_now", False) == True).astype(int)
    g["n_chars_rank"] = pd.to_numeric(g.get("n_chars", 0), errors="coerce").fillna(0).astype(int)

    g = g.sort_values(
        ["text_rank", "fam_rank", "n_chars_rank", "rel_path"],
        ascending=[False, True, False, True]
    )
    return g.iloc[0]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_manifest", default="outputs/docform/manifest_with_docform_clusters.csv.gz")
    ap.add_argument("--out_dir", default="outputs/dedup")
    ap.add_argument("--paths_sample_n", type=int, default=5)
    args = ap.parse_args()

    in_path = Path(args.in_manifest)
    out_dir = Path(args.out_dir)
    ensure_dir(out_dir)

    df = pd.read_csv(in_path, compression="gzip", low_memory=False)
    if "sha256" not in df.columns:
        raise ValueError("Expected sha256 column in manifest.")

    # Map: sha256 -> all paths / counts
    dup = (
        df.groupby("sha256", dropna=False)
          .agg(
              dup_count=("rel_path", "size"),
              rel_paths=("rel_path", lambda x: list(x)),
              source_roots=("source_root", lambda x: sorted(set(x))),
              doc_families=("doc_family", lambda x: sorted(set(x.dropna().astype(str)))),
          )
          .reset_index()
    )
    dup["dup_paths_sample"] = dup["rel_paths"].apply(lambda xs: xs[:args.paths_sample_n])
    dup["source_root_count"] = dup["source_roots"].apply(len)

    dup_map_path = out_dir / "duplicates_map.csv.gz"
    dup.drop(columns=["rel_paths"]).to_csv(dup_map_path, index=False, compression="gzip")

    # Canonical: one row per sha256
    canon_rows = []
    for sha, g in df.groupby("sha256", dropna=False):
        canon_rows.append(pick_canonical(g))

    canon = pd.DataFrame(canon_rows).copy()

    # Join dup stats
    canon = canon.merge(
        dup[["sha256", "dup_count", "dup_paths_sample", "source_root_count", "doc_families"]],
        on="sha256",
        how="left"
    )

    out_path = out_dir / "manifest_dedup.csv.gz"
    canon.to_csv(out_path, index=False, compression="gzip")

    print(f"[DONE] canonical rows: {len(canon):,}")
    print(f"[DONE] wrote: {out_path}")
    print(f"[DONE] wrote: {dup_map_path}")
    print("[TOP] most duplicated:")
    print(canon.sort_values("dup_count", ascending=False)[["sha256","dup_count","rel_path","doc_family","n_chars"]].head(15).to_string(index=False))

if __name__ == "__main__":
    main()
