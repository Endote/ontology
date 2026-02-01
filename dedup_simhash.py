#!/usr/bin/env python3

#run
# python dedup_simhash.py  --manifest outputs/docform/manifest_with_docform_clusters.csv.gz  --preview_jsonl_gz manifest_all_with_preview.jsonl.gz  --text_path_col text_path  --text_root .  --out_dir outputs/near_dedup
"""
Near-duplicate grouping using SimHash (64-bit), with docform-aware normalization.

Works with:
  - manifest_with_docform_clusters.csv.gz (has docform_cluster, doc_id, sha256, etc.)
  - preview JSONL.GZ (doc_id -> preview)
  - optional full text via --text_path_col + --text_root (recommended)

Outputs (under --out_dir):
  - near_dup_groups.csv.gz
  - near_dup_group_stats.csv
  - manifest_with_near_dedup.csv.gz
  - corpus_canonical.csv.gz
"""

import argparse, gzip, json, re
from pathlib import Path
from collections import defaultdict
from typing import Dict, Optional, Tuple

import pandas as pd


# ----------------------------
# SimHash (fast + dependency-free)
# ----------------------------
def _tokenize(text: str):
    return re.findall(r"[a-z0-9]+", (text or "").lower())


def simhash64(text: str) -> int:
    toks = _tokenize(text)
    if not toks:
        return 0
    v = [0] * 64
    import hashlib
    for t in toks:
        h = int.from_bytes(hashlib.md5(t.encode("utf-8")).digest()[:8], "big")
        for i in range(64):
            v[i] += 1 if ((h >> i) & 1) else -1
    out = 0
    for i in range(64):
        if v[i] > 0:
            out |= (1 << i)
    return out


def hamming(a: int, b: int) -> int:
    return (a ^ b).bit_count()


# ----------------------------
# Text normalization (docform-aware)
# ----------------------------
RE_EMAIL_HDR = re.compile(r"^(from|to|cc|bcc|sent|date|subject|attachments|importance)\s*:\s*", re.I)
RE_QP_GARBAGE = re.compile(r"=([0-9A-F]{2})", re.I)
RE_TS_LINE = re.compile(r"^\s*\d{1,2}/\d{1,2}/\d{2,4}\s+\d{1,2}:\d{2}(:\d{2})?\s*(AM|PM)?\s*$", re.I)


def normalize_text(docform_cluster: str, text: str) -> str:
    """
    Uses docform_cluster as the "type" router.
    """
    if not isinstance(docform_cluster, str) or not docform_cluster.strip():
        docform_cluster = "unknown"
    t = (text or "").replace("\r", "\n")

    # kill obvious useless blocks (common in your data)
    t = re.sub(r"Non-Responsive\s*-\s*Redacted", " ", t, flags=re.I)
    t = re.sub(r"Privileged\s*-\s*Redacted", " ", t, flags=re.I)

    lines = [ln.strip() for ln in t.splitlines()]
    out = []

    # iChat exports cluster tends to look like "Source Entry: ... Service: iMessage ..."
    if "source entry:" in t.lower() or "service:" in t.lower():
        for ln in lines:
            l = ln.lower()
            if l.startswith(("source entry:", "service:", "start time:", "end time:", "last message")):
                continue
            if RE_TS_LINE.match(ln):
                continue
            out.append(ln)
        t2 = "\n".join(out)
        t2 = re.sub(r"\s+", " ", t2).strip()
        return t2

    # email-like: remove header lines (From:, Sent:, To:, Subject:, Attachments:, Importance:)
    # You have several email-ish docform clusters; don't hardcode ids, detect by header density.
    hdr_hits = sum(1 for ln in lines[:40] if RE_EMAIL_HDR.match(ln or "") is not None)
    if hdr_hits >= 3:
        for ln in lines:
            if RE_EMAIL_HDR.match(ln):
                continue
            out.append(ln)
        t2 = "\n".join(out)
        t2 = RE_QP_GARBAGE.sub(" ", t2)
        t2 = re.sub(r"\s+", " ", t2).strip()
        return t2

    # general cleanup
    t2 = " ".join([ln for ln in lines if ln])
    t2 = re.sub(r"\s+", " ", t2).strip()
    return t2


def make_slice(text: str, head_chars: int = 4000, tail_chars: int = 4000) -> str:
    t = text or ""
    if len(t) <= head_chars + tail_chars + 50:
        return t
    return t[:head_chars] + "\n...\n" + t[-tail_chars:]


# ----------------------------
# Load preview text map
# ----------------------------
def load_preview_map(path_jsonl_gz: Path) -> Dict[str, str]:
    m = {}
    with gzip.open(path_jsonl_gz, "rt", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)
            did = obj.get("doc_id")
            if did:
                m[did] = obj.get("preview", "") or ""
    return m


# ----------------------------
# Optional full text retrieval
# ----------------------------
def read_text_from_path(text_root: Path, rel_or_abs: str) -> str:
    try:
        p = Path(rel_or_abs)
        if not p.is_absolute():
            p = text_root / p
        if p.exists() and p.is_file():
            return p.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        pass
    return ""


def get_best_text(
    row: pd.Series,
    preview_map: Dict[str, str],
    *,
    text_path_col: Optional[str],
    text_root: Path,
    use_docform_slice: bool,
    head_chars: int,
    tail_chars: int,
) -> str:
    # 1) full text from file
    if text_path_col and text_path_col in row and isinstance(row[text_path_col], str) and row[text_path_col].strip():
        t = read_text_from_path(text_root, row[text_path_col])
        if t.strip():
            return make_slice(t, head_chars=head_chars, tail_chars=tail_chars) if use_docform_slice else t

    # 2) preview
    did = row.get("doc_id", "")
    t = preview_map.get(did, "")
    return make_slice(t, head_chars=head_chars, tail_chars=tail_chars) if use_docform_slice else t


# ----------------------------
# Near-dup grouping (bucketed)
# ----------------------------
def near_dup_groups(
    df: pd.DataFrame,
    preview_map: Dict[str, str],
    *,
    text_path_col: Optional[str],
    text_root: Path,
    bits_prefix: int = 16,
    ham_max: int = 3,
    use_docform_slice: bool = True,
    head_chars: int = 4000,
    tail_chars: int = 4000,
):
    rows = []
    for r in df.itertuples(index=False):
        row = pd.Series(r._asdict())
        docform = str(row.get("docform_cluster", ""))  # may be nan -> "nan"
        txt = get_best_text(
            row, preview_map,
            text_path_col=text_path_col,
            text_root=text_root,
            use_docform_slice=use_docform_slice,
            head_chars=head_chars,
            tail_chars=tail_chars,
        )
        norm = normalize_text(docform, txt)
        sh = simhash64(norm)
        prefix = sh >> (64 - bits_prefix) if bits_prefix < 64 else sh
        rows.append((row.get("doc_id"), row.get("sha256"), docform, sh, prefix, len(norm)))

    work = pd.DataFrame(rows, columns=["doc_id","sha256","docform_cluster","simhash64","bucket","norm_len"])

    buckets = defaultdict(list)
    for i, b in enumerate(work["bucket"].tolist()):
        buckets[b].append(i)

    parent = list(range(len(work)))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    for b, idxs in buckets.items():
        if len(idxs) < 2:
            continue
        for i in range(len(idxs)):
            ii = idxs[i]
            for j in range(i + 1, len(idxs)):
                jj = idxs[j]
                li, lj = int(work.at[ii, "norm_len"]), int(work.at[jj, "norm_len"])
                if min(li, lj) == 0:
                    continue
                if max(li, lj) / max(1, min(li, lj)) > 2.5:
                    continue
                if hamming(int(work.at[ii, "simhash64"]), int(work.at[jj, "simhash64"])) <= ham_max:
                    union(ii, jj)

    # assign group ids
    roots = [find(i) for i in range(len(work))]
    work["near_dup_group_raw"] = roots
    root_map = {r:i for i, r in enumerate(sorted(set(roots)))}
    work["near_dup_group"] = work["near_dup_group_raw"].map(root_map).apply(lambda x: f"g{x:06d}")
    work.drop(columns=["near_dup_group_raw"], inplace=True)

    gstats = (
        work.groupby("near_dup_group")
            .agg(group_size=("doc_id","size"),
                 docform_cluster=("docform_cluster","first"))
            .reset_index()
            .sort_values("group_size", ascending=False)
    )

    return work, gstats


def pick_canonical_per_group(work: pd.DataFrame) -> pd.DataFrame:
    # canonical = longest normalized text
    w = work.copy()
    w["rank_len"] = w["norm_len"].fillna(0).astype(int)
    w = w.sort_values(["near_dup_group","rank_len"], ascending=[True, False])
    canon = (
        w.groupby("near_dup_group")
         .head(1)[["near_dup_group","doc_id"]]
         .rename(columns={"doc_id":"canonical_doc_id"})
    )
    return canon


def safe_sort(df: pd.DataFrame, cols):
    cols2 = [c for c in cols if c in df.columns]
    return df.sort_values(cols2) if cols2 else df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default="outputs/docform/manifest_with_docform_clusters.csv.gz")
    ap.add_argument("--preview_jsonl_gz", default="manifest_all_with_preview.jsonl.gz")
    ap.add_argument("--out_dir", default="outputs/near_dedup")

    # NEW: full text support
    ap.add_argument("--text_path_col", default=None, help="Column in manifest with path to full extracted text.")
    ap.add_argument("--text_root", default=".", help="Root dir for relative text paths.")

    # slicing (recommended)
    ap.add_argument("--use_docform_slice", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--head_chars", type=int, default=4000)
    ap.add_argument("--tail_chars", type=int, default=4000)

    # simhash params
    ap.add_argument("--bits_prefix", type=int, default=16)
    ap.add_argument("--ham_max", type=int, default=3)

    # filter
    ap.add_argument("--doc_families", default="text,pdf",
                    help="Comma-separated doc_family values to include (default: text,pdf).")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.manifest, compression="gzip", low_memory=False)
    prev = load_preview_map(Path(args.preview_jsonl_gz))

    fams = [x.strip() for x in args.doc_families.split(",") if x.strip()]
    if "doc_family" in df.columns and fams:
        df2 = df[df["doc_family"].isin(fams)].copy()
    else:
        df2 = df.copy()

    missing_preview = df2["doc_id"].map(lambda x: x not in prev).sum()
    print(f"[PREVIEW] missing previews for {missing_preview:,}/{len(df2):,} docs")

    work, gstats = near_dup_groups(
        df2, prev,
        text_path_col=args.text_path_col,
        text_root=Path(args.text_root),
        bits_prefix=args.bits_prefix,
        ham_max=args.ham_max,
        use_docform_slice=args.use_docform_slice,
        head_chars=args.head_chars,
        tail_chars=args.tail_chars,
    )
    canon = pick_canonical_per_group(work)

    mf3 = df.merge(work[["doc_id","near_dup_group"]], on="doc_id", how="left")
    mf3 = mf3.merge(canon, on="near_dup_group", how="left")

    corpus = mf3[mf3["doc_id"] == mf3["canonical_doc_id"]].copy()

    dup = int(corpus["doc_id"].duplicated().sum())
    if dup:
        corpus = corpus.sort_values([c for c in ["doc_family","docform_cluster","rel_path","doc_id"] if c in corpus.columns])
        corpus = corpus.drop_duplicates(subset=["doc_id"], keep="first").copy()
        print(f"[CANON] dropped duplicated doc_id inside corpus_canonical: {dup:,}")

    corpus = safe_sort(corpus, ["docform_cluster", "doc_family", "doc_id"])

    work.to_csv(out_dir / "near_dup_groups.csv.gz", index=False, compression="gzip")
    gstats.to_csv(out_dir / "near_dup_group_stats.csv", index=False)
    mf3.to_csv(out_dir / "manifest_with_near_dedup.csv.gz", index=False, compression="gzip")
    corpus.to_csv(out_dir / "corpus_canonical.csv.gz", index=False, compression="gzip")

    print(f"[DONE] groups: {work['near_dup_group'].nunique():,} | docs: {len(work):,}")
    print(f"[WRITE] {out_dir/'near_dup_groups.csv.gz'}")
    print(f"[WRITE] {out_dir/'manifest_with_near_dedup.csv.gz'}")
    print(f"[WRITE] {out_dir/'corpus_canonical.csv.gz'}")


if __name__ == "__main__":
    main()
