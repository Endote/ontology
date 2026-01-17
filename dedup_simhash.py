# python dedup_simhash.py --manifest outputs/docform/manifest_with_doc_types.csv.gz  --preview_jsonl_gz manifest_all_with_preview.jsonl.gz


#!/usr/bin/env python3
import argparse, gzip, json, re
from pathlib import Path
from collections import defaultdict
import pandas as pd

# ----------------------------
# SimHash (remains fast + dependency-free)
# ----------------------------
def _tokenize(text: str):
    # keep it dumb + stable
    return re.findall(r"[a-z0-9]+", text.lower())

def simhash64(text: str) -> int:
    toks = _tokenize(text)
    if not toks:
        return 0
    v = [0] * 64
    for t in toks:
        h = int.from_bytes(__import__("hashlib").md5(t.encode("utf-8")).digest()[:8], "big")
        for i in range(64):
            bit = (h >> i) & 1
            v[i] += 1 if bit else -1
    out = 0
    for i in range(64):
        if v[i] > 0:
            out |= (1 << i)
    return out

def hamming(a: int, b: int) -> int:
    return (a ^ b).bit_count()

# ----------------------------
# Text normalization by doc_type
# ----------------------------
RE_EMAIL_HDR = re.compile(r"^(from|to|cc|bcc|sent|date|subject|attachments|importance)\s*:\s*", re.I)
RE_QP_GARBAGE = re.compile(r"=([0-9A-F]{2})", re.I)
RE_TS_LINE = re.compile(r"^\s*\d{1,2}/\d{1,2}/\d{2,4}\s+\d{1,2}:\d{2}(:\d{2})?\s*(AM|PM)?\s*$", re.I)

def normalize_text(doc_type: str, text: str) -> str:
    if not isinstance(doc_type, str):
        doc_type = "unknown"
    if not isinstance(text, str):
        text = ""

    t = text.replace("\r", "\n")

    # kill obvious useless blocks
    t = re.sub(r"Non-Responsive\s*-\s*Redacted", " ", t, flags=re.I)
    t = re.sub(r"Privileged\s*-\s*Redacted", " ", t, flags=re.I)

    lines = [ln.strip() for ln in t.splitlines()]
    out = []

    if doc_type.startswith("chat_ichat"):
        # iChat exports: remove metadata header lines
        for ln in lines:
            if ln.lower().startswith(("source entry:", "service:", "start time:", "end time:", "last message")):
                continue
            if RE_TS_LINE.match(ln):
                continue
            out.append(ln)

    elif doc_type.startswith("email_"):
        # remove header-ish lines; keep body
        for ln in lines:
            if RE_EMAIL_HDR.match(ln):
                continue
            out.append(ln)
        t2 = "\n".join(out)
        # quoted-printable artifacts
        t2 = RE_QP_GARBAGE.sub(" ", t2)
        t2 = re.sub(r"\s+", " ", t2).strip()
        return t2

    else:
        # general cleanup
        out = lines

    t2 = "\n".join(out)
    t2 = re.sub(r"\s+", " ", t2).strip()
    return t2

# ----------------------------
# Load preview text map
# ----------------------------
def load_preview_map(path_jsonl_gz: Path) -> dict:
    m = {}
    with gzip.open(path_jsonl_gz, "rt", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            # preview file stores doc_id and preview
            m[obj["doc_id"]] = obj.get("preview", "")
    return m

# ----------------------------
# Near-dup grouping (bucketed)
# ----------------------------
def near_dup_groups(df: pd.DataFrame, preview_map: dict, bits_prefix: int = 16, ham_max: int = 3):
    # only work on text-bearing docs
    rows = []
    for r in df.itertuples(index=False):
        txt = preview_map.get(r.doc_id, "")
        doc_type = getattr(r, "human_doc_type", "")
        if not isinstance(doc_type, str) or not doc_type.strip():
            doc_type = "unknown"
        norm = normalize_text(doc_type, txt)
        sh = simhash64(norm)
        prefix = sh >> (64 - bits_prefix) if bits_prefix < 64 else sh
        rows.append((r.doc_id, r.sha256, doc_type, sh, prefix, len(norm)))
    work = pd.DataFrame(rows, columns=["doc_id","sha256","human_doc_type","simhash64","bucket","norm_len"])

    # bucket -> list indices
    buckets = defaultdict(list)
    for i, b in enumerate(work["bucket"].tolist()):
        buckets[b].append(i)

    parent = list(range(len(work)))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a,b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    # Compare only within buckets (small)
    for b, idxs in buckets.items():
        if len(idxs) < 2:
            continue
        for i in range(len(idxs)):
            ii = idxs[i]
            for j in range(i+1, len(idxs)):
                jj = idxs[j]
                # cheap length guard
                li, lj = work.at[ii, "norm_len"], work.at[jj, "norm_len"]
                if min(li, lj) == 0:
                    continue
                if max(li, lj) / max(1, min(li, lj)) > 2.5:
                    continue
                if hamming(int(work.at[ii,"simhash64"]), int(work.at[jj,"simhash64"])) <= ham_max:
                    union(ii, jj)

    # assign group ids
    gid = {}
    for i in range(len(work)):
        root = find(i)
        gid[i] = root
    work["near_dup_group_raw"] = work.index.map(gid)

    # compress group ids
    roots = {r:i for i, r in enumerate(sorted(set(work["near_dup_group_raw"].tolist())))}
    work["near_dup_group"] = work["near_dup_group_raw"].map(roots).apply(lambda x: f"g{x:06d}")
    work.drop(columns=["near_dup_group_raw"], inplace=True)

    # group stats
    gstats = (
        work.groupby("near_dup_group")
            .agg(group_size=("doc_id","size"),
                 doc_type=("human_doc_type","first"))
            .reset_index()
    )

    return work, gstats

def pick_canonical_per_group(work: pd.DataFrame) -> pd.DataFrame:
    # pick canonical doc_id: longest normalized text (proxy for best body)
    w = work.copy()
    w["rank_len"] = w["norm_len"].fillna(0).astype(int)
    w = w.sort_values(["near_dup_group","rank_len"], ascending=[True, False])
    canon = w.groupby("near_dup_group").head(1)[["near_dup_group","doc_id"]].rename(columns={"doc_id":"canonical_doc_id"})
    return canon

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default="manifest_with_doc_types.csv.gz")
    ap.add_argument("--preview_jsonl_gz", default="manifest_all_with_preview.jsonl.gz")
    ap.add_argument("--out_dir", default="outputs/near_dedup")
    ap.add_argument("--bits_prefix", type=int, default=16)
    ap.add_argument("--ham_max", type=int, default=3)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.manifest, compression="gzip", low_memory=False)
    prev = load_preview_map(Path(args.preview_jsonl_gz))

    # Only text-bearing families for now
    df2 = df[df["doc_family"].isin(["text","pdf"])].copy()

    ### SANITY CHECK FOR MISSING PREVIEWS ### 
    missing_preview = df2["doc_id"].map(lambda x: x not in prev).sum()
    print(f"[PREVIEW] missing previews for {missing_preview:,}/{len(df2):,} docs")
    ###

    work, gstats = near_dup_groups(df2, prev, bits_prefix=args.bits_prefix, ham_max=args.ham_max)
    canon = pick_canonical_per_group(work)

    # attach group id back to manifest
    mf3 = df.merge(work[["doc_id","near_dup_group"]], on="doc_id", how="left")
    mf3 = mf3.merge(canon, on="near_dup_group", how="left")

    # corpus canonical: one per group
    corpus = mf3[mf3["doc_id"] == mf3["canonical_doc_id"]].copy()
    corpus = corpus.sort_values(["human_doc_type","docform_cluster","doc_id"])

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
