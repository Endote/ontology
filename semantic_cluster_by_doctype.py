#!/usr/bin/env python3
import argparse, gzip, json, re
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from sklearn.decomposition import TruncatedSVD

import umap
import hdbscan

from scipy import sparse


# ============================
# GLOBAL CONFIG (no long CLI)
# ============================
CONFIG = {
    # --- IO ---
    "CORPUS": "outputs/near_dedup/corpus_canonical.csv.gz",
    "PREVIEW_JSONL_GZ": "manifest_all_with_preview.jsonl.gz",
    "OUT_DIR": "outputs/semantic",

    # --- Filtering ---
    "MIN_DOCS_PER_TYPE": 30,
    "MIN_TOKENS": 30,

    # --- Text cleanup controls ---
    "STOPWORDS_TXT": "stopwords.txt",          # set None to disable
    "DROP_PHRASES_TXT": "drop_phrases.txt",    # set None to disable
    "DROP_PHRASES_REGEX": True,                # patterns in file are regex

    # --- TF-IDF ---
    "MAX_FEATURES": 60000,
    "NGRAM_MIN": 1,
    "NGRAM_MAX": 2,
    "MIN_DF": 2,
    "MAX_DF": 0.85,
    "SUBLINEAR_TF": True,

    # --- SVD -> UMAP ---
    "SVD_COMPONENTS": 1000,
    "UMAP_DIM": 10,
    "UMAP_NEIGHBORS": 20,
    "UMAP_METRIC": "cosine",
    "RANDOM_STATE": 42,

    # --- HDBSCAN ---
    "MIN_CLUSTER_SIZE": 20,
    "MIN_SAMPLES": 7,

    # --- Topic keywords export ---
    "TOPN_KEYWORDS": 15,
    "MIN_DOCS_KEYWORDS": 5,
}

### Fix options (for chaotic noise):

# lower min_samples (less conservative), or
# increase umap_neighbors, or
# increase UMAP_DIM slightly, or
# split that doc_type further by docform_cluster before semantic clustering (often helps a lot)


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
# Cleaning per doc type
# ----------------------------
# RE_EMAIL_HDR = re.compile(r"^(from|to|cc|bcc|sent|date|subject|attachments|importance)\s*:\s*", re.I)
RE_EMAIL_HDR = re.compile(r"^[^\w]{0,3}(from|to|cc|bcc|sent|date|subject|attachments|importance)\s*:\s*", re.I)


RE_QP = re.compile(r"=([0-9A-F]{2})", re.I)
RE_TS_LINE = re.compile(r"^\s*\d{1,2}/\d{1,2}/\d{2,4}\s+\d{1,2}:\d{2}(:\d{2})?\s*(AM|PM)?\s*$", re.I)

# --- Date/time normalization ---
RE_YEAR = re.compile(r"\b(19\d{2}|20\d{2})\b")
RE_TIME = re.compile(r"\b\d{1,2}:\d{2}(?::\d{2})?\b")  # 8:43, 08:43, 08:43:21

# Common date formats:
# 2019-12-05, 2019/12/05, 12/05/2019, 5/12/19, etc.
RE_DATE_NUMERIC = re.compile(
    r"\b("
    r"(?:19\d{2}|20\d{2})[/-](?:0?[1-9]|1[0-2])[/-](?:0?[1-9]|[12]\d|3[01])"
    r"|(?:0?[1-9]|1[0-2])[/-](?:0?[1-9]|[12]\d|3[01])[/-](?:19\d{2}|20\d{2}|\d{2})"
    r")\b"
)

# Month name dates: "Nov 10 2016", "10 Nov 2016", "November 10, 2016"
RE_DATE_MONTHNAME = re.compile(
    r"\b(?:"
    r"(?:jan|january|feb|february|mar|march|apr|april|may|jun|june|jul|july|aug|august|sep|sept|september|oct|october|nov|november|dec|december)"
    r")\s+\d{1,2}(?:st|nd|rd|th)?(?:,\s*)?(?:19\d{2}|20\d{2})?\b"
    r"|\b\d{1,2}(?:st|nd|rd|th)?\s+(?:"
    r"jan|january|feb|february|mar|march|apr|april|may|jun|june|jul|july|aug|august|sep|sept|september|oct|october|nov|november|dec|december"
    r")(?:\s+(?:19\d{2}|20\d{2}))?\b",
    flags=re.I
)

RE_FORWARD_SEP = re.compile(r"^-{2,}\s*(original message|forwarded message)\s*-{2,}$", re.I)
RE_BEGIN_FWD  = re.compile(r"^begin forwarded message\s*:\s*$", re.I)
RE_IPHONE_SIG = re.compile(r"^sent from my (iphone|ipad)\b", re.I)
RE_ON_BEHALF  = re.compile(r"^\s*on behalf of\b", re.I)



def load_wordlist(path: str | None, *, lower: bool = True) -> list[str]:
    if not path:
        return []
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(p)

    out = []
    for ln in p.read_text(encoding="utf-8").splitlines():
        ln = ln.strip()
        if not ln or ln.startswith("#"):
            continue
        out.append(ln.lower() if lower else ln)
    return out


def compile_phrase_regex(phrases: list[str], as_regex: bool, *, flags: int) -> re.Pattern | None:
    """
    Compile a single regex that matches any phrase in `phrases`.

    If as_regex=False: treat phrases literally (tokenized), whitespace flexed with \\s+, bounded by \\b.
    If as_regex=True : treat each line as raw regex fragment.
    """
    if not phrases:
        return None

    if as_regex:
        pat = "|".join(f"(?:{p})" for p in phrases)
    else:
        esc = []
        for p in phrases:
            toks = [re.escape(t) for t in p.split()]
            if not toks:
                continue
            esc.append(r"\b" + r"\s+".join(toks) + r"\b")
        pat = "|".join(esc)

    if not pat:
        return None
    return re.compile(pat, flags=flags)


def split_drop_patterns(phrases: list[str], as_regex: bool) -> tuple[list[str], list[str]]:
    """
    Split patterns into:
      - line_patterns: patterns intended to apply to individual lines (typically start with '^')
      - text_patterns: patterns intended to apply to full text after joining lines
    """
    if not phrases:
        return [], []

    if not as_regex:
        return [], phrases

    line_pats = []
    text_pats = []
    for p in phrases:
        if p.lstrip().startswith("^"):
            line_pats.append(p)
        else:
            text_pats.append(p)
    return line_pats, text_pats


def clean_text(
    doc_type: str,
    text: str,
    *,
    drop_line_re: re.Pattern | None = None,
    drop_text_re: re.Pattern | None = None,
) -> str:
    if not isinstance(doc_type, str) or not doc_type.strip():
        doc_type = "unknown"
    if not isinstance(text, str):
        text = ""

    t = text.replace("\r", "\n")
    t = re.sub(r"Non-Responsive\s*-\s*Redacted", " ", t, flags=re.I)
    t = re.sub(r"Privileged\s*-\s*Redacted", " ", t, flags=re.I)
    # --- normalize time/date/year early (before line splitting) ---
    t = RE_TIME.sub(" __TIME__ ", t)
    t = RE_DATE_NUMERIC.sub(" __DATE__ ", t)
    t = RE_DATE_MONTHNAME.sub(" __DATE__ ", t)
    t = RE_YEAR.sub(" __YEAR__ ", t)


    raw_lines = [ln.rstrip("\n") for ln in t.splitlines()]

    lines = []
    for ln in raw_lines:
        s = ln.replace("\ufeff", "").strip()
        s = re.sub(r"[\u00A0\u2007\u202F]", " ", s).strip()  # nbsp variants
        
        if not s:
            continue

        if RE_EMAIL_HDR.match(s):
            continue

        if RE_FORWARD_SEP.match(s) or RE_BEGIN_FWD.match(s):
            continue
        if RE_IPHONE_SIG.match(s) or RE_ON_BEHALF.match(s):
            continue

        if drop_line_re is not None and drop_line_re.search(s):
            continue

        lines.append(s)

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
        t2 = " ".join(lines)
        t2 = RE_QP.sub(" ", t2)

    else:
        t2 = " ".join(lines)

    t2 = re.sub(r"\s+", " ", t2).strip()

    if drop_text_re is not None and t2:
        t2 = drop_text_re.sub(" ", t2)
        t2 = re.sub(r"\s+", " ", t2).strip()

    return t2


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


def top_keywords_per_cluster(X, labels, feature_names, topn: int = 15, min_docs: int = 5) -> dict:
    labels = np.asarray(labels)
    out = {}

    if not sparse.isspmatrix_csr(X):
        X = X.tocsr()

    for cid in sorted(set(labels.tolist())):
        if cid == -1:
            continue
        idx = np.where(labels == cid)[0]
        if len(idx) < min_docs:
            continue

        Xc = X[idx]
        mean_vec = Xc.mean(axis=0)
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


def main():
    ap = argparse.ArgumentParser(description="Semantic clustering per doc_type using TF-IDF -> SVD -> UMAP -> HDBSCAN")

    ap.add_argument("--corpus", default=CONFIG["CORPUS"])
    ap.add_argument("--preview_jsonl_gz", default=CONFIG["PREVIEW_JSONL_GZ"])
    ap.add_argument("--out_dir", default=CONFIG["OUT_DIR"])

    ap.add_argument("--min_docs", type=int, default=CONFIG["MIN_DOCS_PER_TYPE"])
    ap.add_argument("--min_tokens", type=int, default=CONFIG["MIN_TOKENS"])

    ap.add_argument("--stopwords_txt", default=CONFIG["STOPWORDS_TXT"])
    ap.add_argument("--drop_phrases_txt", default=CONFIG["DROP_PHRASES_TXT"])
    ap.add_argument(
        "--drop_phrases_regex",
        dest="drop_phrases_regex",
        action=argparse.BooleanOptionalAction,
        default=CONFIG["DROP_PHRASES_REGEX"],
    )

    ap.add_argument("--max_features", type=int, default=CONFIG["MAX_FEATURES"])
    ap.add_argument("--ngram_min", type=int, default=CONFIG["NGRAM_MIN"])
    ap.add_argument("--ngram_max", type=int, default=CONFIG["NGRAM_MAX"])
    ap.add_argument("--min_df", type=int, default=CONFIG["MIN_DF"])
    ap.add_argument("--max_df", type=float, default=CONFIG["MAX_DF"])
    ap.add_argument("--sublinear_tf", action="store_true", default=CONFIG["SUBLINEAR_TF"])

    ap.add_argument("--svd_components", type=int, default=CONFIG["SVD_COMPONENTS"])
    ap.add_argument("--umap_dim", type=int, default=CONFIG["UMAP_DIM"])
    ap.add_argument("--umap_neighbors", type=int, default=CONFIG["UMAP_NEIGHBORS"])
    ap.add_argument("--umap_metric", default=CONFIG["UMAP_METRIC"])
    ap.add_argument("--random_state", type=int, default=CONFIG["RANDOM_STATE"])

    ap.add_argument("--min_cluster_size", type=int, default=CONFIG["MIN_CLUSTER_SIZE"])
    ap.add_argument("--min_samples", type=int, default=CONFIG["MIN_SAMPLES"])

    ap.add_argument("--topn_keywords", type=int, default=CONFIG["TOPN_KEYWORDS"])
    ap.add_argument("--min_docs_keywords", type=int, default=CONFIG["MIN_DOCS_KEYWORDS"])

    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.corpus, compression="gzip", low_memory=False)
    prev = load_preview_map(Path(args.preview_jsonl_gz))

    if "human_doc_type" not in df.columns:
        df["human_doc_type"] = "unknown"

    miss = df["doc_id"].map(lambda x: x not in prev).sum()
    print(f"[PREVIEW] missing previews: {miss:,}/{len(df):,}")

    custom_stop = load_wordlist(args.stopwords_txt, lower=True) if args.stopwords_txt else []
    phrases = load_wordlist(args.drop_phrases_txt, lower=True) if args.drop_phrases_txt else []

    stop = sorted(set(ENGLISH_STOP_WORDS) | set(custom_stop))

    line_pats, text_pats = split_drop_patterns(phrases, as_regex=args.drop_phrases_regex)

    drop_line_re = compile_phrase_regex(
        line_pats,
        as_regex=True,
        flags=re.I
    ) if line_pats else None

    drop_text_re = compile_phrase_regex(
        text_pats,
        as_regex=args.drop_phrases_regex,
        flags=re.I | re.MULTILINE
    ) if text_pats else None

    texts = []
    for r in tqdm(df.itertuples(index=False), total=len(df), desc="Clean text"):
        dt = getattr(r, "human_doc_type", "")
        if not isinstance(dt, str) or not dt.strip():
            dt = "unknown"
        raw = prev.get(r.doc_id, "")
        texts.append(clean_text(dt, raw, drop_line_re=drop_line_re, drop_text_re=drop_text_re))

    df["clean_text"] = texts
    df["clean_tok_len"] = df["clean_text"].str.split().map(len).fillna(0).astype(int)
    df = df[df["clean_tok_len"] >= args.min_tokens].copy()

    print(f"[FILTER] usable for semantic clustering: {len(df):,}")

    all_type_stats = []
    all_topics_rows = []  # NEW: global topic summary across all doc_types

    for doc_type, g in df.groupby("human_doc_type"):
        if not isinstance(doc_type, str) or not doc_type.strip():
            doc_type = "unknown"
        if len(g) < args.min_docs:
            continue

        type_dir = out_dir / doc_type
        type_dir.mkdir(parents=True, exist_ok=True)

        vec = TfidfVectorizer(
            max_features=args.max_features,
            ngram_range=(args.ngram_min, args.ngram_max),
            min_df=args.min_df,
            max_df=args.max_df,
            strip_accents="unicode",
            stop_words=stop,
            sublinear_tf=args.sublinear_tf,
            token_pattern=r"(?u)\b[A-Za-z_][A-Za-z_]+\b",

        )

        X = vec.fit_transform(g["clean_text"].tolist())
        feature_names = vec.get_feature_names_out()
        vocab_size = len(vec.vocabulary_)
        nnz = X.nnz
        density = nnz / (X.shape[0] * max(1, X.shape[1]))

        n_svd = min(args.svd_components, X.shape[1] - 1)
        if n_svd < 2:
            print(f"[SKIP] {doc_type} | vocab too small after filtering (vocab={X.shape[1]})")
            continue

        svd = TruncatedSVD(n_components=n_svd, random_state=args.random_state)
        X_svd = svd.fit_transform(X)

        # umap_neighbors = min(args.umap_neighbors, max(5, len(g) // 3))
        umap_neighbors = min(50, max(10, len(g)//15))
    
        emb = umap.UMAP(
            n_neighbors=umap_neighbors,
            n_components=args.umap_dim,
            metric=args.umap_metric,
            random_state=args.random_state
        ).fit_transform(X_svd)

        # clusterer = hdbscan.HDBSCAN(
        #     min_cluster_size=args.min_cluster_size,
        #     min_samples=args.min_samples,
        #     metric="euclidean"
        # )

        ### Adaptive min_cluster_size based on doc count
        n = len(g)
        min_cluster_size = max(5, int(0.05 * n))   # 5% of docs, floor at 5
        # min_samples = max(3, int(0.02 * n))
        min_samples = max(2, int(0.01*n))

        clusterer = hdbscan.HDBSCAN(
            min_cluster_size=min_cluster_size,
            min_samples=min_samples,
            metric="euclidean"
        )

        labels = clusterer.fit_predict(emb)

        min_docs_kw = max(args.min_docs_keywords, int(0.01 * len(g)), 3)
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

        # per-doc_type summary
        topic_summary_df = pd.DataFrame(
            topic_summary_rows,
            columns=["human_doc_type", "topic_cluster", "topic_size", "keywords"]
        )
        if len(topic_summary_df):
            topic_summary_df = topic_summary_df.sort_values(["topic_size"], ascending=False)
        topic_summary_df.to_csv(type_dir / "topic_summary.csv", index=False)

        # NEW: accumulate global topics
        all_topics_rows.extend(topic_summary_rows)

        base_cols = ["doc_id", "sha256", "rel_path", "docform_cluster", "human_doc_type", "clean_tok_len"]
        keep_cols = [c for c in base_cols if c in g.columns]
        out = g[keep_cols].copy()

        out["topic_cluster"] = labels
        out["topic_prob"] = clusterer.probabilities_

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

        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        noise = int((labels == -1).sum())
        all_type_stats.append({
            "human_doc_type": doc_type,
            "n_docs": int(len(g)),
            "vocab_size": int(vocab_size),
            "tfidf_nnz": int(nnz),
            "tfidf_density": float(density),
            "svd_components_used": int(n_svd),
            "umap_dim": int(args.umap_dim),
            "n_clusters": int(n_clusters),
            "noise_docs": noise,
            "clusters_with_keywords": int(len(kw)),
            "min_docs_keywords_used": int(min_docs_kw),
        })

        print(f"[TYPE] {doc_type} | docs={len(g):,} vocab={vocab_size:,} clusters={n_clusters} noise={noise}")

    # NEW: write global topics file
    all_topics_df = pd.DataFrame(
        all_topics_rows,
        columns=["human_doc_type", "topic_cluster", "topic_size", "keywords"]
    )
    if len(all_topics_df):
        all_topics_df = all_topics_df.sort_values(["human_doc_type", "topic_size"], ascending=[True, False])
    all_topics_df.to_csv(out_dir / "semantic_all_topics.csv", index=False)
    print(f"[WRITE] {out_dir/'semantic_all_topics.csv'}")

    stats_df = pd.DataFrame(all_type_stats)
    if len(stats_df):
        stats_df = stats_df.sort_values(["n_docs"], ascending=False)
    stats_df.to_csv(out_dir / "semantic_type_stats.csv", index=False)

    print(f"[WRITE] {out_dir/'semantic_type_stats.csv'}")
    print("[DONE] semantic clustering per doc_type complete.")


if __name__ == "__main__":
    main()
