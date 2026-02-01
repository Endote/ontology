#!/usr/bin/env python3
# run:
# python semantic_cluster_global.py \
#   --corpus outputs/near_dedup/corpus_canonical.csv.gz \
#   --preview_jsonl_gz manifest_all_with_preview.jsonl.gz \
#   --out_dir outputs/semantic_global \
#   --min_tokens 30
#
# Optional: rebalance form kinds so email_generic doesn't dominate:
# python semantic_cluster_global.py ... --rebalance_form_kinds

import argparse, gzip, json, re
from pathlib import Path
from collections import Counter

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
# GLOBAL CONFIG
# ============================
CONFIG = {
    "CORPUS": "outputs/near_dedup/corpus_canonical.csv.gz",
    "PREVIEW_JSONL_GZ": "manifest_all_with_preview.jsonl.gz",
    "OUT_DIR": "outputs/semantic_global",

    # filtering
    "MIN_TOKENS": 30,

    # text cleanup controls
    "STOPWORDS_TXT": "stopwords.txt",
    "DROP_PHRASES_TXT": "drop_phrases.txt",
    "DROP_PHRASES_REGEX": True,

    # TF-IDF
    "MAX_FEATURES": 60000,
    "NGRAM_MIN": 1,
    "NGRAM_MAX": 2,
    "MIN_DF": 2,
    "MAX_DF": 0.85,
    "SUBLINEAR_TF": True,

    # SVD -> UMAP
    "SVD_COMPONENTS": 1000,
    "UMAP_DIM": 10,
    "UMAP_NEIGHBORS": 30,   # a bit higher globally
    "UMAP_METRIC": "cosine",
    "RANDOM_STATE": 42,

    # HDBSCAN (global defaults)
    "MIN_CLUSTER_SIZE": 30, # global clusters should be larger by default
    "MIN_SAMPLES": 10,

    # topic keywords export
    "TOPN_KEYWORDS": 15,
    "MIN_DOCS_KEYWORDS": 7,

    # sanity
    "MAX_DOCS": None,  # set e.g. 50000 for quick experiments
}

TEMPLATE_MIN_DF = 30  # tune: 20–80; start at 30

# ----------------------------
# Preview loader
# ----------------------------
def load_preview_map(path_jsonl_gz: Path) -> dict:
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
# Escaped newline decoding
# ----------------------------
_RE_ESC_NL = re.compile(r"\\r\\n|\\n|\\r")
_RE_ESC_TAB = re.compile(r"\\t")

def unescape_common(text: str) -> str:
    if not isinstance(text, str) or not text:
        return ""
    t = text
    t = _RE_ESC_NL.sub("\n", t)
    t = _RE_ESC_TAB.sub(" ", t)
    # second pass for double-escaped cases
    t = t.replace("\\\\n", "\n").replace("\\\\t", " ")
    return t


# ----------------------------
# Cleaning / detection helpers
# ----------------------------
RE_EMAIL_HDR = re.compile(r"^[^\w]{0,3}(from|to|cc|bcc|sent|date|subject|attachments|importance)\s*:\s*", re.I)
RE_QP = re.compile(r"=([0-9A-F]{2})", re.I)
RE_TS_LINE = re.compile(r"^\s*\d{1,2}/\d{1,2}/\d{2,4}\s+\d{1,2}:\d{2}(:\d{2})?\s*(AM|PM)?\s*$", re.I)

RE_YEAR = re.compile(r"\b(19\d{2}|20\d{2})\b")
RE_TIME = re.compile(r"\b\d{1,2}:\d{2}(?::\d{2})?\b")
RE_DATE_NUMERIC = re.compile(
    r"\b("
    r"(?:19\d{2}|20\d{2})[/-](?:0?[1-9]|1[0-2])[/-](?:0?[1-9]|[12]\d|3[01])"
    r"|(?:0?[1-9]|1[0-2])[/-](?:0?[1-9]|[12]\d|3[01])[/-](?:19\d{2}|20\d{2}|\d{2})"
    r")\b"
)
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

RE_ICHAT_META = re.compile(r"^(source entry:|service:|start time:|end time:|last message)", re.I)

RE_FBI_DELETED = re.compile(r"federal bureau of investigation.*deleted page information sheet|foi/pa", re.I)
RE_REDACTED_NONRESP = re.compile(r"non-responsive\s*-\s*redacted", re.I)
RE_REDACTED_PRIV = re.compile(r"privileged\s*-\s*redacted", re.I)

RE_ON_DATE_WROTE = re.compile(r"^\s*on\s+__DATE__.*wrote\s*:\s*$", re.I)
RE_QUOTED_LINE = re.compile(r"^\s*>")
RE_ORIGINAL_MSG_BLOCK = re.compile(r"^-{2,}\s*original message\s*-{2,}$", re.I)


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
    if not phrases:
        return [], []
    if not as_regex:
        return [], phrases

    line_pats, text_pats = [], []
    for p in phrases:
        if p.lstrip().startswith("^"):
            line_pats.append(p)
        else:
            text_pats.append(p)
    return line_pats, text_pats


def detect_form_kind(text_unescaped: str) -> str:
    if not isinstance(text_unescaped, str) or not text_unescaped.strip():
        return "empty"

    t = text_unescaped.replace("\r", "\n")
    head = "\n".join(t.splitlines()[:40]).strip().lower()

    if RE_ICHAT_META.search(head):
        return "chat_ichat"
    if RE_FBI_DELETED.search(head):
        return "fbi_deleted_sheet"

    hdr_hits = 0
    for ln in head.splitlines():
        if RE_EMAIL_HDR.match(ln.strip()):
            hdr_hits += 1
    if hdr_hits >= 3:
        return "email_generic"

    if RE_REDACTED_NONRESP.search(head) or RE_REDACTED_PRIV.search(head):
        return "redaction_stub"

    return "generic"


def clean_text(
    kind: str,
    text_unescaped: str,
    *,
    drop_line_re: re.Pattern | None = None,
    drop_text_re: re.Pattern | None = None,
    stats: dict | None = None,   # NEW
    template_lines: set[str] | None = None,
) -> str:
    if not isinstance(kind, str) or not kind.strip():
        kind = "generic"

    t = (text_unescaped or "").replace("\r", "\n")

    t = re.sub(r"Non-Responsive\s*-\s*Redacted", " ", t, flags=re.I)
    t = re.sub(r"Privileged\s*-\s*Redacted", " ", t, flags=re.I)

    # normalize time/date/year
    t = RE_TIME.sub(" __TIME__ ", t)
    t = RE_DATE_NUMERIC.sub(" __DATE__ ", t)
    t = RE_DATE_MONTHNAME.sub(" __DATE__ ", t)
    t = RE_YEAR.sub(" __YEAR__ ", t)

    # remove stray backslashes
    t = t.replace("\\", " ")

    raw_lines = [ln.rstrip("\n") for ln in t.splitlines()]

    lines = []
    for ln in raw_lines:
        s = ln.replace("\ufeff", "").strip()
        s = re.sub(r"[\u00A0\u2007\u202F]", " ", s).strip()
        if not s:
            continue

        if RE_FORWARD_SEP.match(s) or RE_BEGIN_FWD.match(s):
            if stats is not None:
                stats["drop_lines_forward_sep"] += 1
            continue
        if RE_IPHONE_SIG.match(s) or RE_ON_BEHALF.match(s):
            if stats is not None:
                stats["drop_lines_sig_onbehalf"] += 1
            continue
        if drop_line_re is not None and drop_line_re.search(s):
            if stats is not None:
                stats["drop_lines_drop_line_re"] += 1
            continue

        if template_lines is not None:
            ns = normalize_line_for_template(s)
            if ns in template_lines:
                if stats is not None:
                    stats["drop_lines_template_df"] += 1
                continue


        lines.append(s)


    if kind == "chat_ichat":
        out = []
        for ln in lines:
            if RE_ICHAT_META.match(ln):
                continue
            if RE_TS_LINE.match(ln):
                continue
            out.append(ln)
        t2 = " ".join(out)

    elif kind == "email_generic":
        out = []
        for ln in lines:
            if RE_EMAIL_HDR.match(ln):
                continue
            # stop at quoted / forwarded blocks
            if RE_ORIGINAL_MSG_BLOCK.match(ln) or RE_FORWARD_SEP.match(ln) or RE_BEGIN_FWD.match(ln):
                break
            if RE_ON_DATE_WROTE.match(ln):
                break
            if RE_QUOTED_LINE.match(ln):
                continue
            out.append(ln)
        t2 = " ".join(out)
        t2 = RE_QP.sub(" ", t2)

    elif kind == "fbi_deleted_sheet":
        t2 = " ".join(lines)
        t2 = re.sub(r"\bpage\s+\d+\b", " ", t2, flags=re.I)
        t2 = re.sub(r"\bb\d+[a-z]?\b", " ", t2, flags=re.I)
        t2 = re.sub(r"\bfoi/pa#?\s*\d+[-]?\d*\b", " __FOI__ ", t2, flags=re.I)

    else:
        t2 = " ".join(lines)

    t2 = re.sub(r"\s+", " ", t2).strip()

    if drop_text_re is not None and t2:
        # count matches BEFORE substitution
        if stats is not None:
            n = len(drop_text_re.findall(t2))
            if n > 0:
                stats["docs_with_drop_text_hits"] += 1
                stats["drop_text_total_matches"] += n

        t2 = drop_text_re.sub(" ", t2)
        t2 = re.sub(r"\s+", " ", t2).strip()
    
    t2 = re.sub(r"[’`]", "'", t2)   # normalize apostrophes
    t2 = re.sub(r"\b(\w+)'(t|re|ve|ll|d|m|s)\b", r"\1\2", t2)  # don't->dont, you're->youre


    return t2


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


def compute_form_weights(form_kinds: pd.Series) -> np.ndarray:
    """
    Simple inverse-frequency weights by form_kind to prevent email_generic dominance.
    Not mandatory, but very useful for cross-form topic discovery.
    """
    counts = form_kinds.fillna("unknown").astype(str).value_counts()
    inv = {k: 1.0 / v for k, v in counts.items()}
    w = form_kinds.fillna("unknown").astype(str).map(inv).astype(float).values
    # normalize mean weight to 1.0
    w = w / np.mean(w)
    return w

def normalize_line_for_template(s: str) -> str:
    s = s.lower().strip()
    s = RE_TIME.sub(" __TIME__ ", s)
    s = RE_DATE_NUMERIC.sub(" __DATE__ ", s)
    s = RE_DATE_MONTHNAME.sub(" __DATE__ ", s)
    s = RE_YEAR.sub(" __YEAR__ ", s)
    s = re.sub(r"\b[\w\.-]+@[\w\.-]+\.\w+\b", " __EMAIL__ ", s)
    s = re.sub(r"https?://\S+|www\.\S+", " __URL__ ", s)
    s = re.sub(r"\b\d+\b", " __NUM__ ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

def build_template_line_counter(df: pd.DataFrame, prev: dict, *, max_lines_per_doc: int = 200) -> Counter:
    c = Counter()
    for r in tqdm(df.itertuples(index=False), total=len(df), desc="Template line DF"):
        raw = prev.get(r.doc_id, "")
        t = unescape_common(raw).replace("\r", "\n")
        # count unique normalized lines per doc (DF, not TF)
        seen = set()
        for ln in t.splitlines()[:max_lines_per_doc]:
            s = ln.strip()
            if not s:
                continue
            ns = normalize_line_for_template(s)
            if len(ns) < 8:
                continue
            seen.add(ns)
        c.update(seen)
    return c

from sklearn.feature_extraction.text import CountVectorizer

def ctfidf_keywords(texts, labels, topn=15, min_docs=7, *, stop_words=None,
                    ngram_range=(1, 2), min_df=1, max_df=1.0):
    """
    Class-based TF-IDF keywords (c-TF-IDF) with stopword support.
    IMPORTANT: pass the SAME stopwords list used by TF-IDF vectorizer.
    """

    # aggregate docs per cluster into one "class document"
    dfc = pd.DataFrame({"text": texts, "label": labels})
    dfc = dfc[dfc["label"] != -1]

    rows = []
    for cid, g in dfc.groupby("label"):
        if len(g) < min_docs:
            continue
        rows.append((int(cid), " ".join(g["text"].tolist()), int(len(g))))

    if not rows:
        return {}

    cids, class_docs, sizes = zip(*rows)

    vec = CountVectorizer(
        token_pattern=r"(?u)\b[A-Za-z_][A-Za-z_]+\b",
        ngram_range=ngram_range,
        stop_words=stop_words,     # ✅ critical
        min_df=min_df,
        max_df=max_df,
    )
    Xc = vec.fit_transform(class_docs)

    # c-TF-IDF
    tf = Xc.astype(np.float64)

    row_sums = np.asarray(tf.sum(axis=1)).ravel()
    row_sums[row_sums == 0] = 1.0
    tf = sparse.diags(1.0 / row_sums) @ tf

    df_term = np.asarray((Xc > 0).sum(axis=0)).ravel()
    idf = np.log((len(class_docs) + 1) / (df_term + 1)) + 1.0

    ctfidf = (tf.multiply(idf)).tocsr()

    terms = np.array(vec.get_feature_names_out())
    out = {}

    for i, cid in enumerate(cids):
        row = ctfidf.getrow(i).toarray().ravel()
        top_idx = row.argsort()[::-1][:topn]
        kw = [terms[j] for j in top_idx if row[j] > 0]
        out[int(cid)] = {
            "topic_cluster": int(cid),
            "topic_size": int(sizes[i]),
            "keywords": ", ".join(kw),
        }

    return out


####===========================


def main():
    ap = argparse.ArgumentParser(
        description="Global semantic clustering across ALL docs using TF-IDF -> SVD -> UMAP -> HDBSCAN "
                    "(no grouping by docform_cluster)."
    )

    ap.add_argument("--corpus", default=CONFIG["CORPUS"])
    ap.add_argument("--preview_jsonl_gz", default=CONFIG["PREVIEW_JSONL_GZ"])
    ap.add_argument("--out_dir", default=CONFIG["OUT_DIR"])

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

    ap.add_argument("--max_docs", type=int, default=CONFIG["MAX_DOCS"])

    ap.add_argument("--umap_min_dist", type=float, default=0.05,
                    help="Lower = tighter local clusters; 0.0–0.2 typical")
    ap.add_argument("--umap_spread", type=float, default=1.0,
                    help="UMAP spread; leave ~1.0 unless you know why")
    ap.add_argument("--hdb_metric", default="euclidean",
                    choices=["euclidean", "manhattan", "l2"],
                    help="Metric for HDBSCAN on embedding")
    ap.add_argument("--cluster_selection_method", default="eom",
                    choices=["eom", "leaf"],
                    help="leaf yields more clusters (often)")
    ap.add_argument("--cluster_selection_epsilon", type=float, default=0.0,
                    help=">0 can merge nearby clusters; keep 0.0 for more splits")

    ap.add_argument(
        "--rebalance_form_kinds",
        action="store_true",
        help="Use inverse-frequency sample weights by form_kind during TF-IDF fit to reduce form dominance."
    )

    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.corpus, compression="gzip", low_memory=False)
    prev = load_preview_map(Path(args.preview_jsonl_gz))

    template_df = build_template_line_counter(df, prev)
    template_lines = {k for k, v in template_df.items() if v >= TEMPLATE_MIN_DF}
    print(f"[TEMPLATE] lines_kept_as_boilerplate={len(template_lines):,} (min_df={TEMPLATE_MIN_DF})")


    if "doc_id" not in df.columns:
        raise KeyError("Corpus must contain doc_id")

    n0 = len(df)
    dup_ids = int(df["doc_id"].duplicated().sum())
    print(f"[LOAD] corpus={args.corpus} rows={n0:,} duplicated_doc_id={dup_ids:,}")

    # deterministically drop duplicates if they exist
    if dup_ids > 0:
        df = df.sort_values([c for c in ["doc_family", "docform_cluster", "rel_path", "doc_id"] if c in df.columns])
        df = df.drop_duplicates(subset=["doc_id"], keep="first").copy()
        print(f"[LOAD] dropped duplicates -> rows={len(df):,}")

    miss = df["doc_id"].map(lambda x: x not in prev).sum()
    print(f"[PREVIEW] missing previews: {miss:,}/{len(df):,}")

    # stopwords + drop phrases
    custom_stop = load_wordlist(args.stopwords_txt, lower=True) if args.stopwords_txt else []
    # IMPORTANT: do NOT lowercase regex patterns
    phrases = load_wordlist(args.drop_phrases_txt, lower=(not args.drop_phrases_regex)) if args.drop_phrases_txt else []

    stop_set = set(ENGLISH_STOP_WORDS) | set(custom_stop)
    stop_set |= {
        "__date__", "__time__", "__year__", "__foi__",
        "jpg", "jpeg", "png", "gif", "tif", "tiff", "bmp", "pdf",
        "http", "https", "www", "com",
    }
    stop = sorted(stop_set)  # sklearn requires list/None/"english"

    line_pats, text_pats = split_drop_patterns(phrases, as_regex=args.drop_phrases_regex)
    drop_line_re = compile_phrase_regex(line_pats, as_regex=True, flags=re.I) if line_pats else None
    drop_text_re = compile_phrase_regex(text_pats, as_regex=args.drop_phrases_regex, flags=re.I | re.MULTILINE) if text_pats else None

    # --- DROP PHRASES / CLEANING STATS (Step 4) ---
    stats = Counter()
    stats.update({
        "drop_lines_forward_sep": 0,
        "drop_lines_sig_onbehalf": 0,
        "drop_lines_drop_line_re": 0,
        "docs_with_drop_text_hits": 0,
        "drop_text_total_matches": 0,
    })

    # clean all docs
    kinds, texts = [], []
    for r in tqdm(df.itertuples(index=False), total=len(df), desc="Clean text"):
        raw = prev.get(r.doc_id, "")
        raw_u = unescape_common(raw)
        kind = detect_form_kind(raw_u)
        kinds.append(kind)
        texts.append(clean_text(kind, raw_u, drop_line_re=drop_line_re, drop_text_re=drop_text_re, stats=stats, template_lines=template_lines))

    df["form_kind"] = kinds
    df["clean_text"] = texts
    df["clean_tok_len"] = df["clean_text"].str.split().map(len).fillna(0).astype(int)

    print("[FORM_KIND] distribution:", dict(Counter(df["form_kind"].tolist())))

    print(
        "[DROP] "
        f"docs_with_drop_text_hits={stats['docs_with_drop_text_hits']:,}/{len(df):,} | "
        f"drop_text_total_matches={stats['drop_text_total_matches']:,} | "
        f"drop_lines_forward_sep={stats['drop_lines_forward_sep']:,} | "
        f"drop_lines_sig_onbehalf={stats['drop_lines_sig_onbehalf']:,} | "
        f"drop_lines_drop_line_re={stats['drop_lines_drop_line_re']:,} | "
        f"drop_lines_template_df={stats.get('drop_lines_template_df', 0):,}"
    )


    # filter by min tokens
    df = df[df["clean_tok_len"] >= args.min_tokens].copy()
    print(f"[FILTER] usable for semantic clustering: {len(df):,} (min_tokens={args.min_tokens})")

    # optional sub-sampling for speed
    if args.max_docs is not None and len(df) > args.max_docs:
        df = df.sample(n=args.max_docs, random_state=args.random_state).copy()
        print(f"[SAMPLE] truncated to max_docs={args.max_docs:,}")

    # vectorize (global)
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

    sample_weight = None
    if args.rebalance_form_kinds:
        sample_weight = compute_form_weights(df["form_kind"])
        print("[WEIGHT] using inverse-frequency weights by form_kind for TF-IDF fit")

    X = vec.fit_transform(df["clean_text"].tolist())

    # Optional: rebalance AFTER TF-IDF (sklearn does not support sample_weight here)
    if sample_weight is not None:
        # scale each document vector (row) by its weight
        X = sparse.diags(sample_weight, offsets=0, format="csr") @ X
        feature_names = vec.get_feature_names_out()

    # ✅ MUST be defined unconditionally before keyword extraction
    feature_names = vec.get_feature_names_out()
    vocab_size = len(vec.vocabulary_)


    vocab_size = len(vec.vocabulary_)
    nnz = X.nnz
    density = nnz / (X.shape[0] * max(1, X.shape[1]))
    print(f"[TFIDF] docs={X.shape[0]:,} vocab={vocab_size:,} nnz={nnz:,} density={density:.6f}")

    # SVD
    n_svd = min(args.svd_components, X.shape[1] - 1)
    if n_svd < 2:
        raise RuntimeError(f"Vocab too small for SVD: {X.shape[1]}")

    svd = TruncatedSVD(n_components=n_svd, random_state=args.random_state)
    X_svd = svd.fit_transform(X)
    print(f"[SVD] components_used={n_svd}")

    # UMAP
    emb = umap.UMAP(
        n_neighbors=args.umap_neighbors,      # try 10–20 for more clusters
        n_components=args.umap_dim,
        min_dist=args.umap_min_dist,          # key: 0.0–0.1
        spread=args.umap_spread,
        metric=args.umap_metric,              # cosine is fine with SVD output
        random_state=args.random_state,
    ).fit_transform(X_svd)
    print(f"[UMAP] dim={args.umap_dim} neighbors={args.umap_neighbors} "
          f"min_dist={args.umap_min_dist} metric={args.umap_metric}")

    # HDBSCAN: allow more topic splits
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=args.min_cluster_size,        # try 15–25
        min_samples=args.min_samples,                  # try 3–8
        metric=args.hdb_metric,
        cluster_selection_method=args.cluster_selection_method,  # leaf => more clusters
        cluster_selection_epsilon=args.cluster_selection_epsilon,
    )
    labels = clusterer.fit_predict(emb)


    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    noise = int((labels == -1).sum())
    print(f"[HDBSCAN] clusters={n_clusters} noise_docs={noise:,}/{len(labels):,}")

    # keywords per cluster
    # kw = top_keywords_per_cluster(
    #     X, labels, feature_names,
    #     topn=args.topn_keywords,
    #     min_docs=max(args.min_docs_keywords, int(0.01 * len(df)), 3),
    # )

    kw = ctfidf_keywords(
        df["clean_text"].tolist(),
        labels,
        topn=args.topn_keywords,
        min_docs= 10, # max(args.min_docs_keywords, int(0.01 * len(df)), 3),
        stop_words=stop,                    # ✅ same stopwords as TF-IDF
        ngram_range=(args.ngram_min, args.ngram_max),
        min_df=2,                           # mirror your TF-IDF-ish
        max_df=1.0                          # keep 1.0 here; df pruning is less stable for class-docs
    )



    # topic summary
    topic_summary_rows = []
    for cid in sorted(set(labels.tolist())):
        if cid == -1:
            continue
        topic_summary_rows.append({
            "group_col": "ALL",
            "group_value": "ALL",
            "topic_cluster": int(cid),
            "topic_size": int((labels == cid).sum()),
            "keywords": kw.get(int(cid), {}).get("keywords", ""),
        })

    topic_summary_df = pd.DataFrame(topic_summary_rows, columns=["group_col","group_value","topic_cluster","topic_size","keywords"])
    if len(topic_summary_df):
        topic_summary_df = topic_summary_df.sort_values(["topic_size"], ascending=False)

    topic_summary_df.to_csv(out_dir / "topic_summary.csv", index=False)
    (out_dir / "semantic_all_topics.csv").write_text(topic_summary_df.to_csv(index=False), encoding="utf-8")
    print(f"[WRITE] {out_dir/'topic_summary.csv'}")
    print(f"[WRITE] {out_dir/'semantic_all_topics.csv'}")

    # docs with clusters
    base_cols = ["doc_id", "sha256", "rel_path", "docform_cluster", "doc_family", "human_doc_type", "form_kind", "clean_tok_len"]
    keep_cols = [c for c in base_cols if c in df.columns]
    out = df[keep_cols].copy()

    out["topic_cluster"] = labels
    out["topic_prob"] = clusterer.probabilities_

    # ----------------------------
    # INSPECTION PACK (top docs per cluster + purity stats)
    # ----------------------------
    # 1) top-N docs by topic_prob per cluster
    TOP_DOCS_PER_CLUSTER = 15
    top_docs = (
        out[out["topic_cluster"] != -1]
        .sort_values(["topic_cluster", "topic_prob"], ascending=[True, False])
        .groupby("topic_cluster")
        .head(TOP_DOCS_PER_CLUSTER)
        .copy()
    )
    top_docs.to_csv(out_dir / "top_docs_per_cluster.csv", index=False)

    # 2) cluster purity summary across metadata columns
    purity_cols = [c for c in ["form_kind", "docform_cluster", "doc_family", "human_doc_type"] if c in out.columns]
    purity_rows = []
    for cid, g in out[out["topic_cluster"] != -1].groupby("topic_cluster"):
        row = {"topic_cluster": int(cid), "topic_size": int(len(g)), "mean_prob": float(g["topic_prob"].mean())}
        for c in purity_cols:
            vc = g[c].fillna("NA").astype(str).value_counts()
            top = vc.index[0]
            row[f"{c}_top"] = top
            row[f"{c}_top_share"] = float(vc.iloc[0] / len(g))
            row[f"{c}_n_unique"] = int(vc.shape[0])
        purity_rows.append(row)

    purity_df = pd.DataFrame(purity_rows).sort_values(["topic_size"], ascending=False)
    purity_df.to_csv(out_dir / "cluster_purity.csv", index=False)

    # 3) noise breakdown
    noise_df = out[out["topic_cluster"] == -1]
    noise_stats = {"noise_docs": int(len(noise_df))}
    for c in purity_cols:
        noise_stats[f"noise_{c}_top"] = str(noise_df[c].fillna("NA").astype(str).value_counts().index[0]) if len(noise_df) else ""
    pd.DataFrame([noise_stats]).to_csv(out_dir / "noise_summary.csv", index=False)

    print(f"[WRITE] {out_dir/'top_docs_per_cluster.csv'}")
    print(f"[WRITE] {out_dir/'cluster_purity.csv'}")
    print(f"[WRITE] {out_dir/'noise_summary.csv'}")

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

    out.to_csv(out_dir / "docs_with_topic_clusters.csv.gz", index=False, compression="gzip")
    print(f"[WRITE] {out_dir/'docs_with_topic_clusters.csv.gz'}")

    # embedding export
    emb_df = pd.DataFrame(emb, columns=[f"u{i}" for i in range(emb.shape[1])])
    emb_df.insert(0, "doc_id", df["doc_id"].values)
    emb_df["topic_cluster"] = labels
    emb_df.to_csv(out_dir / "umap_embedding.csv", index=False)
    print(f"[WRITE] {out_dir/'umap_embedding.csv'}")

    # group stats (single row)
    form_mode = ""
    m = df["form_kind"].mode(dropna=True)
    form_mode = str(m.iloc[0]) if len(m) else ""

    stats_df = pd.DataFrame([{
        "group_col": "ALL",
        "group_value": "ALL",
        "n_docs": int(len(df)),
        "form_kind_mode": form_mode,
        "vocab_size": int(vocab_size),
        "tfidf_nnz": int(nnz),
        "tfidf_density": float(density),
        "svd_components_used": int(n_svd),
        "umap_dim": int(args.umap_dim),
        "umap_neighbors_used": int(args.umap_neighbors),
        "n_clusters": int(n_clusters),
        "noise_docs": int(noise),
        "clusters_with_keywords": int(len(kw)),
        "min_docs_keywords_used": int(max(args.min_docs_keywords, int(0.01 * len(df)), 3)),
        "min_cluster_size_used": int(args.min_cluster_size),
        "min_samples_used": int(args.min_samples),
        "rebalance_form_kinds": bool(args.rebalance_form_kinds),
    }])
    stats_df.to_csv(out_dir / "semantic_group_stats.csv", index=False)
    print(f"[WRITE] {out_dir/'semantic_group_stats.csv'}")

    print("[DONE] global semantic clustering complete.")


if __name__ == "__main__":
    main()
