# Ontology TextMining — Pipeline README (Runbook)

This repo builds a structured corpus manifest, discovers **doc-form clusters**, applies **human doc-type labels**, deduplicates **near-duplicates** (SimHash), and then runs **semantic clustering per doc-type** to prepare for later extraction + targeted OCR.

> Ground rule: **don’t OCR images yet**. We OCR PDFs only when embedded text is missing (first page only), and we defer image OCR until after doc-type discovery.

---

## 0) Setup

### Python + venv
```bash
python -m venv venv
source venv/bin/activate
pip install -U pip

### Install dependencies
pip install pandas numpy scikit-learn umap-learn hdbscan tqdm charset-normalizer pymupdf pillow pytesseract

### Install Tesseract (macOS)
brew install tesseract
```

## 1) Data directory

We scan recursively from data/. So the structure is not that much important.

## 2) Step 0 — Build manifest (fast, content-hash IDs, no image OCR)
`python manifest.py`

Outputs:
    manifest_all.csv.gz
    manifest_all_with_preview.jsonl.gz

## 3) Step 1 — Doc-form clustering (shape/form types)
Goal: cluster documents by “form” (email-like, lists, transcripts, etc.) using numeric stats (and optionally char n-grams).
`python docform_clusters.py --manifest_csv manifest_all.csv.gz --previews_jsonl manifest_all_with_preview.jsonl.gz  --text_path_col text_path  --text_root . --out_dir outputs/docform --min_cluster_size 35 --min_samples 6 --hdbscan_selection_method eom --umap_components 30 --refine_min_docs 200 --char_max_features 15000 --char_min_df 10`

Outputs (under outputs/docform/):
    manifest_with_docform_clusters.csv.gz
    docform_cluster_feature_means.csv
    docform_cluster_samples.tsv
    docform_labeling_sheet.csv
    umap_embedding.csv

What to inspect:
    outputs/docform/docform_cluster_samples.tsv (quick “what is this cluster?” view)
    outputs/docform/docform_labeling_sheet.csv (the sheet you fill in)

## 4) Step 2 — Human doc-type labeling (manual)
Goal: map doc-form clusters → human doc types (email variants, lists, articles, etc.).

Open notebook
`review_clusters.ipynb`
After inspection fill columns like:
    human_doc_type (string label)
    keep (yes/no)
    notes
in
`outputs/docform/docform_labeling_sheet.csv`
or just use cells in
`review_clusters.ipynb`

Outputs
    outputs/docform/manifest_with_doc_types.csv.gz
    outputs/docform/docform_labeling_sheet_labeled.csv


## 6) Step 2.7 — Near-duplicate dedup (SimHash)
Goal: collapse “same content with small differences” (email headers, minor formatting) into canonical docs before semantic clustering.
`python dedup_simhash.py  --manifest outputs/docform/manifest_with_docform_clusters.csv.gz  --preview_jsonl_gz manifest_all_with_preview.jsonl.gz  --text_path_col text_path  --text_root .  --out_dir outputs/near_dedup`

Outputs (under outputs/near_dedup/):
    near_dup_groups.csv.gz (group definitions)
    manifest_with_near_dedup.csv.gz (each doc assigned a near-dup group)
    corpus_canonical.csv.gz (one canonical doc per group)

Sanity checks, should print something like:
    [PREVIEW] missing previews for 0/... docs
    [DONE] groups: X | docs: Y
    corpus_canonical.csv.gz is the input to semantic clustering

## 7) Step 3 — Semantic clustering per human_doc_type
Goal: within each human_doc_type, discover semantic topic structure using TF-IDF → UMAP → HDBSCAN.
`python semantic_cluster_by_doctype.py  --corpus outputs/near_dedup/corpus_canonical.csv.gz  --preview_jsonl_gz manifest_all_with_preview.jsonl.gz  --out_dir outputs/semantic  --group_by docform_cluster  --min_docs 30 --min_tokens 30`

Outputs:
    outputs/semantic/semantic_type_stats.csv

For each doc type folder:
    docs_with_topic_clusters.csv.gz
    umap_embedding.csv
    topic_labeling_sheet.csv

Some doc types won’t cluster (all noise). That’s normal for:
    short messages
    numeric lists
    tiny doc-type sizes


## 8) What happens next (not implemented yet)

#### Step 4 — Targeted image OCR policy (after doc types)

We do NOT OCR all images.
We will OCR selectively:
    only doc types where OCR adds value 
    using dedup/heuristics (pHash, near-dup, adjacency to valuable text)

#### Step 5 — Entity extraction + event modeling

Once doc types + topics are stable, we define:
    event schemas per type
    entity schemas (PERSON/ORG/LOC/EMAIL/PHONE/MONEY/DATE/etc.)
    evidence/provenance links