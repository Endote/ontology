# build_manifest_v3_fast_parallel_cache.py
import json
import hashlib
import gzip
import os
import re
import time
from pathlib import Path
from typing import Tuple, Optional, Dict
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
from tqdm import tqdm  # pip install tqdm

# ---- optional deps ----
try:
    import fitz  # PyMuPDF
except Exception:
    fitz = None

try:
    from PIL import Image
except Exception:
    Image = None

try:
    import pytesseract
except Exception:
    pytesseract = None

try:
    from charset_normalizer import from_bytes as cn_from_bytes
except Exception:
    cn_from_bytes = None

MIN_TEXT_CHARS = 50
# -----------------------------
ROOT = Path("data")

OUT_CSV_GZ   = Path("manifest_all.csv.gz")
OUT_JSONL_GZ = Path("manifest_all_with_preview.jsonl.gz")

# OCR cache
CACHE_DIR = Path("cache/ocr")
CACHE_DIR.mkdir(parents=True, exist_ok=True)

# Full text cache (NEW)
TEXT_DIR = Path("cache/fulltext")
TEXT_DIR.mkdir(parents=True, exist_ok=True)

# Store full text for these doc families (recommend text+pdf only)
STORE_FULLTEXT_FOR_FAMILIES = {"text", "pdf"}

# Hard safety cap per document
MAX_FULLTEXT_CHARS = 2_000_000

EXT_TEXTLIKE = {".txt", ".csv", ".tsv", ".json", ".xml", ".html", ".md", ".rtf"}
EXT_PDF      = {".pdf"}
EXT_IMAGE    = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp"}
EXT_BINARY_INTERESTING = {".xls", ".xlsx", ".doc", ".docx", ".eml", ".msg"}

SKIP_FILENAMES = {".DS_Store"}
SKIP_PREFIXES  = ("._",)
SKIP_DIRS      = {"__MACOSX"}

# Tuning
DEFAULT_MAX_BYTES_TEXTLIKE = 2_000_000
DEFAULT_PREVIEW_CHARS = 800

# Parallel OCR threads (PDF OCR only)
MAX_OCR_WORKERS = min(6, (os.cpu_count() or 4))


# -----------------------------
# Utilities
# -----------------------------
def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def decode_bytes_safely(b: bytes) -> Tuple[str, str, int]:
    try:
        text = b.decode("utf-8", errors="strict")
        return text, "utf-8", 0
    except Exception:
        pass

    if cn_from_bytes is not None:
        try:
            best = cn_from_bytes(b).best()
            if best is not None:
                text = str(best)
                return text, f"cn:{best.encoding}", text.count("\ufffd")
        except Exception:
            pass

    text = b.decode("latin-1", errors="replace")
    return text, "latin-1", text.count("\ufffd")


def read_textlike(path: Path, max_bytes: int = DEFAULT_MAX_BYTES_TEXTLIKE) -> Tuple[str, str, int, int]:
    with path.open("rb") as f:
        b = f.read(max_bytes)
    text, enc, repl = decode_bytes_safely(b)
    return text, enc, repl, len(b)


def ocr_available() -> bool:
    if pytesseract is None:
        return False
    try:
        _ = pytesseract.get_tesseract_version()
        return True
    except Exception:
        return False


def extract_pdf_text(path: Path, max_pages: int = 2, max_chars: int = 200_000) -> Tuple[str, int]:
    """
    Returns (text, page_count)
    """
    if fitz is None:
        return "", 0
    doc = fitz.open(path)
    parts = []
    try:
        page_count = doc.page_count
        for i in range(min(max_pages, page_count)):
            page = doc.load_page(i)
            parts.append(page.get_text("text"))
            if sum(len(x) for x in parts) >= max_chars:
                break
    finally:
        doc.close()
    return "\n".join(parts).strip(), page_count


def cache_path_for_sha(sha256: str) -> Path:
    return CACHE_DIR / f"{sha256}.txt.gz"


def load_ocr_cache(sha256: str) -> Optional[str]:
    fp = cache_path_for_sha(sha256)
    if not fp.exists():
        return None
    try:
        with gzip.open(fp, "rt", encoding="utf-8") as f:
            return f.read()
    except Exception:
        return None


def save_ocr_cache(sha256: str, text: str) -> None:
    fp = cache_path_for_sha(sha256)
    tmp = fp.with_suffix(fp.suffix + ".tmp")
    with gzip.open(tmp, "wt", encoding="utf-8") as f:
        f.write(text)
    tmp.replace(fp)  # atomic-ish on same filesystem


def ocr_pdf_first_page(path: Path, dpi: int = 200) -> Tuple[str, str]:
    """
    OCR only the first page. Returns (text, err)
    """
    if fitz is None:
        return "", "pymupdf_missing"
    if Image is None or pytesseract is None:
        return "", "pytesseract_or_pillow_missing"
    if not ocr_available():
        return "", "tesseract_binary_not_found"

    try:
        doc = fitz.open(path)
        try:
            if doc.page_count == 0:
                return "", "pdf_no_pages"
            page = doc.load_page(0)
            mat = fitz.Matrix(dpi / 72, dpi / 72)
            pix = page.get_pixmap(matrix=mat, alpha=False)
            img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        finally:
            doc.close()

        txt = pytesseract.image_to_string(img)
        return txt.strip(), ""
    except Exception as e:
        return "", f"ocr_pdf_failed:{type(e).__name__}:{e}"


def get_image_metadata(path: Path) -> Dict[str, Optional[str]]:
    """
    Returns width/height/format/mode safely.
    """
    meta = {"img_width": None, "img_height": None, "img_format": None, "img_mode": None}
    if Image is None:
        return meta
    try:
        with Image.open(path) as im:
            meta["img_width"] = im.width
            meta["img_height"] = im.height
            meta["img_format"] = im.format
            meta["img_mode"] = im.mode
    except Exception:
        pass
    return meta


def text_stats(text: str):
    if not text:
        return {
            "n_chars": 0, "n_lines": 0, "n_tokens": 0, "blank_lines": 0,
            "avg_line_len": 0.0, "max_line_len": 0,
            "digit_ratio": 0.0, "upper_ratio": 0.0, "non_ascii_ratio": 0.0,
            "email_count": 0, "phone_like_count": 0, "date_like_count": 0, "money_like_count": 0,
        }

    lines = text.splitlines()
    n_lines = len(lines)
    blank_lines = sum(1 for ln in lines if not ln.strip())
    line_lens = [len(ln) for ln in lines] if lines else [0]
    avg_line_len = float(sum(line_lens) / max(1, len(line_lens)))
    max_line_len = max(line_lens) if line_lens else 0

    n_chars = len(text)
    digit_ratio = sum(c.isdigit() for c in text) / n_chars
    upper_ratio = sum(c.isupper() for c in text) / n_chars
    non_ascii_ratio = sum(ord(c) > 127 for c in text) / n_chars

    email_count = len(re.findall(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}", text))
    phone_like_count = len(re.findall(r"\+?\d[\d\s().-]{7,}\d", text))
    date_like_count = len(re.findall(r"\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b|\b\d{4}-\d{2}-\d{2}\b", text))
    money_like_count = len(re.findall(r"\$\s?\d|\b\d+[,.]?\d*\s?(USD|EUR|PLN|GBP)\b", text, flags=re.IGNORECASE))

    n_tokens = len(text.split())

    return {
        "n_chars": n_chars,
        "n_lines": n_lines,
        "n_tokens": n_tokens,
        "blank_lines": blank_lines,
        "avg_line_len": avg_line_len,
        "max_line_len": max_line_len,
        "digit_ratio": digit_ratio,
        "upper_ratio": upper_ratio,
        "non_ascii_ratio": non_ascii_ratio,
        "email_count": email_count,
        "phone_like_count": phone_like_count,
        "date_like_count": date_like_count,
        "money_like_count": money_like_count,
    }


def should_skip(path: Path) -> bool:
    if path.name in SKIP_FILENAMES:
        return True
    if any(path.name.startswith(pfx) for pfx in SKIP_PREFIXES):
        return True
    if any(part in SKIP_DIRS for part in path.parts):
        return True
    return False


def pdf_needs_ocr(pdf_text: str, min_chars: int = 30) -> bool:
    # embedded PDF text is often empty for scanned docs
    return len(pdf_text.strip()) < min_chars


# -----------------------------
# Fulltext cache helpers (NEW)
# -----------------------------
def text_cache_path(doc_id: str) -> Path:
    return TEXT_DIR / f"{doc_id}.txt.gz"


def save_fulltext(doc_id: str, text: str) -> str:
    """
    Save full text (gz) and return relative/posix path to store in manifest.
    """
    if text is None:
        text = ""
    text = text[:MAX_FULLTEXT_CHARS]
    fp = text_cache_path(doc_id)
    tmp = fp.with_suffix(fp.suffix + ".tmp")
    with gzip.open(tmp, "wt", encoding="utf-8") as f:
        f.write(text)
    tmp.replace(fp)
    return fp.as_posix()


# -----------------------------
# Parallel OCR task (PDF only)
# -----------------------------
def pdf_ocr_task(path: Path, sha256: str, dpi: int) -> Tuple[str, str, str]:
    """
    Returns (sha256, ocr_text, ocr_error)
    Uses cache if present.
    """
    cached = load_ocr_cache(sha256)
    if cached is not None:
        return sha256, cached, ""  # cached -> no error

    txt, err = ocr_pdf_first_page(path, dpi=dpi)
    if txt:
        save_ocr_cache(sha256, txt)
    return sha256, txt, err


# -----------------------------
# Main
# -----------------------------
def build_manifest(
    pdf_text_pages: int = 2,
    pdf_ocr_dpi: int = 200,
    verbose_every: int = 500,
) -> pd.DataFrame:
    t0 = time.time()

    # Collect files
    files = []
    for p in ROOT.rglob("*"):
        if not p.is_file():
            continue
        if should_skip(p):
            continue
        ext = p.suffix.lower()
        if ext in EXT_TEXTLIKE or ext in EXT_PDF or ext in EXT_IMAGE or ext in EXT_BINARY_INTERESTING:
            files.append(p)

    files = sorted(files)
    print(f"[SCAN] Found {len(files):,} candidate files under {ROOT}")
    print(
        f"[ENV] PyMuPDF: {'OK' if fitz is not None else 'MISSING'} | "
        f"Pillow: {'OK' if Image is not None else 'MISSING'} | "
        f"pytesseract: {'OK' if pytesseract is not None else 'MISSING'} | "
        f"tesseract-bin: {'OK' if ocr_available() else 'MISSING'} | "
        f"OCR workers: {MAX_OCR_WORKERS} | OCR cache: {CACHE_DIR} | Fulltext cache: {TEXT_DIR}"
    )

    counters = Counter()
    timings = Counter()

    # First pass: compute hashes + cheap extraction, decide which PDFs need OCR.
    per_file: list[dict] = []
    pdfs_to_ocr: list[tuple[Path, str]] = []

    for idx, path in enumerate(tqdm(files, desc="Pass1: hash + cheap extract", unit="file")):
        t_file0 = time.time()

        rel_path = path.relative_to(ROOT).as_posix()
        source_root = rel_path.split("/", 1)[0] if "/" in rel_path else rel_path
        ext = path.suffix.lower()

        if ext in EXT_TEXTLIKE:
            doc_family = "text"
        elif ext in EXT_PDF:
            doc_family = "pdf"
        elif ext in EXT_IMAGE:
            doc_family = "image"
        elif ext in EXT_BINARY_INTERESTING:
            doc_family = "binary"
        else:
            doc_family = "other"

        st = path.stat()
        size_bytes = st.st_size
        mtime = int(st.st_mtime)

        t_hash0 = time.time()
        file_hash = sha256_file(path)
        timings["sha256_s"] += time.time() - t_hash0

        # content-only identifiers
        doc_id = file_hash[:24]
        doc_sha256 = file_hash

        extraction_method = "binary"
        encoding = ""
        decode_replacements = 0
        bytes_read = 0
        text = ""
        ocr_error = ""
        pdf_page_count = None
        img_meta = {"img_width": None, "img_height": None, "img_format": None, "img_mode": None}

        t_extract0 = time.time()
        if ext in EXT_TEXTLIKE:
            counters["textlike"] += 1
            text, encoding, decode_replacements, bytes_read = read_textlike(path)
            extraction_method = "textlike"

        elif ext in EXT_PDF:
            counters["pdf"] += 1
            pdf_txt, page_count = extract_pdf_text(path, max_pages=pdf_text_pages)
            pdf_page_count = page_count
            text = pdf_txt
            extraction_method = "pdf_text_only"

            if pdf_needs_ocr(pdf_txt):
                counters["pdf_needs_ocr"] += 1
                pdfs_to_ocr.append((path, doc_sha256))
                extraction_method = "pdf_text_pending_ocr"

        elif ext in EXT_IMAGE:
            counters["image"] += 1
            img_meta = get_image_metadata(path)
            extraction_method = "image_meta_only"

        else:
            counters["binary_interesting"] += 1
            extraction_method = "binary_interesting"

        timings["extract_s"] += time.time() - t_extract0

        # stats computed from whatever text we currently have (cheap pass)
        t_stats0 = time.time()
        stats = text_stats(text)
        timings["stats_s"] += time.time() - t_stats0

        per_file.append({
            "path": path,
            "doc_id": doc_id,
            "sha256": doc_sha256,
            "source_root": source_root,
            "rel_path": rel_path,
            "basename": path.name,
            "ext": ext,
            "size_bytes": size_bytes,
            "mtime_unix": mtime,
            "extraction_method": extraction_method,
            "ocr_error": ocr_error,
            "encoding": encoding,
            "bytes_read_for_preview": bytes_read,
            "decode_replacements": decode_replacements,
            "pdf_page_count": pdf_page_count,
            **img_meta,
            **stats,
            # store text temporarily for preview + potential merge
            "_text": text,
            # NEW: path to persisted full text (filled in Pass3)
            "text_path": "",
            "doc_family": doc_family,
        })

        timings["per_file_pass1_s"] += time.time() - t_file0

        if verbose_every and (idx + 1) % verbose_every == 0:
            print(
                f"[PASS1] {idx+1:,}/{len(files):,} | pdf_need_ocr={counters['pdf_needs_ocr']:,} "
                f"| ocr_cache_files={len(list(CACHE_DIR.glob('*.txt.gz'))):,} "
                f"| fulltext_files={len(list(TEXT_DIR.glob('*.txt.gz'))):,}"
            )

    # Second pass: parallel OCR for needed PDFs + cache
    ocr_map: Dict[str, Tuple[str, str]] = {}  # sha256 -> (ocr_text, ocr_error)

    if pdfs_to_ocr:
        if not ocr_available():
            print("[WARN] PDFs need OCR but tesseract-bin is missing; skipping OCR step.")
            for _, sha in pdfs_to_ocr:
                ocr_map[sha] = ("", "tesseract_binary_not_found")
        else:
            print(f"[OCR] Scheduling OCR for {len(pdfs_to_ocr):,} PDFs (first page only) with {MAX_OCR_WORKERS} workers...")
            t_ocr0 = time.time()
            with ThreadPoolExecutor(max_workers=MAX_OCR_WORKERS) as ex:
                futures = []
                for p, sha in pdfs_to_ocr:
                    futures.append(ex.submit(pdf_ocr_task, p, sha, pdf_ocr_dpi))

                for fut in tqdm(as_completed(futures), total=len(futures), desc="Pass2: PDF OCR", unit="pdf"):
                    sha, txt, err = fut.result()
                    ocr_map[sha] = (txt, err)
                    if err and not txt:
                        counters["pdf_ocr_errors"] += 1

            timings["pdf_ocr_s"] += time.time() - t_ocr0
            print(f"[OCR] Done PDF OCR in {timings['pdf_ocr_s']:.1f}s | errors={counters['pdf_ocr_errors']}")

    # Finalize: merge OCR where applicable, recompute stats if text changed, write outputs
    rows = []
    with gzip.open(OUT_JSONL_GZ, "wt", encoding="utf-8") as jout:
        for item in tqdm(per_file, desc="Pass3: finalize + write", unit="file"):
            ext = item["ext"]
            text = item["_text"]
            extraction_method = item["extraction_method"]
            ocr_error = item["ocr_error"]

            if ext in EXT_PDF and extraction_method == "pdf_text_pending_ocr":
                ocr_txt, err = ocr_map.get(item["sha256"], ("", "ocr_missing_result"))
                # Combine embedded + OCR
                combined = "\n\n".join([t for t in (text, ocr_txt) if t]).strip()
                text = combined
                ocr_error = err

                if text and ocr_txt and item["_text"]:
                    extraction_method = "pdf_text+pdf_ocr_first_page"
                elif text and ocr_txt:
                    extraction_method = "pdf_ocr_only_first_page"
                elif item["_text"]:
                    extraction_method = "pdf_text_only"
                else:
                    extraction_method = "pdf_no_text"

                # recompute stats on merged text
                stats = text_stats(text)
                for k, v in stats.items():
                    item[k] = v

            # NEW: persist full text to per-doc gz, store text_path in manifest
            if item.get("doc_family") in STORE_FULLTEXT_FOR_FAMILIES and text and text.strip():
                try:
                    item["text_path"] = save_fulltext(item["doc_id"], text)
                except Exception as e:
                    # don't fail the manifest build if fulltext write fails for one doc
                    item["text_path"] = ""
                    counters["fulltext_write_errors"] += 1
                    # stash a short note
                    item["ocr_error"] = (ocr_error + f" | fulltext_write_failed:{type(e).__name__}").note
            else:
                item["text_path"] = ""

            # preview and jsonl (escape both CR and LF for stability)
            preview = text[:DEFAULT_PREVIEW_CHARS].replace("\r", "\\r").replace("\n", "\\n")
            jout.write(json.dumps(
                {"doc_id": item["doc_id"], "sha256": item["sha256"], "rel_path": item["rel_path"],
                 "extraction_method": extraction_method, "preview": preview},
                ensure_ascii=False
            ) + "\n")

            # build final row (drop temp text)
            item["extraction_method"] = extraction_method
            item["ocr_error"] = ocr_error
            item.pop("_text", None)
            rows.append(item)

    df = pd.DataFrame(rows)

    # --- triage flags (fixed) ---
    df["has_text"] = (df["n_chars"] >= MIN_TEXT_CHARS)

    # text_missing means: "we expected text eventually but we don't have it yet"
    df["text_missing"] = (
        df["doc_family"].isin(["pdf", "image"]) & (~df["has_text"])
    )

    # for convenience: what is usable for text-based clustering/NER *right now*
    df["text_usable_now"] = df["doc_family"].isin(["text", "pdf"]) & df["has_text"]

    # pattern flags (unchanged)
    df["has_emails"] = (df["email_count"] > 0)
    df["has_phones"] = (df["phone_like_count"] > 0)
    df["has_dates"]  = (df["date_like_count"] > 0)
    df["has_money"]  = (df["money_like_count"] > 0)

    df = df.sort_values(["source_root", "rel_path"]).reset_index(drop=True)
    df.to_csv(OUT_CSV_GZ, index=False, compression="gzip")

    total_s = time.time() - t0
    print(f"[DONE] Wrote {len(df):,} rows to {OUT_CSV_GZ}")
    print(f"[DONE] Wrote previews to {OUT_JSONL_GZ}")
    print(
        f"[TIME] Total {total_s:.1f}s | sha256 {timings['sha256_s']:.1f}s | "
        f"extract {timings['extract_s']:.1f}s | stats {timings['stats_s']:.1f}s | "
        f"pdf_ocr {timings.get('pdf_ocr_s', 0.0):.1f}s"
    )
    print(f"[COUNTS] {dict(counters)}")

    return df


if __name__ == "__main__":
    df = build_manifest(
        pdf_text_pages=2,
        pdf_ocr_dpi=200,
        verbose_every=1000,
    )
    print(df[["doc_family", "ext", "extraction_method", "has_text", "text_missing"]].value_counts().head(25))
    # NEW: quick check of fulltext persistence
    if "text_path" in df.columns:
        print("[FULLTEXT] non-empty text_path ratio:", (df["text_path"].astype(str).str.len() > 0).mean())
        print(df.loc[df["text_path"].astype(str).str.len() > 0, ["doc_id", "doc_family", "text_path"]].head(10))
