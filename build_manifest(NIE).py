import json
import hashlib
import gzip
import os
import re
from pathlib import Path

import pandas as pd

ROOT = Path('data')
OUT_CSV_GZ = Path('manifest_text_files.csv.gz')
OUT_JSONL_GZ = Path('manifest_text_files_with_preview.jsonl.gz')

EXT_KEEP = {'.txt', '.csv', '.tsv', '.json', '.xml', '.html', '.pdf', '.md'}


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def safe_read_text(path: Path, max_bytes: int = 2_000_000):
    """Read up to max_bytes and decode as UTF-8 with replacement.

    Returns (text, bytes_read, num_replacements).
    """
    with path.open('rb') as f:
        b = f.read(max_bytes)
    # Decode with replacement so we never crash on bad OCR/encoding.
    text = b.decode('utf-8', errors='replace')
    num_repl = text.count('\ufffd')
    return text, len(b), num_repl


def text_stats(text: str):
    if not text:
        return {
            'n_chars': 0,
            'n_lines': 0,
            'n_tokens': 0,
            'blank_lines': 0,
            'avg_line_len': 0.0,
            'max_line_len': 0,
            'digit_ratio': 0.0,
            'upper_ratio': 0.0,
            'non_ascii_ratio': 0.0,
            'email_count': 0,
            'phone_like_count': 0,
            'date_like_count': 0,
            'money_like_count': 0,
        }

    lines = text.splitlines()
    n_lines = len(lines)
    blank_lines = sum(1 for ln in lines if not ln.strip())
    line_lens = [len(ln) for ln in lines] if lines else [0]
    avg_line_len = float(sum(line_lens) / max(1, len(line_lens)))
    max_line_len = max(line_lens) if line_lens else 0

    chars = list(text)
    n_chars = len(chars)
    if n_chars:
        digit_ratio = sum(c.isdigit() for c in chars) / n_chars
        upper_ratio = sum(c.isupper() for c in chars) / n_chars
        non_ascii_ratio = sum(ord(c) > 127 for c in chars) / n_chars
    else:
        digit_ratio = upper_ratio = non_ascii_ratio = 0.0

    # very lightweight pattern counts (for doc-form clustering)
    email_count = len(re.findall(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}", text))
    phone_like_count = len(re.findall(r"\+?\d[\d\s().-]{7,}\d", text))
    date_like_count = len(re.findall(r"\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b|\b\d{4}-\d{2}-\d{2}\b", text))
    money_like_count = len(re.findall(r"\$\s?\d|\b\d+[,.]?\d*\s?(USD|EUR|PLN|GBP)\b", text, flags=re.IGNORECASE))

    # tokens: cheap whitespace split (don’t overthink for manifest)
    n_tokens = len(text.split())

    return {
        'n_chars': n_chars,
        'n_lines': n_lines,
        'n_tokens': n_tokens,
        'blank_lines': blank_lines,
        'avg_line_len': avg_line_len,
        'max_line_len': max_line_len,
        'digit_ratio': digit_ratio,
        'upper_ratio': upper_ratio,
        'non_ascii_ratio': non_ascii_ratio,
        'email_count': email_count,
        'phone_like_count': phone_like_count,
        'date_like_count': date_like_count,
        'money_like_count': money_like_count,
    }


def build_manifest():
    files = []
    for p in ROOT.rglob('*'):
        if not p.is_file():
            continue
        if p.name in {'.DS_Store'} or p.name.startswith('._'):
            continue
        if p.suffix.lower() in EXT_KEEP:
            files.append(p)

    rows = []

    with gzip.open(OUT_JSONL_GZ, 'wt', encoding='utf-8') as jout:
        for i, path in enumerate(sorted(files)):
            rel_path = path.relative_to(ROOT).as_posix()
            source_root = rel_path.split('/', 1)[0] if '/' in rel_path else rel_path

            st = path.stat()
            size_bytes = st.st_size
            mtime = int(st.st_mtime)

            # doc_id is deterministic: sha256 of relative path + sha256 of bytes
            file_hash = sha256_file(path)
            doc_id = hashlib.sha256(f"{rel_path}|{file_hash}".encode('utf-8')).hexdigest()[:24]

            text, bytes_read, decode_replacements = safe_read_text(path)
            preview = text[:800].replace('\r', '\\r')

            stats = text_stats(text)

            row = {
                'doc_id': doc_id,
                'source_root': source_root,
                'rel_path': rel_path,
                'basename': path.name,
                'ext': path.suffix.lower(),
                'size_bytes': size_bytes,
                'mtime_unix': mtime,
                'sha256': file_hash,
                'bytes_read_for_preview': bytes_read,
                'decode_replacements': decode_replacements,
                **stats,
            }
            rows.append(row)

            # JSONL keeps the preview (not full text) so it stays manageable
            jout.write(json.dumps({
                'doc_id': doc_id,
                'rel_path': rel_path,
                'preview': preview,
            }, ensure_ascii=False) + "\n")

    df = pd.DataFrame(rows)

    # convenience flags for triage
    df['is_emptyish'] = (df['n_chars'] < 50)
    df['has_emails'] = (df['email_count'] > 0)
    df['has_phones'] = (df['phone_like_count'] > 0)
    df['has_dates'] = (df['date_like_count'] > 0)
    df['has_money'] = (df['money_like_count'] > 0)

    # stable ordering
    df = df.sort_values(['source_root', 'rel_path']).reset_index(drop=True)

    df.to_csv(OUT_CSV_GZ, index=False, compression='gzip')
    return df


if __name__ == '__main__':
    df = build_manifest()
    print(f"Wrote {len(df):,} rows to {OUT_CSV_GZ}")
    print(f"Wrote previews to {OUT_JSONL_GZ}")
    print(df[['source_root','ext','is_emptyish']].value_counts().head(15))
