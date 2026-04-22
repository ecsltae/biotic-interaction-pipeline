#!/usr/bin/env python3
"""
Batch article processor — biotic interaction sentence detection.

Two-stage pipeline:
  1. GloBI pre-filter  (local, ~1ms/sentence)
  2. Distilled BiomedBERT via HTTP API  (~14ms/sentence on GPU)

Usage:
    # Folder of .txt files
    python process_articles.py --input articles/ --output results.csv

    # CSV with a text column
    python process_articles.py --input abstracts.csv --text-col abstract --output results.csv

    # Override API URL and threshold
    python process_articles.py --input articles/ --output results.csv \\
        --api http://HOST:8003 --threshold 0.4

    # Custom filter dictionaries
    python process_articles.py --input articles/ --output results.csv \\
        --interaction-dict data/interaction_dict.csv \\
        --species-dict data/species_dict.csv
"""

import argparse
import csv
import sys
import time
from pathlib import Path
from typing import Iterator

import requests

from biotic_pipeline.filter import build_filter, split_sentences

# ── Article loaders ───────────────────────────────────────────────────────

def iter_txt_folder(folder: Path) -> Iterator[tuple[str, str]]:
    files = sorted(folder.glob("*.txt"))
    if not files:
        print(f"WARNING: no .txt files found in {folder}", file=sys.stderr)
    for f in files:
        yield f.stem, f.read_text(encoding="utf-8", errors="replace")


def iter_csv_file(path: Path, text_col: str, id_col: str | None) -> Iterator[tuple[str, str]]:
    import pandas as pd
    df = pd.read_csv(path)
    if text_col not in df.columns:
        raise ValueError(f"Column '{text_col}' not found. Available: {df.columns.tolist()}")
    resolved_id = id_col if (id_col and id_col in df.columns) else df.columns[0]
    for _, row in df.iterrows():
        yield str(row[resolved_id]), str(row[text_col])


# ── API caller ────────────────────────────────────────────────────────────

def classify_batch(sentences: list[str], api_url: str, threshold: float,
                   retries: int = 3) -> list[dict]:
    payload = {"sentences": sentences, "threshold": threshold}
    for attempt in range(retries):
        try:
            r = requests.post(f"{api_url}/batch", json=payload, timeout=60)
            r.raise_for_status()
            return r.json()["results"]
        except Exception as e:
            if attempt == retries - 1:
                raise
            time.sleep(2 ** attempt)
    return []


# ── Main pipeline ─────────────────────────────────────────────────────────

def process(articles: Iterator[tuple[str, str]], api_url: str, output: Path,
            threshold: float, batch_size: int,
            interaction_dict: Path | None, species_dict: Path | None) -> None:

    passes = build_filter(
        interaction_dict_csv=str(interaction_dict) if interaction_dict else None,
        species_dict_csv=str(species_dict) if species_dict else None,
    )

    total_articles = total_sentences = total_filtered = total_positive = 0
    t_start = time.time()

    with open(output, "w", newline="", encoding="utf-8") as fout:
        writer = csv.DictWriter(fout, fieldnames=[
            "article_id", "sentence", "label", "probability", "threshold_used"
        ])
        writer.writeheader()

        pending_sentences: list[str] = []
        pending_meta: list[tuple[str, str]] = []

        def flush_batch():
            nonlocal total_positive
            if not pending_sentences:
                return
            results = classify_batch(pending_sentences, api_url, threshold)
            for (aid, sent), res in zip(pending_meta, results):
                writer.writerow({
                    "article_id": aid,
                    "sentence": sent,
                    "label": res["label"],
                    "probability": res["probability"],
                    "threshold_used": res["threshold_used"],
                })
                if res["label"] == 1:
                    total_positive += 1
            fout.flush()
            pending_sentences.clear()
            pending_meta.clear()

        for article_id, text in articles:
            total_articles += 1
            sentences = split_sentences(text)
            total_sentences += len(sentences)

            for sent in sentences:
                if not passes(sent):
                    continue
                total_filtered += 1
                pending_sentences.append(sent)
                pending_meta.append((article_id, sent))
                if len(pending_sentences) >= batch_size:
                    flush_batch()

            if total_articles % 100 == 0:
                elapsed = time.time() - t_start
                rate = total_articles / elapsed
                print(
                    f"  {total_articles} articles | {total_sentences} sentences | "
                    f"{total_filtered} passed filter | {total_positive} positive | "
                    f"{rate:.1f} art/s",
                    flush=True,
                )

        flush_batch()

    elapsed = time.time() - t_start
    filter_rate = total_filtered / total_sentences * 100 if total_sentences else 0
    print(f"\n=== Done ===")
    print(f"  Articles:        {total_articles}")
    print(f"  Total sentences: {total_sentences}")
    print(f"  Passed filter:   {total_filtered} ({filter_rate:.1f}%)")
    print(f"  Positive:        {total_positive}")
    print(f"  Time:            {elapsed:.0f}s ({elapsed/60:.1f} min)")
    print(f"  Output:          {output}")


# ── CLI ───────────────────────────────────────────────────────────────────

def main():
    # Read defaults from config.toml if present
    try:
        from biotic_pipeline.config import get_config
        cfg = get_config()
        default_api = f"http://{cfg.server.host}:{cfg.server.port}"
        default_threshold = cfg.model.threshold
        default_int_dict  = cfg.data.interaction_dict
        default_sp_dict   = cfg.data.species_dict
    except Exception:
        default_api       = "http://localhost:8003"
        default_threshold = 0.25
        default_int_dict  = None
        default_sp_dict   = None

    parser = argparse.ArgumentParser(description="Batch biotic interaction detector")
    parser.add_argument("--input",            required=True)
    parser.add_argument("--output",           required=True)
    parser.add_argument("--api",              default=default_api)
    parser.add_argument("--threshold",        type=float, default=default_threshold)
    parser.add_argument("--batch-size",       type=int,   default=500)
    parser.add_argument("--text-col",         default="full_text")
    parser.add_argument("--id-col",           default=None)
    parser.add_argument("--interaction-dict", default=default_int_dict)
    parser.add_argument("--species-dict",     default=default_sp_dict)
    args = parser.parse_args()

    inp = Path(args.input)
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    int_dict = Path(args.interaction_dict) if args.interaction_dict else None
    sp_dict  = Path(args.species_dict)     if args.species_dict     else None

    print(f"Checking API at {args.api} ...", flush=True)
    try:
        r = requests.get(f"{args.api}/health", timeout=5)
        info = r.json()
        print(f"  API ok — model_dir={info.get('model_dir', '?')}  device={info.get('device', '?')}", flush=True)
    except Exception as e:
        print(f"ERROR: cannot reach API at {args.api}: {e}", file=sys.stderr)
        sys.exit(1)

    if inp.is_dir():
        articles = iter_txt_folder(inp)
    elif inp.suffix == ".csv":
        articles = iter_csv_file(inp, args.text_col, args.id_col)
    else:
        print("ERROR: --input must be a folder of .txt files or a .csv file", file=sys.stderr)
        sys.exit(1)

    process(articles, args.api, out, args.threshold, args.batch_size, int_dict, sp_dict)


if __name__ == "__main__":
    main()
