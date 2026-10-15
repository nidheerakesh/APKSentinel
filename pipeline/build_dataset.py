"""
Download AndroZoo's latest.csv.gz, stream-decompress, and filter for:
- Malware: vt_detection >= 15
- Benign: vt_detection == 0, from Google Play

Usage: python -m pipeline.build_dataset --limit 300
"""
from __future__ import annotations

import argparse
import csv
import gzip
import io
import random
from pathlib import Path

import requests

from config.settings import ANDROZOO_API_KEY, LABELS_DIR

LATEST_CSV_URL = "https://androzoo.uni.lu/static/lists/latest.csv.gz"
LATEST_CSV_LOCAL = LABELS_DIR / "latest.csv.gz"

BANKING_KEYWORDS = [
    "bank", "secure", "update", "flash", "player",
    "security", "certificate", "chrome", "system.service",
]


def download_latest_csv(output_path: Path):
    """Download latest.csv.gz with resume support."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    existing_size = output_path.stat().st_size if output_path.exists() else 0

    headers = {}
    if existing_size > 0:
        headers["Range"] = f"bytes={existing_size}-"
        print(f"Resuming from {existing_size / (1024*1024):.0f} MB...")

    print(f"Downloading latest.csv.gz (this is ~8GB, will take a while)...")

    session = requests.Session()
    adapter = requests.adapters.HTTPAdapter(max_retries=3)
    session.mount("https://", adapter)

    resp = session.get(LATEST_CSV_URL, stream=True, timeout=(30, 300), headers=headers)
    resp.raise_for_status()

    total = int(resp.headers.get("content-length", 0)) + existing_size
    downloaded = existing_size
    mode = "ab" if existing_size > 0 else "wb"

    with open(output_path, mode) as f:
        for chunk in resp.iter_content(chunk_size=1024 * 1024):
            f.write(chunk)
            downloaded += len(chunk)
            mb = downloaded / (1024 * 1024)
            if total:
                pct = 100 * downloaded / total
                print(f"\r  {mb:.0f} MB / {total/(1024*1024):.0f} MB ({pct:.1f}%)", end="", flush=True)
            else:
                print(f"\r  {mb:.0f} MB", end="", flush=True)

    print(f"\nDone: {output_path}")


def filter_samples(
    csv_gz_path: Path,
    malware_limit: int = 150,
    benign_limit: int = 150,
    min_vt: int = 15,
    year_min: int = 2020,
    min_size: int = 100_000,
    max_size: int = 50_000_000,
) -> tuple[list[dict], list[dict]]:
    malware = []
    benign = []
    rows = 0

    print(f"Scanning for {malware_limit} malware + {benign_limit} benign (year >= {year_min})...")

    with gzip.open(csv_gz_path, "rt", encoding="utf-8", errors="ignore") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows += 1
            if rows % 500_000 == 0:
                print(f"  scanned {rows:,} rows — {len(malware)} malware, {len(benign)} benign")

            if len(malware) >= malware_limit and len(benign) >= benign_limit:
                break

            try:
                sha256 = row.get("sha256", "").strip()
                vt = int(row.get("vt_detection", -1))
                size = int(row.get("apk_size", 0))
                pkg = row.get("pkg_name", "").strip()
                dex_date = row.get("dex_date", "")
                markets = row.get("markets", "")

                if not sha256 or size < min_size or size > max_size:
                    continue

                year = int(dex_date[:4]) if dex_date and len(dex_date) >= 4 else 0
                if year < year_min:
                    continue

                if vt >= min_vt and len(malware) < malware_limit:
                    is_banking = any(kw in pkg.lower() for kw in BANKING_KEYWORDS)
                    malware.append({
                        "sha256": sha256,
                        "label": "malware",
                        "family": "banking_suspect" if is_banking else "malware",
                        "first_seen": dex_date,
                        "vt_detection": vt,
                        "pkg_name": pkg,
                        "apk_size": size,
                    })

                elif vt == 0 and len(benign) < benign_limit and "play.google.com" in markets:
                    benign.append({
                        "sha256": sha256,
                        "label": "benign",
                        "family": "benign",
                        "first_seen": dex_date,
                        "vt_detection": 0,
                        "pkg_name": pkg,
                        "apk_size": size,
                    })

            except (ValueError, KeyError):
                continue

    print(f"Scanned {rows:,} rows total")
    return malware, benign


def save_csv(malware: list[dict], benign: list[dict], path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    samples = malware + benign
    random.shuffle(samples)

    fields = ["sha256", "label", "family", "first_seen", "vt_detection", "pkg_name", "apk_size"]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(samples)

    print(f"Saved {len(samples)} to {path}")
    print(f"  Malware: {len(malware)}, Benign: {len(benign)}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=300)
    parser.add_argument("--min-vt", type=int, default=15)
    parser.add_argument("--year-min", type=int, default=2020)
    parser.add_argument("--skip-download", action="store_true")
    args = parser.parse_args()

    if not args.skip_download:
        download_latest_csv(LATEST_CSV_LOCAL)

    half = args.limit // 2
    malware, benign = filter_samples(
        LATEST_CSV_LOCAL, malware_limit=half, benign_limit=half,
        min_vt=args.min_vt, year_min=args.year_min,
    )

    save_csv(malware, benign, LABELS_DIR / "labels.csv")


if __name__ == "__main__":
    main()
