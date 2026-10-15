"""Download APKs from AndroZoo using their API.

Usage:
    python -m pipeline.download --input data/labels/samples.csv --output data/raw/ --limit 200
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import requests
from tqdm import tqdm

from config.settings import ANDROZOO_API_KEY, RAW_DIR, LABELS_DIR

ANDROZOO_URL = "https://androzoo.uni.lu/api/download"


def download_apk(sha256: str, output_dir: Path, api_key: str = "") -> Path | None:
    api_key = api_key or ANDROZOO_API_KEY
    if not api_key:
        raise ValueError("ANDROZOO_API_KEY not set in .env")

    output_path = output_dir / f"{sha256}.apk"
    if output_path.exists():
        return output_path

    try:
        resp = requests.get(
            ANDROZOO_URL,
            params={"apikey": api_key, "sha256": sha256},
            stream=True,
            timeout=120,
        )
        resp.raise_for_status()

        output_dir.mkdir(parents=True, exist_ok=True)
        with open(output_path, "wb") as f:
            for chunk in resp.iter_content(chunk_size=8192):
                f.write(chunk)

        return output_path
    except Exception as e:
        print(f"Failed to download {sha256}: {e}")
        return None


def download_batch(
    csv_path: str | Path,
    output_dir: str | Path | None = None,
    limit: int = 0,
    api_key: str = "",
):
    csv_path = Path(csv_path)
    output_dir = Path(output_dir or RAW_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)

    if limit > 0:
        rows = rows[:limit]

    print(f"Downloading {len(rows)} APKs to {output_dir}")
    success = 0
    for row in tqdm(rows, desc="Downloading"):
        sha256 = row.get("sha256", row.get("SHA256", "")).strip()
        if not sha256:
            continue
        result = download_apk(sha256, output_dir, api_key)
        if result:
            success += 1

    print(f"Downloaded {success}/{len(rows)} APKs")


def main():
    parser = argparse.ArgumentParser(description="Download APKs from AndroZoo")
    parser.add_argument("--input", required=True, help="CSV with sha256 column")
    parser.add_argument("--output", default=str(RAW_DIR))
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    download_batch(args.input, args.output, args.limit)


if __name__ == "__main__":
    main()
