"""Process APKs: run agents, build graphs, save .pt files.

Usage:
    python -m pipeline.process --input data/raw/ --labels data/labels/labels.csv
"""
from __future__ import annotations

import argparse
import csv
import gc
from pathlib import Path

import torch
from tqdm import tqdm

from agents.coordinator import Coordinator
from graph.builder import GraphBuilder
from config.settings import RAW_DIR, LABELS_DIR, EVIDENCE_DIR, GRAPHS_DIR


def load_labels(labels_path: Path) -> dict[str, int]:
    mapping = {}
    if not labels_path.exists():
        return mapping

    with open(labels_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            sha = row.get("sha256", row.get("SHA256", "")).strip()
            label = row.get("label", row.get("class", "")).strip().lower()
            if sha:
                mapping[sha] = 1 if label in ("malware", "1", "malicious") else 0

    return mapping


def process_apk(
    apk_path: Path,
    coordinator: Coordinator,
    builder: GraphBuilder,
    label: int = -1,
) -> bool:
    sha256 = apk_path.stem
    graph_path = GRAPHS_DIR / f"{sha256}.pt"
    evidence_path = EVIDENCE_DIR / f"{sha256}.json"

    if graph_path.exists():
        return True

    try:
        result = coordinator.analyze(str(apk_path), sha256=sha256)

        EVIDENCE_DIR.mkdir(parents=True, exist_ok=True)
        result.save(evidence_path)

        data = builder.build(result.evidence, label=label)

        GRAPHS_DIR.mkdir(parents=True, exist_ok=True)
        torch.save(data, graph_path)

        return True
    except Exception as e:
        print(f"Error processing {sha256}: {e}")
        return False


def process_batch(
    apk_dir: str | Path,
    labels_path: str | Path | None = None,
):
    apk_dir = Path(apk_dir)
    labels = {}
    if labels_path:
        labels = load_labels(Path(labels_path))

    apk_files = sorted(apk_dir.glob("*.apk"))
    if not apk_files:
        print(f"No APK files found in {apk_dir}")
        return

    print(f"Processing {len(apk_files)} APKs")
    coordinator = Coordinator()
    builder = GraphBuilder()

    success = 0
    for apk_path in tqdm(apk_files, desc="Processing"):
        sha256 = apk_path.stem
        label = labels.get(sha256, -1)
        if process_apk(apk_path, coordinator, builder, label):
            success += 1
        gc.collect()

    print(f"Processed {success}/{len(apk_files)} APKs")


def main():
    parser = argparse.ArgumentParser(description="Process APKs into graphs")
    parser.add_argument("--input", default=str(RAW_DIR))
    parser.add_argument("--labels", default=str(LABELS_DIR / "labels.csv"))
    args = parser.parse_args()

    process_batch(args.input, args.labels)


if __name__ == "__main__":
    main()
