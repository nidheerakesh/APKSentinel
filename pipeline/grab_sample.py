"""
Grab a single APK from AndroZoo and run the full agent pipeline on it.

Usage:
    python -m pipeline.grab_sample <sha256> [--label malware|benign]
    python -m pipeline.grab_sample --demo   # grab a few known samples
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from config.settings import RAW_DIR, EVIDENCE_DIR, GRAPHS_DIR
from pipeline.download import download_apk
from pipeline.process import process_apk
from agents.coordinator import Coordinator
from graph.builder import GraphBuilder

# Known banking trojan samples from public threat reports (VirusTotal positive)
DEMO_SAMPLES = {
    # Anubis banker - well-documented family
    "a]": ("anubis", "malware"),
    # You can add more known hashes here from threat intel reports
}


def grab_and_analyze(sha256: str, label: str = "unknown"):
    print(f"Downloading {sha256[:16]}...")
    apk_path = download_apk(sha256, RAW_DIR)
    if not apk_path:
        print("Download failed")
        return

    print(f"Downloaded: {apk_path}")
    print(f"Running 4-agent analysis...")

    coordinator = Coordinator()
    builder = GraphBuilder()

    label_int = 1 if label == "malware" else (0 if label == "benign" else -1)
    success = process_apk(apk_path, coordinator, builder, label=label_int)

    if success:
        print(f"\nEvidence: {EVIDENCE_DIR / f'{sha256}.json'}")
        print(f"Graph:    {GRAPHS_DIR / f'{sha256}.pt'}")
        print(f"\nTo inspect: python -m tests.test_with_real_apk {apk_path}")
    else:
        print("Processing failed")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("sha256", nargs="?")
    parser.add_argument("--label", default="unknown", choices=["malware", "benign", "unknown"])
    parser.add_argument("--demo", action="store_true", help="Download demo samples")
    args = parser.parse_args()

    if args.demo:
        for sha, (family, label) in DEMO_SAMPLES.items():
            print(f"\n{'='*60}")
            print(f"Sample: {family} ({label})")
            grab_and_analyze(sha, label)
    elif args.sha256:
        grab_and_analyze(args.sha256, args.label)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
