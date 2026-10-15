"""Rebuild all graphs from evidence JSONs with updated feature encoding."""
import json
import csv
import torch
from pathlib import Path
from agents.base import EvidenceItem, EvidenceType
from graph.builder import GraphBuilder

EVIDENCE_DIR = Path("data/evidence")
GRAPHS_DIR = Path("data/graphs")
LABELS_FILE = Path("data/labels/labels.csv")


def load_labels():
    mapping = {}
    with open(LABELS_FILE) as f:
        for row in csv.DictReader(f):
            sha = row.get("sha256", "").strip()
            label = row.get("label", "").strip().lower()
            if sha:
                mapping[sha] = 1 if label in ("malware", "1", "malicious") else 0
    return mapping


def evidence_from_json(data: dict) -> list[EvidenceItem]:
    items = []
    for e in data.get("evidence", []):
        items.append(EvidenceItem(
            evidence_type=EvidenceType(e["evidence_type"]),
            source_agent=e["source_agent"],
            entity=e["entity"],
            attributes=e.get("attributes", {}),
            related_entities=e.get("related_entities", []),
            confidence=e.get("confidence", 1.0),
            description=e.get("description", ""),
        ))
    return items


def main():
    labels = load_labels()
    builder = GraphBuilder()
    GRAPHS_DIR.mkdir(parents=True, exist_ok=True)

    evidence_files = sorted(EVIDENCE_DIR.glob("*.json"))
    print(f"Rebuilding {len(evidence_files)} graphs from evidence...", flush=True)

    success = 0
    failed = 0
    for i, ef in enumerate(evidence_files):
        sha = ef.stem
        label = labels.get(sha, -1)
        if label < 0:
            continue

        try:
            data = json.loads(ef.read_text())
            evidence = evidence_from_json(data)
            graph = builder.build(evidence, label=label)
            torch.save(graph, GRAPHS_DIR / f"{sha}.pt")
            success += 1
        except Exception as e:
            print(f"  Failed {sha[:16]}: {e}", flush=True)
            failed += 1

        if (i + 1) % 50 == 0:
            print(f"  [{i+1}/{len(evidence_files)}] {success} ok, {failed} failed", flush=True)

    print(f"\nDone: {success} rebuilt, {failed} failed", flush=True)


if __name__ == "__main__":
    main()
