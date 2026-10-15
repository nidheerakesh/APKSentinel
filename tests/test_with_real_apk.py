"""
Test the full agent pipeline on a real APK.
Usage: python -m tests.test_with_real_apk path/to/sample.apk
"""
import sys
import json
from pathlib import Path

from agents.coordinator import Coordinator
from graph.builder import GraphBuilder


def test_real_apk(apk_path: str):
    print(f"Analyzing: {apk_path}\n")

    coordinator = Coordinator()
    result = coordinator.analyze(apk_path)

    print(f"Duration: {result.duration_seconds}s")
    print(f"Total evidence: {len(result.evidence)}")
    print(f"Agent stats: {result.agent_stats}")
    if result.errors:
        print(f"Errors: {result.errors}")

    print("\n--- Evidence by type ---")
    by_type: dict[str, int] = {}
    for item in result.evidence:
        by_type[item.evidence_type.value] = by_type.get(item.evidence_type.value, 0) + 1
    for t, count in sorted(by_type.items()):
        print(f"  {t}: {count}")

    print("\n--- High-risk findings ---")
    for item in result.evidence:
        risk = item.attributes.get("risk_level", item.attributes.get("sensitivity_tier", 0))
        if risk >= 3:
            print(f"  [{item.source_agent}] {item.description}")

    builder = GraphBuilder()
    data = builder.build(result.evidence, label=-1)
    print(f"\n--- Graph ---")
    print(f"  Nodes: {data.num_nodes}")
    print(f"  Edges: {data.edge_index.size(1)}")
    print(f"  Feature shape: {data.x.shape}")

    evidence_path = Path(f"data/evidence/{result.sha256}.json")
    result.save(evidence_path)
    print(f"\nEvidence saved to {evidence_path}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python -m tests.test_with_real_apk <path_to_apk>")
        sys.exit(1)
    test_real_apk(sys.argv[1])
