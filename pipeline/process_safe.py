"""Memory-safe APK processing — one at a time, smallest first, with size limit."""
import gc
import os
import sys
import csv
import subprocess
from pathlib import Path

GRAPHS_DIR = Path("data/graphs")
RAW_DIR = Path("data/raw")
LABELS_FILE = Path("data/labels/labels.csv")
MAX_APK_SIZE_MB = 20

def load_labels():
    mapping = {}
    with open(LABELS_FILE) as f:
        for row in csv.DictReader(f):
            sha = row.get("sha256", "").strip()
            label = row.get("label", "").strip().lower()
            if sha:
                mapping[sha] = 1 if label in ("malware", "1", "malicious") else 0
    return mapping

def get_unprocessed():
    raw = {p.stem: p for p in RAW_DIR.glob("*.apk")}
    done = {p.stem for p in GRAPHS_DIR.glob("*.pt")}
    todo = {k: v for k, v in raw.items() if k not in done}
    return sorted(todo.items(), key=lambda x: x[1].stat().st_size)

def process_one(sha256, apk_path, label):
    """Process single APK in subprocess to isolate memory."""
    code = f'''
import gc, torch, sys
sys.path.insert(0, ".")
from agents.coordinator import Coordinator
from graph.builder import GraphBuilder
from pathlib import Path

try:
    c = Coordinator()
    b = GraphBuilder()
    result = c.analyze("{apk_path}", sha256="{sha256}")
    Path("data/evidence").mkdir(parents=True, exist_ok=True)
    result.save(Path("data/evidence/{sha256}.json"))
    data = b.build(result.evidence, label={label})
    Path("data/graphs").mkdir(parents=True, exist_ok=True)
    torch.save(data, Path("data/graphs/{sha256}.pt"))
    print("OK")
except Exception as e:
    print(f"ERR: {{e}}")
finally:
    gc.collect()
'''
    try:
        result = subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True, text=True, timeout=120
        )
        output = result.stdout.strip()
        if "OK" in output:
            return True
        print(f"  Failed: {output[-200:] if output else result.stderr[-200:]}", flush=True)
        return False
    except subprocess.TimeoutExpired:
        print(f"  Timeout", flush=True)
        return False

def main():
    labels = load_labels()
    todo = get_unprocessed()
    print(f"Unprocessed: {len(todo)} APKs", flush=True)

    skipped = 0
    success = 0
    failed = 0

    for sha, path in todo:
        size_mb = path.stat().st_size / (1024 * 1024)
        if size_mb > MAX_APK_SIZE_MB:
            skipped += 1
            continue

        label = labels.get(sha, -1)
        print(f"[{success+failed+1}/{len(todo)-skipped}] {sha[:16]}... ({size_mb:.1f}MB) label={label}", flush=True)
        if process_one(sha, str(path), label):
            success += 1
        else:
            failed += 1
        gc.collect()

    print(f"\nDone: {success} ok, {failed} failed, {skipped} skipped (>{MAX_APK_SIZE_MB}MB)", flush=True)

if __name__ == "__main__":
    main()
