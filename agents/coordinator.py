from __future__ import annotations

import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .base import AgentBase, EvidenceItem
from .manifest_agent import ManifestAgent
from .bytecode_agent import BytecodeAgent
from .network_agent import NetworkAgent
from .crypto_agent import CryptoAgent


@dataclass
class AnalysisResult:
    sha256: str
    apk_path: str
    evidence: list[EvidenceItem]
    agent_stats: dict[str, int]
    duration_seconds: float
    errors: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "sha256": self.sha256,
            "apk_path": self.apk_path,
            "evidence": [e.to_dict() for e in self.evidence],
            "agent_stats": self.agent_stats,
            "duration_seconds": self.duration_seconds,
            "errors": self.errors,
        }

    def save(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2))

    @classmethod
    def load(cls, path: Path) -> AnalysisResult:
        data = json.loads(path.read_text())
        return cls(
            sha256=data["sha256"],
            apk_path=data["apk_path"],
            evidence=[EvidenceItem.from_dict(e) for e in data["evidence"]],
            agent_stats=data["agent_stats"],
            duration_seconds=data["duration_seconds"],
            errors=data.get("errors", []),
        )


class Coordinator:
    def __init__(self):
        self.agents: list[AgentBase] = [
            ManifestAgent(),
            BytecodeAgent(),
            NetworkAgent(),
            CryptoAgent(),
        ]

    def analyze(self, apk_path: str, sha256: str = "") -> AnalysisResult:
        import logging
        logging.getLogger("androguard").setLevel(logging.ERROR)
        logging.getLogger("loguru").setLevel(logging.ERROR)

        start = time.time()
        all_evidence: list[EvidenceItem] = []
        agent_stats: dict[str, int] = {}
        errors: list[str] = []

        from androguard.misc import AnalyzeAPK
        print(f"  Decompiling APK...")
        apk, dalvik_objects, analysis = AnalyzeAPK(apk_path)
        print(f"  Decompiled in {time.time() - start:.1f}s")

        for agent in self.agents:
            try:
                if agent.name == "manifest":
                    items = agent.analyze(apk_path)
                elif agent.name in ("bytecode", "crypto"):
                    items = agent.analyze(apk_path, dalvik_objects=dalvik_objects, analysis=analysis)
                elif agent.name == "network":
                    items = agent.analyze(apk_path, dalvik_objects=dalvik_objects)
                else:
                    items = agent.analyze(apk_path)
                all_evidence.extend(items)
                agent_stats[agent.name] = len(items)
                print(f"  {agent.name}: {len(items)} findings")
            except Exception as e:
                errors.append(f"{agent.name}: {e}")
                agent_stats[agent.name] = 0

        all_evidence = self._deduplicate(all_evidence)

        return AnalysisResult(
            sha256=sha256 or Path(apk_path).stem,
            apk_path=apk_path,
            evidence=all_evidence,
            agent_stats=agent_stats,
            duration_seconds=round(time.time() - start, 2),
            errors=errors,
        )

    @staticmethod
    def _deduplicate(evidence: list[EvidenceItem]) -> list[EvidenceItem]:
        seen: dict[str, EvidenceItem] = {}
        for item in evidence:
            key = f"{item.source_agent}:{item.entity}"
            if key not in seen:
                seen[key] = item
        return list(seen.values())
