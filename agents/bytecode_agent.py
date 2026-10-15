from __future__ import annotations

import json
import re
from pathlib import Path

from .base import AgentBase, EvidenceItem, EvidenceType
from config.settings import SENSITIVE_APIS_FILE


def _load_sensitive_apis() -> dict[str, tuple[int, str]]:
    """Returns {api_substring: (tier, category)}."""
    mapping = {}
    if SENSITIVE_APIS_FILE.exists():
        data = json.loads(SENSITIVE_APIS_FILE.read_text())
        for category, info in data.items():
            tier = info["tier"]
            for api in info["apis"]:
                mapping[api] = (tier, category)
    return mapping


SENSITIVE_API_MAP = _load_sensitive_apis()

OBFUSCATION_PATTERN = re.compile(r"^[a-zA-Z]{1,2}$")


class BytecodeAgent(AgentBase):
    name = "bytecode"

    def analyze(self, apk_path: str, dalvik_objects=None, analysis=None) -> list[EvidenceItem]:
        if analysis is None:
            from androguard.misc import AnalyzeAPK
            _, dalvik_objects, analysis = AnalyzeAPK(apk_path)

        evidence: list[EvidenceItem] = []

        evidence.extend(self._extract_sensitive_api_calls(analysis))
        evidence.extend(self._extract_call_graph_structure(analysis))
        evidence.extend(self._detect_obfuscation(analysis))

        return evidence

    def _extract_sensitive_api_calls(self, analysis) -> list[EvidenceItem]:
        items = []
        seen = set()

        for method in analysis.get_methods():
            method_name = str(method.get_method())
            for api_pattern, (tier, category) in SENSITIVE_API_MAP.items():
                if api_pattern in method_name and method_name not in seen:
                    seen.add(method_name)

                    callers = []
                    try:
                        for _, caller, _ in method.get_xref_from():
                            callers.append(str(caller))
                    except Exception:
                        pass

                    items.append(EvidenceItem(
                        evidence_type=EvidenceType.API_USAGE,
                        source_agent=self.name,
                        entity=method_name,
                        attributes={
                            "sensitivity_tier": tier,
                            "category": category,
                            "caller_count": len(callers),
                        },
                        related_entities=callers[:10],
                        description=f"Sensitive API ({category}, tier {tier}): {method_name}",
                    ))

        return items

    def _extract_call_graph_structure(self, analysis) -> list[EvidenceItem]:
        items = []
        method_calls = []

        for method in analysis.get_methods():
            if method.is_external():
                continue

            method_name = str(method.get_method())
            callees = []
            try:
                for _, callee, _ in method.get_xref_to():
                    callees.append(str(callee))
            except Exception:
                continue

            if callees:
                method_calls.append((method_name, callees))

        for method_name, callees in method_calls[:500]:
            items.append(EvidenceItem(
                evidence_type=EvidenceType.METHOD_CALL,
                source_agent=self.name,
                entity=method_name,
                attributes={
                    "is_internal": True,
                    "callee_count": len(callees),
                },
                related_entities=callees[:20],
                description=f"Method with {len(callees)} callees",
            ))

        return items

    def _detect_obfuscation(self, analysis) -> list[EvidenceItem]:
        items = []
        obfuscated_count = 0
        total_classes = 0

        for cls in analysis.get_classes():
            total_classes += 1
            class_name = str(cls.name)
            parts = class_name.rstrip(";").split("/")
            short_name = parts[-1] if parts else class_name

            if OBFUSCATION_PATTERN.match(short_name):
                obfuscated_count += 1

        if total_classes > 0:
            ratio = obfuscated_count / total_classes
            items.append(EvidenceItem(
                evidence_type=EvidenceType.CLASS,
                source_agent=self.name,
                entity="obfuscation_metric",
                attributes={
                    "obfuscated_classes": obfuscated_count,
                    "total_classes": total_classes,
                    "obfuscation_ratio": round(ratio, 3),
                    "is_likely_obfuscated": ratio > 0.3,
                },
                confidence=min(1.0, ratio * 2),
                description=f"Obfuscation: {obfuscated_count}/{total_classes} classes ({ratio:.1%})",
            ))

        return items
