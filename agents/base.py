from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class EvidenceType(str, Enum):
    METHOD_CALL = "method_call"
    PERMISSION = "permission"
    API_USAGE = "api_usage"
    NETWORK_INDICATOR = "network_indicator"
    CRYPTO_OPERATION = "crypto_operation"
    DATA_FLOW = "data_flow"
    COMPONENT = "component"
    CLASS = "class"


@dataclass
class EvidenceItem:
    evidence_type: EvidenceType
    source_agent: str
    entity: str
    attributes: dict[str, Any] = field(default_factory=dict)
    related_entities: list[str] = field(default_factory=list)
    confidence: float = 1.0
    description: str = ""

    def to_dict(self) -> dict:
        return {
            "evidence_type": self.evidence_type.value,
            "source_agent": self.source_agent,
            "entity": self.entity,
            "attributes": self.attributes,
            "related_entities": self.related_entities,
            "confidence": self.confidence,
            "description": self.description,
        }

    @classmethod
    def from_dict(cls, d: dict) -> EvidenceItem:
        return cls(
            evidence_type=EvidenceType(d["evidence_type"]),
            source_agent=d["source_agent"],
            entity=d["entity"],
            attributes=d.get("attributes", {}),
            related_entities=d.get("related_entities", []),
            confidence=d.get("confidence", 1.0),
            description=d.get("description", ""),
        )


class AgentBase(ABC):
    name: str = "base"

    @abstractmethod
    def analyze(self, apk_path: str) -> list[EvidenceItem]:
        ...
