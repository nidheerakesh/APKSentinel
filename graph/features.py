from __future__ import annotations

import torch
from .schema import NodeType

NUM_NODE_TYPES = 6
FEATURE_DIM = 25


def encode_node_features(
    node_type: NodeType,
    attributes: dict,
) -> torch.Tensor:
    features = torch.zeros(FEATURE_DIM)

    features[int(node_type)] = 1.0

    features[6] = _normalize(attributes.get("sensitivity_tier",
                  attributes.get("risk_level", 0)), max_val=3)
    features[7] = float(attributes.get("is_likely_obfuscated",
                  attributes.get("is_obfuscated", False)))
    features[8] = float(attributes.get("confidence", 1.0))
    features[9] = _normalize(attributes.get("risk_level",
                  attributes.get("perm_risk", 0)), max_val=3)
    features[10] = float(attributes.get("is_exported", False))
    features[11] = float(attributes.get("has_intent_filter", False))
    features[12] = _normalize(attributes.get("entropy",
                   attributes.get("net_entropy", 0)), max_val=5)
    features[13] = float(attributes.get("uses_reflection", False))
    features[14] = float(attributes.get("uses_dynamic_loading", False))
    features[15] = _crypto_strength(attributes)
    features[16] = _normalize(attributes.get("string_entropy", 0), max_val=5)
    features[17] = _normalize(attributes.get("callee_count",
                   attributes.get("caller_count", 0)), max_val=20)
    features[18] = float(attributes.get("is_dangerous", False))
    features[19] = float(attributes.get("is_known_benign", False))
    features[20] = float(attributes.get("is_https", False))
    features[21] = float(attributes.get("is_raw_ip", False))
    features[22] = float(attributes.get("is_internal", False))
    features[23] = float(attributes.get("non_standard_port", False))
    features[24] = _normalize(attributes.get("length", 0), max_val=200)

    return features


def _normalize(value, max_val: float = 1.0) -> float:
    if isinstance(value, bool):
        return float(value)
    try:
        return min(1.0, float(value) / max_val)
    except (TypeError, ValueError):
        return 0.0


def _crypto_strength(attrs: dict) -> float:
    category = attrs.get("crypto_category", "")
    if category in ("ssl_bypass", "hardcoded_key"):
        return 0.1
    if category in ("cipher", "keygen", "keypair"):
        return 0.7
    if category in ("digest", "mac"):
        return 0.5
    if category == "android_keystore":
        return 0.9
    return 0.0
