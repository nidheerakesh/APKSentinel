from enum import IntEnum


class NodeType(IntEnum):
    METHOD = 0
    CLASS = 1
    PERMISSION = 2
    API_CALL = 3
    NETWORK_ENDPOINT = 4
    CRYPTO_OP = 5


class EdgeType(IntEnum):
    CALLS = 0
    USES_PERMISSION = 1
    ACCESSES_API = 2
    CONNECTS_TO = 3
    USES_CRYPTO = 4
    BELONGS_TO = 5


EVIDENCE_TYPE_TO_NODE_TYPE = {
    "method_call": NodeType.METHOD,
    "permission": NodeType.PERMISSION,
    "api_usage": NodeType.API_CALL,
    "network_indicator": NodeType.NETWORK_ENDPOINT,
    "crypto_operation": NodeType.CRYPTO_OP,
    "component": NodeType.CLASS,
    "class": NodeType.CLASS,
    "data_flow": NodeType.METHOD,
}


def infer_edge_type(src_type: NodeType, dst_type: NodeType) -> EdgeType:
    if src_type == NodeType.METHOD and dst_type == NodeType.METHOD:
        return EdgeType.CALLS
    if dst_type == NodeType.PERMISSION:
        return EdgeType.USES_PERMISSION
    if dst_type == NodeType.API_CALL:
        return EdgeType.ACCESSES_API
    if dst_type == NodeType.NETWORK_ENDPOINT:
        return EdgeType.CONNECTS_TO
    if dst_type == NodeType.CRYPTO_OP:
        return EdgeType.USES_CRYPTO
    if src_type == NodeType.METHOD and dst_type == NodeType.CLASS:
        return EdgeType.BELONGS_TO
    return EdgeType.CALLS
