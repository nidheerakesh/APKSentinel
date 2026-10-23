"""
End-to-end pipeline test using synthetic evidence (no real APK needed).
Verifies: evidence → graph → GAT → explainer → attribution.
"""
import torch
from agents.base import EvidenceItem, EvidenceType
from graph.builder import GraphBuilder
from model.gat import MalwareGAT
from explain.explainer import explain_prediction, ExplanationResult
from explain.attribution import attribute_to_agents, generate_explanation_text
from config.settings import ExplainConfig


def make_malware_evidence() -> list[EvidenceItem]:
    """Simulate evidence a banking trojan would produce."""
    return [
        EvidenceItem(
            evidence_type=EvidenceType.PERMISSION,
            source_agent="manifest",
            entity="android.permission.RECEIVE_SMS",
            attributes={"risk_level": 3, "is_dangerous": True, "category": "sms"},
            description="Permission: RECEIVE_SMS",
        ),
        EvidenceItem(
            evidence_type=EvidenceType.PERMISSION,
            source_agent="manifest",
            entity="android.permission.BIND_ACCESSIBILITY_SERVICE",
            attributes={"risk_level": 3, "is_dangerous": True, "category": "accessibility"},
            description="Permission: BIND_ACCESSIBILITY_SERVICE",
        ),
        EvidenceItem(
            evidence_type=EvidenceType.PERMISSION,
            source_agent="manifest",
            entity="android.permission.INTERNET",
            attributes={"risk_level": 1, "is_dangerous": False, "category": "network"},
            description="Permission: INTERNET",
        ),
        EvidenceItem(
            evidence_type=EvidenceType.API_USAGE,
            source_agent="bytecode",
            entity="Landroid/telephony/SmsManager;->sendTextMessage",
            attributes={"sensitivity_tier": 3, "category": "sms", "caller_count": 2},
            related_entities=["Lcom/malware/Stealer;->exfiltrate"],
            description="Sensitive API (sms, tier 3)",
        ),
        EvidenceItem(
            evidence_type=EvidenceType.METHOD_CALL,
            source_agent="bytecode",
            entity="Lcom/malware/Stealer;->exfiltrate",
            attributes={"is_internal": True, "callee_count": 3},
            related_entities=[
                "Ljava/net/HttpURLConnection;->connect",
                "Landroid/telephony/SmsManager;->sendTextMessage",
            ],
            description="Method with 3 callees",
        ),
        EvidenceItem(
            evidence_type=EvidenceType.METHOD_CALL,
            source_agent="bytecode",
            entity="Lcom/malware/Overlay;->showFakeLogin",
            attributes={"is_internal": True, "callee_count": 1},
            related_entities=["Landroid/view/WindowManager;->addView"],
            description="Overlay method",
        ),
        EvidenceItem(
            evidence_type=EvidenceType.API_USAGE,
            source_agent="bytecode",
            entity="Landroid/view/WindowManager;->addView",
            attributes={"sensitivity_tier": 3, "category": "overlay"},
            description="Overlay API usage",
        ),
        EvidenceItem(
            evidence_type=EvidenceType.NETWORK_INDICATOR,
            source_agent="network",
            entity="http://185.234.72.15:8443/gate.php",
            attributes={"indicator_type": "url", "entropy": 3.8, "is_known_benign": False, "is_https": False},
            description="Suspicious URL with raw IP",
        ),
        EvidenceItem(
            evidence_type=EvidenceType.NETWORK_INDICATOR,
            source_agent="network",
            entity="185.234.72.15",
            attributes={"indicator_type": "ip", "is_raw_ip": True, "port": 8443, "non_standard_port": True},
            description="Raw IP with non-standard port",
        ),
        EvidenceItem(
            evidence_type=EvidenceType.CRYPTO_OPERATION,
            source_agent="crypto",
            entity="Ljavax/crypto/Cipher;->getInstance",
            attributes={"crypto_category": "cipher"},
            related_entities=["Lcom/malware/Stealer;->exfiltrate"],
            description="Cipher usage",
        ),
        EvidenceItem(
            evidence_type=EvidenceType.CRYPTO_OPERATION,
            source_agent="crypto",
            entity="hardcoded_key:a1b2c3d4e5f6...",
            attributes={"crypto_category": "hardcoded_key", "key_length": 16, "is_hardcoded": True},
            confidence=0.7,
            description="Hardcoded encryption key",
        ),
    ]


def make_benign_evidence() -> list[EvidenceItem]:
    """Simulate evidence a benign app would produce."""
    return [
        EvidenceItem(
            evidence_type=EvidenceType.PERMISSION,
            source_agent="manifest",
            entity="android.permission.INTERNET",
            attributes={"risk_level": 1, "is_dangerous": False, "category": "network"},
        ),
        EvidenceItem(
            evidence_type=EvidenceType.PERMISSION,
            source_agent="manifest",
            entity="android.permission.ACCESS_NETWORK_STATE",
            attributes={"risk_level": 0, "is_dangerous": False, "category": "network"},
        ),
        EvidenceItem(
            evidence_type=EvidenceType.COMPONENT,
            source_agent="manifest",
            entity="activity:com.example.MainActivity",
            attributes={"component_type": "activity", "is_exported": True, "has_intent_filter": True},
        ),
        EvidenceItem(
            evidence_type=EvidenceType.METHOD_CALL,
            source_agent="bytecode",
            entity="Lcom/example/MainActivity;->onCreate",
            attributes={"is_internal": True, "callee_count": 5},
            related_entities=["Landroid/widget/TextView;->setText"],
        ),
        EvidenceItem(
            evidence_type=EvidenceType.NETWORK_INDICATOR,
            source_agent="network",
            entity="https://api.example.com/data",
            attributes={"indicator_type": "url", "entropy": 3.2, "is_known_benign": False, "is_https": True},
        ),
    ]


def test_graph_construction():
    print("=== Test: Graph Construction ===")
    builder = GraphBuilder()

    mal_evidence = make_malware_evidence()
    mal_graph = builder.build(mal_evidence, label=1)
    print(f"Malware graph: {mal_graph.num_nodes} nodes, {mal_graph.edge_index.size(1)} edges")
    assert mal_graph.x.shape[1] == 25, f"Expected 25 features, got {mal_graph.x.shape[1]}"
    assert mal_graph.y.item() == 1
    assert mal_graph.num_nodes > 5

    ben_evidence = make_benign_evidence()
    ben_graph = builder.build(ben_evidence, label=0)
    print(f"Benign graph: {ben_graph.num_nodes} nodes, {ben_graph.edge_index.size(1)} edges")
    assert ben_graph.y.item() == 0

    print("PASS\n")
    return mal_graph, ben_graph


def test_model_forward(mal_graph, ben_graph):
    print("=== Test: GAT Forward Pass ===")
    model = MalwareGAT()

    model.eval()
    with torch.no_grad():
        out_mal = model(mal_graph.x, mal_graph.edge_index)
        out_ben = model(ben_graph.x, ben_graph.edge_index)

    print(f"Malware output: {out_mal.tolist()}")
    print(f"Benign output:  {out_ben.tolist()}")
    assert out_mal.shape == (1, 2)
    assert out_ben.shape == (1, 2)
    print("PASS\n")
    return model


def test_training_loop(mal_graph, ben_graph):
    print("=== Test: Mini Training Loop ===")
    from torch_geometric.loader import DataLoader

    dataset = [mal_graph, ben_graph] * 10
    loader = DataLoader(dataset, batch_size=4, shuffle=True)

    model = MalwareGAT()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = torch.nn.CrossEntropyLoss()

    model.train()
    losses = []
    for epoch in range(20):
        epoch_loss = 0
        for batch in loader:
            optimizer.zero_grad()
            out = model(batch.x, batch.edge_index, batch.batch)
            loss = criterion(out, batch.y.view(-1))
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        losses.append(epoch_loss / len(loader))

    print(f"Loss: {losses[0]:.4f} → {losses[-1]:.4f}")
    assert losses[-1] < losses[0], "Loss should decrease"
    print("PASS\n")
    return model


def test_explainer(model, mal_graph):
    print("=== Test: GNNExplainer ===")
    config = ExplainConfig(explainer_epochs=50, top_k_nodes=5)
    result = explain_prediction(model, mal_graph, config)

    print(f"Prediction: {result.prediction} ({result.risk_label()})")
    print(f"Risk score: {result.risk_score:.4f}")
    print(f"Confidence: {result.confidence:.4f}")
    print(f"Top nodes: {len(result.top_nodes)}")
    for name, imp in result.top_nodes[:3]:
        short = name.split(";->")[-1] if ";->" in name else name
        print(f"  {short}: {imp:.4f}")

    assert isinstance(result, ExplanationResult)
    assert 0.0 <= result.risk_score <= 1.0
    assert len(result.top_nodes) > 0
    print("PASS\n")
    return result


def test_attribution(explanation, mal_graph):
    print("=== Test: Agent Attribution ===")
    attributions = attribute_to_agents(explanation, mal_graph)

    for a in attributions:
        print(f"  {a.agent_name}: {a.contribution_pct}% ({a.evidence_count} items)")

    assert len(attributions) > 0
    total_pct = sum(a.contribution_pct for a in attributions)
    assert abs(total_pct - 100.0) < 1.0, f"Expected ~100%, got {total_pct}%"

    text = generate_explanation_text(explanation, attributions)
    print(f"\n{text}")
    print("PASS\n")


def main():
    print("=" * 60)
    print("SYNTHETIC END-TO-END PIPELINE TEST")
    print("=" * 60 + "\n")

    mal_graph, ben_graph = test_graph_construction()
    test_model_forward(mal_graph, ben_graph)
    model = test_training_loop(mal_graph, ben_graph)
    explanation = test_explainer(model, mal_graph)
    test_attribution(explanation, mal_graph)

    print("=" * 60)
    print("ALL TESTS PASSED")
    print("=" * 60)


if __name__ == "__main__":
    main()
