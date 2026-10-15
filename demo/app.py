"""
Streamlit demo app for Android Banking Malware Analyzer.
Run: streamlit run demo/app.py
"""
from __future__ import annotations

import sys
import json
import tempfile
from pathlib import Path

import streamlit as st
import plotly.graph_objects as go
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent.parent))

from agents.coordinator import Coordinator, AnalysisResult
from graph.builder import GraphBuilder
from model.gat import MalwareGAT
from explain.explainer import explain_prediction, ExplanationResult
from explain.attribution import attribute_to_agents, generate_explanation_text, AgentAttribution
from config.settings import MODEL_DIR, EVIDENCE_DIR, GRAPHS_DIR, ExplainConfig

SOC_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500;600&display=swap');

:root {
    --bg: #0b1326;
    --surface: #0F172A;
    --surface-high: #1E293B;
    --surface-highest: #2d3449;
    --border: rgba(51,65,85,0.45);
    --text-primary: #F8FAFC;
    --text-muted: #94A3B8;
    --cyan: #06B6D4;
    --cyan-light: #4CD7F6;
    --red: #EF4444;
    --amber: #F59E0B;
    --green: #10B981;
}

.stApp {
    background-color: var(--bg) !important;
}

.stApp header { background-color: rgba(11,19,38,0.85) !important; backdrop-filter: blur(12px); }

section[data-testid="stSidebar"] { background-color: #131b2e !important; }

h1, h2, h3 { font-family: 'Inter', sans-serif !important; color: var(--text-primary) !important; letter-spacing: -0.01em; }
p, span, label, div { font-family: 'Inter', sans-serif; }

.soc-header {
    background: linear-gradient(135deg, var(--surface) 0%, rgba(6,182,212,0.05) 100%);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 1.5rem;
    margin-bottom: 1rem;
}
.soc-title {
    font-family: 'Inter', sans-serif;
    font-size: 28px;
    font-weight: 700;
    color: var(--text-primary);
    letter-spacing: -0.02em;
    margin: 0;
}
.soc-subtitle {
    font-family: 'JetBrains Mono', monospace;
    font-size: 10px;
    font-weight: 600;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    color: var(--cyan);
    margin-top: 4px;
}
.soc-status-bar {
    display: flex;
    align-items: center;
    gap: 8px;
    margin-top: 12px;
}
.soc-status-dot {
    width: 8px; height: 8px;
    border-radius: 50%;
    background: var(--green);
    animation: pulse-dot 2s infinite;
}
@keyframes pulse-dot {
    0%, 100% { opacity: 1; box-shadow: 0 0 0 0 rgba(16,185,129,0.4); }
    50% { opacity: 0.8; box-shadow: 0 0 0 4px rgba(16,185,129,0); }
}
.soc-status-text {
    font-family: 'JetBrains Mono', monospace;
    font-size: 10px;
    font-weight: 600;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    color: var(--text-muted);
}

.soc-card {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 1rem;
    margin-bottom: 0.75rem;
}
.soc-card-glow-red {
    background: var(--surface);
    border: 1px solid rgba(239,68,68,0.35);
    border-radius: 8px;
    padding: 1rem;
    margin-bottom: 0.75rem;
    box-shadow: 0 0 16px rgba(239,68,68,0.15);
}
.soc-card-glow-cyan {
    background: var(--surface);
    border: 1px solid rgba(6,182,212,0.25);
    border-radius: 8px;
    padding: 1rem;
    margin-bottom: 0.75rem;
    box-shadow: 0 0 12px rgba(6,182,212,0.12);
}
.soc-card-glow-green {
    background: var(--surface);
    border: 1px solid rgba(16,185,129,0.35);
    border-radius: 8px;
    padding: 1rem;
    margin-bottom: 0.75rem;
    box-shadow: 0 0 12px rgba(16,185,129,0.12);
}

.soc-score-badge {
    display: inline-flex;
    align-items: center;
    justify-content: center;
    flex-direction: column;
    width: 80px; height: 80px;
    border-radius: 12px;
    font-family: 'Inter', sans-serif;
}
.soc-score-badge.critical {
    background: rgba(239,68,68,0.15);
    box-shadow: 0 0 24px rgba(239,68,68,0.4);
}
.soc-score-badge.warning {
    background: rgba(245,158,11,0.15);
    box-shadow: 0 0 24px rgba(245,158,11,0.3);
}
.soc-score-badge.nominal {
    background: rgba(16,185,129,0.15);
    box-shadow: 0 0 24px rgba(16,185,129,0.3);
}
.soc-score-num {
    font-size: 28px; font-weight: 700; line-height: 1;
}
.soc-score-label {
    font-family: 'JetBrains Mono', monospace;
    font-size: 9px; font-weight: 600;
    letter-spacing: 0.08em; text-transform: uppercase;
    margin-top: 4px; opacity: 0.8;
}

.chip-critical {
    display: inline-block;
    background: rgba(239,68,68,0.12);
    border: 1px solid rgba(239,68,68,0.35);
    color: #EF4444;
    font-family: 'JetBrains Mono', monospace;
    font-size: 10px; font-weight: 600;
    letter-spacing: 0.08em; text-transform: uppercase;
    padding: 2px 8px; border-radius: 4px;
}
.chip-warning {
    display: inline-block;
    background: rgba(245,158,11,0.12);
    border: 1px solid rgba(245,158,11,0.35);
    color: #F59E0B;
    font-family: 'JetBrains Mono', monospace;
    font-size: 10px; font-weight: 600;
    letter-spacing: 0.08em; text-transform: uppercase;
    padding: 2px 8px; border-radius: 4px;
}
.chip-nominal {
    display: inline-block;
    background: rgba(16,185,129,0.12);
    border: 1px solid rgba(16,185,129,0.35);
    color: #10B981;
    font-family: 'JetBrains Mono', monospace;
    font-size: 10px; font-weight: 600;
    letter-spacing: 0.08em; text-transform: uppercase;
    padding: 2px 8px; border-radius: 4px;
}
.chip-cyan {
    display: inline-block;
    background: rgba(6,182,212,0.12);
    border: 1px solid rgba(6,182,212,0.35);
    color: #06B6D4;
    font-family: 'JetBrains Mono', monospace;
    font-size: 10px; font-weight: 600;
    letter-spacing: 0.08em; text-transform: uppercase;
    padding: 2px 8px; border-radius: 4px;
}

.soc-metric-row {
    display: flex;
    gap: 12px;
    margin: 8px 0;
}
.soc-metric {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 6px;
    padding: 12px 16px;
    flex: 1;
    text-align: center;
}
.soc-metric-value {
    font-family: 'Inter', sans-serif;
    font-size: 22px; font-weight: 700;
    color: var(--text-primary);
}
.soc-metric-label {
    font-family: 'JetBrains Mono', monospace;
    font-size: 10px; font-weight: 600;
    letter-spacing: 0.08em; text-transform: uppercase;
    color: var(--text-muted);
    margin-top: 4px;
}

.mono { font-family: 'JetBrains Mono', monospace !important; }
.mono-sm { font-family: 'JetBrains Mono', monospace; font-size: 11px; color: var(--text-muted); }
.label-caps {
    font-family: 'JetBrains Mono', monospace;
    font-size: 10px; font-weight: 600;
    letter-spacing: 0.08em; text-transform: uppercase;
    color: var(--text-muted);
}
.text-cyan { color: var(--cyan) !important; }
.text-red { color: var(--red) !important; }
.text-amber { color: var(--amber) !important; }
.text-green { color: var(--green) !important; }

.evidence-item {
    background: var(--surface);
    border-left: 3px solid var(--border);
    padding: 8px 12px;
    margin: 4px 0;
    border-radius: 0 4px 4px 0;
    font-size: 13px;
}
.evidence-item.high { border-left-color: var(--red); }
.evidence-item.med { border-left-color: var(--amber); }
.evidence-item.low { border-left-color: var(--cyan); }

.agent-contrib-bar {
    height: 6px;
    border-radius: 3px;
    margin-top: 4px;
}

div[data-testid="stTabs"] button[data-baseweb="tab"] {
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 11px !important;
    font-weight: 600 !important;
    letter-spacing: 0.06em !important;
    text-transform: uppercase !important;
    color: var(--text-muted) !important;
}
div[data-testid="stTabs"] button[aria-selected="true"] {
    color: var(--cyan) !important;
    border-bottom-color: var(--cyan) !important;
}

div[data-testid="stExpander"] {
    background: var(--surface) !important;
    border: 1px solid var(--border) !important;
    border-radius: 6px !important;
}

div[data-testid="stMetric"] {
    background: var(--surface) !important;
    border: 1px solid var(--border) !important;
    border-radius: 6px !important;
    padding: 12px !important;
}
div[data-testid="stMetric"] label { color: var(--text-muted) !important; font-family: 'JetBrains Mono', monospace !important; font-size: 10px !important; letter-spacing: 0.06em !important; text-transform: uppercase !important; }
div[data-testid="stMetric"] div[data-testid="stMetricValue"] { color: var(--cyan) !important; font-family: 'Inter', sans-serif !important; }

table { border-collapse: collapse; width: 100%; }
th {
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 10px !important; font-weight: 600 !important;
    letter-spacing: 0.06em !important; text-transform: uppercase !important;
    color: var(--text-muted) !important;
    border-bottom: 1px solid var(--border) !important;
    padding: 8px 12px !important;
    text-align: left !important;
}
td {
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 12px !important;
    color: var(--text-primary) !important;
    border-bottom: 1px solid rgba(51,65,85,0.2) !important;
    padding: 6px 12px !important;
}

div[data-testid="stFileUploader"] {
    background: var(--surface) !important;
    border: 1px dashed rgba(6,182,212,0.3) !important;
    border-radius: 8px !important;
}
</style>
"""


@st.cache_resource
def load_model():
    model = MalwareGAT()
    model_path = MODEL_DIR / "best_gat.pt"
    if model_path.exists():
        model.load_state_dict(torch.load(model_path, weights_only=True))
    model.eval()
    return model


@st.cache_resource
def load_ensemble():
    import pickle
    gbm_path = MODEL_DIR / "ensemble_gbm.pkl"
    if gbm_path.exists():
        with open(gbm_path, "rb") as f:
            return pickle.load(f)
    return None


def ensemble_predict(model, data):
    import numpy as np
    from model.ensemble import _extra_structural_features
    gbm = load_ensemble()
    with torch.no_grad():
        out = model(data.x, data.edge_index)
        gat_prob = F.softmax(out, dim=1)[0, 1].item()
    if gbm is None:
        return gat_prob
    batch = torch.zeros(data.x.size(0), dtype=torch.long)
    gf = model._graph_features(data.x, data.edge_index, batch)
    base_features = gf.squeeze(0).numpy()
    extra = _extra_structural_features(data)
    features = np.concatenate([base_features, extra])
    expected = gbm.n_features_in_
    if expected == len(features) + 1:
        features = np.append(features, gat_prob)
    return float(gbm.predict_proba(features.reshape(1, -1))[0, 1])


def create_risk_gauge(score: float) -> go.Figure:
    if score < 0.3:
        color = "#10B981"
        label = "NOMINAL"
    elif score < 0.6:
        color = "#F59E0B"
        label = "SUSPICIOUS"
    else:
        color = "#EF4444"
        label = "CRITICAL THREAT"

    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=score * 100,
        number={"font": {"size": 48, "color": color, "family": "Inter"}, "suffix": ""},
        title={"text": f"<b>{label}</b>", "font": {"size": 14, "color": color, "family": "JetBrains Mono"}},
        gauge={
            "axis": {
                "range": [0, 100],
                "tickwidth": 1,
                "tickcolor": "#334155",
                "dtick": 20,
                "tickfont": {"color": "#94A3B8", "size": 10, "family": "JetBrains Mono"},
            },
            "bar": {"color": color, "thickness": 0.8},
            "bgcolor": "#1E293B",
            "borderwidth": 1,
            "bordercolor": "#334155",
            "steps": [
                {"range": [0, 30], "color": "rgba(16,185,129,0.08)"},
                {"range": [30, 60], "color": "rgba(245,158,11,0.08)"},
                {"range": [60, 100], "color": "rgba(239,68,68,0.08)"},
            ],
            "threshold": {
                "line": {"color": color, "width": 3},
                "thickness": 0.85,
                "value": score * 100,
            },
        },
    ))
    fig.update_layout(
        height=260,
        margin=dict(l=30, r=30, t=30, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font={"family": "Inter"},
    )
    return fig


def create_graph_html(data, explanation: ExplanationResult | None = None) -> str:
    from pyvis.network import Network

    net = Network(height="500px", width="100%", directed=True, bgcolor="#0b1326", font_color="#F8FAFC")
    net.barnes_hut(gravity=-3000, central_gravity=0.3, spring_length=100)

    node_names = getattr(data, "node_names", [f"node_{i}" for i in range(data.num_nodes)])
    node_types = getattr(data, "node_types", torch.zeros(data.num_nodes, dtype=torch.long))

    type_colors = {
        0: "#38BDF8",
        1: "#A78BFA",
        2: "#F59E0B",
        3: "#EF4444",
        4: "#10B981",
        5: "#F472B6",
    }
    type_labels = {
        0: "Method", 1: "Class", 2: "Permission",
        3: "API Call", 4: "Network", 5: "Crypto",
    }

    important_nodes = set()
    if explanation:
        important_nodes = {name for name, _ in explanation.top_nodes}

    max_nodes = 150
    display_names = node_names[:max_nodes]

    for i, name in enumerate(display_names):
        ntype = int(node_types[i]) if i < len(node_types) else 0
        color = type_colors.get(ntype, "#94A3B8")
        label = name.split(";->")[-1] if ";->" in name else name.split(":")[-1]
        if len(label) > 30:
            label = label[:27] + "..."

        size = 12
        border_width = 1
        border_color = "#334155"
        if name in important_nodes:
            size = 28
            border_width = 3
            border_color = "#EF4444"
            color = "#EF4444"

        net.add_node(
            i, label=label, color=color, size=size,
            title=f"{type_labels.get(ntype, 'Unknown')}: {name}",
            borderWidth=border_width, borderWidthSelected=border_width + 1,
        )

    for j in range(data.edge_index.size(1)):
        src = data.edge_index[0, j].item()
        dst = data.edge_index[1, j].item()
        if src < max_nodes and dst < max_nodes:
            edge_color = "rgba(51,65,85,0.5)"
            width = 0.5
            if explanation:
                src_name = node_names[src] if src < len(node_names) else ""
                dst_name = node_names[dst] if dst < len(node_names) else ""
                if src_name in important_nodes or dst_name in important_nodes:
                    edge_color = "rgba(239,68,68,0.6)"
                    width = 1.5
            net.add_edge(src, dst, color=edge_color, width=width)

    html = net.generate_html()
    return html


def run_analysis(apk_path: str) -> tuple[AnalysisResult, object, ExplanationResult, list[AgentAttribution]]:
    coordinator = Coordinator()
    builder = GraphBuilder()
    model = load_model()

    result = coordinator.analyze(apk_path)
    data = builder.build(result.evidence, label=-1)

    explanation = explain_prediction(model, data, ExplainConfig())
    ens_score = ensemble_predict(model, data)
    explanation.risk_score = ens_score
    explanation.prediction = 1 if ens_score >= 0.5 else 0
    explanation.confidence = ens_score if ens_score >= 0.5 else 1 - ens_score
    attributions = attribute_to_agents(explanation, data)

    return result, data, explanation, attributions


def render_threat_badge(score: float) -> str:
    if score >= 0.6:
        return '<span class="chip-critical">CRITICAL THREAT</span>'
    elif score >= 0.3:
        return '<span class="chip-warning">SUSPICIOUS</span>'
    else:
        return '<span class="chip-nominal">NOMINAL</span>'


def render_score_badge(score: float) -> str:
    val = int(score * 100)
    if score >= 0.6:
        cls = "critical"
        col = "#EF4444"
    elif score >= 0.3:
        cls = "warning"
        col = "#F59E0B"
    else:
        cls = "nominal"
        col = "#10B981"
    return f'''<div class="soc-score-badge {cls}">
        <span class="soc-score-num" style="color:{col}">{val}</span>
        <span class="soc-score-label" style="color:{col}">/100 RISK</span>
    </div>'''


AGENT_COLORS = {
    "manifest": "#F59E0B",
    "bytecode": "#38BDF8",
    "network": "#10B981",
    "crypto": "#F472B6",
}
AGENT_ICONS = {
    "manifest": "description",
    "bytecode": "code",
    "network": "language",
    "crypto": "lock",
}


@st.cache_data(show_spinner="Running GNNExplainer...")
def load_preanalyzed(_sha):
    evidence_path = EVIDENCE_DIR / f"{_sha}.json"
    graph_path = GRAPHS_DIR / f"{_sha}.pt"

    result = AnalysisResult.load(evidence_path)
    builder = GraphBuilder()
    model = load_model()

    if graph_path.exists():
        data = torch.load(graph_path, weights_only=False)
    else:
        data = builder.build(result.evidence, label=-1)

    explanation = explain_prediction(model, data, ExplainConfig())
    ens_score = ensemble_predict(model, data)
    explanation.risk_score = ens_score
    explanation.prediction = 1 if ens_score >= 0.5 else 0
    explanation.confidence = ens_score if ens_score >= 0.5 else 1 - ens_score
    attributions = attribute_to_agents(explanation, data)
    return result, data, explanation, attributions


def render_landing():
    st.markdown('''
    <div class="soc-header" style="text-align:center; padding:2rem 1.5rem;">
        <div class="soc-title" style="font-size:36px;">🛡️ APKSentinel</div>
        <div style="font-family:'Inter',sans-serif; font-size:16px; color:#94A3B8; margin-top:8px; max-width:640px; margin-left:auto; margin-right:auto; line-height:1.6;">
            Automated multi-agent analysis and explainable risk scoring of fraudulent Android banking applications
        </div>
    </div>
    ''', unsafe_allow_html=True)

    st.markdown('''
    <div style="display:flex; gap:12px; justify-content:center; flex-wrap:wrap; margin:1.5rem 0;">
        <div class="soc-metric"><div class="soc-metric-value text-cyan">94.05%</div><div class="soc-metric-label">Accuracy</div></div>
        <div class="soc-metric"><div class="soc-metric-value text-cyan">0.958</div><div class="soc-metric-label">F1 Score</div></div>
        <div class="soc-metric"><div class="soc-metric-value text-cyan">98.3%</div><div class="soc-metric-label">Recall</div></div>
        <div class="soc-metric"><div class="soc-metric-value text-cyan">554</div><div class="soc-metric-label">APKs Analyzed</div></div>
    </div>
    ''', unsafe_allow_html=True)

    # Pipeline
    st.markdown('<div class="label-caps" style="margin:24px 0 12px 0; text-align:center; font-size:12px;">Analysis Pipeline</div>', unsafe_allow_html=True)
    st.markdown('''
    <div style="display:flex; align-items:center; justify-content:center; gap:0; flex-wrap:wrap; margin-bottom:24px;">
        <div class="soc-metric" style="min-width:100px; flex:0;">
            <div style="font-size:20px;">📦</div>
            <div style="font-size:12px; font-weight:600; color:#F8FAFC; margin-top:4px;">APK Input</div>
            <div style="font-size:10px; color:#94A3B8;">Android package</div>
        </div>
        <span style="color:#334155; font-size:18px; padding:0 6px;">▸</span>
        <div class="soc-metric" style="min-width:100px; flex:0;">
            <div style="font-size:20px;">🤖</div>
            <div style="font-size:12px; font-weight:600; color:#F8FAFC; margin-top:4px;">4 Agents</div>
            <div style="font-size:10px; color:#94A3B8;">Static analysis</div>
        </div>
        <span style="color:#334155; font-size:18px; padding:0 6px;">▸</span>
        <div class="soc-metric" style="min-width:100px; flex:0;">
            <div style="font-size:20px;">🕸️</div>
            <div style="font-size:12px; font-weight:600; color:#F8FAFC; margin-top:4px;">Graph</div>
            <div style="font-size:10px; color:#94A3B8;">Behaviour graph</div>
        </div>
        <span style="color:#334155; font-size:18px; padding:0 6px;">▸</span>
        <div class="soc-metric" style="min-width:100px; flex:0;">
            <div style="font-size:20px;">🧠</div>
            <div style="font-size:12px; font-weight:600; color:#F8FAFC; margin-top:4px;">GAT</div>
            <div style="font-size:10px; color:#94A3B8;">Classification</div>
        </div>
        <span style="color:#334155; font-size:18px; padding:0 6px;">▸</span>
        <div class="soc-metric" style="min-width:100px; flex:0;">
            <div style="font-size:20px;">🔍</div>
            <div style="font-size:12px; font-weight:600; color:#F8FAFC; margin-top:4px;">Explain</div>
            <div style="font-size:10px; color:#94A3B8;">XAI + MITRE</div>
        </div>
    </div>
    ''', unsafe_allow_html=True)

    # Agents
    st.markdown('<div class="label-caps" style="margin:16px 0 12px 0; text-align:center; font-size:12px;">Specialized Analysis Agents</div>', unsafe_allow_html=True)
    col1, col2 = st.columns(2)

    with col1:
        st.markdown('''
        <div class="soc-card" style="border-left:3px solid #F59E0B;">
            <div style="display:flex; align-items:center; gap:8px; margin-bottom:8px;">
                <span style="font-size:18px;">📋</span>
                <span style="font-family:'JetBrains Mono',monospace; font-size:12px; font-weight:600; color:#F59E0B; text-transform:uppercase; letter-spacing:0.06em;">Manifest Agent</span>
            </div>
            <div style="font-size:13px; color:#94A3B8; line-height:1.7;">
                • Parses AndroidManifest.xml<br>
                • Extracts 30 dangerous permissions<br>
                • Identifies exported components<br>
                • Flags accessibility service abuse
            </div>
        </div>
        ''', unsafe_allow_html=True)

        st.markdown('''
        <div class="soc-card" style="border-left:3px solid #10B981;">
            <div style="display:flex; align-items:center; gap:8px; margin-bottom:8px;">
                <span style="font-size:18px;">🌐</span>
                <span style="font-family:'JetBrains Mono',monospace; font-size:12px; font-weight:600; color:#10B981; text-transform:uppercase; letter-spacing:0.06em;">Network Agent</span>
            </div>
            <div style="font-size:13px; color:#94A3B8; line-height:1.7;">
                • Extracts URLs, IPs, domains<br>
                • Decodes base64-encoded URLs<br>
                • Detects C2 indicators<br>
                • Flags high-entropy domains
            </div>
        </div>
        ''', unsafe_allow_html=True)

    with col2:
        st.markdown('''
        <div class="soc-card" style="border-left:3px solid #38BDF8;">
            <div style="display:flex; align-items:center; gap:8px; margin-bottom:8px;">
                <span style="font-size:18px;">⚙️</span>
                <span style="font-family:'JetBrains Mono',monospace; font-size:12px; font-weight:600; color:#38BDF8; text-transform:uppercase; letter-spacing:0.06em;">Bytecode Agent</span>
            </div>
            <div style="font-size:13px; color:#94A3B8; line-height:1.7;">
                • Analyzes DEX via Androguard<br>
                • Builds function call graph<br>
                • Flags ~150 sensitive API calls<br>
                • Detects reflection & dynamic loading
            </div>
        </div>
        ''', unsafe_allow_html=True)

        st.markdown('''
        <div class="soc-card" style="border-left:3px solid #F472B6;">
            <div style="display:flex; align-items:center; gap:8px; margin-bottom:8px;">
                <span style="font-size:18px;">🔐</span>
                <span style="font-family:'JetBrains Mono',monospace; font-size:12px; font-weight:600; color:#F472B6; text-transform:uppercase; letter-spacing:0.06em;">Crypto Agent</span>
            </div>
            <div style="font-size:13px; color:#94A3B8; line-height:1.7;">
                • Scans javax.crypto / java.security<br>
                • Detects hardcoded keys<br>
                • Flags weak algorithms (DES, MD5)<br>
                • Identifies custom crypto patterns
            </div>
        </div>
        ''', unsafe_allow_html=True)

    # Coordinator
    st.markdown('''
    <div class="soc-card-glow-cyan" style="margin-top:8px;">
        <div style="display:flex; align-items:center; gap:8px; margin-bottom:8px;">
            <span style="font-size:18px;">🎯</span>
            <span style="font-family:'JetBrains Mono',monospace; font-size:12px; font-weight:600; color:#06B6D4; text-transform:uppercase; letter-spacing:0.06em;">Coordinator</span>
        </div>
        <div style="font-size:13px; color:#94A3B8; line-height:1.7;">
            Dispatches all 4 agents in parallel on each APK. Collects standardized EvidenceItem objects, deduplicates by entity, and passes merged evidence to the graph builder. Star topology — agents communicate only through the coordinator.
        </div>
    </div>
    ''', unsafe_allow_html=True)

    # Classification
    st.markdown('<div class="label-caps" style="margin:24px 0 12px 0; text-align:center; font-size:12px;">Classification Models</div>', unsafe_allow_html=True)
    col_gat, col_ens = st.columns(2)

    with col_gat:
        st.markdown('''
        <div class="soc-card">
            <div style="display:flex; align-items:center; gap:8px; margin-bottom:8px;">
                <span style="font-size:18px;">🧠</span>
                <span style="font-family:'JetBrains Mono',monospace; font-size:12px; font-weight:600; color:#06B6D4; text-transform:uppercase; letter-spacing:0.06em;">Graph Attention Network</span>
            </div>
            <div style="font-size:13px; color:#94A3B8; line-height:1.6; margin-bottom:12px;">
                2-layer GAT with 4 attention heads. Learns which nodes matter most for classification. 36,258 parameters, CPU-trained.
            </div>
            <div style="margin:4px 0;">
                <div style="display:flex; justify-content:space-between; font-size:11px; color:#94A3B8;"><span>Accuracy</span><span style="color:#F8FAFC; font-weight:600;">85.7%</span></div>
                <div style="background:rgba(51,65,85,0.3); height:6px; border-radius:3px; margin-top:4px; overflow:hidden;"><div style="width:85.7%; height:100%; background:#06B6D4; border-radius:3px;"></div></div>
            </div>
            <div style="margin:8px 0;">
                <div style="display:flex; justify-content:space-between; font-size:11px; color:#94A3B8;"><span>F1 Score</span><span style="color:#F8FAFC; font-weight:600;">89.7%</span></div>
                <div style="background:rgba(51,65,85,0.3); height:6px; border-radius:3px; margin-top:4px; overflow:hidden;"><div style="width:89.7%; height:100%; background:#06B6D4; border-radius:3px;"></div></div>
            </div>
        </div>
        ''', unsafe_allow_html=True)

    with col_ens:
        st.markdown('''
        <div class="soc-card-glow-green">
            <div style="display:flex; align-items:center; gap:8px; margin-bottom:8px;">
                <span style="font-size:18px;">📊</span>
                <span style="font-family:'JetBrains Mono',monospace; font-size:12px; font-weight:600; color:#10B981; text-transform:uppercase; letter-spacing:0.06em;">Ensemble (GAT + GBM)</span>
            </div>
            <div style="font-size:13px; color:#94A3B8; line-height:1.6; margin-bottom:12px;">
                Gradient Boosting over 33 features: 15 base graph features + 18 structural features + GAT probability.
            </div>
            <div style="margin:4px 0;">
                <div style="display:flex; justify-content:space-between; font-size:11px; color:#94A3B8;"><span>Accuracy</span><span style="color:#F8FAFC; font-weight:600;">94.05%</span></div>
                <div style="background:rgba(51,65,85,0.3); height:6px; border-radius:3px; margin-top:4px; overflow:hidden;"><div style="width:94%; height:100%; background:#10B981; border-radius:3px;"></div></div>
            </div>
            <div style="margin:8px 0;">
                <div style="display:flex; justify-content:space-between; font-size:11px; color:#94A3B8;"><span>F1 Score</span><span style="color:#F8FAFC; font-weight:600;">95.8%</span></div>
                <div style="background:rgba(51,65,85,0.3); height:6px; border-radius:3px; margin-top:4px; overflow:hidden;"><div style="width:95.8%; height:100%; background:#10B981; border-radius:3px;"></div></div>
            </div>
            <div style="margin:8px 0;">
                <div style="display:flex; justify-content:space-between; font-size:11px; color:#94A3B8;"><span>Recall</span><span style="color:#F8FAFC; font-weight:600;">98.3%</span></div>
                <div style="background:rgba(51,65,85,0.3); height:6px; border-radius:3px; margin-top:4px; overflow:hidden;"><div style="width:98.3%; height:100%; background:#10B981; border-radius:3px;"></div></div>
            </div>
        </div>
        ''', unsafe_allow_html=True)

    # Explainability
    st.markdown('<div class="label-caps" style="margin:24px 0 12px 0; text-align:center; font-size:12px;">Explainability</div>', unsafe_allow_html=True)
    col_xai, col_mitre = st.columns(2)

    with col_xai:
        st.markdown('''
        <div class="soc-card" style="border-left:3px solid #10B981;">
            <div style="display:flex; align-items:center; gap:8px; margin-bottom:8px;">
                <span style="font-size:18px;">🔬</span>
                <span style="font-family:'JetBrains Mono',monospace; font-size:12px; font-weight:600; color:#10B981; text-transform:uppercase; letter-spacing:0.06em;">GNNExplainer</span>
            </div>
            <div style="font-size:13px; color:#94A3B8; line-height:1.7;">
                Generates node importance masks over the behaviour graph. Top-15 nodes form the explanation subgraph. Each node traces back to its source agent for attribution percentages.
            </div>
        </div>
        ''', unsafe_allow_html=True)

    with col_mitre:
        st.markdown('''
        <div class="soc-card" style="border-left:3px solid #F59E0B;">
            <div style="display:flex; align-items:center; gap:8px; margin-bottom:8px;">
                <span style="font-size:18px;">⚔️</span>
                <span style="font-family:'JetBrains Mono',monospace; font-size:12px; font-weight:600; color:#F59E0B; text-transform:uppercase; letter-spacing:0.06em;">MITRE ATT&CK Mapping</span>
            </div>
            <div style="font-size:13px; color:#94A3B8; line-height:1.7;">
                Maps 28 permissions and 29 API calls to ATT&CK for Mobile techniques and tactics. Based on Tanha & Kafaie (IEEE Access, 2025).
            </div>
        </div>
        ''', unsafe_allow_html=True)

    # Reference
    st.markdown('''
    <div class="soc-card" style="margin-top:16px; text-align:center;">
        <div class="label-caps" style="margin-bottom:6px;">Reference Paper</div>
        <div style="font-size:13px; color:#94A3B8; line-height:1.6;">
            M. Tanha and S. Kafaie, "A Review of Explainable AI for Android Malware Detection and Analysis,"<br>
            <span style="color:#06B6D4;">IEEE Access</span>, vol. 13, pp. 141958–141974, 2025. DOI: 10.1109/ACCESS.2025.3597575
        </div>
    </div>
    ''', unsafe_allow_html=True)


def main():
    st.set_page_config(
        page_title="APKSentinel // Threat Analysis",
        page_icon="🛡️",
        layout="wide",
    )

    st.markdown(SOC_CSS, unsafe_allow_html=True)

    page = st.sidebar.radio(
        "Navigation",
        ["🏠 Home", "🔬 Analyze"],
        label_visibility="collapsed",
    )

    if page == "🏠 Home":
        render_landing()
        return

    st.markdown('''
    <div class="soc-header">
        <div class="soc-title">APKSentinel</div>
        <div class="soc-subtitle">Multi-Agent Static Analysis &bull; Explainable GAT Classification</div>
        <div class="soc-status-bar">
            <div class="soc-status-dot"></div>
            <span class="soc-status-text">ANALYSIS ENGINE ACTIVE &bull; 4 AGENTS ONLINE &bull; GAT MODEL LOADED</span>
        </div>
    </div>
    ''', unsafe_allow_html=True)

    uploaded_file = st.file_uploader("Drop APK for analysis", type=["apk"])

    pre_analyzed = None
    evidence_dir = EVIDENCE_DIR
    if evidence_dir.exists():
        evidence_files = sorted(evidence_dir.glob("*.json"))
        if evidence_files:
            options = [f.stem for f in evidence_files]
            demo_default = "0016B3D270A9C97E2A5DF8C7F063DABECB09BBF93BE090F9DE2BE65E91E178BE"
            default_idx = options.index(demo_default) if demo_default in options else 0
            selected = st.selectbox(
                "Or select a pre-analyzed sample",
                options,
                index=default_idx,
            )
            pre_analyzed = selected

    result = data = explanation = attributions = None

    if uploaded_file is not None:
        with tempfile.NamedTemporaryFile(suffix=".apk", delete=False) as tmp:
            tmp.write(uploaded_file.getbuffer())
            apk_path = tmp.name

        with st.spinner("Detonating APK across 4 analysis agents..."):
            result, data, explanation, attributions = run_analysis(apk_path)
    elif pre_analyzed:
        result, data, explanation, attributions = load_preanalyzed(pre_analyzed)

    if result and explanation:
        sha_display = pre_analyzed or result.sha256
        st.markdown(f'''
        <div style="display:flex; align-items:center; gap:8px; margin-bottom:12px;">
            <span class="label-caps text-cyan">SAMPLE</span>
            <span class="mono-sm">{sha_display}</span>
            <span style="margin-left:auto">{render_threat_badge(explanation.risk_score)}</span>
        </div>
        ''', unsafe_allow_html=True)

        tab_verdict, tab_agents, tab_graph, tab_model = st.tabs(
            ["Threat Verdict", "Agent Evidence", "Behaviour Graph", "Model Details"]
        )

        with tab_verdict:
            col1, col2 = st.columns([2, 3])

            with col1:
                st.markdown(f'''
                <div style="display:flex; align-items:center; gap:16px; margin-bottom:16px;">
                    {render_score_badge(explanation.risk_score)}
                    <div>
                        <div style="font-family:'JetBrains Mono',monospace; font-size:10px; font-weight:600; letter-spacing:0.08em; text-transform:uppercase; color:#94A3B8;">Threat Classification</div>
                        <div style="font-size:20px; font-weight:700; color:#F8FAFC; margin-top:2px;">
                            {'Trojan Dropper' if explanation.risk_score >= 0.6 else 'Anomalous Activity' if explanation.risk_score >= 0.3 else 'Clean Binary'}
                        </div>
                        <div class="mono-sm" style="margin-top:4px;">
                            Confidence: {explanation.confidence:.1%}
                        </div>
                    </div>
                </div>
                ''', unsafe_allow_html=True)

                st.plotly_chart(create_risk_gauge(explanation.risk_score), width="stretch")

            with col2:
                text = generate_explanation_text(explanation, attributions)
                glow_class = "soc-card-glow-red" if explanation.risk_score >= 0.6 else "soc-card-glow-cyan" if explanation.risk_score >= 0.3 else "soc-card-glow-green"
                st.markdown(f'<div class="{glow_class}">', unsafe_allow_html=True)
                st.markdown(text)
                st.markdown('</div>', unsafe_allow_html=True)

                if attributions:
                    st.markdown('<div class="label-caps" style="margin:12px 0 6px 0;">Agent Attribution Breakdown</div>', unsafe_allow_html=True)
                    for attr in attributions:
                        color = AGENT_COLORS.get(attr.agent_name, "#94A3B8")
                        st.markdown(f'''
                        <div style="display:flex; align-items:center; justify-content:space-between; margin:4px 0;">
                            <span style="font-family:'JetBrains Mono',monospace; font-size:12px; color:{color}; font-weight:600; text-transform:uppercase;">{attr.agent_name}</span>
                            <span style="font-family:'Inter',sans-serif; font-size:18px; font-weight:700; color:{color};">{attr.contribution_pct}%</span>
                        </div>
                        <div style="background:rgba(51,65,85,0.3); height:4px; border-radius:2px; overflow:hidden;">
                            <div style="width:{attr.contribution_pct}%; height:100%; background:{color}; border-radius:2px;"></div>
                        </div>
                        ''', unsafe_allow_html=True)

            st.markdown('<div class="label-caps" style="margin:16px 0 8px 0;">Top Risk Indicators</div>', unsafe_allow_html=True)
            for name, imp in explanation.top_nodes[:10]:
                short = name.split(";->")[-1] if ";->" in name else name.split(":")[-1]
                bar_width = int(imp / (explanation.top_nodes[0][1] + 0.001) * 100)
                severity = "high" if imp > 0.1 else "med" if imp > 0.05 else "low"
                st.markdown(f'''
                <div class="evidence-item {severity}">
                    <div style="display:flex; justify-content:space-between; align-items:center;">
                        <span class="mono" style="font-size:12px; color:#F8FAFC;">{short}</span>
                        <span class="mono" style="font-size:11px; color:#94A3B8;">{imp:.4f}</span>
                    </div>
                    <div style="background:rgba(51,65,85,0.3); height:3px; border-radius:2px; margin-top:4px; overflow:hidden;">
                        <div style="width:{bar_width}%; height:100%; background:{'#EF4444' if severity=='high' else '#F59E0B' if severity=='med' else '#06B6D4'}; border-radius:2px;"></div>
                    </div>
                    <div class="mono-sm" style="margin-top:2px; font-size:10px;">{name}</div>
                </div>
                ''', unsafe_allow_html=True)

            # MITRE ATT&CK Mapping (Tanha & Kafaie, IEEE Access 2025)
            from explain.mitre_mapping import map_features_to_mitre, get_tactic_summary
            mitre_maps = map_features_to_mitre(explanation.node_importances)
            if mitre_maps:
                tactic_summary = get_tactic_summary(mitre_maps)
                st.markdown('<div class="label-caps" style="margin:16px 0 8px 0;">MITRE ATT&CK Mapping</div>', unsafe_allow_html=True)

                tactic_colors = {
                    "Collection": "#F59E0B", "Discovery": "#06B6D4", "Impact": "#EF4444",
                    "Execution": "#DC2626", "Persistence": "#8B5CF6", "Command and Control": "#F97316",
                    "Defense Evasion": "#10B981", "Privilege Escalation": "#EC4899",
                    "Initial Access": "#F43F5E", "Credential Access": "#A855F7",
                    "Exfiltration": "#E11D48", "Lateral Movement": "#6366F1",
                }
                for tactic, techniques in tactic_summary.items():
                    color = tactic_colors.get(tactic, "#94A3B8")
                    tech_html = " ".join(
                        f'<span style="background:rgba(51,65,85,0.5); padding:2px 6px; border-radius:3px; font-size:10px; color:#CBD5E1;">{t}</span>'
                        for t in techniques
                    )
                    st.markdown(f'''
                    <div style="margin:6px 0; padding:8px 12px; background:rgba(15,23,42,0.6); border-left:3px solid {color}; border-radius:0 4px 4px 0;">
                        <div style="font-family:'JetBrains Mono',monospace; font-size:11px; font-weight:600; color:{color}; text-transform:uppercase; letter-spacing:0.05em;">{tactic}</div>
                        <div style="margin-top:4px; display:flex; flex-wrap:wrap; gap:4px;">{tech_html}</div>
                    </div>
                    ''', unsafe_allow_html=True)

        with tab_agents:
            st.markdown(f'''
            <div style="display:flex; gap:12px; margin-bottom:16px; flex-wrap:wrap;">
                <div class="soc-metric"><div class="soc-metric-value">{len(result.evidence)}</div><div class="soc-metric-label">Total Evidence</div></div>
                <div class="soc-metric"><div class="soc-metric-value">{result.duration_seconds:.1f}s</div><div class="soc-metric-label">Analysis Time</div></div>
                <div class="soc-metric"><div class="soc-metric-value">{len(result.agent_stats)}</div><div class="soc-metric-label">Agents Active</div></div>
                <div class="soc-metric"><div class="soc-metric-value">{len(result.errors)}</div><div class="soc-metric-label">Errors</div></div>
            </div>
            ''', unsafe_allow_html=True)

            if result.errors:
                for err in result.errors:
                    st.error(err)

            agent_groups: dict[str, list] = {}
            for item in result.evidence:
                agent_groups.setdefault(item.source_agent, []).append(item)

            for agent_name, items in sorted(agent_groups.items()):
                color = AGENT_COLORS.get(agent_name, "#94A3B8")
                with st.expander(f"{agent_name.upper()} Agent -- {len(items)} findings", expanded=False):
                    for item in items[:50]:
                        risk = item.attributes.get("risk_level", item.attributes.get("sensitivity_tier", 0))
                        if risk >= 3:
                            severity = "high"
                        elif risk >= 2:
                            severity = "med"
                        else:
                            severity = "low"
                        st.markdown(f'''
                        <div class="evidence-item {severity}">
                            <span class="chip-{'critical' if severity=='high' else 'warning' if severity=='med' else 'cyan'}">{item.evidence_type.value}</span>
                            <span style="font-size:13px; color:#F8FAFC; margin-left:8px;">{item.description}</span>
                        </div>
                        ''', unsafe_allow_html=True)

        with tab_graph:
            st.markdown(f'''
            <div style="display:flex; gap:24px; margin-bottom:12px;">
                <span class="label-caps">Nodes: <span style="color:#F8FAFC; font-size:14px; font-weight:700;">{data.num_nodes}</span></span>
                <span class="label-caps">Edges: <span style="color:#F8FAFC; font-size:14px; font-weight:700;">{data.edge_index.size(1)}</span></span>
            </div>
            ''', unsafe_allow_html=True)

            show_explanation = st.checkbox("Highlight GNNExplainer subgraph", value=True)
            graph_html = create_graph_html(
                data, explanation if show_explanation else None
            )
            st.components.v1.html(graph_html, height=550, scrolling=True)

            type_colors_legend = {
                "Method": ("#38BDF8", "functions"),
                "Class": ("#A78BFA", "school"),
                "Permission": ("#F59E0B", "security"),
                "API Call": ("#EF4444", "api"),
                "Network": ("#10B981", "language"),
                "Crypto": ("#F472B6", "lock"),
            }
            legend_html = '<div style="display:flex; gap:16px; flex-wrap:wrap; margin-top:8px;">'
            for name, (color, _) in type_colors_legend.items():
                legend_html += f'<span style="display:flex; align-items:center; gap:4px;"><span style="width:10px;height:10px;border-radius:50%;background:{color};display:inline-block;"></span><span class="label-caps">{name}</span></span>'
            legend_html += '</div>'
            st.markdown(legend_html, unsafe_allow_html=True)

        with tab_model:
            model = load_model()
            total_params = sum(p.numel() for p in model.parameters())

            st.markdown(f'''
            <div class="soc-card">
                <div class="label-caps" style="margin-bottom:8px;">Architecture Specification</div>
                <table style="width:100%;">
                    <tr><td class="label-caps" style="width:40%;">Architecture</td><td style="color:#06B6D4;">2-Layer Graph Attention Network (GATConv)</td></tr>
                    <tr><td class="label-caps">Parameters</td><td style="color:#F8FAFC; font-weight:600;">{total_params:,}</td></tr>
                    <tr><td class="label-caps">Input Features</td><td>25-dim node features + 2 degree features (runtime)</td></tr>
                    <tr><td class="label-caps">Hidden Channels</td><td>64 channels, 4 attention heads</td></tr>
                    <tr><td class="label-caps">Readout</td><td>Mean + Max pooling + 15 graph-level structural features</td></tr>
                    <tr><td class="label-caps">Classifier</td><td>Linear(143&rarr;64) &rarr; ReLU &rarr; Linear(64&rarr;32) &rarr; ReLU &rarr; Linear(32&rarr;2)</td></tr>
                    <tr><td class="label-caps">Ensemble</td><td>GAT probabilities + 15 graph features &rarr; Gradient Boosting (200 trees)</td></tr>
                </table>
            </div>
            ''', unsafe_allow_html=True)

            ens_path = MODEL_DIR / "ensemble_results.json"
            results_path = ens_path if ens_path.exists() else MODEL_DIR / "results.json"
            if results_path.exists():
                results = json.loads(results_path.read_text())
                metrics = results.get("test_metrics", {})
                label = "Ensemble Metrics" if ens_path.exists() else "GAT Metrics"
                metrics_html = '<div style="display:flex; gap:12px; flex-wrap:wrap; margin-top:12px;">'
                for name, val in metrics.items():
                    metrics_html += f'<div class="soc-metric"><div class="soc-metric-value text-cyan">{val:.4f}</div><div class="soc-metric-label">{name}</div></div>'
                metrics_html += '</div>'
                st.markdown(f'<div class="label-caps" style="margin-top:16px;">{label}</div>{metrics_html}', unsafe_allow_html=True)

            st.markdown(f'''
            <div class="soc-card" style="margin-top:16px;">
                <div class="label-caps" style="margin-bottom:8px;">Current Prediction</div>
                <table style="width:100%;">
                    <tr><td class="label-caps" style="width:40%;">Predicted Class</td><td>{render_threat_badge(explanation.risk_score)} <span style="color:#F8FAFC; margin-left:8px;">{'Malware' if explanation.prediction == 1 else 'Benign'}</span></td></tr>
                    <tr><td class="label-caps">Confidence</td><td style="color:#06B6D4; font-weight:600;">{explanation.confidence:.4f}</td></tr>
                    <tr><td class="label-caps">Risk Score</td><td style="color:{'#EF4444' if explanation.risk_score >= 0.6 else '#F59E0B' if explanation.risk_score >= 0.3 else '#10B981'}; font-weight:600;">{explanation.risk_score:.4f}</td></tr>
                    <tr><td class="label-caps">Processing Time</td><td>{result.duration_seconds:.2f}s</td></tr>
                </table>
            </div>
            ''', unsafe_allow_html=True)


if __name__ == "__main__":
    main()
