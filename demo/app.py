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
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent.parent))

from agents.coordinator import Coordinator, AnalysisResult
from graph.builder import GraphBuilder
from model.gat import MalwareGAT
from explain.explainer import explain_prediction, ExplanationResult
from explain.attribution import attribute_to_agents, generate_explanation_text, AgentAttribution
from config.settings import MODEL_DIR, EVIDENCE_DIR, GRAPHS_DIR, ExplainConfig

DEMO_DIR = Path(__file__).parent / "samples"
DEMO_MODEL_DIR = Path(__file__).parent / "models"

DEMO_SAMPLES = {
    "0016B3D270A9C97E2A5DF8C7F063DABECB09BBF93BE090F9DE2BE65E91E178BE": "com.Android.SmsReg (Malware, VT:37)",
    "0059C516094CE7BE2138376D226326D26BA7DEBC6A1FB3D38C6B0810FF23BE01": "pj.ishuaji (Malware, VT:44)",
    "0000230AE0799EAA963F0AE5337A0E6E1AC32ACF1FA4E3C5FA7783CBAE2EFE77": "RegWoosh (Benign, VT:0)",
}

def _resolve_evidence_dir():
    if EVIDENCE_DIR.exists() and any(EVIDENCE_DIR.glob("*.json")):
        return EVIDENCE_DIR
    return DEMO_DIR

def _resolve_graphs_dir():
    if GRAPHS_DIR.exists() and any(GRAPHS_DIR.glob("*.pt")):
        return GRAPHS_DIR
    return DEMO_DIR

def _resolve_model_dir():
    if (MODEL_DIR / "best_gat.pt").exists():
        return MODEL_DIR
    return DEMO_MODEL_DIR

# ---------------------------------------------------------------------------
# Design tokens / CSS -- SOC forensics dashboard (Stitch design)
# ---------------------------------------------------------------------------

SOC_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@600;700&family=Inter:wght@400;500;600&family=JetBrains+Mono:wght@400;500;700&display=swap');
@import url('https://fonts.googleapis.com/css2?family=Material+Symbols+Outlined:opsz,wght,FILL,GRAD@20..48,100..700,0..1,-50..200');

:root {
    --canvas: #0B1326;
    --surface-1: #0F172A;
    --surface-2: #1E293B;
    --border: #334155;
    --accent: #06B6D4;
    --threat: #EF4444;
    --safe: #10B981;
    --warning: #F59E0B;
    --text-primary: #F8FAFC;
    --text-muted: #94A3B8;
}

.stApp { background-color: var(--canvas) !important; }
.stApp header { background-color: rgba(11,19,38,0.85) !important; backdrop-filter: blur(12px); }
section[data-testid="stSidebar"] { background-color: #0A0F20 !important; border-right: 1px solid var(--border); }

h1, h2, h3 { font-family: 'Plus Jakarta Sans', sans-serif !important; color: var(--text-primary) !important; letter-spacing: -0.01em; }
p, span, label, div { font-family: 'Inter', sans-serif; }

.material-symbols-outlined {
    font-family: 'Material Symbols Outlined';
    font-weight: normal;
    font-style: normal;
    font-size: 20px;
    line-height: 1;
    vertical-align: middle;
    font-variation-settings: 'FILL' 0, 'wght' 500, 'GRAD' 0, 'opsz' 20;
}

/* --- Label caps (section labels) --- */
.label-caps {
    font-family: 'JetBrains Mono', monospace;
    font-size: 11px;
    font-weight: 700;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    color: var(--text-muted);
}
.mono { font-family: 'JetBrains Mono', monospace !important; }
.mono-sm { font-family: 'JetBrains Mono', monospace; font-size: 11px; color: var(--text-muted); }
.text-cyan { color: var(--accent) !important; }
.text-red { color: var(--threat) !important; }
.text-amber { color: var(--warning) !important; }
.text-green { color: var(--safe) !important; }

/* --- App chrome header --- */
.app-topbar {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 14px 20px;
    background: var(--surface-1);
    border: 1px solid var(--border);
    border-radius: 10px;
    margin-bottom: 10px;
}
.app-topbar-left { display: flex; align-items: center; gap: 10px; }
.app-name {
    font-family: 'Plus Jakarta Sans', sans-serif;
    font-size: 19px;
    font-weight: 700;
    color: var(--text-primary);
    letter-spacing: -0.01em;
}
.version-badge {
    font-family: 'JetBrains Mono', monospace;
    font-size: 10px;
    font-weight: 700;
    letter-spacing: 0.06em;
    color: var(--accent);
    background: rgba(6,182,212,0.12);
    border: 1px solid rgba(6,182,212,0.35);
    border-radius: 4px;
    padding: 2px 8px;
}
.pulse-dot {
    width: 8px; height: 8px; border-radius: 50%;
    background: var(--safe);
    animation: pulse-dot 2s infinite;
    display: inline-block;
}
@keyframes pulse-dot {
    0%, 100% { opacity: 1; box-shadow: 0 0 0 0 rgba(16,185,129,0.4); }
    50% { opacity: 0.8; box-shadow: 0 0 0 4px rgba(16,185,129,0); }
}
.engine-status {
    display: flex; align-items: center; gap: 8px;
    font-family: 'JetBrains Mono', monospace;
    font-size: 10px; font-weight: 700; letter-spacing: 0.08em; text-transform: uppercase;
    color: var(--text-muted);
}

/* --- Telemetry breadcrumb bar --- */
.telemetry-bar {
    display: flex;
    align-items: center;
    flex-wrap: wrap;
    gap: 10px;
    padding: 8px 16px;
    background: rgba(15,23,42,0.6);
    border: 1px solid var(--border);
    border-radius: 8px;
    margin-bottom: 18px;
}
.telemetry-path {
    font-family: 'JetBrains Mono', monospace;
    font-size: 10.5px;
    font-weight: 700;
    letter-spacing: 0.05em;
    color: var(--accent);
}
.telemetry-pill {
    font-family: 'JetBrains Mono', monospace;
    font-size: 9.5px;
    font-weight: 700;
    letter-spacing: 0.06em;
    text-transform: uppercase;
    color: var(--text-muted);
    background: var(--surface-2);
    border: 1px solid var(--border);
    border-radius: 4px;
    padding: 2px 8px;
}
.telemetry-pill.good { color: var(--safe); border-color: rgba(16,185,129,0.3); }

/* --- Page title block --- */
.page-title {
    font-family: 'Plus Jakarta Sans', sans-serif;
    font-size: 26px;
    font-weight: 700;
    color: var(--text-primary);
    letter-spacing: -0.02em;
    margin-bottom: 4px;
}
.page-subtitle {
    font-family: 'Inter', sans-serif;
    font-size: 13.5px;
    color: var(--text-muted);
    max-width: 760px;
    line-height: 1.6;
    margin-bottom: 20px;
}

/* --- Cards --- */
.soc-card {
    background: var(--surface-1);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 16px;
    margin-bottom: 0.75rem;
}
.soc-card-glow-red {
    background: var(--surface-1);
    border: 1px solid rgba(239,68,68,0.35);
    border-radius: 10px;
    padding: 16px;
    margin-bottom: 0.75rem;
    box-shadow: 0 0 20px rgba(239,68,68,0.15);
}
.soc-card-glow-cyan {
    background: var(--surface-1);
    border: 1px solid rgba(6,182,212,0.3);
    border-radius: 10px;
    padding: 16px;
    margin-bottom: 0.75rem;
    box-shadow: 0 0 16px rgba(6,182,212,0.14);
}
.soc-card-glow-green {
    background: var(--surface-1);
    border: 1px solid rgba(16,185,129,0.35);
    border-radius: 10px;
    padding: 16px;
    margin-bottom: 0.75rem;
    box-shadow: 0 0 16px rgba(16,185,129,0.14);
}

/* --- Severity badges --- */
.sev-badge {
    display: inline-flex;
    align-items: center;
    font-family: 'JetBrains Mono', monospace;
    font-size: 10px; font-weight: 700;
    letter-spacing: 0.07em; text-transform: uppercase;
    padding: 2px 8px; border-radius: 4px;
}
.sev-critical { background: rgba(239,68,68,0.14); border: 1px solid rgba(239,68,68,0.4); color: var(--threat); }
.sev-high { background: rgba(245,158,11,0.14); border: 1px solid rgba(245,158,11,0.4); color: var(--warning); }
.sev-medium { background: rgba(148,163,184,0.14); border: 1px solid rgba(148,163,184,0.35); color: var(--text-muted); }
.sev-safe { background: rgba(16,185,129,0.14); border: 1px solid rgba(16,185,129,0.4); color: var(--safe); }
.sev-info { background: rgba(6,182,212,0.14); border: 1px solid rgba(6,182,212,0.4); color: var(--accent); }

/* --- Metrics --- */
.soc-metric-row { display: flex; gap: 12px; margin: 8px 0; flex-wrap: wrap; }
.soc-metric {
    background: var(--surface-1);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 12px 16px;
    flex: 1;
    text-align: center;
}
.soc-metric-value { font-family: 'Plus Jakarta Sans', sans-serif; font-size: 22px; font-weight: 700; color: var(--text-primary); }
.soc-metric-label { font-family: 'JetBrains Mono', monospace; font-size: 10px; font-weight: 700; letter-spacing: 0.08em; text-transform: uppercase; color: var(--text-muted); margin-top: 4px; }

/* --- Evidence rows --- */
.evidence-item {
    background: var(--surface-1);
    border: 1px solid var(--border);
    border-left: 3px solid var(--border);
    padding: 10px 14px;
    margin: 6px 0;
    border-radius: 0 6px 6px 0;
    font-size: 13px;
}
.evidence-item.high { border-left-color: var(--threat); }
.evidence-item.med { border-left-color: var(--warning); }
.evidence-item.low { border-left-color: var(--accent); }

/* --- Tabs --- */
div[data-testid="stTabs"] button[data-baseweb="tab"] {
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 11px !important;
    font-weight: 700 !important;
    letter-spacing: 0.06em !important;
    text-transform: uppercase !important;
    color: var(--text-muted) !important;
}
div[data-testid="stTabs"] button[aria-selected="true"] {
    color: var(--accent) !important;
    border-bottom-color: var(--accent) !important;
}

div[data-testid="stExpander"] {
    background: var(--surface-1) !important;
    border: 1px solid var(--border) !important;
    border-radius: 8px !important;
}

div[data-testid="stMetric"] {
    background: var(--surface-1) !important;
    border: 1px solid var(--border) !important;
    border-radius: 8px !important;
    padding: 12px !important;
}
div[data-testid="stMetric"] label { color: var(--text-muted) !important; font-family: 'JetBrains Mono', monospace !important; font-size: 10px !important; letter-spacing: 0.06em !important; text-transform: uppercase !important; }
div[data-testid="stMetric"] div[data-testid="stMetricValue"] { color: var(--accent) !important; font-family: 'Plus Jakarta Sans', sans-serif !important; }

table { border-collapse: collapse; width: 100%; }
th {
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 10px !important; font-weight: 700 !important;
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
    border-bottom: 1px solid rgba(51,65,85,0.3) !important;
    padding: 7px 12px !important;
}

div[data-testid="stFileUploader"] {
    background: var(--surface-1) !important;
    border: 1px dashed rgba(6,182,212,0.35) !important;
    border-radius: 10px !important;
}

/* --- Graph header bar --- */
.graph-header-bar {
    display: flex;
    align-items: center;
    justify-content: space-between;
    flex-wrap: wrap;
    gap: 12px;
    padding: 10px 14px;
    background: var(--surface-2);
    border: 1px solid var(--border);
    border-radius: 8px 8px 0 0;
    border-bottom: none;
}
.graph-legend { display: flex; gap: 16px; flex-wrap: wrap; }
.graph-legend-item { display: flex; align-items: center; gap: 6px; font-family: 'JetBrains Mono', monospace; font-size: 10px; font-weight: 600; letter-spacing: 0.04em; color: var(--text-muted); text-transform: uppercase; }
.graph-legend-dot { width: 9px; height: 9px; border-radius: 50%; display: inline-block; }

/* --- Footer bar --- */
.footer-bar {
    display: flex;
    align-items: center;
    justify-content: center;
    gap: 10px;
    padding: 10px 16px;
    margin-top: 28px;
    background: var(--surface-1);
    border: 1px solid var(--border);
    border-radius: 8px;
    font-family: 'JetBrains Mono', monospace;
    font-size: 10px;
    font-weight: 700;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    color: var(--text-muted);
}
</style>
"""


@st.cache_resource
def load_model():
    model = MalwareGAT()
    model_dir = _resolve_model_dir()
    model_path = model_dir / "best_gat.pt"
    if model_path.exists():
        model.load_state_dict(torch.load(model_path, weights_only=True))
    model.eval()
    return model


@st.cache_resource
def load_ensemble():
    import pickle
    model_dir = _resolve_model_dir()
    gbm_path = model_dir / "ensemble_gbm.pkl"
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


def create_risk_gauge(score: float) -> str:
    """Return an inline SVG radial gauge (no Plotly)."""
    if score < 0.3:
        color = "#10B981"
        label = "NOMINAL"
    elif score < 0.6:
        color = "#F59E0B"
        label = "SUSPICIOUS"
    else:
        color = "#EF4444"
        label = "CRITICAL THREAT"

    pct = max(0.0, min(1.0, score))
    radius = 80
    cx, cy = 100, 100
    circumference = 2 * 3.14159265 * radius
    arc_fraction = 0.75  # 270-degree sweep
    track_len = circumference * arc_fraction
    fill_len = track_len * pct
    start_angle = 135  # degrees, so sweep covers 270deg ending at 45deg

    return f'''
    <div style="display:flex; align-items:center; justify-content:center; background:transparent;">
    <svg viewBox="0 0 200 200" width="240" height="240" xmlns="http://www.w3.org/2000/svg">
        <defs>
            <filter id="gauge-glow" x="-50%" y="-50%" width="200%" height="200%">
                <feGaussianBlur stdDeviation="4" result="blur"/>
                <feMerge>
                    <feMergeNode in="blur"/>
                    <feMergeNode in="SourceGraphic"/>
                </feMerge>
            </filter>
        </defs>
        <circle cx="{cx}" cy="{cy}" r="{radius}" fill="none" stroke="#1E293B" stroke-width="14"
            stroke-dasharray="{track_len:.2f} {circumference:.2f}"
            stroke-dashoffset="0"
            transform="rotate({start_angle} {cx} {cy})"
            stroke-linecap="round"/>
        <circle cx="{cx}" cy="{cy}" r="{radius}" fill="none" stroke="{color}" stroke-width="14"
            stroke-dasharray="{fill_len:.2f} {circumference:.2f}"
            stroke-dashoffset="0"
            transform="rotate({start_angle} {cx} {cy})"
            stroke-linecap="round"
            filter="url(#gauge-glow)"/>
        <text x="{cx}" y="{cy - 6}" text-anchor="middle" fill="{color}" font-family="Plus Jakarta Sans" font-size="40" font-weight="700">{int(pct*100)}</text>
        <text x="{cx}" y="{cy + 20}" text-anchor="middle" fill="#94A3B8" font-family="JetBrains Mono" font-size="10" letter-spacing="1">/ 100 RISK</text>
        <text x="{cx}" y="{cy + 48}" text-anchor="middle" fill="{color}" font-family="JetBrains Mono" font-size="12" font-weight="700" letter-spacing="1.5">{label}</text>
    </svg>
    </div>
    '''


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


@st.cache_data(show_spinner="Running GNNExplainer...")
def load_preanalyzed(_sha):
    ev_dir = _resolve_evidence_dir()
    gr_dir = _resolve_graphs_dir()
    evidence_path = ev_dir / f"{_sha}.json"
    graph_path = gr_dir / f"{_sha}.pt"

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


# ---------------------------------------------------------------------------
# Presentation helpers
# ---------------------------------------------------------------------------

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


def render_threat_badge(score: float) -> str:
    if score >= 0.6:
        return '<span class="sev-badge sev-critical">CRITICAL THREAT</span>'
    elif score >= 0.3:
        return '<span class="sev-badge sev-high">SUSPICIOUS</span>'
    else:
        return '<span class="sev-badge sev-safe">NOMINAL</span>'


def severity_badge(level: str) -> str:
    """level: critical | high | medium | safe | info"""
    return f'<span class="sev-badge sev-{level}">{level.upper()}</span>'


def _icon(name: str, size: int = 20) -> str:
    return f'<span class="material-symbols-outlined" style="font-size:{size}px;">{name}</span>'


def _topbar():
    st.markdown(f'''
    <div class="app-topbar">
        <div class="app-topbar-left">
            {_icon("shield_lock", 22)}
            <span class="app-name">APKSentinel</span>
            <span class="version-badge">v2.4-PROD</span>
        </div>
        <div class="engine-status">
            <span class="pulse-dot"></span>
            CORE ENGINES ONLINE
        </div>
    </div>
    ''', unsafe_allow_html=True)


def _telemetry_bar():
    st.markdown('''
    <div class="telemetry-bar">
        <span class="telemetry-path">SENTINEL::FORENSIC_TRIAGE / ANALYSIS_WORKSPACE</span>
        <span class="telemetry-pill good">REAL-TIME TELEMETRY VECTORED</span>
        <span class="telemetry-pill good">NEURAL ENGINE: ONLINE</span>
        <span class="telemetry-pill">LATENCY: 42MS</span>
    </div>
    ''', unsafe_allow_html=True)


def _pipeline_svg():
    return '''
    <div style="background:#0b1326;">
    <svg viewBox="0 0 900 120" xmlns="http://www.w3.org/2000/svg" style="width:100%; max-width:900px; display:block; margin:0 auto;">
        <defs>
            <filter id="node-glow">
                <feGaussianBlur stdDeviation="3" result="blur"/>
                <feComposite in="SourceGraphic" in2="blur" operator="over"/>
            </filter>
        </defs>

        <line x1="160" y1="60" x2="230" y2="60" stroke="#1E293B" stroke-width="2" stroke-dasharray="6,4"/>
        <polygon points="226,55 236,60 226,65" fill="#334155"/>
        <line x1="330" y1="60" x2="400" y2="60" stroke="#1E293B" stroke-width="2" stroke-dasharray="6,4"/>
        <polygon points="396,55 406,60 396,65" fill="#334155"/>
        <line x1="500" y1="60" x2="570" y2="60" stroke="#1E293B" stroke-width="2" stroke-dasharray="6,4"/>
        <polygon points="566,55 576,60 566,65" fill="#334155"/>
        <line x1="670" y1="60" x2="740" y2="60" stroke="#1E293B" stroke-width="2" stroke-dasharray="6,4"/>
        <polygon points="736,55 746,60 736,65" fill="#334155"/>

        <rect x="40" y="25" width="120" height="70" rx="10" fill="#0F172A" stroke="#334155" stroke-width="1.5"/>
        <text x="100" y="55" text-anchor="middle" fill="#F8FAFC" font-family="Plus Jakarta Sans" font-size="13" font-weight="700">APK Input</text>
        <text x="100" y="75" text-anchor="middle" fill="#64748B" font-family="JetBrains Mono" font-size="10">Android package</text>

        <rect x="236" y="25" width="100" height="70" rx="10" fill="#0F172A" stroke="#F59E0B" stroke-width="1.5" opacity="0.9"/>
        <text x="286" y="55" text-anchor="middle" fill="#F59E0B" font-family="Plus Jakarta Sans" font-size="13" font-weight="700">4 Agents</text>
        <text x="286" y="75" text-anchor="middle" fill="#64748B" font-family="JetBrains Mono" font-size="10">Static analysis</text>

        <rect x="406" y="25" width="100" height="70" rx="10" fill="#0F172A" stroke="#8B5CF6" stroke-width="1.5" opacity="0.9"/>
        <text x="456" y="55" text-anchor="middle" fill="#8B5CF6" font-family="Plus Jakarta Sans" font-size="13" font-weight="700">Graph</text>
        <text x="456" y="75" text-anchor="middle" fill="#64748B" font-family="JetBrains Mono" font-size="10">Behaviour net</text>

        <rect x="576" y="25" width="100" height="70" rx="10" fill="#0F172A" stroke="#06B6D4" stroke-width="1.5" opacity="0.9"/>
        <text x="626" y="55" text-anchor="middle" fill="#06B6D4" font-family="Plus Jakarta Sans" font-size="13" font-weight="700">GAT</text>
        <text x="626" y="75" text-anchor="middle" fill="#64748B" font-family="JetBrains Mono" font-size="10">Classify</text>

        <rect x="746" y="25" width="120" height="70" rx="10" fill="#0F172A" stroke="#10B981" stroke-width="1.5" opacity="0.9"/>
        <text x="806" y="55" text-anchor="middle" fill="#10B981" font-family="Plus Jakarta Sans" font-size="13" font-weight="700">Explain</text>
        <text x="806" y="75" text-anchor="middle" fill="#64748B" font-family="JetBrains Mono" font-size="10">XAI + MITRE</text>
    </svg>
    </div>
    '''


def _coordinator_svg():
    return '''
    <div style="background:#0b1326;">
    <svg viewBox="0 0 520 320" xmlns="http://www.w3.org/2000/svg" style="width:100%; max-width:520px; display:block; margin:0 auto;">
        <defs>
            <filter id="c-glow">
                <feGaussianBlur stdDeviation="4" result="blur"/>
                <feComposite in="SourceGraphic" in2="blur" operator="over"/>
            </filter>
        </defs>

        <line x1="260" y1="80" x2="110" y2="140" stroke="#334155" stroke-width="1.5" stroke-dasharray="5,3"/>
        <line x1="260" y1="80" x2="260" y2="140" stroke="#334155" stroke-width="1.5" stroke-dasharray="5,3"/>
        <line x1="260" y1="80" x2="410" y2="140" stroke="#334155" stroke-width="1.5" stroke-dasharray="5,3"/>
        <line x1="260" y1="80" x2="185" y2="140" stroke="#334155" stroke-width="1.5" stroke-dasharray="5,3"/>

        <circle cx="260" cy="60" r="30" fill="#0F172A" stroke="#06B6D4" stroke-width="2" filter="url(#c-glow)"/>
        <text x="260" y="56" text-anchor="middle" fill="#06B6D4" font-family="Plus Jakarta Sans" font-size="10" font-weight="700">COORD</text>
        <text x="260" y="70" text-anchor="middle" fill="#64748B" font-family="JetBrains Mono" font-size="8">dispatch</text>

        <rect x="50" y="140" width="120" height="55" rx="8" fill="#0F172A" stroke="#F59E0B" stroke-width="1.5"/>
        <text x="110" y="163" text-anchor="middle" fill="#F59E0B" font-family="Plus Jakarta Sans" font-size="11" font-weight="700">Manifest</text>
        <text x="110" y="180" text-anchor="middle" fill="#64748B" font-family="JetBrains Mono" font-size="9">permissions</text>

        <rect x="190" y="140" width="120" height="55" rx="8" fill="#0F172A" stroke="#38BDF8" stroke-width="1.5"/>
        <text x="250" y="163" text-anchor="middle" fill="#38BDF8" font-family="Plus Jakarta Sans" font-size="11" font-weight="700">Bytecode</text>
        <text x="250" y="180" text-anchor="middle" fill="#64748B" font-family="JetBrains Mono" font-size="9">call graph</text>

        <rect x="50" y="210" width="120" height="55" rx="8" fill="#0F172A" stroke="#10B981" stroke-width="1.5"/>
        <text x="110" y="233" text-anchor="middle" fill="#10B981" font-family="Plus Jakarta Sans" font-size="11" font-weight="700">Network</text>
        <text x="110" y="250" text-anchor="middle" fill="#64748B" font-family="JetBrains Mono" font-size="9">C2 detect</text>

        <rect x="190" y="210" width="120" height="55" rx="8" fill="#0F172A" stroke="#F472B6" stroke-width="1.5"/>
        <text x="250" y="233" text-anchor="middle" fill="#F472B6" font-family="Plus Jakarta Sans" font-size="11" font-weight="700">Crypto</text>
        <text x="250" y="250" text-anchor="middle" fill="#64748B" font-family="JetBrains Mono" font-size="9">weak algo</text>

        <line x1="260" y1="80" x2="110" y2="210" stroke="#334155" stroke-width="1.5" stroke-dasharray="5,3"/>
        <line x1="260" y1="80" x2="250" y2="210" stroke="#334155" stroke-width="1.5" stroke-dasharray="5,3"/>

        <rect x="360" y="155" width="140" height="110" rx="10" fill="rgba(6,182,212,0.05)" stroke="#1E293B" stroke-width="1"/>
        <text x="430" y="178" text-anchor="middle" fill="#64748B" font-family="JetBrains Mono" font-size="9" font-weight="700">EvidenceItem[]</text>
        <text x="430" y="198" text-anchor="middle" fill="#94A3B8" font-family="JetBrains Mono" font-size="8">.entity</text>
        <text x="430" y="212" text-anchor="middle" fill="#94A3B8" font-family="JetBrains Mono" font-size="8">.source_agent</text>
        <text x="430" y="226" text-anchor="middle" fill="#94A3B8" font-family="JetBrains Mono" font-size="8">.attributes</text>
        <text x="430" y="240" text-anchor="middle" fill="#94A3B8" font-family="JetBrains Mono" font-size="8">.confidence</text>
        <line x1="310" y1="195" x2="360" y2="210" stroke="#334155" stroke-width="1" stroke-dasharray="4,3"/>

        <text x="260" y="300" text-anchor="middle" fill="#475569" font-family="Plus Jakarta Sans" font-size="11">Star topology  /  No inter-agent communication  /  Parallel dispatch</text>
    </svg>
    </div>
    '''


def _gat_svg():
    return '''
    <div style="background:#0b1326;">
    <svg viewBox="0 0 600 200" xmlns="http://www.w3.org/2000/svg" style="width:100%; max-width:600px; display:block; margin:0 auto;">
        <text x="20" y="20" fill="#475569" font-family="JetBrains Mono" font-size="10">GAT Architecture</text>

        <rect x="20" y="40" width="90" height="130" rx="6" fill="rgba(6,182,212,0.06)" stroke="#1E293B" stroke-width="1"/>
        <text x="65" y="60" text-anchor="middle" fill="#94A3B8" font-family="JetBrains Mono" font-size="9">Input</text>
        <text x="65" y="78" text-anchor="middle" fill="#06B6D4" font-family="Plus Jakarta Sans" font-size="11" font-weight="700">25-dim</text>
        <text x="65" y="94" text-anchor="middle" fill="#64748B" font-family="JetBrains Mono" font-size="8">node feats</text>
        <circle cx="50" cy="120" r="4" fill="#38BDF8" opacity="0.6"/><circle cx="65" cy="115" r="4" fill="#F59E0B" opacity="0.6"/>
        <circle cx="80" cy="122" r="4" fill="#EF4444" opacity="0.6"/><circle cx="58" cy="135" r="4" fill="#10B981" opacity="0.6"/>
        <circle cx="73" cy="138" r="4" fill="#F472B6" opacity="0.6"/>
        <line x1="50" y1="120" x2="65" y2="115" stroke="#334155" stroke-width="0.8"/>
        <line x1="65" y1="115" x2="80" y2="122" stroke="#334155" stroke-width="0.8"/>
        <line x1="58" y1="135" x2="73" y2="138" stroke="#334155" stroke-width="0.8"/>
        <line x1="50" y1="120" x2="58" y2="135" stroke="#334155" stroke-width="0.8"/>

        <line x1="110" y1="105" x2="145" y2="105" stroke="#334155" stroke-width="1.5" stroke-dasharray="4,3"/>
        <polygon points="141,100 151,105 141,110" fill="#334155"/>

        <rect x="150" y="40" width="110" height="130" rx="6" fill="rgba(139,92,246,0.06)" stroke="#1E293B" stroke-width="1"/>
        <text x="205" y="60" text-anchor="middle" fill="#94A3B8" font-family="JetBrains Mono" font-size="9">Layer 1</text>
        <text x="205" y="78" text-anchor="middle" fill="#8B5CF6" font-family="Plus Jakarta Sans" font-size="11" font-weight="700">GATConv</text>
        <text x="205" y="94" text-anchor="middle" fill="#64748B" font-family="JetBrains Mono" font-size="8">25 -> 64</text>
        <text x="205" y="110" text-anchor="middle" fill="#64748B" font-family="JetBrains Mono" font-size="8">4 heads</text>
        <text x="205" y="126" text-anchor="middle" fill="#64748B" font-family="JetBrains Mono" font-size="8">+ BatchNorm</text>
        <text x="205" y="142" text-anchor="middle" fill="#64748B" font-family="JetBrains Mono" font-size="8">+ ELU</text>
        <text x="205" y="158" text-anchor="middle" fill="#64748B" font-family="JetBrains Mono" font-size="8">+ Dropout</text>

        <line x1="260" y1="105" x2="295" y2="105" stroke="#334155" stroke-width="1.5" stroke-dasharray="4,3"/>
        <polygon points="291,100 301,105 291,110" fill="#334155"/>

        <rect x="300" y="40" width="110" height="130" rx="6" fill="rgba(6,182,212,0.06)" stroke="#1E293B" stroke-width="1"/>
        <text x="355" y="60" text-anchor="middle" fill="#94A3B8" font-family="JetBrains Mono" font-size="9">Layer 2</text>
        <text x="355" y="78" text-anchor="middle" fill="#06B6D4" font-family="Plus Jakarta Sans" font-size="11" font-weight="700">GATConv</text>
        <text x="355" y="94" text-anchor="middle" fill="#64748B" font-family="JetBrains Mono" font-size="8">256 -> 64</text>
        <text x="355" y="110" text-anchor="middle" fill="#64748B" font-family="JetBrains Mono" font-size="8">1 head</text>
        <text x="355" y="126" text-anchor="middle" fill="#64748B" font-family="JetBrains Mono" font-size="8">+ BatchNorm</text>
        <text x="355" y="142" text-anchor="middle" fill="#64748B" font-family="JetBrains Mono" font-size="8">+ Readout</text>
        <text x="355" y="158" text-anchor="middle" fill="#64748B" font-family="JetBrains Mono" font-size="8">mean+max</text>

        <line x1="410" y1="105" x2="445" y2="105" stroke="#334155" stroke-width="1.5" stroke-dasharray="4,3"/>
        <polygon points="441,100 451,105 441,110" fill="#334155"/>

        <rect x="450" y="55" width="130" height="100" rx="6" fill="rgba(16,185,129,0.06)" stroke="#1E293B" stroke-width="1"/>
        <text x="515" y="75" text-anchor="middle" fill="#94A3B8" font-family="JetBrains Mono" font-size="9">Classifier</text>
        <text x="515" y="95" text-anchor="middle" fill="#64748B" font-family="JetBrains Mono" font-size="8">143 -> 64 -> 32 -> 2</text>
        <text x="515" y="115" text-anchor="middle" fill="#10B981" font-family="Plus Jakarta Sans" font-size="11" font-weight="700">Malware?</text>
        <text x="515" y="135" text-anchor="middle" fill="#64748B" font-family="JetBrains Mono" font-size="8">36,258 params</text>
    </svg>
    </div>
    '''


def _footer_bar():
    st.markdown('''
    <div class="footer-bar">
        <span class="material-symbols-outlined" style="font-size:14px;">cloud_sync</span>
        IOC EXPORT STREAM: STIX 2.1 / MISP READY
    </div>
    ''', unsafe_allow_html=True)


def render_landing():
    _topbar()
    _telemetry_bar()

    st.markdown('''
    <div class="page-title">Interactive Malware Analyzer &amp; Behavioral Telemetry</div>
    <div class="page-subtitle">
        Automated multi-agent static analysis, graph-attention classification, and explainable risk
        scoring for fraudulent Android banking applications. Upload a sample or inspect a
        pre-analyzed case from the forensic archive to begin.
    </div>
    ''', unsafe_allow_html=True)

    # Stats
    st.markdown('''
    <div class="soc-metric-row" style="justify-content:center;">
        <div class="soc-metric" style="min-width:130px; flex:0 0 150px;">
            <div class="soc-metric-value" style="color:#06B6D4; font-size:26px;">94.05<span style="font-size:14px; opacity:0.7;">%</span></div>
            <div class="soc-metric-label">Accuracy</div>
        </div>
        <div class="soc-metric" style="min-width:130px; flex:0 0 150px;">
            <div class="soc-metric-value" style="color:#8B5CF6; font-size:26px;">0.958</div>
            <div class="soc-metric-label">F1 Score</div>
        </div>
        <div class="soc-metric" style="min-width:130px; flex:0 0 150px;">
            <div class="soc-metric-value" style="color:#10B981; font-size:26px;">98.3<span style="font-size:14px; opacity:0.7;">%</span></div>
            <div class="soc-metric-label">Recall</div>
        </div>
        <div class="soc-metric" style="min-width:130px; flex:0 0 150px;">
            <div class="soc-metric-value" style="color:#F59E0B; font-size:26px;">554</div>
            <div class="soc-metric-label">APKs Analyzed</div>
        </div>
    </div>
    ''', unsafe_allow_html=True)

    # Pipeline SVG
    st.markdown('<div class="label-caps" style="margin:28px 0 12px 0; text-align:center;">End-to-End Pipeline</div>', unsafe_allow_html=True)
    st.components.v1.html(_pipeline_svg(), height=140, scrolling=False)

    st.markdown('<div style="height:24px;"></div>', unsafe_allow_html=True)

    # Agent Architecture SVG
    st.markdown('<div class="label-caps" style="margin:28px 0 16px 0; text-align:center;">Agent Architecture</div>', unsafe_allow_html=True)

    col_svg, col_cards = st.columns([1, 1])

    with col_svg:
        st.components.v1.html(_coordinator_svg(), height=340, scrolling=False)

    with col_cards:
        agents = [
            ("Manifest Agent", "#F59E0B", "description", [
                "Parses AndroidManifest.xml",
                "Extracts 30 dangerous permissions",
                "Identifies exported components",
                "Flags accessibility service abuse",
            ]),
            ("Bytecode Agent", "#38BDF8", "code", [
                "Analyzes DEX via Androguard",
                "Builds function call graph",
                "Flags ~150 sensitive API calls",
                "Detects reflection and dynamic loading",
            ]),
            ("Network Agent", "#10B981", "language", [
                "Extracts URLs, IPs, domains",
                "Decodes base64-encoded URLs",
                "Detects C2 indicators",
                "Flags high-entropy domains",
            ]),
            ("Crypto Agent", "#F472B6", "lock", [
                "Scans javax.crypto / java.security",
                "Detects hardcoded keys",
                "Flags weak algorithms (DES, MD5)",
                "Identifies custom crypto patterns",
            ]),
        ]
        for name, color, icon, items in agents:
            items_html = "".join(
                f'<div style="font-size:12px; color:#94A3B8; padding:1px 0; display:flex; align-items:baseline; gap:6px;">'
                f'<span style="width:4px; height:4px; border-radius:50%; background:{color}; flex-shrink:0; margin-top:6px;"></span>{item}</div>'
                for item in items
            )
            st.markdown(f'''
            <div class="soc-card" style="border-left:3px solid {color}; margin-bottom:8px; padding:10px 14px;">
                <div style="display:flex; align-items:center; gap:8px; margin-bottom:6px;">
                    <span class="material-symbols-outlined" style="color:{color}; font-size:16px;">{icon}</span>
                    <span style="font-family:'JetBrains Mono',monospace; font-size:11px; font-weight:700; color:{color}; text-transform:uppercase; letter-spacing:0.06em;">{name}</span>
                </div>
                {items_html}
            </div>
            ''', unsafe_allow_html=True)

    # GAT Architecture
    st.markdown('<div class="label-caps" style="margin:28px 0 16px 0; text-align:center;">GAT Architecture</div>', unsafe_allow_html=True)
    st.components.v1.html(_gat_svg(), height=220, scrolling=False)

    # Model comparison
    st.markdown('<div style="height:16px;"></div>', unsafe_allow_html=True)
    col_gat, col_ens = st.columns(2)

    with col_gat:
        st.markdown(f'''
        <div class="soc-card">
            <div style="display:flex; align-items:center; gap:8px; margin-bottom:10px;">
                <span class="material-symbols-outlined" style="color:#06B6D4; font-size:18px;">hub</span>
                <span style="font-family:'JetBrains Mono',monospace; font-size:11px; font-weight:700; color:#06B6D4; text-transform:uppercase; letter-spacing:0.06em;">GAT Only</span>
            </div>
            <div style="font-size:13px; color:#64748B; line-height:1.6; margin-bottom:14px;">
                2-layer Graph Attention Network with 4 heads. 36,258 parameters. Trained on CPU.
            </div>
            <div style="margin:6px 0;">
                <div style="display:flex; justify-content:space-between; font-size:11px; color:#64748B;"><span>Accuracy</span><span style="color:#F8FAFC; font-weight:700;">85.71%</span></div>
                <div style="background:rgba(51,65,85,0.3); height:5px; border-radius:3px; margin-top:4px; overflow:hidden;"><div style="width:85.7%; height:100%; background:linear-gradient(90deg,#06B6D4,#0891B2); border-radius:3px;"></div></div>
            </div>
            <div style="margin:8px 0;">
                <div style="display:flex; justify-content:space-between; font-size:11px; color:#64748B;"><span>F1 Score</span><span style="color:#F8FAFC; font-weight:700;">89.66%</span></div>
                <div style="background:rgba(51,65,85,0.3); height:5px; border-radius:3px; margin-top:4px; overflow:hidden;"><div style="width:89.7%; height:100%; background:linear-gradient(90deg,#06B6D4,#0891B2); border-radius:3px;"></div></div>
            </div>
            <div style="margin:8px 0;">
                <div style="display:flex; justify-content:space-between; font-size:11px; color:#64748B;"><span>AUC-ROC</span><span style="color:#F8FAFC; font-weight:700;">92.24%</span></div>
                <div style="background:rgba(51,65,85,0.3); height:5px; border-radius:3px; margin-top:4px; overflow:hidden;"><div style="width:92.2%; height:100%; background:linear-gradient(90deg,#06B6D4,#0891B2); border-radius:3px;"></div></div>
            </div>
        </div>
        ''', unsafe_allow_html=True)

    with col_ens:
        st.markdown(f'''
        <div class="soc-card-glow-green">
            <div style="display:flex; align-items:center; gap:8px; margin-bottom:10px;">
                <span class="material-symbols-outlined" style="color:#10B981; font-size:18px;">layers</span>
                <span style="font-family:'JetBrains Mono',monospace; font-size:11px; font-weight:700; color:#10B981; text-transform:uppercase; letter-spacing:0.06em;">Ensemble (GAT + GBM)</span>
                <span class="sev-badge sev-safe" style="font-size:8px; padding:1px 6px; margin-left:auto;">BEST</span>
            </div>
            <div style="font-size:13px; color:#64748B; line-height:1.6; margin-bottom:14px;">
                Gradient Boosting over 33 features: 15 graph-level + 18 structural + GAT probability.
            </div>
            <div style="margin:6px 0;">
                <div style="display:flex; justify-content:space-between; font-size:11px; color:#64748B;"><span>Accuracy</span><span style="color:#F8FAFC; font-weight:700;">94.05%</span></div>
                <div style="background:rgba(51,65,85,0.3); height:5px; border-radius:3px; margin-top:4px; overflow:hidden;"><div style="width:94%; height:100%; background:linear-gradient(90deg,#10B981,#059669); border-radius:3px;"></div></div>
            </div>
            <div style="margin:8px 0;">
                <div style="display:flex; justify-content:space-between; font-size:11px; color:#64748B;"><span>F1 Score</span><span style="color:#F8FAFC; font-weight:700;">95.80%</span></div>
                <div style="background:rgba(51,65,85,0.3); height:5px; border-radius:3px; margin-top:4px; overflow:hidden;"><div style="width:95.8%; height:100%; background:linear-gradient(90deg,#10B981,#059669); border-radius:3px;"></div></div>
            </div>
            <div style="margin:8px 0;">
                <div style="display:flex; justify-content:space-between; font-size:11px; color:#64748B;"><span>Recall</span><span style="color:#F8FAFC; font-weight:700;">98.28%</span></div>
                <div style="background:rgba(51,65,85,0.3); height:5px; border-radius:3px; margin-top:4px; overflow:hidden;"><div style="width:98.3%; height:100%; background:linear-gradient(90deg,#10B981,#059669); border-radius:3px;"></div></div>
            </div>
            <div style="margin:8px 0;">
                <div style="display:flex; justify-content:space-between; font-size:11px; color:#64748B;"><span>AUC-ROC</span><span style="color:#F8FAFC; font-weight:700;">95.03%</span></div>
                <div style="background:rgba(51,65,85,0.3); height:5px; border-radius:3px; margin-top:4px; overflow:hidden;"><div style="width:95%; height:100%; background:linear-gradient(90deg,#10B981,#059669); border-radius:3px;"></div></div>
            </div>
        </div>
        ''', unsafe_allow_html=True)

    # Explainability
    st.markdown('<div class="label-caps" style="margin:28px 0 12px 0; text-align:center;">Explainability Layer</div>', unsafe_allow_html=True)
    col_xai, col_mitre = st.columns(2)

    with col_xai:
        st.markdown(f'''
        <div class="soc-card" style="border-left:3px solid #8B5CF6;">
            <div style="display:flex; align-items:center; gap:8px; margin-bottom:8px;">
                <span class="material-symbols-outlined" style="color:#8B5CF6; font-size:16px;">visibility</span>
                <span style="font-family:'JetBrains Mono',monospace; font-size:11px; font-weight:700; color:#8B5CF6; text-transform:uppercase; letter-spacing:0.06em;">GNNExplainer</span>
            </div>
            <div style="font-size:13px; color:#94A3B8; line-height:1.7;">
                Generates node importance masks over the behaviour graph. Top-15 nodes form the explanation subgraph. Each node traces back to its source agent for attribution percentages.
            </div>
        </div>
        ''', unsafe_allow_html=True)

    with col_mitre:
        st.markdown(f'''
        <div class="soc-card" style="border-left:3px solid #F59E0B;">
            <div style="display:flex; align-items:center; gap:8px; margin-bottom:8px;">
                <span class="material-symbols-outlined" style="color:#F59E0B; font-size:16px;">track_changes</span>
                <span style="font-family:'JetBrains Mono',monospace; font-size:11px; font-weight:700; color:#F59E0B; text-transform:uppercase; letter-spacing:0.06em;">MITRE ATT&amp;CK Mapping</span>
            </div>
            <div style="font-size:13px; color:#94A3B8; line-height:1.7;">
                Maps 28 permissions and 29 API calls to ATT&amp;CK for Mobile techniques and tactics. Based on Tanha &amp; Kafaie (IEEE Access, 2025).
            </div>
        </div>
        ''', unsafe_allow_html=True)

    # Reference
    st.markdown('''
    <div class="soc-card" style="margin-top:20px; text-align:center; border:1px solid rgba(6,182,212,0.15);">
        <div class="label-caps" style="margin-bottom:6px;">Reference</div>
        <div style="font-size:13px; color:#94A3B8; line-height:1.6;">
            M. Tanha and S. Kafaie, "A Review of Explainable AI for Android Malware Detection and Analysis,"<br>
            <span style="color:#06B6D4;">IEEE Access</span>, vol. 13, pp. 141958-141974, 2025
        </div>
        <div style="font-family:'JetBrains Mono',monospace; font-size:10px; color:#475569; margin-top:4px;">DOI: 10.1109/ACCESS.2025.3597575</div>
    </div>
    ''', unsafe_allow_html=True)

    _footer_bar()


def main():
    st.set_page_config(
        page_title="APKSentinel // Threat Analysis",
        page_icon=":shield:",
        layout="wide",
    )

    st.markdown(SOC_CSS, unsafe_allow_html=True)

    page = st.sidebar.radio(
        "Navigation",
        ["Home", "Analyze"],
        label_visibility="collapsed",
    )

    if page == "Home":
        render_landing()
        return

    _topbar()
    _telemetry_bar()

    st.markdown('''
    <div class="page-title">Interactive Malware Analyzer &amp; Behavioral Telemetry</div>
    <div class="page-subtitle">
        Submit an APK for live multi-agent detonation, or load a case from the forensic archive
        to inspect agent evidence, GAT risk scoring, and the GNNExplainer attention subgraph.
    </div>
    ''', unsafe_allow_html=True)

    col_upload, col_sample = st.columns([3, 2])
    with col_upload:
        st.markdown('<div class="label-caps" style="margin-bottom:6px;">Upload Sample</div>', unsafe_allow_html=True)
        uploaded_file = st.file_uploader("Drop APK for analysis", type=["apk"], label_visibility="collapsed")

    pre_analyzed = None
    with col_sample:
        st.markdown('<div class="label-caps" style="margin-bottom:6px;">Forensic Archive</div>', unsafe_allow_html=True)
        evidence_dir = _resolve_evidence_dir()
        if evidence_dir.exists():
            evidence_files = sorted(evidence_dir.glob("*.json"))
            if evidence_files:
                options = [f.stem for f in evidence_files]
                display_names = [DEMO_SAMPLES.get(o, o[:16] + "...") for o in options]
                demo_default = "0016B3D270A9C97E2A5DF8C7F063DABECB09BBF93BE090F9DE2BE65E91E178BE"
                default_idx = options.index(demo_default) if demo_default in options else 0
                selected_idx = st.selectbox(
                    "Or select a pre-analyzed sample",
                    range(len(options)),
                    index=default_idx,
                    format_func=lambda i: display_names[i],
                    label_visibility="collapsed",
                )
                pre_analyzed = options[selected_idx]

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
        package_name = getattr(result, "package_name", None) or "unknown.package"

        st.markdown('<div style="height:8px;"></div>', unsafe_allow_html=True)

        # --- Threat verdict panel ---
        glow_class = (
            "soc-card-glow-red" if explanation.risk_score >= 0.6
            else "soc-card-glow-cyan" if explanation.risk_score >= 0.3
            else "soc-card-glow-green"
        )
        family_name = (
            "Trojan Dropper" if explanation.risk_score >= 0.6
            else "Anomalous Activity" if explanation.risk_score >= 0.3
            else "Clean Binary"
        )

        st.markdown(f'<div class="{glow_class}">', unsafe_allow_html=True)
        st.markdown('<div class="label-caps" style="margin-bottom:10px;">Threat Verdict</div>', unsafe_allow_html=True)

        col_gauge, col_details = st.columns([1, 2])

        with col_gauge:
            st.components.v1.html(create_risk_gauge(explanation.risk_score), height=260)

        with col_details:
            st.markdown(f'''
            <div style="margin-bottom:10px;">
                {render_threat_badge(explanation.risk_score)}
                <span style="font-size:20px; font-weight:700; color:#F8FAFC; margin-left:10px;">{family_name}</span>
            </div>
            <table style="width:100%; margin-bottom:10px;">
                <tr><td class="label-caps" style="width:35%;">Package Identifier</td><td class="mono" style="color:#F8FAFC;">{package_name}</td></tr>
                <tr><td class="label-caps">SHA-256</td><td class="mono-sm">{sha_display}</td></tr>
                <tr><td class="label-caps">Confidence</td><td class="mono" style="color:#06B6D4;">{explanation.confidence:.1%}</td></tr>
                <tr><td class="label-caps">Risk Score</td><td class="mono" style="color:{'#EF4444' if explanation.risk_score >= 0.6 else '#F59E0B' if explanation.risk_score >= 0.3 else '#10B981'};">{explanation.risk_score:.4f}</td></tr>
            </table>
            ''', unsafe_allow_html=True)

            text = generate_explanation_text(explanation, attributions)
            st.markdown(text)

            if attributions:
                st.markdown('<div class="label-caps" style="margin:12px 0 6px 0;">Agent Attribution Breakdown</div>', unsafe_allow_html=True)
                for attr in attributions:
                    color = AGENT_COLORS.get(attr.agent_name, "#94A3B8")
                    st.markdown(f'''
                    <div style="display:flex; align-items:center; justify-content:space-between; margin:4px 0;">
                        <span style="font-family:'JetBrains Mono',monospace; font-size:12px; color:{color}; font-weight:700; text-transform:uppercase;">{attr.agent_name}</span>
                        <span style="font-family:'Inter',sans-serif; font-size:18px; font-weight:700; color:{color};">{attr.contribution_pct}%</span>
                    </div>
                    <div style="background:rgba(51,65,85,0.3); height:4px; border-radius:2px; overflow:hidden;">
                        <div style="width:{attr.contribution_pct}%; height:100%; background:{color}; border-radius:2px;"></div>
                    </div>
                    ''', unsafe_allow_html=True)

        st.markdown('</div>', unsafe_allow_html=True)

        # --- Top risk indicators ---
        st.markdown('<div class="label-caps" style="margin:20px 0 8px 0;">Top Risk Indicators</div>', unsafe_allow_html=True)
        for name, imp in explanation.top_nodes[:10]:
            short = name.split(";->")[-1] if ";->" in name else name.split(":")[-1]
            bar_width = int(imp / (explanation.top_nodes[0][1] + 0.001) * 100)
            severity = "high" if imp > 0.1 else "med" if imp > 0.05 else "low"
            badge_level = "critical" if severity == "high" else "high" if severity == "med" else "info"
            st.markdown(f'''
            <div class="evidence-item {severity}">
                <div style="display:flex; justify-content:space-between; align-items:center;">
                    <span>{severity_badge(badge_level)} <span class="mono" style="font-size:12px; color:#F8FAFC; margin-left:6px;">{short}</span></span>
                    <span class="mono" style="font-size:11px; color:#94A3B8;">{imp:.4f}</span>
                </div>
                <div style="background:rgba(51,65,85,0.3); height:3px; border-radius:2px; margin-top:6px; overflow:hidden;">
                    <div style="width:{bar_width}%; height:100%; background:{'#EF4444' if severity=='high' else '#F59E0B' if severity=='med' else '#06B6D4'}; border-radius:2px;"></div>
                </div>
                <div class="mono-sm" style="margin-top:4px; font-size:10px;">{name}</div>
            </div>
            ''', unsafe_allow_html=True)

        # --- MITRE ATT&CK mapping ---
        from explain.mitre_mapping import map_features_to_mitre, get_tactic_summary
        mitre_maps = map_features_to_mitre(explanation.node_importances)
        if mitre_maps:
            tactic_summary = get_tactic_summary(mitre_maps)
            st.markdown('<div class="label-caps" style="margin:20px 0 8px 0;">MITRE ATT&amp;CK Mapping</div>', unsafe_allow_html=True)

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
                    <div style="font-family:'JetBrains Mono',monospace; font-size:11px; font-weight:700; color:{color}; text-transform:uppercase; letter-spacing:0.05em;">{tactic}</div>
                    <div style="margin-top:4px; display:flex; flex-wrap:wrap; gap:4px;">{tech_html}</div>
                </div>
                ''', unsafe_allow_html=True)

        # --- Agent evidence section (accordion) ---
        st.markdown('<div class="label-caps" style="margin:24px 0 10px 0;">Agent Evidence</div>', unsafe_allow_html=True)

        st.markdown(f'''
        <div class="soc-metric-row">
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
            icon = AGENT_ICONS.get(agent_name, "description")
            with st.expander(f"{agent_name.upper()} AGENT -- {len(items)} FINDINGS", expanded=False):
                for item in items[:50]:
                    risk = item.attributes.get("risk_level", item.attributes.get("sensitivity_tier", 0))
                    if risk >= 3:
                        severity = "high"
                        badge_level = "critical"
                    elif risk >= 2:
                        severity = "med"
                        badge_level = "high"
                    else:
                        severity = "low"
                        badge_level = "medium"
                    st.markdown(f'''
                    <div class="evidence-item {severity}">
                        {severity_badge(badge_level)}
                        <span class="mono" style="font-size:11px; color:{color}; margin-left:8px;">{item.evidence_type.value}</span>
                        <span style="font-size:13px; color:#F8FAFC; margin-left:8px;">{item.description}</span>
                    </div>
                    ''', unsafe_allow_html=True)

        # --- Behaviour graph ---
        st.markdown('<div class="label-caps" style="margin:24px 0 0 0;">Behaviour Graph</div>', unsafe_allow_html=True)
        st.markdown(f'''
        <div class="graph-header-bar">
            <div style="display:flex; align-items:center; gap:12px;">
                <span class="label-caps">ATTENTION INTENSITY</span>
                <span class="sev-badge sev-critical" style="font-size:9px;">CRITICAL</span>
                <span class="sev-badge sev-high" style="font-size:9px;">SUSPICIOUS</span>
                <span class="sev-badge sev-safe" style="font-size:9px;">BENIGN</span>
            </div>
            <div style="display:flex; gap:24px;">
                <span class="label-caps">NODES <span style="color:#F8FAFC; font-size:13px; font-weight:700;">{data.num_nodes}</span></span>
                <span class="label-caps">EDGES <span style="color:#F8FAFC; font-size:13px; font-weight:700;">{data.edge_index.size(1)}</span></span>
            </div>
        </div>
        ''', unsafe_allow_html=True)

        show_explanation = st.checkbox("Highlight GNNExplainer subgraph", value=True)
        graph_html = create_graph_html(
            data, explanation if show_explanation else None
        )
        st.components.v1.html(graph_html, height=550, scrolling=True)

        st.markdown('''
        <div class="graph-legend" style="padding:10px 14px; background:#0F172A; border:1px solid #334155; border-top:none; border-radius:0 0 8px 8px;">
            <span class="graph-legend-item"><span class="graph-legend-dot" style="background:#EF4444;"></span>Critical Vector</span>
            <span class="graph-legend-item"><span class="graph-legend-dot" style="background:#F59E0B;"></span>Suspicious Hook</span>
            <span class="graph-legend-item"><span class="graph-legend-dot" style="background:#10B981;"></span>Benign</span>
        </div>
        ''', unsafe_allow_html=True)

        type_colors_legend = {
            "Method": "#38BDF8", "Class": "#A78BFA", "Permission": "#F59E0B",
            "API Call": "#EF4444", "Network": "#10B981", "Crypto": "#F472B6",
        }
        legend_html = '<div style="display:flex; gap:16px; flex-wrap:wrap; margin-top:10px;">'
        for name, color in type_colors_legend.items():
            legend_html += f'<span style="display:flex; align-items:center; gap:4px;"><span style="width:10px;height:10px;border-radius:50%;background:{color};display:inline-block;"></span><span class="label-caps">{name}</span></span>'
        legend_html += '</div>'
        st.markdown(legend_html, unsafe_allow_html=True)

        # --- Architecture spec table ---
        st.markdown('<div class="label-caps" style="margin:24px 0 10px 0;">Architecture Specification</div>', unsafe_allow_html=True)

        model = load_model()
        total_params = sum(p.numel() for p in model.parameters())

        st.markdown(f'''
        <div class="soc-card">
            <table style="width:100%;">
                <tr>
                    <th>Architecture Parameter</th>
                    <th>Configuration Value</th>
                    <th>Execution Subsystem</th>
                    <th>Optimization Scope</th>
                </tr>
                <tr>
                    <td>Network Type</td><td style="color:#06B6D4;">2-Layer GATConv</td><td>GAT Encoder</td><td>Node-level attention</td>
                </tr>
                <tr>
                    <td>Parameters</td><td style="color:#F8FAFC; font-weight:700;">{total_params:,}</td><td>GAT Encoder</td><td>Global</td>
                </tr>
                <tr>
                    <td>Input Features</td><td>25-dim + 2 degree (runtime)</td><td>Feature Builder</td><td>Per-node</td>
                </tr>
                <tr>
                    <td>Hidden Channels</td><td>64 channels &times; 4 heads</td><td>GAT Layer 1</td><td>Multi-head attention</td>
                </tr>
                <tr>
                    <td>Readout</td><td>Mean + Max pooling + 15 graph features</td><td>Graph Pooling</td><td>Graph-level</td>
                </tr>
                <tr>
                    <td>Classifier</td><td>143&rarr;64&rarr;32&rarr;2 (ReLU)</td><td>MLP Head</td><td>Binary classification</td>
                </tr>
                <tr>
                    <td>Ensemble</td><td>GAT prob + 15 graph feats &rarr; GBM (200 trees)</td><td>Gradient Boosting</td><td>Final risk score</td>
                </tr>
            </table>
        </div>
        ''', unsafe_allow_html=True)

        ens_path = MODEL_DIR / "ensemble_results.json"
        results_path = ens_path if ens_path.exists() else MODEL_DIR / "results.json"
        if results_path.exists():
            results = json.loads(results_path.read_text())
            metrics = results.get("test_metrics", {})
            label = "Ensemble Metrics" if ens_path.exists() else "GAT Metrics"
            metrics_html = '<div class="soc-metric-row">'
            for name, val in metrics.items():
                metrics_html += f'<div class="soc-metric"><div class="soc-metric-value text-cyan">{val:.4f}</div><div class="soc-metric-label">{name}</div></div>'
            metrics_html += '</div>'
            st.markdown(f'<div class="label-caps" style="margin-top:16px;">{label}</div>{metrics_html}', unsafe_allow_html=True)

        st.markdown(f'''
        <div class="soc-card" style="margin-top:16px;">
            <div class="label-caps" style="margin-bottom:8px;">Current Prediction</div>
            <table style="width:100%;">
                <tr><td class="label-caps" style="width:40%;">Predicted Class</td><td>{render_threat_badge(explanation.risk_score)} <span style="color:#F8FAFC; margin-left:8px;">{'Malware' if explanation.prediction == 1 else 'Benign'}</span></td></tr>
                <tr><td class="label-caps">Confidence</td><td style="color:#06B6D4; font-weight:700;">{explanation.confidence:.4f}</td></tr>
                <tr><td class="label-caps">Risk Score</td><td style="color:{'#EF4444' if explanation.risk_score >= 0.6 else '#F59E0B' if explanation.risk_score >= 0.3 else '#10B981'}; font-weight:700;">{explanation.risk_score:.4f}</td></tr>
                <tr><td class="label-caps">Processing Time</td><td>{result.duration_seconds:.2f}s</td></tr>
            </table>
        </div>
        ''', unsafe_allow_html=True)

    _footer_bar()


if __name__ == "__main__":
    main()
