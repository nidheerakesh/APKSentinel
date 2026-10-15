from pathlib import Path
from dataclasses import dataclass, field
import os
from dotenv import load_dotenv

load_dotenv()

PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
LABELS_DIR = DATA_DIR / "labels"
EVIDENCE_DIR = DATA_DIR / "evidence"
GRAPHS_DIR = DATA_DIR / "graphs"
MODEL_DIR = DATA_DIR / "models"

ANDROZOO_API_KEY = os.getenv("ANDROZOO_API_KEY", "")

NODE_FEATURE_DIM = 25
NUM_NODE_TYPES = 6
NUM_EDGE_TYPES = 6

SENSITIVE_APIS_FILE = PROJECT_ROOT / "config" / "sensitive_apis.json"

RISK_THRESHOLDS = {"benign": 0.3, "suspicious": 0.6}


@dataclass
class TrainConfig:
    hidden_channels: int = 64
    num_heads: int = 4
    dropout: float = 0.3
    lr: float = 5e-4
    weight_decay: float = 1e-3
    epochs: int = 300
    patience: int = 30
    batch_size: int = 32
    num_classes: int = 2
    train_ratio: float = 0.7
    val_ratio: float = 0.15


@dataclass
class ExplainConfig:
    top_k_nodes: int = 15
    explainer_epochs: int = 20
    explainer_lr: float = 0.01
