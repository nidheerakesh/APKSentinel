from __future__ import annotations

import os
from pathlib import Path

import torch
from torch_geometric.data import InMemoryDataset, Data


class MalwareDataset(InMemoryDataset):
    def __init__(self, root: str, split: str = "train", train_ratio: float = 0.7,
                 val_ratio: float = 0.15, transform=None, pre_transform=None):
        self.split = split
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        super().__init__(root, transform, pre_transform)
        self.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return []

    @property
    def processed_file_names(self):
        return [f"{self.split}_data.pt"]

    def process(self):
        graphs_dir = Path(self.root) / "graphs"
        if not graphs_dir.exists():
            raise FileNotFoundError(f"No graphs directory at {graphs_dir}")

        all_data = []
        for pt_file in sorted(graphs_dir.glob("*.pt")):
            try:
                data = torch.load(pt_file, weights_only=False)
                if data.y is not None:
                    all_data.append(data)
            except Exception:
                continue

        if not all_data:
            raise ValueError("No valid graph data found")

        n = len(all_data)
        train_end = int(n * self.train_ratio)
        val_end = int(n * (self.train_ratio + self.val_ratio))

        if self.split == "train":
            split_data = all_data[:train_end]
        elif self.split == "val":
            split_data = all_data[train_end:val_end]
        else:
            split_data = all_data[val_end:]

        self.save(split_data, self.processed_paths[0])


def load_all_graphs(graphs_dir: str | Path) -> list[Data]:
    graphs_dir = Path(graphs_dir)
    all_data = []
    for pt_file in sorted(graphs_dir.glob("*.pt")):
        try:
            data = torch.load(pt_file, weights_only=False)
            if data.y is not None:
                all_data.append(data)
        except Exception:
            continue
    return all_data


def temporal_split(data_list: list[Data], train_ratio=0.7, val_ratio=0.15):
    import random
    random.seed(42)

    class0 = [d for d in data_list if d.y.item() == 0]
    class1 = [d for d in data_list if d.y.item() == 1]
    random.shuffle(class0)
    random.shuffle(class1)

    def split_class(items):
        n = len(items)
        t = int(n * train_ratio)
        v = int(n * (train_ratio + val_ratio))
        return items[:t], items[t:v], items[v:]

    tr0, va0, te0 = split_class(class0)
    tr1, va1, te1 = split_class(class1)

    train = tr0 + tr1
    val = va0 + va1
    test = te0 + te1
    random.shuffle(train)
    random.shuffle(val)
    random.shuffle(test)
    return train, val, test
