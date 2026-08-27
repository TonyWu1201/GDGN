from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

from .constants import CELL_FEATURE_PATH, HETERO_GRAPH_PATH, MODEL_DIR
from .io import read_json
from .preprocessing import FoldPreprocessor, transform_drug_features
from .splits import load_pair_table

RAW_CELL_FEATURE_PATH = HETERO_GRAPH_PATH.parent / "cell_line_features_raw.pt"
RAW_DRUG_FEATURE_PATH = HETERO_GRAPH_PATH.parent / "drug_features_raw.pt"


class PairDataset(Dataset):
    def __init__(self, pairs: pd.DataFrame):
        self.pairs = pairs.reset_index(drop=True)

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        row = self.pairs.iloc[index]
        return {
            "pair_id": torch.tensor(int(row.pair_id), dtype=torch.long),
            "cell_idx": torch.tensor(int(row.cell_idx), dtype=torch.long),
            "drug_idx": torch.tensor(int(row.drug_idx), dtype=torch.long),
            "y": torch.tensor(float(row.ic50), dtype=torch.float32),
        }


def load_fold_pairs(split_path: str | Path) -> tuple[dict, dict[str, pd.DataFrame]]:
    payload = read_json(split_path)
    pairs = load_pair_table().set_index("pair_id", drop=False)
    frames = {
        name: pairs.loc[item["pair_ids"]].reset_index(drop=True)
        for name, item in payload["splits"].items()
    }
    for name in ("val", "test"):
        mappings = payload.get("similarity_by_split", {}).get(name, {})
        frames[name]["drug_max_similarity"] = frames[name]["drug_idx"].map(
            {int(key): value for key, value in mappings.get("drug_max_to_train", {}).items()}
        )
        frames[name]["cell_max_similarity"] = frames[name]["cell_idx"].map(
            {int(key): value for key, value in mappings.get("cell_max_to_train", {}).items()}
        )
        protocol = payload["protocol"]
        if protocol in {"eval-ldo-kt", "eval-ldo-so"}:
            frames[name]["protocol_similarity"] = frames[name]["drug_max_similarity"]
        elif protocol == "eval-lco":
            frames[name]["protocol_similarity"] = frames[name]["cell_max_similarity"]
        elif protocol == "eval-db":
            frames[name]["protocol_similarity"] = frames[name][
                ["drug_max_similarity", "cell_max_similarity"]
            ].max(axis=1)
        else:
            frames[name]["protocol_similarity"] = np.nan
    return payload, frames


def build_fold_loaders(
    frames: dict[str, pd.DataFrame],
    batch_size: int,
    seed: int,
    num_workers: int = 0,
    smoke: bool = False,
) -> dict[str, DataLoader]:
    generator = torch.Generator().manual_seed(seed)
    loaders = {}
    for name in ("train", "val", "test"):
        frame = frames[name]
        if smoke:
            limit = max(batch_size * 3, batch_size + 1)
            frame = frame.iloc[:limit]
        loaders[name] = DataLoader(
            PairDataset(frame),
            batch_size=batch_size,
            shuffle=(name == "train"),
            drop_last=(name == "train" and len(frame) >= batch_size),
            num_workers=num_workers,
            generator=generator,
        )
    return loaders


def _tensor_dict_to_numpy(obj: dict) -> dict[str, np.ndarray]:
    mapping = {
        "expression": "expression",
        "mutation": "mutation",
        "copynumber": "copynumber",
        "methylation": "methylation",
        "pathway_activity": "pathway_activity",
    }
    return {
        dest: obj[source].detach().cpu().numpy().astype(np.float32)
        for dest, source in mapping.items()
    }


def load_fold_features(
    split_payload: dict,
    scaler_dir: str | Path,
    allow_legacy_fallback: bool = False,
) -> tuple[dict[str, torch.Tensor], torch.Tensor, dict]:
    """Load features and fit all continuous statistics on the training fold."""
    source = RAW_CELL_FEATURE_PATH if RAW_CELL_FEATURE_PATH.exists() else CELL_FEATURE_PATH
    cell_obj = torch.load(source, map_location="cpu", weights_only=False)
    strict_status = cell_obj.get("preprocessing_status", {}).get("strict_unscaled", False)
    if not strict_status and not allow_legacy_fallback:
        raise RuntimeError(
            "strict unscaled cell features are missing; rerun compute_ssgsea.py, "
            "methylation_tss1kb.py and build_cell_line_features.py"
        )
    arrays = _tensor_dict_to_numpy(cell_obj)
    train_cells = split_payload["splits"]["train"]["cell_idx"]
    fold_preprocessor = FoldPreprocessor().fit(arrays, train_cells)
    transformed = fold_preprocessor.transform(arrays)
    fold_preprocessor.save(scaler_dir)

    if RAW_DRUG_FEATURE_PATH.exists():
        drug_obj = torch.load(RAW_DRUG_FEATURE_PATH, map_location="cpu", weights_only=False)
        fingerprints = drug_obj["fingerprint"].detach().cpu().numpy()
        physicochemical = drug_obj["physicochemical"].detach().cpu().numpy()
        drug_strict = True
    else:
        legacy = torch.load(HETERO_GRAPH_PATH.parent / "drug_features.pt", map_location="cpu", weights_only=False)
        legacy = legacy.detach().cpu().numpy()
        high, low = legacy[:, :1024].max(axis=0), legacy[:, :1024].min(axis=0)
        fingerprints = ((legacy[:, :1024] == high) & (high > low)).astype(np.float32)
        physicochemical = legacy[:, 1024:]
        drug_strict = False
        if not allow_legacy_fallback:
            raise RuntimeError("drug_features_raw.pt is missing; rerun program/model/build_graph.py")
    train_drugs = split_payload["splits"]["train"]["drug_idx"]
    drug_features, drug_scaler = transform_drug_features(fingerprints, physicochemical, train_drugs)
    np.savez_compressed(
        Path(scaler_dir) / "drug_scaler.npz",
        mean=drug_scaler.mean,
        scale=drug_scaler.scale,
        train_drug_idx=np.asarray(train_drugs, dtype=np.int64),
    )
    tensors = {name: torch.from_numpy(values) for name, values in transformed.items()}
    metadata = {
        "cell_feature_source": str(source),
        "strict_unscaled_cell_features": bool(strict_status),
        "strict_unscaled_drug_features": drug_strict,
    }
    return tensors, torch.from_numpy(drug_features.astype(np.float32)), metadata


def predictions_frame(
    pair_ids: Iterable[int],
    cell_idx: Iterable[int],
    drug_idx: Iterable[int],
    y_true: Iterable[float],
    y_pred: Iterable[float],
    split: str,
) -> pd.DataFrame:
    return pd.DataFrame({
        "pair_id": list(pair_ids),
        "cell_idx": list(cell_idx),
        "drug_idx": list(drug_idx),
        "y_true": list(y_true),
        "y_pred": list(y_pred),
        "split": split,
    })
