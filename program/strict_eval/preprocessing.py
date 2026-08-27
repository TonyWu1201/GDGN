from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np

CONTINUOUS_CELL_MODALITIES = ("expression", "copynumber", "methylation", "pathway_activity")
BINARY_CELL_MODALITIES = ("mutation",)


@dataclass
class ArrayScaler:
    mean: np.ndarray
    scale: np.ndarray

    @classmethod
    def fit(cls, values: np.ndarray) -> "ArrayScaler":
        values = np.asarray(values, dtype=np.float64)
        mean = np.nanmean(values, axis=0)
        scale = np.nanstd(values, axis=0)
        mean = np.where(np.isfinite(mean), mean, 0.0)
        scale = np.where(np.isfinite(scale) & (scale > 0), scale, 1.0)
        return cls(mean=mean.astype(np.float32), scale=scale.astype(np.float32))

    def transform(self, values: np.ndarray) -> np.ndarray:
        out = (np.asarray(values, dtype=np.float32) - self.mean) / self.scale
        return np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)


class FoldPreprocessor:
    """Fit continuous feature statistics on training entities only."""

    def __init__(self, modalities: Iterable[str] = CONTINUOUS_CELL_MODALITIES):
        self.modalities = tuple(modalities)
        self.scalers: dict[str, ArrayScaler] = {}
        self.train_cell_idx: tuple[int, ...] = ()

    def fit(self, features: dict[str, np.ndarray], train_cell_idx: Iterable[int]) -> "FoldPreprocessor":
        idx = np.asarray(sorted(set(int(i) for i in train_cell_idx)), dtype=np.int64)
        if idx.size == 0:
            raise ValueError("train_cell_idx cannot be empty")
        self.train_cell_idx = tuple(idx.tolist())
        for name in self.modalities:
            if name not in features:
                raise KeyError(f"missing modality: {name}")
            self.scalers[name] = ArrayScaler.fit(np.asarray(features[name])[idx])
        return self

    def transform(self, features: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        if not self.scalers:
            raise RuntimeError("fit must be called before transform")
        out: dict[str, np.ndarray] = {}
        for name, values in features.items():
            arr = np.asarray(values)
            out[name] = self.scalers[name].transform(arr) if name in self.scalers else arr.astype(np.float32)
        return out

    def save(self, directory: str | Path) -> None:
        target = Path(directory)
        target.mkdir(parents=True, exist_ok=True)
        arrays: dict[str, np.ndarray] = {"train_cell_idx": np.asarray(self.train_cell_idx, dtype=np.int64)}
        for name, scaler in self.scalers.items():
            arrays[f"{name}__mean"] = scaler.mean
            arrays[f"{name}__scale"] = scaler.scale
        np.savez_compressed(target / "scalers.npz", **arrays)

    @classmethod
    def load(cls, path: str | Path) -> "FoldPreprocessor":
        payload = np.load(path)
        names = sorted({key.rsplit("__", 1)[0] for key in payload.files if key.endswith("__mean")})
        obj = cls(names)
        obj.train_cell_idx = tuple(payload["train_cell_idx"].astype(int).tolist())
        obj.scalers = {
            name: ArrayScaler(payload[f"{name}__mean"], payload[f"{name}__scale"])
            for name in names
        }
        return obj


def transform_drug_features(
    fingerprints: np.ndarray,
    physicochemical: np.ndarray,
    train_drug_idx: Iterable[int],
) -> tuple[np.ndarray, ArrayScaler]:
    """Keep Morgan bits binary; standardize only physicochemical descriptors."""
    idx = np.asarray(sorted(set(int(i) for i in train_drug_idx)), dtype=np.int64)
    if idx.size == 0:
        raise ValueError("train_drug_idx cannot be empty")
    scaler = ArrayScaler.fit(np.asarray(physicochemical)[idx])
    scaled = scaler.transform(physicochemical)
    return np.concatenate([np.asarray(fingerprints, dtype=np.float32), scaled], axis=1), scaler
