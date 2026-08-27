from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.linear_model import ElasticNet, Ridge


class StatisticalBaseline:
    def fit(self, train: pd.DataFrame) -> "StatisticalBaseline":
        raise NotImplementedError

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        raise NotImplementedError


class GlobalMeanBaseline(StatisticalBaseline):
    def fit(self, train: pd.DataFrame) -> "GlobalMeanBaseline":
        self.mean_ = float(train["ic50"].mean())
        return self

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        return np.full(len(frame), self.mean_, dtype=np.float32)


class EntityMeanBaseline(StatisticalBaseline):
    def __init__(self, entity: str):
        if entity not in {"drug_idx", "cell_idx"}:
            raise ValueError("entity must be drug_idx or cell_idx")
        self.entity = entity

    def fit(self, train: pd.DataFrame) -> "EntityMeanBaseline":
        self.global_mean_ = float(train["ic50"].mean())
        self.means_ = train.groupby(self.entity)["ic50"].mean().to_dict()
        return self

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        return np.asarray(
            [self.means_.get(value, self.global_mean_) for value in frame[self.entity]],
            dtype=np.float32,
        )


class TwoWayAdditiveBaseline(StatisticalBaseline):
    """y = global + drug effect + cell effect with cold-start zero effects."""

    def fit(self, train: pd.DataFrame) -> "TwoWayAdditiveBaseline":
        self.global_mean_ = float(train["ic50"].mean())
        self.drug_effect_ = (train.groupby("drug_idx")["ic50"].mean() - self.global_mean_).to_dict()
        self.cell_effect_ = (train.groupby("cell_idx")["ic50"].mean() - self.global_mean_).to_dict()
        return self

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        return np.asarray([
            self.global_mean_ + self.drug_effect_.get(drug, 0.0) + self.cell_effect_.get(cell, 0.0)
            for cell, drug in zip(frame["cell_idx"], frame["drug_idx"])
        ], dtype=np.float32)


class LinearFeatureBaseline(StatisticalBaseline):
    def __init__(self, kind: str = "ridge", alpha: float = 1.0, l1_ratio: float = 0.5):
        if kind == "ridge":
            self.model = Ridge(alpha=alpha)
        elif kind == "elastic-net":
            self.model = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, max_iter=5000)
        else:
            raise ValueError(f"unknown linear baseline: {kind}")

    def fit_features(self, x: np.ndarray, y: np.ndarray) -> "LinearFeatureBaseline":
        self.model.fit(x, y)
        return self

    def predict_features(self, x: np.ndarray) -> np.ndarray:
        return self.model.predict(x).astype(np.float32)


def build_statistical_baseline(model_id: str) -> StatisticalBaseline:
    mapping = {
        "global-mean": GlobalMeanBaseline,
        "drug-mean": lambda: EntityMeanBaseline("drug_idx"),
        "cell-mean": lambda: EntityMeanBaseline("cell_idx"),
        "two-way-additive": TwoWayAdditiveBaseline,
    }
    if model_id not in mapping:
        raise ValueError(f"not a statistical baseline: {model_id}")
    return mapping[model_id]()
