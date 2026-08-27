from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import torch
from sklearn.cluster import AgglomerativeClustering
from sklearn.decomposition import PCA
from sklearn.model_selection import KFold

from .constants import (
    CANCER_TYPE_PATH,
    CELL_FEATURE_PATH,
    CELL_INDEX_PATH,
    DEFAULT_N_FOLDS,
    DRUG_INDEX_PATH,
    HETERO_GRAPH_PATH,
    IC50_PATH,
    PROTOCOLS,
    PROJECT_ROOT,
    SPLITS_DIR,
)
from .io import git_commit, sha256_file, write_json


@dataclass(frozen=True)
class SplitOptions:
    n_folds: int = DEFAULT_N_FOLDS
    seed: int = 42
    similarity_control: bool = True
    # 60 keeps similarity neighborhoods together without allowing one broad
    # cluster to dominate a five-fold test set on the current 184/404 entities.
    n_drug_clusters: int = 60
    n_cell_clusters: int = 60

    def validate(self) -> None:
        if self.n_folds < 3:
            raise ValueError("n_folds must be at least 3 (train/validation/test)")
        if self.n_drug_clusters < self.n_folds or self.n_cell_clusters < self.n_folds:
            raise ValueError("similarity clusters must be at least n_folds")


def load_pair_table() -> pd.DataFrame:
    """Return the canonical observed response table with stable integer pair ids."""
    ic50 = pd.read_csv(IC50_PATH, index_col=0)
    cell_to_idx = json.loads(CELL_INDEX_PATH.read_text(encoding="utf-8"))
    drug_to_idx = json.loads(DRUG_INDEX_PATH.read_text(encoding="utf-8"))
    cancer_of = json.loads(CANCER_TYPE_PATH.read_text(encoding="utf-8"))
    rows: list[dict] = []
    for cell_name, values in ic50.iterrows():
        if cell_name not in cell_to_idx:
            continue
        for cid, response in values.items():
            if pd.isna(response) or str(cid) not in drug_to_idx:
                continue
            rows.append({
                "pair_id": len(rows),
                "cell_idx": int(cell_to_idx[cell_name]),
                "drug_idx": int(drug_to_idx[str(cid)]),
                "cell_name": str(cell_name),
                "drug_cid": str(cid),
                "cancer_type": str(cancer_of.get(cell_name, "other")),
                "ic50": float(response),
            })
    frame = pd.DataFrame(rows)
    if frame.empty:
        raise RuntimeError(f"no response pairs found in {IC50_PATH}")
    if frame["pair_id"].duplicated().any():
        raise AssertionError("pair_id must be unique")
    return frame


def _balanced_group_assignment(
    group_values: Iterable[object],
    n_folds: int,
    seed: int,
) -> dict[object, int]:
    """Assign groups greedily so observed-pair counts remain reasonably balanced."""
    counts = Counter(group_values)
    rng = np.random.default_rng(seed)
    tie_break = {key: float(rng.random()) for key in counts}
    ordered = sorted(counts, key=lambda key: (-counts[key], tie_break[key], str(key)))
    loads = [0] * n_folds
    assignment: dict[object, int] = {}
    for key in ordered:
        fold = min(range(n_folds), key=lambda idx: (loads[idx], idx))
        assignment[key] = fold
        loads[fold] += counts[key]
    return assignment


def _groups_to_folds(groups: pd.Series, options: SplitOptions) -> np.ndarray:
    assignment = _balanced_group_assignment(groups.tolist(), options.n_folds, options.seed)
    return groups.map(assignment).to_numpy(dtype=np.int64)


def drug_similarity_groups(n_clusters: int, seed: int = 42) -> dict[int, str]:
    """Cluster drugs by inferred Morgan bits from the stored 1024-bit feature block.

    The legacy graph artifact Z-scored each fingerprint column. A binary bit can
    still be recovered because the larger of the (at most) two column values is
    the original one state. This is used only to define similarity-controlled
    groups, never as a fold-fitted model input.
    """
    features = torch.load(
        HETERO_GRAPH_PATH.parent / "drug_features.pt", map_location="cpu", weights_only=False
    ).detach().cpu().numpy()[:, :1024]
    high = features.max(axis=0, keepdims=True)
    low = features.min(axis=0, keepdims=True)
    bits = ((features == high) & (high > low)).astype(np.float32)
    intersection = bits @ bits.T
    sizes = bits.sum(axis=1)
    union = sizes[:, None] + sizes[None, :] - intersection
    similarity = np.divide(intersection, union, out=np.zeros_like(intersection), where=union > 0)
    distance = np.clip(1.0 - similarity, 0.0, 1.0)
    model = AgglomerativeClustering(
        n_clusters=min(n_clusters, len(features)), metric="precomputed", linkage="average"
    )
    labels = model.fit_predict(distance)
    return {idx: f"morgan-{int(label):03d}" for idx, label in enumerate(labels)}


def drug_similarity_matrix() -> np.ndarray:
    features = torch.load(
        HETERO_GRAPH_PATH.parent / "drug_features.pt", map_location="cpu", weights_only=False
    ).detach().cpu().numpy()[:, :1024]
    high, low = features.max(axis=0, keepdims=True), features.min(axis=0, keepdims=True)
    bits = ((features == high) & (high > low)).astype(np.float32)
    intersection = bits @ bits.T
    sizes = bits.sum(axis=1)
    union = sizes[:, None] + sizes[None, :] - intersection
    return np.divide(intersection, union, out=np.zeros_like(intersection), where=union > 0)


def cell_similarity_groups(n_clusters: int, seed: int = 42) -> dict[int, str]:
    """Cluster cell lines from expression only; response labels are never used."""
    obj = torch.load(CELL_FEATURE_PATH, map_location="cpu", weights_only=False)
    expression = obj["expression"].detach().cpu().numpy().astype(np.float32)
    n_components = min(50, expression.shape[0] - 1, expression.shape[1])
    reduced = PCA(
        n_components=n_components, svd_solver="randomized", random_state=seed
    ).fit_transform(expression)
    model = AgglomerativeClustering(
        n_clusters=min(n_clusters, len(expression)), metric="cosine", linkage="average"
    )
    labels = model.fit_predict(reduced)
    return {idx: f"expression-{int(label):03d}" for idx, label in enumerate(labels)}


def cell_similarity_matrix() -> np.ndarray:
    obj = torch.load(CELL_FEATURE_PATH, map_location="cpu", weights_only=False)
    expression = obj["expression"].detach().cpu().numpy().astype(np.float32)
    reduced = PCA(
        n_components=min(50, expression.shape[0] - 1, expression.shape[1]),
        svd_solver="randomized", random_state=42,
    ).fit_transform(expression)
    norms = np.linalg.norm(reduced, axis=1, keepdims=True)
    normalized = np.divide(reduced, norms, out=np.zeros_like(reduced), where=norms > 0)
    return np.clip(normalized @ normalized.T, -1.0, 1.0)


def _max_similarity_summary(matrix: np.ndarray, train_idx: list[int], test_idx: list[int]) -> dict:
    if not train_idx or not test_idx:
        return {"n": 0, "mean": float("nan"), "median": float("nan"), "min": float("nan"), "max": float("nan")}
    values = matrix[np.asarray(test_idx, dtype=int)][:, np.asarray(train_idx, dtype=int)].max(axis=1)
    return {
        "n": int(len(values)), "mean": float(values.mean()), "median": float(np.median(values)),
        "min": float(values.min()), "max": float(values.max()),
        "low_lt_0_4": int((values < 0.4).sum()),
        "medium_0_4_to_0_7": int(((values >= 0.4) & (values < 0.7)).sum()),
        "high_ge_0_7": int((values >= 0.7).sum()),
    }


def _max_similarity_by_entity(
    matrix: np.ndarray, train_idx: list[int], query_idx: list[int]
) -> dict[str, float]:
    if not train_idx or not query_idx:
        return {}
    train = np.asarray(train_idx, dtype=int)
    return {
        str(index): float(matrix[int(index), train].max())
        for index in query_idx
    }


def _fold_payload(
    pairs: pd.DataFrame,
    train_mask: np.ndarray,
    val_mask: np.ndarray,
    test_mask: np.ndarray,
    protocol: str,
    fold: int,
    graph_policy: str,
) -> dict:
    split_masks = {"train": train_mask, "val": val_mask, "test": test_mask}
    split_rows: dict[str, dict] = {}
    for name, mask in split_masks.items():
        subset = pairs.loc[mask]
        split_rows[name] = {
            "pair_ids": subset["pair_id"].astype(int).tolist(),
            "cell_idx": sorted(subset["cell_idx"].astype(int).unique().tolist()),
            "drug_idx": sorted(subset["drug_idx"].astype(int).unique().tolist()),
            "cancer_types": sorted(subset["cancer_type"].astype(str).unique().tolist()),
            "n_pairs": int(len(subset)),
        }
    used = train_mask | val_mask | test_mask
    return {
        "protocol": protocol,
        "fold": fold,
        "graph_policy": graph_policy,
        "splits": split_rows,
        "excluded_pair_ids": pairs.loc[~used, "pair_id"].astype(int).tolist(),
        "n_excluded_pairs": int((~used).sum()),
    }


def audit_fold(payload: dict) -> dict[str, object]:
    splits = payload["splits"]
    pair_sets = {name: set(item["pair_ids"]) for name, item in splits.items()}
    pair_overlap = {
        "train_val": len(pair_sets["train"] & pair_sets["val"]),
        "train_test": len(pair_sets["train"] & pair_sets["test"]),
        "val_test": len(pair_sets["val"] & pair_sets["test"]),
    }
    protocol = payload["protocol"]
    entity_checks: dict[str, int] = {}
    if protocol in {"eval-lco", "eval-db"}:
        entity_checks["train_test_cell_overlap"] = len(
            set(splits["train"]["cell_idx"]) & set(splits["test"]["cell_idx"])
        )
        entity_checks["train_val_cell_overlap"] = len(
            set(splits["train"]["cell_idx"]) & set(splits["val"]["cell_idx"])
        )
        entity_checks["val_test_cell_overlap"] = len(
            set(splits["val"]["cell_idx"]) & set(splits["test"]["cell_idx"])
        )
    if protocol in {"eval-ldo-kt", "eval-ldo-so", "eval-db"}:
        entity_checks["train_test_drug_overlap"] = len(
            set(splits["train"]["drug_idx"]) & set(splits["test"]["drug_idx"])
        )
        entity_checks["train_val_drug_overlap"] = len(
            set(splits["train"]["drug_idx"]) & set(splits["val"]["drug_idx"])
        )
        entity_checks["val_test_drug_overlap"] = len(
            set(splits["val"]["drug_idx"]) & set(splits["test"]["drug_idx"])
        )
    if protocol == "eval-lto":
        entity_checks["train_test_tissue_overlap"] = len(
            set(splits["train"]["cancer_types"]) & set(splits["test"]["cancer_types"])
        )
        entity_checks["train_val_tissue_overlap"] = len(
            set(splits["train"]["cancer_types"]) & set(splits["val"]["cancer_types"])
        )
        entity_checks["val_test_tissue_overlap"] = len(
            set(splits["val"]["cancer_types"]) & set(splits["test"]["cancer_types"])
        )
    all_values = list(pair_overlap.values()) + list(entity_checks.values())
    nonempty = all(splits[name]["n_pairs"] > 0 for name in ("train", "val", "test"))
    return {
        "passed": bool(nonempty and all(value == 0 for value in all_values)),
        "nonempty": nonempty,
        "pair_overlap": pair_overlap,
        "entity_overlap": entity_checks,
    }


def make_protocol_folds(
    pairs: pd.DataFrame,
    protocol: str,
    options: SplitOptions,
    drug_groups: dict[int, str] | None = None,
    cell_groups: dict[int, str] | None = None,
) -> list[dict]:
    options.validate()
    if protocol not in PROTOCOLS:
        raise ValueError(f"unknown protocol: {protocol}")
    n = len(pairs)
    pair_fold = np.full(n, -1, dtype=np.int64)
    drug_fold: np.ndarray | None = None
    cell_fold: np.ndarray | None = None

    if protocol == "eval-lpo":
        kfold = KFold(options.n_folds, shuffle=True, random_state=options.seed)
        for fold, (_, test_pos) in enumerate(kfold.split(np.arange(n))):
            pair_fold[test_pos] = fold
    elif protocol in {"eval-ldo-kt", "eval-ldo-so"}:
        groups = pairs["drug_idx"].map(drug_groups) if drug_groups else pairs["drug_idx"]
        pair_fold = _groups_to_folds(groups, options)
    elif protocol == "eval-lco":
        groups = pairs["cell_idx"].map(cell_groups) if cell_groups else pairs["cell_idx"]
        pair_fold = _groups_to_folds(groups, options)
    elif protocol == "eval-lto":
        pair_fold = _groups_to_folds(pairs["cancer_type"], options)
    else:
        drug_series = pairs["drug_idx"].map(drug_groups) if drug_groups else pairs["drug_idx"]
        cell_series = pairs["cell_idx"].map(cell_groups) if cell_groups else pairs["cell_idx"]
        drug_fold = _groups_to_folds(drug_series, options)
        cell_fold = _groups_to_folds(cell_series, options)

    graph_policy = {
        "eval-ldo-kt": "known-targets",
        "eval-ldo-so": "structure-only",
        "eval-db": "structure-only",
    }.get(protocol, "training-entities-only")
    folds = []
    for fold in range(options.n_folds):
        val_fold = (fold + 1) % options.n_folds
        if protocol == "eval-db":
            assert drug_fold is not None and cell_fold is not None
            train_mask = ~np.isin(drug_fold, [fold, val_fold]) & ~np.isin(cell_fold, [fold, val_fold])
            val_mask = (drug_fold == val_fold) & (cell_fold == val_fold)
            test_mask = (drug_fold == fold) & (cell_fold == fold)
        else:
            test_mask = pair_fold == fold
            val_mask = pair_fold == val_fold
            train_mask = ~(test_mask | val_mask)
        payload = _fold_payload(
            pairs, train_mask, val_mask, test_mask, protocol, fold, graph_policy
        )
        payload["audit"] = audit_fold(payload)
        if not payload["audit"]["passed"]:
            raise AssertionError(f"split audit failed for {protocol} fold {fold}: {payload['audit']}")
        folds.append(payload)
    return folds


def _source_manifest(options: SplitOptions, pairs: pd.DataFrame) -> dict:
    paths = [IC50_PATH, CELL_INDEX_PATH, DRUG_INDEX_PATH, CANCER_TYPE_PATH, CELL_FEATURE_PATH]
    return {
        "created_on": date.today().isoformat(),
        "git_commit": git_commit(PROJECT_ROOT),
        "options": options.__dict__,
        "n_pairs": int(len(pairs)),
        "n_cells": int(pairs["cell_idx"].nunique()),
        "n_drugs": int(pairs["drug_idx"].nunique()),
        "source_files": {
            str(path.relative_to(PROJECT_ROOT)).replace("\\", "/"): sha256_file(path)
            for path in paths if path.exists()
        },
    }


def build_all_splits(
    output_root: str | Path = SPLITS_DIR,
    options: SplitOptions | None = None,
    protocols: Iterable[str] = PROTOCOLS,
) -> dict[str, Path]:
    options = options or SplitOptions()
    options.validate()
    pairs = load_pair_table()
    output_root = Path(output_root)
    drug_groups = drug_similarity_groups(options.n_drug_clusters, options.seed) if options.similarity_control else None
    cell_groups = cell_similarity_groups(options.n_cell_clusters, options.seed) if options.similarity_control else None
    drug_similarity = drug_similarity_matrix()
    cell_similarity = cell_similarity_matrix()
    written: dict[str, Path] = {}
    for protocol in protocols:
        directory = output_root / protocol
        directory.mkdir(parents=True, exist_ok=True)
        folds = make_protocol_folds(pairs, protocol, options, drug_groups, cell_groups)
        for payload in folds:
            train = payload["splits"]["train"]
            test = payload["splits"]["test"]
            payload["similarity_audit"] = {
                "drug_test_to_train_max": _max_similarity_summary(
                    drug_similarity, train["drug_idx"], test["drug_idx"]
                ),
                "cell_test_to_train_max": _max_similarity_summary(
                    cell_similarity, train["cell_idx"], test["cell_idx"]
                ),
            }
            payload["similarity_by_split"] = {}
            for split_name in ("val", "test"):
                query = payload["splits"][split_name]
                payload["similarity_by_split"][split_name] = {
                    "drug_max_to_train": _max_similarity_by_entity(
                        drug_similarity, train["drug_idx"], query["drug_idx"]
                    ),
                    "cell_max_to_train": _max_similarity_by_entity(
                        cell_similarity, train["cell_idx"], query["cell_idx"]
                    ),
                }
            write_json(directory / f"fold-{payload['fold']:02d}.json", payload)
        manifest = _source_manifest(options, pairs) | {
            "split_id": protocol,
            "protocol": protocol,
            "folds": [
                {
                    "fold": payload["fold"],
                    "counts": {name: item["n_pairs"] for name, item in payload["splits"].items()},
                    "n_excluded_pairs": payload["n_excluded_pairs"],
                    "audit": payload["audit"],
                }
                for payload in folds
            ],
            "drug_group_method": "morgan-tanimoto-agglomerative" if drug_groups else "entity-id",
            "cell_group_method": "expression-pca-cosine-agglomerative" if cell_groups else "entity-id",
        }
        write_json(directory / "manifest.json", manifest)
        written[protocol] = directory
    return written
