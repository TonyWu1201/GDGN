from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable

import numpy as np
from scipy.stats import hypergeom, mannwhitneyu
from sklearn.metrics import average_precision_score, ndcg_score

from .statistics import benjamini_hochberg


def target_retrieval_metrics(importance: np.ndarray, target_mask: np.ndarray) -> dict[str, float | int]:
    scores = np.asarray(importance, dtype=np.float64)
    targets = np.asarray(target_mask, dtype=bool)
    if scores.shape != targets.shape:
        raise ValueError("importance and target_mask shapes differ")
    n_targets = int(targets.sum())
    if n_targets == 0:
        return {"n_targets": 0, "auprc": float("nan"), "ndcg": float("nan"), "mean_rank_percentile": float("nan")}
    order = np.argsort(-scores)
    ranks = np.empty_like(order)
    ranks[order] = np.arange(len(order))
    return {
        "n_targets": n_targets,
        "auprc": float(average_precision_score(targets.astype(int), scores)),
        "ndcg": float(ndcg_score(targets.astype(float)[None, :], scores[None, :])),
        "mean_rank_percentile": float(np.mean(ranks[targets] / max(1, len(scores) - 1))),
    }


def neighborhood_enrichment(
    ranked_gene_idx: Iterable[int],
    gene_sets: dict[str, set[int]],
    universe_size: int,
    top_k: int,
) -> list[dict]:
    top = set(list(ranked_gene_idx)[:top_k])
    rows = []
    for name, members in gene_sets.items():
        overlap = len(top & members)
        p_value = float(hypergeom.sf(overlap - 1, universe_size, len(members), len(top)))
        rows.append({"set": name, "overlap": overlap, "set_size": len(members), "p_value": p_value})
    adjusted = benjamini_hochberg([row["p_value"] for row in rows]) if rows else []
    for row, q_value in zip(rows, adjusted):
        row["q_value"] = float(q_value)
    return sorted(rows, key=lambda row: (row["q_value"], -row["overlap"]))


@dataclass
class DeletionCurve:
    fractions: list[float]
    top_drop: list[float]
    random_drop: list[float]
    matched_drop: list[float]

    @property
    def comprehensiveness(self) -> float:
        return float(np.trapezoid(self.top_drop, self.fractions))


def deletion_curve(
    importance: np.ndarray,
    base_prediction: float,
    predict_with_deleted: Callable[[np.ndarray], float],
    matched_order: np.ndarray,
    fractions: Iterable[float] = (0.01, 0.05, 0.1, 0.2),
    seed: int = 42,
) -> DeletionCurve:
    scores = np.asarray(importance)
    n = len(scores)
    top_order = np.argsort(-np.abs(scores))
    rng = np.random.default_rng(seed)
    random_order = rng.permutation(n)
    fs, top, random, matched = [], [], [], []
    for fraction in fractions:
        k = max(1, int(round(n * fraction)))
        fs.append(float(fraction))
        top.append(float(base_prediction - predict_with_deleted(top_order[:k])))
        random.append(float(base_prediction - predict_with_deleted(random_order[:k])))
        matched.append(float(base_prediction - predict_with_deleted(np.asarray(matched_order)[:k])))
    return DeletionCurve(fs, top, random, matched)


def sufficiency(
    importance: np.ndarray,
    base_prediction: float,
    predict_with_retained: Callable[[np.ndarray], float],
    fraction: float = 0.1,
) -> float:
    k = max(1, int(round(len(importance) * fraction)))
    keep = np.argsort(-np.abs(importance))[:k]
    return float(abs(base_prediction - predict_with_retained(keep)))


def attention_randomization_test(
    original_prediction: float,
    predict_with_attention: Callable[[np.ndarray], float],
    attention: np.ndarray,
    n_random: int = 100,
    seed: int = 42,
) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    original = float(predict_with_attention(attention))
    changes = []
    for _ in range(n_random):
        shuffled = attention[rng.permutation(len(attention))]
        changes.append(abs(float(predict_with_attention(shuffled)) - original_prediction))
    return {
        "original_reconstruction_error": abs(original - original_prediction),
        "mean_shuffle_change": float(np.mean(changes)),
        "p95_shuffle_change": float(np.percentile(changes, 95)),
    }


def jaccard_stability(rankings: list[Iterable[int]], top_k: int = 20) -> dict[str, float | int]:
    sets = [set(list(ranking)[:top_k]) for ranking in rankings]
    values = []
    for i in range(len(sets)):
        for j in range(i + 1, len(sets)):
            union = sets[i] | sets[j]
            values.append(len(sets[i] & sets[j]) / len(union) if union else 1.0)
    return {"n_pairs": len(values), "mean_jaccard": float(np.mean(values)) if values else float("nan")}


def target_vs_control_test(importance: np.ndarray, target_mask: np.ndarray, control_mask: np.ndarray) -> dict:
    target = np.abs(np.asarray(importance)[np.asarray(target_mask, dtype=bool)])
    control = np.abs(np.asarray(importance)[np.asarray(control_mask, dtype=bool)])
    if len(target) == 0 or len(control) == 0:
        return {"statistic": float("nan"), "p_value": float("nan")}
    result = mannwhitneyu(target, control, alternative="greater")
    return {"statistic": float(result.statistic), "p_value": float(result.pvalue)}


def integrated_gradients_delta(
    input_values: np.ndarray,
    baseline: np.ndarray,
    attributions: np.ndarray,
    prediction: float,
    baseline_prediction: float,
) -> float:
    del input_values, baseline
    return float(prediction - baseline_prediction - np.asarray(attributions).sum())
