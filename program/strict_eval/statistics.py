from __future__ import annotations

from typing import Callable

import numpy as np
import pandas as pd

from .metrics import regression_metrics


def hierarchical_bootstrap(
    predictions: pd.DataFrame,
    cluster_column: str,
    n_boot: int = 2000,
    seed: int = 42,
) -> dict[str, dict[str, float]]:
    """Cluster bootstrap that resamples whole drugs or cell lines."""
    groups = {key: frame for key, frame in predictions.groupby(cluster_column, sort=True)}
    keys = np.asarray(list(groups), dtype=object)
    if keys.size < 2:
        raise ValueError("hierarchical bootstrap needs at least two clusters")
    rng = np.random.default_rng(seed)
    values = {"pcc": [], "rmse": [], "mae": [], "r2": []}
    for _ in range(n_boot):
        sampled = rng.choice(keys, size=len(keys), replace=True)
        frame = pd.concat([groups[key] for key in sampled], ignore_index=True)
        metrics = regression_metrics(frame["y_true"], frame["y_pred"])
        for name in values:
            values[name].append(float(metrics[name]))
    return {
        name: {
            "mean": float(np.nanmean(samples)),
            "ci_low": float(np.nanpercentile(samples, 2.5)),
            "ci_high": float(np.nanpercentile(samples, 97.5)),
        }
        for name, samples in values.items()
    }


def paired_run_differences(
    summary: pd.DataFrame,
    model_a: str,
    model_b: str,
    metric: str,
    key_columns: tuple[str, ...] = ("experiment_id", "fold", "seed"),
    n_boot: int = 10000,
    seed: int = 42,
) -> dict[str, float | int]:
    cols = list(key_columns) + ["model_id", metric]
    wide = summary[cols].pivot_table(index=list(key_columns), columns="model_id", values=metric)
    if model_a not in wide or model_b not in wide:
        raise ValueError(f"both models are required: {model_a}, {model_b}")
    diff = (wide[model_a] - wide[model_b]).dropna().to_numpy(dtype=np.float64)
    if diff.size == 0:
        raise ValueError("no paired runs")
    rng = np.random.default_rng(seed)
    boot = np.asarray([rng.choice(diff, size=len(diff), replace=True).mean() for _ in range(n_boot)])
    return {
        "n_pairs": int(diff.size),
        "mean_difference": float(diff.mean()),
        "std_difference": float(diff.std(ddof=1)) if diff.size > 1 else 0.0,
        "ci_low": float(np.percentile(boot, 2.5)),
        "ci_high": float(np.percentile(boot, 97.5)),
    }


def benjamini_hochberg(p_values: list[float] | np.ndarray) -> np.ndarray:
    values = np.asarray(p_values, dtype=np.float64)
    if values.ndim != 1:
        raise ValueError("p_values must be one-dimensional")
    order = np.argsort(values)
    ranked = values[order]
    adjusted = ranked * len(values) / np.arange(1, len(values) + 1)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    out = np.empty_like(adjusted)
    out[order] = np.clip(adjusted, 0.0, 1.0)
    return out
