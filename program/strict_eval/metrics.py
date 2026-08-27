from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def _correlation(x: np.ndarray, y: np.ndarray, kind: str) -> float:
    if x.size < 2 or np.std(x) == 0 or np.std(y) == 0:
        return float("nan")
    fn = pearsonr if kind == "pcc" else spearmanr
    return float(fn(x, y).statistic)


def regression_metrics(y_true: Iterable[float], y_pred: Iterable[float]) -> dict[str, float | int]:
    true = np.asarray(list(y_true), dtype=np.float64)
    pred = np.asarray(list(y_pred), dtype=np.float64)
    valid = np.isfinite(true) & np.isfinite(pred)
    true, pred = true[valid], pred[valid]
    if true.size == 0:
        return {k: float("nan") for k in ("pcc", "spearman", "rmse", "mae", "r2")} | {"n": 0}
    return {
        "pcc": _correlation(pred, true, "pcc"),
        "spearman": _correlation(pred, true, "spearman"),
        "rmse": float(np.sqrt(mean_squared_error(true, pred))),
        "mae": float(mean_absolute_error(true, pred)),
        "r2": float(r2_score(true, pred)) if true.size > 1 else float("nan"),
        "n": int(true.size),
    }


def macro_metrics(
    predictions: pd.DataFrame,
    group_column: str,
    min_samples: int = 5,
) -> dict[str, float | int]:
    required = {"y_true", "y_pred", group_column}
    if missing := required - set(predictions.columns):
        raise KeyError(f"missing prediction columns: {sorted(missing)}")
    rows = []
    excluded = 0
    for _, group in predictions.groupby(group_column, sort=True):
        if len(group) < min_samples:
            excluded += 1
            continue
        rows.append(regression_metrics(group["y_true"], group["y_pred"]))
    keys = ("pcc", "spearman", "rmse", "mae", "r2")
    result = {}
    for key in keys:
        values = np.asarray([row[key] for row in rows], dtype=np.float64)
        finite = values[np.isfinite(values)]
        result[f"macro_{key}"] = float(finite.mean()) if finite.size else float("nan")
    result.update({"n_groups": len(rows), "n_groups_excluded": excluded, "min_samples": min_samples})
    return result


def full_metric_bundle(predictions: pd.DataFrame, min_samples: int = 5) -> dict:
    result = {"global": regression_metrics(predictions["y_true"], predictions["y_pred"])}
    result["by_drug"] = macro_metrics(predictions, "drug_idx", min_samples)
    result["by_cell"] = macro_metrics(predictions, "cell_idx", min_samples)
    return result
