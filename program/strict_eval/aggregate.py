from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from .constants import EXPERIMENTS_DIR
from .io import atomic_write_text
from .statistics import hierarchical_bootstrap, paired_run_differences

SIMILARITY_BANDS = ("low_lt_0_4", "medium_0_4_to_0_7", "high_ge_0_7")


def collect_metrics(root: str | Path = EXPERIMENTS_DIR) -> pd.DataFrame:
    rows = []
    for path in Path(root).glob("*/*/fold-*/seed-*/metrics.json"):
        status_path = path.with_name("status.json")
        config_path = path.with_name("config.json")
        if not status_path.exists() or not config_path.exists():
            continue
        status = json.loads(status_path.read_text(encoding="utf-8"))
        if status.get("state") != "complete":
            continue
        config = json.loads(config_path.read_text(encoding="utf-8"))
        metrics = json.loads(path.read_text(encoding="utf-8"))["test"]
        global_metrics = metrics["global"]
        row = {
            "experiment_id": config["experiment_id"],
            "split_id": config["split_id"],
            "model_id": config["model_id"],
            "variant": config.get("variant", "default"),
            "fold": int(config["fold"]),
            "seed": int(config["seed"]),
            **{name: global_metrics.get(name) for name in ("pcc", "spearman", "rmse", "mae", "r2", "n")},
            "drug_macro_pcc": metrics["by_drug"].get("macro_pcc"),
            "drug_macro_rmse": metrics["by_drug"].get("macro_rmse"),
            "cell_macro_pcc": metrics["by_cell"].get("macro_pcc"),
            "cell_macro_rmse": metrics["by_cell"].get("macro_rmse"),
            "run_dir": str(path.parent),
        }
        strata = metrics.get("similarity_strata", {})
        for band in SIMILARITY_BANDS:
            row[f"{band}_pcc"] = strata.get(band, {}).get("pcc")
            row[f"{band}_rmse"] = strata.get(band, {}).get("rmse")
            row[f"{band}_n"] = strata.get(band, {}).get("n", 0)
        rows.append(row)
    return pd.DataFrame(rows)


def summarize_runs(rows: pd.DataFrame) -> pd.DataFrame:
    if rows.empty:
        return pd.DataFrame()
    metrics = [
        "pcc", "spearman", "rmse", "mae", "r2", "drug_macro_pcc", "cell_macro_pcc",
        *[f"{band}_{metric}" for band in SIMILARITY_BANDS for metric in ("pcc", "rmse")],
    ]
    records = []
    group_cols = ["experiment_id", "split_id", "model_id", "variant"]
    for keys, group in rows.groupby(group_cols, dropna=False, sort=True):
        row = dict(zip(group_cols, keys))
        row["n_runs"] = len(group)
        row["n_folds"] = group["fold"].nunique()
        row["n_seeds"] = group["seed"].nunique()
        for metric in metrics:
            values = group[metric].to_numpy(dtype=float)
            finite = values[np.isfinite(values)]
            if finite.size == 0:
                row[f"{metric}_mean"] = float("nan")
                row[f"{metric}_std"] = float("nan")
                row[f"{metric}_ci_low"] = float("nan")
                row[f"{metric}_ci_high"] = float("nan")
                continue
            row[f"{metric}_mean"] = float(np.mean(finite))
            row[f"{metric}_std"] = float(np.std(finite, ddof=1)) if len(finite) > 1 else 0.0
            rng = np.random.default_rng(42)
            boot = np.asarray([
                rng.choice(finite, size=len(finite), replace=True).mean() for _ in range(5000)
            ])
            row[f"{metric}_ci_low"] = float(np.nanpercentile(boot, 2.5))
            row[f"{metric}_ci_high"] = float(np.nanpercentile(boot, 97.5))
        records.append(row)
    return pd.DataFrame(records)


def hierarchical_group_intervals(rows: pd.DataFrame, n_boot: int = 500) -> pd.DataFrame:
    records = []
    if rows.empty or n_boot <= 0:
        return pd.DataFrame(records)
    keys = ["experiment_id", "split_id", "model_id", "variant"]
    for values, group in rows.groupby(keys, dropna=False, sort=True):
        frames = []
        for run_dir in group["run_dir"]:
            path = Path(run_dir) / "predictions.parquet"
            if path.exists():
                frame = pd.read_parquet(path)
                frames.append(frame[frame["split"] == "test"])
        if not frames:
            continue
        predictions = pd.concat(frames, ignore_index=True)
        predictions = predictions.groupby("pair_id", as_index=False).agg({
            "drug_idx": "first", "cell_idx": "first", "y_true": "first", "y_pred": "mean",
        })
        record = dict(zip(keys, values))
        for cluster in ("drug_idx", "cell_idx"):
            if predictions[cluster].nunique() < 2:
                continue
            intervals = hierarchical_bootstrap(predictions, cluster, n_boot=n_boot)
            prefix = "drug" if cluster == "drug_idx" else "cell"
            for metric, interval in intervals.items():
                for name, value in interval.items():
                    record[f"{prefix}_{metric}_{name}"] = value
        records.append(record)
    return pd.DataFrame(records)


def paired_comparisons(rows: pd.DataFrame) -> pd.DataFrame:
    records = []
    if rows.empty:
        return pd.DataFrame(records)
    references = {
        "eval": ("model_id", "two-way-additive"),
        "abl-graph": ("variant", "graph-real"),
        "abl-pretrain": ("variant", "a00"),
        "abl-esm": ("variant", "esm-id"),
        "abl-modality": ("variant", "expression"),
        "model-pathway-residual": ("model_id", "model-m0"),
    }
    for (experiment_id, split_id), group in rows.groupby(["experiment_id", "split_id"], sort=True):
        family = "eval" if experiment_id.startswith("eval-") else experiment_id
        column, reference = references.get(family, ("model_id", "two-way-additive"))
        if reference not in set(group[column]):
            continue
        for candidate in sorted(set(group[column]) - {reference}):
            comparison = group[group[column].isin([candidate, reference])].copy()
            comparison["model_id"] = comparison[column]
            for metric in ("pcc", "rmse"):
                try:
                    result = paired_run_differences(
                        comparison, candidate, reference, metric,
                        key_columns=("split_id", "fold", "seed"), n_boot=5000,
                    )
                except ValueError:
                    continue
                favorable = (
                    result["mean_difference"] if metric == "pcc"
                    else -result["mean_difference"]
                )
                records.append({
                    "experiment_id": experiment_id, "split_id": split_id,
                    "comparison_axis": column, "candidate": candidate, "reference": reference,
                    "metric": metric, "higher_is_better": metric == "pcc", **result,
                    "favorable_improvement": favorable,
                })
    return pd.DataFrame(records)


def render_markdown(summary: pd.DataFrame) -> str:
    lines = ["# Strict benchmark report", ""]
    if summary.empty:
        return "\n".join(lines + ["No completed formal runs were found.", ""])
    lines.extend([
        "| Protocol | Model | Variant | Runs | PCC mean | PCC 95% CI | RMSE mean |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: |",
    ])
    for row in summary.itertuples(index=False):
        lines.append(
            f"| {row.split_id} | {row.model_id} | {row.variant} | {row.n_runs} | "
            f"{row.pcc_mean:.4f} | [{row.pcc_ci_low:.4f}, {row.pcc_ci_high:.4f}] | "
            f"{row.rmse_mean:.4f} |"
        )
    lines.extend(["", "Results are aggregated over fold/seed runs; no sample-pair bootstrap is used here.", ""])
    return "\n".join(lines)


def aggregate_experiments(
    root: str | Path, output_dir: str | Path, hierarchical_bootstrap_runs: int = 500
) -> tuple[Path, Path, Path, Path]:
    rows = collect_metrics(root)
    summary = summarize_runs(rows)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    csv_path = output / "benchmark-summary.csv"
    report_path = output / "benchmark-report.md"
    hierarchical_path = output / "hierarchical-bootstrap.csv"
    paired_path = output / "paired-comparisons.csv"
    summary.to_csv(csv_path, index=False)
    hierarchical_group_intervals(rows, hierarchical_bootstrap_runs).to_csv(hierarchical_path, index=False)
    paired_comparisons(rows).to_csv(paired_path, index=False)
    atomic_write_text(report_path, render_markdown(summary))
    return csv_path, report_path, hierarchical_path, paired_path
