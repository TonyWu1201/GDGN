import json

import pandas as pd

from program.strict_eval.aggregate import aggregate_experiments, collect_metrics


def _write_run(root, model, pcc, predictions):
    run = root / "eval-lpo" / model / "fold-00" / "seed-0042"
    run.mkdir(parents=True)
    config = {
        "experiment_id": "eval-lpo", "split_id": "eval-lpo", "model_id": model,
        "variant": "default", "fold": 0, "seed": 42,
    }
    metrics = {
        "test": {
            "global": {"pcc": pcc, "spearman": pcc, "rmse": 1 - pcc, "mae": 1 - pcc, "r2": pcc, "n": 4},
            "by_drug": {"macro_pcc": pcc, "macro_rmse": 1 - pcc},
            "by_cell": {"macro_pcc": pcc, "macro_rmse": 1 - pcc},
            "similarity_strata": {},
        }
    }
    (run / "config.json").write_text(json.dumps(config), encoding="utf-8")
    (run / "status.json").write_text(json.dumps({"state": "complete"}), encoding="utf-8")
    (run / "metrics.json").write_text(json.dumps(metrics), encoding="utf-8")
    predictions.assign(split="test").to_parquet(run / "predictions.parquet", index=False)


def test_aggregate_writes_structured_outputs(tmp_path):
    predictions = pd.DataFrame({
        "pair_id": range(4), "drug_idx": [0, 0, 1, 1], "cell_idx": [0, 1, 0, 1],
        "y_true": [0.0, 1.0, 2.0, 3.0], "y_pred": [0.1, 0.9, 2.1, 2.9],
    })
    _write_run(tmp_path, "two-way-additive", 0.8, predictions)
    _write_run(tmp_path, "model-m0", 0.9, predictions)
    assert len(collect_metrics(tmp_path)) == 2
    outputs = aggregate_experiments(tmp_path, tmp_path / "summary", hierarchical_bootstrap_runs=5)
    assert all(path.exists() for path in outputs)
    paired = pd.read_csv(outputs[3])
    assert set(paired["candidate"]) == {"model-m0"}
