import numpy as np
import pandas as pd

from program.strict_eval.metrics import full_metric_bundle, regression_metrics
from program.strict_eval.statistics import benjamini_hochberg, hierarchical_bootstrap, paired_run_differences


def prediction_frame():
    return pd.DataFrame({
        "y_true": [0, 1, 2, 3, 4, 5, 6, 7],
        "y_pred": [0, 1, 2, 3, 4, 5, 6, 7],
        "drug_idx": [0, 0, 0, 0, 1, 1, 1, 1],
        "cell_idx": [0, 1, 2, 3, 0, 1, 2, 3],
    })


def test_metric_bundle_perfect_prediction():
    result = full_metric_bundle(prediction_frame(), min_samples=2)
    assert np.isclose(result["global"]["pcc"], 1.0)
    assert result["global"]["rmse"] == 0.0
    assert result["by_drug"]["n_groups"] == 2


def test_cluster_bootstrap_and_bh():
    bootstrap = hierarchical_bootstrap(prediction_frame(), "drug_idx", n_boot=20)
    assert bootstrap["rmse"]["mean"] == 0.0
    adjusted = benjamini_hochberg([0.01, 0.04, 0.03])
    assert np.all((adjusted >= 0) & (adjusted <= 1))
    assert np.allclose(adjusted, [0.03, 0.04, 0.04])


def test_paired_run_difference():
    rows = pd.DataFrame([
        {"experiment_id": "e", "fold": fold, "seed": 42, "model_id": model, "pcc": value}
        for fold, values in enumerate([(0.8, 0.7), (0.9, 0.8)])
        for model, value in zip(("a", "b"), values)
    ])
    result = paired_run_differences(rows, "a", "b", "pcc", n_boot=100)
    assert np.isclose(result["mean_difference"], 0.1)
