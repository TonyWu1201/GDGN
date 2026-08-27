import numpy as np

from program.strict_eval.faithfulness import (
    deletion_curve, integrated_gradients_delta, jaccard_stability, target_retrieval_metrics,
)


def test_target_metrics_prioritize_true_target():
    result = target_retrieval_metrics(np.array([10.0, 1.0, 0.0]), np.array([True, False, False]))
    assert result["auprc"] == 1.0
    assert result["mean_rank_percentile"] == 0.0


def test_deletion_and_stability():
    curve = deletion_curve(
        np.array([4.0, 3.0, 2.0, 1.0]), 10.0,
        lambda deleted: 10.0 - len(deleted), np.array([1, 0, 2, 3]), fractions=[0.25, 0.5],
    )
    assert curve.top_drop == [1.0, 2.0]
    stability = jaccard_stability([[1, 2, 3], [1, 2, 4]], top_k=2)
    assert stability["mean_jaccard"] == 1.0


def test_ig_delta():
    assert integrated_gradients_delta(np.zeros(2), np.zeros(2), np.array([1.0, 2.0]), 5.0, 2.0) == 0.0
