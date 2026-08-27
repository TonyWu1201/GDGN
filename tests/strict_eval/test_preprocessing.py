import numpy as np

from program.strict_eval.preprocessing import ArrayScaler, FoldPreprocessor, transform_drug_features


def test_scaler_uses_training_rows_only():
    features = {
        "expression": np.array([[0.0], [2.0], [1000.0]], dtype=np.float32),
        "copynumber": np.array([[0.0], [2.0], [1000.0]], dtype=np.float32),
        "methylation": np.array([[0.0], [2.0], [1000.0]], dtype=np.float32),
        "pathway_activity": np.array([[0.0], [2.0], [1000.0]], dtype=np.float32),
        "mutation": np.array([[0.0], [1.0], [1.0]], dtype=np.float32),
    }
    transformed = FoldPreprocessor().fit(features, [0, 1]).transform(features)
    assert np.allclose(transformed["expression"][:2].mean(), 0.0)
    assert transformed["expression"][2, 0] == 999.0
    assert np.array_equal(transformed["mutation"], features["mutation"])


def test_drug_fingerprint_remains_binary():
    fp = np.array([[0, 1], [1, 0], [1, 1]], dtype=np.float32)
    phys = np.array([[0.0], [2.0], [100.0]], dtype=np.float32)
    transformed, _ = transform_drug_features(fp, phys, [0, 1])
    assert np.array_equal(transformed[:, :2], fp)
    assert transformed[2, 2] == 99.0
