import numpy as np

from program.preprocess.compute_ssgsea import ssgsea
from program.strict_eval.pathway_recompute import ssgsea_sample


def test_single_sample_ssgsea_matches_preprocessing_implementation():
    genes = ["A", "B", "C", "D", "E", "F"]
    sets = {"first": ["A", "C"], "second": ["B", "D", "F"]}
    expression = np.asarray([3.0, 1.0, 5.0, 0.0, 2.0, 4.0])
    expected = ssgsea(expression[:, None], genes, sets)[0]
    actual = ssgsea_sample(expression, genes, sets, list(sets))
    assert np.allclose(actual, expected)
