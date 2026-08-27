import pytest
import torch

from program.model.gdgn_model import initialize_gene_features


@pytest.mark.parametrize(
    ("mode", "trainable"),
    [("esm-mean", False), ("esm-parti", False), ("esm-none", False), ("esm-random", False), ("esm-id", True)],
)
def test_gene_feature_ablation_modes(mode, trainable):
    original = torch.arange(12, dtype=torch.float32).reshape(3, 4)
    values, actual_trainable = initialize_gene_features(original, mode, seed=42)
    assert values.shape == original.shape
    assert actual_trainable is trainable
    if mode in {"esm-mean", "esm-parti"}:
        assert torch.equal(values, original)
    elif mode == "esm-none":
        assert torch.count_nonzero(values) == 0
    else:
        repeated, _ = initialize_gene_features(original, mode, seed=42)
        assert torch.equal(values, repeated)


def test_unknown_gene_feature_mode_is_rejected():
    with pytest.raises(ValueError):
        initialize_gene_features(torch.zeros(2, 3), "bad-mode")
