import numpy as np
import torch

from program.verify_pretrain import embedding_stats


def test_effective_rank_identity_is_full_rank():
    values = torch.eye(4)
    result = embedding_stats(values)
    # Centering removes one dimension, leaving three equal non-zero singular values.
    assert np.isclose(result["effective_rank"], 3.0, atol=1e-5)
