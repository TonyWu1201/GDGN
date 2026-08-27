import pytest
import torch

from program.strict_eval.models import ChunkAttentionPooler, MaskedOmicsAutoencoder, PathwayResidualModel


def features():
    return {
        "expression": torch.randn(8, 12),
        "mutation": torch.randint(0, 2, (8, 12)).float(),
        "copynumber": torch.randn(8, 12),
        "methylation": torch.randn(8, 12),
        "pathway_activity": torch.randn(8, 5),
    }


@pytest.mark.parametrize("level", range(5))
def test_m0_to_m4_forward_backward(level):
    kwargs = {}
    if level >= 2:
        kwargs["pathway_adjacency"] = torch.eye(5)
    if level >= 3:
        kwargs["known_target_pathways"] = torch.softmax(torch.randn(6, 5), dim=-1)
    model = PathwayResidualModel(features(), torch.randn(6, 10), f"model-m{level}", hidden_dim=8, **kwargs)
    prediction, extra = model(torch.tensor([0, 1, 2, 3]), torch.tensor([0, 1, 2, 3]), "structure-only")
    assert prediction.shape == (4, 1)
    prediction.sum().backward()
    assert any(parameter.grad is not None for parameter in model.parameters())
    if level >= 2:
        assert 0 < float(extra["alpha"].detach()) < 0.1


def test_chunk_attention_ignores_padding():
    pooler = ChunkAttentionPooler(6)
    chunks = torch.randn(2, 4, 6)
    mask = torch.tensor([[True, True, False, False], [True, True, True, False]])
    output, weights = pooler(chunks, mask)
    assert output.shape == (2, 6)
    assert torch.allclose(weights[~mask], torch.zeros_like(weights[~mask]))
    assert torch.allclose(weights.sum(dim=1), torch.ones(2))


def test_chunk_attention_all_invalid_row_is_finite_zero():
    pooler = ChunkAttentionPooler(6)
    chunks = torch.zeros(1, 2, 6)
    output, weights = pooler(chunks, torch.zeros(1, 2, dtype=torch.bool))
    assert torch.isfinite(weights).all()
    assert torch.equal(output, torch.zeros_like(output))


def test_m4_accepts_training_entity_visibility_policy():
    model = PathwayResidualModel(
        features(), torch.randn(6, 10), "model-m4", hidden_dim=8,
        pathway_adjacency=torch.eye(5),
        known_target_pathways=torch.softmax(torch.randn(6, 5), dim=-1),
    )
    prediction, extra = model(
        torch.tensor([0, 1]), torch.tensor([0, 1]), "training-entities-only"
    )
    assert prediction.shape == (2, 1)
    assert torch.allclose(extra["pathway_gate"].sum(dim=1), torch.ones(2))


def test_masked_omics_checkpoint_can_be_loaded_and_frozen(tmp_path):
    pretrained = MaskedOmicsAutoencoder(12, latent_dim=8)
    checkpoint = tmp_path / "checkpoint.pt"
    torch.save({
        "model_state_dict": pretrained.state_dict(), "input_dim": 12, "latent_dim": 8,
    }, checkpoint)
    model = PathwayResidualModel(
        features(), torch.randn(6, 10), "model-m0", hidden_dim=8,
        modalities=("expression",), omics_pretrain_ckpt=checkpoint,
        omics_finetune_policy="freeze",
    )
    assert all(not parameter.requires_grad for parameter in model.cell_encoder.parameters())
    assert torch.equal(model.cell_encoder.net[0].weight, pretrained.encoder.net[0].weight)
