from program.model.pretrain_dataloader import PretrainCellSampler


def test_pretrain_sampler_uses_only_explicit_fold_cells():
    sampler = PretrainCellSampler(
        n_cells=10, batch_condition=2, seed=42, shuffle=False, cell_indices=[2, 5, 9]
    )
    assert [batch.tolist() for batch in sampler] == [[2, 5], [9]]
    assert len(sampler) == 2
