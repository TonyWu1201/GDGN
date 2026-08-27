import numpy as np
import torch
from torch_geometric.data import HeteroData

from program.model.edge_mask import EdgeMaskSampler, _edge_key_asym, _edge_key_sym, _make_cfg


def tiny_graph():
    graph = HeteroData()
    graph["gene"].num_nodes = 8
    graph["drug"].num_nodes = 4
    undirected = torch.tensor([[0, 1, 1, 2, 3, 4, 5, 6], [1, 0, 2, 1, 4, 3, 6, 5]])
    graph["gene", "ppi", "gene"].edge_index = undirected
    graph["drug", "targets", "gene"].edge_index = torch.tensor([[0, 1, 2, 3], [0, 2, 4, 6]])
    return graph


def test_strict_edges_never_include_validation_or_test():
    sampler = EdgeMaskSampler(tiny_graph(), _make_cfg(heldout_ppi_ratio=0.5, heldout_dti_ratio=0.5))
    heldout = sampler.build_heldout()
    train_ppi, _, train_dti, _ = sampler.training_message_edges()
    train_ppi_keys = _edge_key_sym(train_ppi[0].numpy(), train_ppi[1].numpy(), sampler.n_genes)
    train_dti_keys = _edge_key_asym(train_dti[0].numpy(), train_dti[1].numpy(), sampler.n_genes)
    for split in ("val", "test"):
        ppi = heldout[f"ppi_{split}_pos"]
        dti = heldout[f"dti_{split}_pos"]
        assert np.intersect1d(train_ppi_keys, _edge_key_sym(ppi[0].numpy(), ppi[1].numpy(), sampler.n_genes)).size == 0
        assert np.intersect1d(train_dti_keys, _edge_key_asym(dti[0].numpy(), dti[1].numpy(), sampler.n_genes)).size == 0
