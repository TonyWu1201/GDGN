import numpy as np
import pytest
import torch
from torch_geometric.data import HeteroData

from program.strict_eval.graph_policies import (
    apply_dti_visibility,
    apply_ppi_policy,
    degree_preserving_rewire,
    random_graph,
)


def degrees(edges, n):
    result = np.zeros(n, dtype=int)
    for a, b in edges:
        result[a] += 1
        result[b] += 1
    return result


def test_degree_rewire_preserves_degrees():
    edges = np.array([[0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 0], [0, 3], [1, 4]])
    rewired = degree_preserving_rewire(edges, seed=42)
    assert np.array_equal(degrees(edges, 6), degrees(rewired, 6))


def test_random_graph_has_exact_edge_count_and_no_loops():
    edges = random_graph(20, 30, seed=42)
    assert len(edges) == 30
    assert np.all(edges[:, 0] < edges[:, 1])


def test_random_graph_rejects_impossible_size():
    with pytest.raises(ValueError):
        random_graph(3, 4, seed=42)


def graph_fixture():
    graph = HeteroData()
    graph["gene"].num_nodes = 6
    graph["drug"].num_nodes = 3
    undirected = np.array([[0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 0], [0, 3], [1, 4]])
    graph["gene", "ppi", "gene"].edge_index = torch.from_numpy(
        np.concatenate([undirected, undirected[:, ::-1]], axis=0).T
    ).long()
    graph["gene", "ppi", "gene"].edge_weight = torch.arange(16).float()
    graph["drug", "targets", "gene"].edge_index = torch.tensor([[0, 1, 2], [1, 2, 3]])
    return graph


@pytest.mark.parametrize(
    "policy", ["graph-real", "graph-none", "graph-degree", "graph-random", "graph-weight-shuffle", "graph-pathway"]
)
def test_all_ppi_retraining_policies_return_valid_graph(policy):
    graph = graph_fixture()
    changed = apply_ppi_policy(graph, policy, seed=42, pathway_gene_mask=torch.ones(6, dtype=torch.bool))
    relation = changed["gene", "ppi", "gene"]
    assert relation.edge_index.shape[0] == 2
    assert relation.edge_weight.shape[0] == relation.edge_index.shape[1]
    assert changed is not graph


def test_dti_visibility_filters_drug_sources():
    filtered = apply_dti_visibility(graph_fixture(), {0, 2})
    assert set(filtered["drug", "targets", "gene"].edge_index[0].tolist()) == {0, 2}
