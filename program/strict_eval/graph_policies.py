from __future__ import annotations

from copy import deepcopy

import numpy as np
import torch


def _unique_undirected(edge_index: torch.Tensor) -> np.ndarray:
    edges = edge_index.detach().cpu().numpy()
    lo = np.minimum(edges[0], edges[1])
    hi = np.maximum(edges[0], edges[1])
    mask = lo != hi
    return np.unique(np.stack([lo[mask], hi[mask]], axis=1), axis=0)


def _bidirectional(edges: np.ndarray, device: torch.device) -> torch.Tensor:
    if len(edges) == 0:
        return torch.empty((2, 0), dtype=torch.long, device=device)
    both = np.concatenate([edges, edges[:, ::-1]], axis=0)
    return torch.from_numpy(both.T.copy()).long().to(device)


def degree_preserving_rewire(edges: np.ndarray, seed: int, swaps_per_edge: int = 3) -> np.ndarray:
    """Undirected double-edge swap with fixed degree sequence and no self-loops."""
    rng = np.random.default_rng(seed)
    current = {tuple(sorted(map(int, edge))) for edge in edges}
    edge_list = list(current)
    if len(edge_list) < 2:
        return np.asarray(edge_list, dtype=np.int64).reshape(-1, 2)
    target_swaps = max(1, len(edge_list) * swaps_per_edge)
    accepted = 0
    attempts = 0
    while accepted < target_swaps and attempts < target_swaps * 20:
        attempts += 1
        ia, ib = rng.choice(len(edge_list), size=2, replace=False)
        a, b = edge_list[ia]
        c, d = edge_list[ib]
        old_first, old_second = (a, b), (c, d)
        if len({a, b, c, d}) < 4:
            continue
        if rng.random() < 0.5:
            c, d = d, c
        first, second = tuple(sorted((a, d))), tuple(sorted((c, b)))
        if first in current or second in current or first[0] == first[1] or second[0] == second[1]:
            continue
        current.remove(old_first)
        current.remove(old_second)
        current.add(first)
        current.add(second)
        edge_list[ia], edge_list[ib] = first, second
        accepted += 1
    return np.asarray(edge_list, dtype=np.int64)


def random_graph(n_nodes: int, n_edges: int, seed: int) -> np.ndarray:
    maximum = n_nodes * (n_nodes - 1) // 2
    if n_nodes < 0 or n_edges < 0 or n_edges > maximum:
        raise ValueError(f"cannot sample {n_edges} undirected edges from {n_nodes} nodes")
    rng = np.random.default_rng(seed)
    edges: set[tuple[int, int]] = set()
    while len(edges) < n_edges:
        values = rng.integers(0, n_nodes, size=(max(64, (n_edges - len(edges)) * 2), 2))
        for a, b in values:
            if a != b:
                edges.add(tuple(sorted((int(a), int(b)))))
            if len(edges) == n_edges:
                break
    return np.asarray(sorted(edges), dtype=np.int64)


def apply_ppi_policy(hetero, policy: str, seed: int = 42, pathway_gene_mask: torch.Tensor | None = None):
    """Return a cloned graph for a retrained PPI control, never an in-place perturbation."""
    graph = deepcopy(hetero)
    relation = graph["gene", "ppi", "gene"]
    device = relation.edge_index.device
    original = _unique_undirected(relation.edge_index)
    n_nodes = int(graph["gene"].num_nodes)
    weights = relation.edge_weight.detach().clone() if hasattr(relation, "edge_weight") else None

    if policy == "graph-real":
        return graph
    if policy == "graph-none":
        idx = torch.arange(n_nodes, device=device)
        relation.edge_index = torch.stack([idx, idx], dim=0)
        relation.edge_weight = torch.ones(n_nodes, device=device)
        return graph
    if policy == "graph-degree":
        changed = degree_preserving_rewire(original, seed)
    elif policy == "graph-random":
        changed = random_graph(n_nodes, len(original), seed)
    elif policy == "graph-pathway":
        if pathway_gene_mask is None:
            raise ValueError("graph-pathway requires pathway_gene_mask")
        keep_nodes = set(torch.where(pathway_gene_mask.cpu())[0].tolist())
        changed = np.asarray([edge for edge in original if edge[0] in keep_nodes and edge[1] in keep_nodes])
    elif policy == "graph-weight-shuffle":
        if weights is None:
            raise ValueError("graph-weight-shuffle requires edge weights")
        generator = torch.Generator(device="cpu").manual_seed(seed)
        relation.edge_weight = weights[torch.randperm(len(weights), generator=generator).to(weights.device)]
        return graph
    else:
        raise ValueError(f"unknown PPI policy: {policy}")

    relation.edge_index = _bidirectional(changed, device)
    relation.edge_weight = torch.ones(relation.edge_index.shape[1], device=device)
    return graph


def apply_dti_visibility(hetero, allowed_drug_idx: set[int]):
    """Clone a graph and retain DTI edges for the explicitly allowed drugs."""
    graph = deepcopy(hetero)
    relation = graph["drug", "targets", "gene"]
    allowed = torch.zeros(int(graph["drug"].num_nodes), dtype=torch.bool, device=relation.edge_index.device)
    if allowed_drug_idx:
        allowed[torch.tensor(sorted(allowed_drug_idx), device=allowed.device)] = True
    keep = allowed[relation.edge_index[0]]
    relation.edge_index = relation.edge_index[:, keep]
    if hasattr(relation, "edge_weight"):
        relation.edge_weight = relation.edge_weight[keep]
    return graph


def set_model_dti_edges(model, hetero) -> None:
    """Switch DTI visibility for validation/test without changing learned weights."""
    if not hasattr(model, "gene_enc"):
        return
    forward = hetero["drug", "targets", "gene"].edge_index.to(model.drug_x.device)
    reverse = torch.stack([forward[1], forward[0]], dim=0)
    model.gene_enc.dti_fwd = forward
    model.gene_enc.dti_rev = reverse
