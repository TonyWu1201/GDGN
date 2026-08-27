from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr

from .biological import pathway_gene_mask, pathway_membership
from .constants import DRUG_INDEX_PATH, HETERO_GRAPH_PATH
from .data import load_fold_features, load_fold_pairs
from .faithfulness import (
    attention_randomization_test,
    deletion_curve,
    neighborhood_enrichment,
    sufficiency,
    target_retrieval_metrics,
    target_vs_control_test,
)
from .graph_policies import apply_dti_visibility, apply_ppi_policy
from .io import write_json
from .pathway_recompute import FoldPathwayRecomputer
from .training import PATHWAY_MODELS, _build_neural_model


def _matched_control_order(
    importance: np.ndarray, expression: np.ndarray, degree: np.ndarray, seed: int
) -> np.ndarray:
    """Match each ranked gene on expression and PPI-degree quintiles."""
    rng = np.random.default_rng(seed)
    rank = np.argsort(-np.abs(importance))
    expression_bin = pd.qcut(pd.Series(expression).rank(method="first"), 5, labels=False).to_numpy()
    degree_bin = pd.qcut(pd.Series(degree).rank(method="first"), 5, labels=False).to_numpy()
    selected: set[int] = set()
    result = []
    for gene in rank:
        candidates = np.where(
            (expression_bin == expression_bin[gene]) & (degree_bin == degree_bin[gene])
        )[0]
        candidates = np.asarray([item for item in candidates if item not in selected and item != gene])
        if not len(candidates):
            candidates = np.asarray([item for item in range(len(rank)) if item not in selected and item != gene])
        choice = int(rng.choice(candidates)) if len(candidates) else int(gene)
        selected.add(choice)
        result.append(choice)
    return np.asarray(result, dtype=np.int64)


def _gene_attribution(
    attribution: torch.Tensor, modalities: tuple[str, ...], cell_features: dict[str, torch.Tensor]
) -> tuple[np.ndarray, np.ndarray]:
    blocks, start = [], 0
    n_genes = int(cell_features["expression"].shape[1])
    for modality in modalities:
        width = int(cell_features[modality].shape[1])
        block = attribution[0, start:start + width]
        if width == n_genes:
            blocks.append(block)
        start += width
    if not blocks:
        raise ValueError("no gene-aligned modality is available for attribution")
    stacked = torch.stack(blocks)
    return (
        torch.sqrt(torch.square(stacked).sum(dim=0)).detach().cpu().numpy(),
        stacked.sum(dim=0).detach().cpu().numpy(),
    )


def _ppi_degree_and_neighborhoods() -> tuple[np.ndarray, list[set[int]]]:
    graph = torch.load(HETERO_GRAPH_PATH, map_location="cpu", weights_only=False)
    edge_index = graph["gene", "ppi", "gene"].edge_index.detach().cpu().numpy()
    n_genes = int(graph["gene"].num_nodes)
    neighbors = [set() for _ in range(n_genes)]
    for source, target in edge_index.T:
        if source != target:
            neighbors[int(source)].add(int(target))
    return np.asarray([len(item) for item in neighbors], dtype=np.float32), neighbors


def _target_sets(drug_idx: int, gene_order: list[str]) -> tuple[np.ndarray, set[int]]:
    graph = torch.load(HETERO_GRAPH_PATH, map_location="cpu", weights_only=False)
    edges = graph["drug", "targets", "gene"].edge_index.detach().cpu().numpy()
    target_idx = set(edges[1, edges[0] == int(drug_idx)].astype(int).tolist())
    mask = np.asarray([index in target_idx for index in range(len(gene_order))], dtype=bool)
    return mask, target_idx


def _neighborhood_sets(targets: set[int], neighbors: list[set[int]]) -> dict[str, set[int]]:
    one = set().union(*(neighbors[index] for index in targets)) if targets else set()
    two = set(one)
    for index in list(one):
        two.update(neighbors[index])
    return {"known-targets": targets, "ppi-one-hop": one, "ppi-two-hop": two}


def _predict_explicit(model, cell_input, pathway, drug_idx, dti_policy, gate=None) -> float:
    with torch.no_grad():
        prediction, _ = model.forward_features(
            cell_input, pathway, drug_idx, dti_policy, pathway_gate_override=gate
        )
    return float(prediction.squeeze().detach().cpu())


def interpret_case(
    model,
    cell_features: dict[str, torch.Tensor],
    recomputer: FoldPathwayRecomputer,
    cell_idx: int,
    drug_idx: int,
    dti_policy: str,
    train_cells: list[int],
    n_steps: int = 50,
    n_attention_random: int = 100,
    seed: int = 42,
) -> dict:
    from captum.attr import IntegratedGradients

    device = next(model.parameters()).device
    cell_tensor = torch.tensor([cell_idx], device=device)
    drug_tensor = torch.tensor([drug_idx], device=device)
    cell_input = model._cell_input(cell_tensor).detach()
    pathway = model.cell_pathway_activity[cell_tensor].detach()
    train_tensor = torch.tensor(train_cells, device=device)
    train_mean_input = model._cell_input(train_tensor).mean(dim=0, keepdim=True)
    train_mean_pathway = model.cell_pathway_activity[train_tensor].mean(dim=0, keepdim=True)

    def forward(explicit_cell, explicit_pathway):
        return model.forward_features(explicit_cell, explicit_pathway, drug_tensor, dti_policy)[0]

    ig = IntegratedGradients(forward)
    baseline_specs = {
        "train-mean": (train_mean_input, train_mean_pathway),
        "zero": (torch.zeros_like(cell_input), torch.zeros_like(pathway)),
    }
    baseline_results = {}
    importance_values, signed_values = [], []
    original_states = [parameter.requires_grad for parameter in model.parameters()]
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    try:
        for name, baselines in baseline_specs.items():
            (cell_attr, pathway_attr), delta = ig.attribute(
                (cell_input.clone().requires_grad_(True), pathway.clone().requires_grad_(True)),
                baselines=baselines, n_steps=n_steps, internal_batch_size=1,
                return_convergence_delta=True,
            )
            importance, signed = _gene_attribution(cell_attr, model.modalities, cell_features)
            importance_values.append(importance)
            signed_values.append(signed)
            baseline_results[name] = {
                "convergence_delta": float(delta.detach().abs().mean().cpu()),
                "pathway_attribution": pathway_attr.detach().cpu().squeeze(0).tolist(),
            }
    finally:
        for parameter, state in zip(model.parameters(), original_states):
            parameter.requires_grad_(state)
        model.eval()
    importance = np.mean(importance_values, axis=0)
    signed = np.mean(signed_values, axis=0)
    baseline_sensitivity = float(spearmanr(importance_values[0], importance_values[1]).statistic)

    original_prediction, extras = model.forward_features(cell_input, pathway, drug_tensor, dti_policy)
    base_prediction = float(original_prediction.squeeze().detach().cpu())
    degree, neighbors = _ppi_degree_and_neighborhoods()
    expression = cell_features["expression"][cell_idx].detach().cpu().numpy()
    matched = _matched_control_order(importance, expression, degree, seed)
    n_genes = len(importance)

    def changed_inputs(indices: np.ndarray, retain: bool = False):
        changed = cell_input.clone()
        offset = 0
        selected = np.asarray(indices, dtype=int)
        for modality in model.modalities:
            width = int(cell_features[modality].shape[1])
            if width == n_genes:
                if retain:
                    original = changed[:, offset:offset + width].clone()
                    changed[:, offset:offset + width] = 0.0
                    changed[:, offset + torch.as_tensor(selected, device=device)] = original[
                        :, torch.as_tensor(selected, device=device)
                    ]
                else:
                    changed[:, offset + torch.as_tensor(selected, device=device)] = 0.0
            offset += width
        pathway_values = recomputer.recompute(cell_idx, selected, retain=retain)
        return changed, torch.from_numpy(pathway_values).unsqueeze(0).to(device)

    def predict_deleted(indices: np.ndarray) -> float:
        changed, changed_pathway = changed_inputs(indices, retain=False)
        return _predict_explicit(model, changed, changed_pathway, drug_tensor, dti_policy)

    def predict_retained(indices: np.ndarray) -> float:
        changed, changed_pathway = changed_inputs(indices, retain=True)
        return _predict_explicit(model, changed, changed_pathway, drug_tensor, dti_policy)

    curve = deletion_curve(importance, base_prediction, predict_deleted, matched, seed=seed)
    sufficient = sufficiency(importance, base_prediction, predict_retained)
    membership, pathway_names, gene_order = pathway_membership()
    target_mask, targets = _target_sets(drug_idx, gene_order)
    retrieval = target_retrieval_metrics(importance, target_mask)
    control = np.zeros(n_genes, dtype=bool)
    control[matched[:max(1, int(target_mask.sum()))]] = True
    target_test = target_vs_control_test(importance, target_mask, control)
    enrichment_sets = _neighborhood_sets(targets, neighbors)
    enrichment_sets.update({
        f"pathway:{name}": set(torch.where(membership[index] > 0)[0].tolist())
        for index, name in enumerate(pathway_names)
    })
    enrichment = neighborhood_enrichment(
        np.argsort(-np.abs(importance)), enrichment_sets, n_genes, top_k=100
    )
    attention = None
    if "pathway_gate" in extras:
        gate = extras["pathway_gate"].detach().cpu().squeeze(0).numpy()

        def predict_gate(values: np.ndarray) -> float:
            tensor = torch.from_numpy(np.asarray(values, dtype=np.float32)).unsqueeze(0).to(device)
            return _predict_explicit(model, cell_input, pathway, drug_tensor, dti_policy, tensor)

        attention = attention_randomization_test(
            base_prediction, predict_gate, gate, n_random=n_attention_random, seed=seed
        )
    return {
        "drug_idx": drug_idx, "cell_idx": cell_idx,
        "prediction": base_prediction,
        "ig_per_gene": importance.tolist(), "ig_signed_per_gene": signed.tolist(),
        "ig_baselines": baseline_results,
        "ig_convergence_delta": max(item["convergence_delta"] for item in baseline_results.values()),
        "ig_baseline_spearman": baseline_sensitivity,
        "pathway_recomputed_for_interventions": True,
        "target_metrics": retrieval,
        "target_vs_matched_p": target_test["p_value"],
        "neighborhood_enrichment": enrichment,
        "deletion_curve": {
            "fractions": curve.fractions, "top_drop": curve.top_drop,
            "random_drop": curve.random_drop, "matched_drop": curve.matched_drop,
        },
        "comprehensiveness": curve.comprehensiveness,
        "sufficiency": sufficient,
        "attention_randomization": attention,
    }


def run_strict_interpretation(
    run_dir: str | Path,
    output_dir: str | Path,
    min_drug_samples: int = 5,
    max_drugs: int = 0,
    n_steps: int = 50,
    n_attention_random: int = 100,
    smoke: bool = False,
) -> list[Path]:
    run_dir = Path(run_dir)
    config = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
    if config["model_id"] not in PATHWAY_MODELS:
        raise ValueError("strict interpretation currently supports M0-M4 pathway-family checkpoints")
    split_path = Path(f"data/model/splits/{config['split_id']}/fold-{int(config['fold']):02d}.json")
    split_payload, frames = load_fold_pairs(split_path)
    cell_features, drug_features, _ = load_fold_features(
        split_payload, run_dir, allow_legacy_fallback=smoke
    )
    graph = torch.load(HETERO_GRAPH_PATH, map_location="cpu", weights_only=False)
    graph["drug"].x = drug_features
    graph_policy = config.get("graph_policy", "graph-real")
    graph = apply_ppi_policy(
        graph, graph_policy, int(config["seed"]),
        pathway_gene_mask() if graph_policy == "graph-pathway" else None,
    )
    train_drugs = set(split_payload["splits"]["train"]["drug_idx"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = _build_neural_model(
        config, cell_features, drug_features, apply_dti_visibility(graph, train_drugs).to(device), device
    )
    checkpoint = torch.load(run_dir / "checkpoint-best.pt", map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    recomputer = FoldPathwayRecomputer(split_payload, run_dir / "scalers.npz")
    eligible = [
        (int(drug), group) for drug, group in frames["test"].groupby("drug_idx")
        if len(group) >= min_drug_samples
    ]
    if max_drugs > 0:
        eligible = eligible[:max_drugs]
    if smoke:
        eligible, n_steps, n_attention_random = eligible[:1], min(n_steps, 8), min(n_attention_random, 5)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    cid_to_idx = json.loads(DRUG_INDEX_PATH.read_text(encoding="utf-8"))
    cid_of = {int(index): str(cid) for cid, index in cid_to_idx.items()}
    written = []
    for drug_idx, group in eligible:
        rows = [group.loc[group["ic50"].idxmin()], group.loc[group["ic50"].idxmax()]]
        for label, row in zip(("sensitive", "resistant"), rows):
            result = interpret_case(
                model, cell_features, recomputer, int(row.cell_idx), drug_idx,
                split_payload["graph_policy"], split_payload["splits"]["train"]["cell_idx"],
                n_steps=n_steps, n_attention_random=n_attention_random, seed=int(config["seed"]),
            )
            result["case_type"] = label
            result["cid"] = cid_of.get(drug_idx, str(drug_idx))
            result["seed"] = int(config["seed"])
            result["fold"] = int(config["fold"])
            path = output / f"drug{drug_idx}_cell{int(row.cell_idx)}_{label}.json"
            write_json(path, result)
            written.append(path)
    write_json(output / "manifest.json", {
        "run_dir": str(run_dir), "n_cases": len(written), "smoke": smoke,
        "min_drug_samples": min_drug_samples,
    })
    return written
