from __future__ import annotations

import copy
import json
import math
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.decomposition import PCA
from tqdm import tqdm

from program.model.dataset import load_hetero_graph
from program.model.train_gdgn import build_model as build_legacy_model

from .baselines import LinearFeatureBaseline, build_statistical_baseline
from .biological import build_weighted_pathway_adjacency, known_target_pathway_distribution, pathway_gene_mask
from .data import build_fold_loaders, load_fold_features, load_fold_pairs, predictions_frame
from .graph_policies import apply_dti_visibility, apply_ppi_policy, set_model_dti_edges
from .io import atomic_write_text, seed_everything, write_json
from .metrics import full_metric_bundle, regression_metrics
from .models import IDPairMLP, PathwayResidualModel
from .registry import ExperimentRun, RunSpec

STATISTICAL_MODELS = {"global-mean", "drug-mean", "cell-mean", "two-way-additive"}
LINEAR_MODELS = {"ridge", "elastic-net"}
LEGACY_MODELS = {"gdgn", "baseline_simple", "cdr_baseline"}
PATHWAY_MODELS = {f"model-m{i}" for i in range(5)} | {
    "morgan-expression-mlp", "morgan-multiomics-mlp"
}


def resolve_config_templates(config: dict[str, Any]) -> dict[str, Any]:
    context = {
        "experiment_id": str(config["experiment_id"]), "model_id": str(config["model_id"]),
        "split_id": str(config["split_id"]), "fold": int(config["fold"]), "seed": int(config["seed"]),
        "variant": str(config.get("variant", "default")),
    }
    return {
        key: (value.format(**context) if isinstance(value, str) and "{" in value else value)
        for key, value in config.items()
    }


def _write_predictions(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        frame.to_parquet(path, index=False)
    except ImportError as exc:
        raise RuntimeError("Parquet output requires pyarrow; run `uv sync --all-groups`") from exc


def _attach_metadata(output: pd.DataFrame, frame: pd.DataFrame) -> pd.DataFrame:
    columns = [
        name for name in (
            "pair_id", "cancer_type", "drug_max_similarity",
            "cell_max_similarity", "protocol_similarity",
        )
        if name in frame.columns
    ]
    if columns == ["pair_id"]:
        return output
    return output.merge(frame[columns].drop_duplicates("pair_id"), on="pair_id", how="left")


def _metrics_with_similarity(output: pd.DataFrame) -> dict:
    metrics = full_metric_bundle(output)
    if "protocol_similarity" not in output or output["protocol_similarity"].notna().sum() == 0:
        metrics["similarity_strata"] = {}
        return metrics
    bands = pd.cut(
        output["protocol_similarity"], bins=[-np.inf, 0.4, 0.7, np.inf],
        labels=["low_lt_0_4", "medium_0_4_to_0_7", "high_ge_0_7"], right=False,
    )
    metrics["similarity_strata"] = {
        str(label): regression_metrics(output.loc[bands == label, "y_true"], output.loc[bands == label, "y_pred"])
        for label in bands.cat.categories
        if int((bands == label).sum()) > 0
    }
    return metrics


def _evaluate_frame(frame: pd.DataFrame, pred: np.ndarray, split: str) -> tuple[pd.DataFrame, dict]:
    output = predictions_frame(
        frame["pair_id"], frame["cell_idx"], frame["drug_idx"], frame["ic50"], pred, split
    )
    output = _attach_metadata(output, frame)
    return output, _metrics_with_similarity(output)


def _fit_statistical(config: dict, frames: dict[str, pd.DataFrame], run: ExperimentRun) -> dict:
    model = build_statistical_baseline(config["model_id"]).fit(frames["train"])
    outputs, metrics = [], {}
    for split in ("val", "test"):
        output, metric = _evaluate_frame(frames[split], model.predict(frames[split]), split)
        outputs.append(output)
        metrics[split] = metric
    torch.save({"model": model, "config": config}, run.run_dir / "checkpoint-best.pt")
    _write_predictions(run.run_dir / "predictions.parquet", pd.concat(outputs, ignore_index=True))
    write_json(run.run_dir / "metrics.json", metrics)
    return metrics


def _linear_features(
    frames: dict[str, pd.DataFrame],
    cell_features: dict[str, torch.Tensor],
    drug_features: torch.Tensor,
    max_components: int,
) -> tuple[dict[str, np.ndarray], PCA]:
    expression = cell_features["expression"].numpy()
    train_cells = sorted(frames["train"]["cell_idx"].unique().tolist())
    n_components = min(max_components, len(train_cells) - 1, expression.shape[1])
    pca = PCA(n_components=n_components, random_state=42).fit(expression[train_cells])
    cell_repr = pca.transform(expression).astype(np.float32)
    drug = drug_features.numpy()
    return {
        split: np.concatenate([
            cell_repr[frame["cell_idx"].to_numpy(dtype=int)],
            drug[frame["drug_idx"].to_numpy(dtype=int)],
        ], axis=1)
        for split, frame in frames.items()
    }, pca


def _fit_linear(
    config: dict,
    frames: dict[str, pd.DataFrame],
    cell_features: dict[str, torch.Tensor],
    drug_features: torch.Tensor,
    run: ExperimentRun,
) -> dict:
    features, pca = _linear_features(
        frames, cell_features, drug_features, int(config.get("pca_components", 128))
    )
    model = LinearFeatureBaseline(
        config["model_id"], alpha=float(config.get("linear_alpha", 1.0)),
        l1_ratio=float(config.get("l1_ratio", 0.5)),
    ).fit_features(features["train"], frames["train"]["ic50"].to_numpy())
    outputs, metrics = [], {}
    for split in ("val", "test"):
        output, metric = _evaluate_frame(frames[split], model.predict_features(features[split]), split)
        outputs.append(output)
        metrics[split] = metric
    torch.save({"model": model, "pca": pca, "config": config}, run.run_dir / "checkpoint-best.pt")
    _write_predictions(run.run_dir / "predictions.parquet", pd.concat(outputs, ignore_index=True))
    write_json(run.run_dir / "metrics.json", metrics)
    return metrics


def _omics_batch(cell_features: dict[str, torch.Tensor], cell_idx: torch.Tensor, device) -> dict:
    return {
        "expr": cell_features["expression"][cell_idx].to(device),
        "mut": cell_features["mutation"][cell_idx].to(device),
        "cnv": cell_features["copynumber"][cell_idx].to(device),
        "meth": cell_features["methylation"][cell_idx].to(device),
        "pathway": cell_features["pathway_activity"][cell_idx].to(device),
    }


def _predict_neural(
    model,
    loader,
    model_id: str,
    cell_features: dict[str, torch.Tensor],
    device,
    dti_policy: str,
    unseen_cells: set[int],
    unseen_drugs: set[int],
) -> pd.DataFrame:
    model.eval()
    rows = {key: [] for key in ("pair_id", "cell_idx", "drug_idx", "y_true", "y_pred")}
    with torch.no_grad():
        for batch in loader:
            cell_idx = batch["cell_idx"].to(device)
            drug_idx = batch["drug_idx"].to(device)
            if model_id == "id-mlp":
                mapped_cell = cell_idx.clone()
                mapped_drug = drug_idx.clone()
                for value in unseen_cells:
                    mapped_cell[cell_idx == value] = model.cell_unk
                for value in unseen_drugs:
                    mapped_drug[drug_idx == value] = model.drug_unk
                pred = model(mapped_cell, mapped_drug)
            elif model_id in PATHWAY_MODELS:
                pred, _ = model(cell_idx, drug_idx, dti_policy=dti_policy)
            else:
                pred, _ = model(cell_idx, drug_idx, _omics_batch(cell_features, batch["cell_idx"], device))
            rows["pair_id"].extend(batch["pair_id"].tolist())
            rows["cell_idx"].extend(batch["cell_idx"].tolist())
            rows["drug_idx"].extend(batch["drug_idx"].tolist())
            rows["y_true"].extend(batch["y"].tolist())
            rows["y_pred"].extend(pred.squeeze(-1).detach().cpu().tolist())
    return pd.DataFrame(rows)


def _build_neural_model(
    config: dict,
    cell_features: dict[str, torch.Tensor],
    drug_features: torch.Tensor,
    train_graph,
    device,
):
    model_id = config["model_id"]
    if model_id == "id-mlp":
        return IDPairMLP(len(cell_features["expression"]), len(drug_features), config.get("hidden_dim", 128)).to(device)
    if model_id in PATHWAY_MODELS:
        actual_id = {
            "morgan-expression-mlp": "model-m0",
            "morgan-multiomics-mlp": "model-m0",
        }.get(model_id, model_id)
        modalities = tuple(config.get("modalities", ["expression"]))
        if model_id == "morgan-multiomics-mlp":
            modalities = ("expression", "mutation", "copynumber", "methylation")
        ppi_relation = train_graph["gene", "ppi", "gene"]
        ppi_weight = (
            ppi_relation.edge_weight if hasattr(ppi_relation, "edge_weight")
            else torch.ones(ppi_relation.edge_index.shape[1])
        )
        pathway_adjacency = None
        if int(actual_id[-1]) >= 2:
            if config.get("graph_policy", "graph-real") == "graph-none":
                pathway_adjacency = torch.eye(cell_features["pathway_activity"].shape[1])
            else:
                pathway_adjacency = build_weighted_pathway_adjacency(
                    ppi_relation.edge_index, ppi_weight
                )
        return PathwayResidualModel(
            {name: value.to(device) for name, value in cell_features.items()},
            drug_features.to(device),
            model_id=actual_id,
            hidden_dim=int(config.get("hidden_dim", 128)),
            pathway_adjacency=(pathway_adjacency.to(device) if pathway_adjacency is not None else None),
            known_target_pathways=(known_target_pathway_distribution().to(device) if int(actual_id[-1]) >= 3 else None),
            modalities=modalities,
            dropout=float(config.get("dropout", 0.2)),
            omics_pretrain_ckpt=config.get("omics_pretrain_ckpt"),
            omics_finetune_policy=config.get("omics_finetune_policy", "full"),
        ).to(device)
    legacy_config = dict(config)
    legacy_config["model"] = model_id
    return build_legacy_model(model_id, train_graph, legacy_config, device)


def _fit_neural(
    config: dict,
    split_payload: dict,
    frames: dict[str, pd.DataFrame],
    loaders,
    cell_features: dict[str, torch.Tensor],
    drug_features: torch.Tensor,
    run: ExperimentRun,
    smoke: bool,
) -> dict:
    seed_everything(int(config["seed"]), deterministic=bool(config.get("deterministic", False)))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_id = config["model_id"]
    full_graph = load_hetero_graph("cpu")
    full_graph["drug"].x = drug_features
    graph_policy = config.get("graph_policy", "graph-real")
    full_graph = apply_ppi_policy(
        full_graph, graph_policy, int(config["seed"]),
        pathway_gene_mask() if graph_policy == "graph-pathway" else None,
    )
    train_drugs = set(split_payload["splits"]["train"]["drug_idx"])
    train_graph = apply_dti_visibility(full_graph, train_drugs).to(device)
    if config.get("gene_feature_mode") == "esm-parti":
        parti_path = Path(config["esm_parti_path"])
        train_graph["gene"].x = torch.load(parti_path, map_location=device, weights_only=False)
    model = _build_neural_model(config, cell_features, drug_features, train_graph, device)
    if model_id == "model-m4" and split_payload["graph_policy"] == "structure-only":
        allowed = torch.zeros(
            model.known_target_pathways.shape[0], dtype=torch.bool,
            device=model.known_target_pathways.device,
        )
        allowed[torch.tensor(sorted(train_drugs), device=allowed.device)] = True
        model.known_target_pathways[~allowed] = 0.0

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=float(config.get("lr", 1e-3)),
        weight_decay=float(config.get("weight_decay", 0.0)),
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", patience=int(config.get("lr_patience", 3)), factor=0.5
    )
    max_epochs = 1 if smoke else int(config.get("max_epochs", 50))
    patience = int(config.get("early_stopping_patience", 10))
    best_pcc, best_state, best_epoch = -math.inf, None, -1
    patience_left = patience
    history = []
    alpha_trace = []
    train_cells = set(split_payload["splits"]["train"]["cell_idx"])

    for epoch in range(max_epochs):
        if (
            getattr(model, "omics_finetune_policy", None) == "staged"
            and epoch == int(config.get("omics_unfreeze_epoch", 5))
        ):
            for parameter in model.cell_encoder.parameters():
                parameter.requires_grad_(True)
        model.train()
        losses = []
        batch_bar = tqdm(enumerate(loaders["train"]), total=len(loaders["train"]),
                         desc=f"[train] epoch {epoch}", unit="batch", leave=False)
        for step, batch in batch_bar:
            cell_idx = batch["cell_idx"].to(device)
            drug_idx = batch["drug_idx"].to(device)
            y = batch["y"].to(device)
            if model_id == "id-mlp":
                pred = model(cell_idx, drug_idx)
                extras = {}
            elif model_id in PATHWAY_MODELS:
                pred, extras = model(cell_idx, drug_idx, dti_policy=split_payload["graph_policy"])
            else:
                pred, extras = model(cell_idx, drug_idx, _omics_batch(cell_features, batch["cell_idx"], device))
            loss = F.mse_loss(pred.squeeze(-1), y)
            if model_id == "model-m4" and split_payload["graph_policy"] == "structure-only":
                target = model.known_target_pathways[drug_idx]
                logits = model.target_predictor(model.drug_features[drug_idx])
                loss = loss + float(config.get("target_aux_weight", 0.1)) * F.binary_cross_entropy_with_logits(
                    logits, (target > 0).float()
                )
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(config.get("grad_clip", 1.0)))
            optimizer.step()
            losses.append(float(loss.item()))
            batch_bar.set_postfix(loss=float(loss.item()))
            if smoke and step >= 1:
                break
        batch_bar.close()

        if model_id in LEGACY_MODELS:
            val_allowed = train_drugs | set(split_payload["splits"]["val"]["drug_idx"])
            if split_payload["graph_policy"] == "structure-only":
                val_allowed = train_drugs
            set_model_dti_edges(model, apply_dti_visibility(full_graph, val_allowed))
        val_pred = _predict_neural(
            model, loaders["val"], model_id, cell_features, device, split_payload["graph_policy"],
            set(split_payload["splits"]["val"]["cell_idx"]) - train_cells,
            set(split_payload["splits"]["val"]["drug_idx"]) - train_drugs,
        )
        val_metrics = full_metric_bundle(val_pred)["global"]
        pcc = float(val_metrics["pcc"])
        if not np.isfinite(pcc):
            pcc = -math.inf
        history.append({"epoch": epoch, "loss": float(np.mean(losses)), "val": val_metrics})
        if hasattr(model, "alpha"):
            alpha_trace.append(float(torch.sigmoid(model.alpha).detach().cpu()))
        scheduler.step(pcc)
        if pcc > best_pcc:
            best_pcc, best_epoch = pcc, epoch
            best_state = copy.deepcopy({key: value.detach().cpu() for key, value in model.state_dict().items()})
            patience_left = patience
        else:
            patience_left -= 1
            if patience_left <= 0:
                break
        for instance in list(tqdm._instances):
            if not instance.disable:
                instance.clear()
                instance.close()

    if best_state is None:
        best_state = copy.deepcopy({key: value.detach().cpu() for key, value in model.state_dict().items()})
    torch.save({
        "model_state_dict": best_state,
        "config": config,
        "epoch": best_epoch,
        "best_val_pcc": best_pcc,
    }, run.run_dir / "checkpoint-best.pt")
    model.load_state_dict(best_state)

    outputs, metrics = [], {}
    for split in ("val", "test"):
        if model_id in LEGACY_MODELS:
            allowed = set(train_drugs)
            if split_payload["graph_policy"] == "known-targets":
                allowed |= set(split_payload["splits"][split]["drug_idx"])
            set_model_dti_edges(model, apply_dti_visibility(full_graph, allowed))
        output = _predict_neural(
            model, loaders[split], model_id, cell_features, device, split_payload["graph_policy"],
            set(split_payload["splits"][split]["cell_idx"]) - train_cells,
            set(split_payload["splits"][split]["drug_idx"]) - train_drugs,
        )
        output["split"] = split
        output = _attach_metadata(output, frames[split])
        outputs.append(output)
        metrics[split] = _metrics_with_similarity(output)
    write_json(run.run_dir / "metrics.json", metrics)
    write_json(run.run_dir / "train-history.json", {"history": history, "alpha_trace": alpha_trace})
    _write_predictions(run.run_dir / "predictions.parquet", pd.concat(outputs, ignore_index=True))
    return metrics


def run_experiment(config: dict[str, Any], smoke: bool = False) -> dict:
    required = {"experiment_id", "model_id", "split_id", "fold", "seed"}
    if missing := required - set(config):
        raise KeyError(f"missing required config fields: {sorted(missing)}")
    config = resolve_config_templates(config)
    spec = RunSpec(
        experiment_id=str(config["experiment_id"]), model_id=str(config["model_id"]),
        split_id=str(config["split_id"]), fold=int(config["fold"]), seed=int(config["seed"]),
        variant=str(config.get("variant", "default")),
    )
    run = ExperimentRun(spec, config, Path(config["output_dir"]) if config.get("output_dir") else None)
    if not run.start(force=smoke):
        return json.loads((run.run_dir / "metrics.json").read_text(encoding="utf-8"))
    log_lines = [f"start={time.time()}", f"smoke={smoke}", json.dumps(config, ensure_ascii=False)]
    try:
        split_path = Path(config.get(
            "split_path", f"data/model/splits/{spec.split_id}/fold-{spec.fold:02d}.json"
        ))
        split_payload, frames = load_fold_pairs(split_path)
        if config["model_id"] in STATISTICAL_MODELS:
            metrics = _fit_statistical(config, frames, run)
        else:
            cell_features, drug_features, feature_meta = load_fold_features(
                split_payload, run.run_dir, allow_legacy_fallback=smoke
            )
            write_json(run.run_dir / "feature-manifest.json", feature_meta)
            if config["model_id"] in LINEAR_MODELS:
                metrics = _fit_linear(config, frames, cell_features, drug_features, run)
            else:
                loaders = build_fold_loaders(
                    frames, int(config.get("batch_size", 32)), int(config["seed"]),
                    int(config.get("num_workers", 0)), smoke=smoke,
                )
                metrics = _fit_neural(
                    config, split_payload, frames, loaders, cell_features, drug_features, run, smoke
                )
        run.complete({"metrics_path": "metrics.json"})
        log_lines.append("state=complete")
        return metrics
    except BaseException as exc:
        run.fail(exc)
        log_lines.extend(["state=failed", f"error={type(exc).__name__}: {exc}"])
        raise
    finally:
        atomic_write_text(run.run_dir / "run.log", "\n".join(log_lines) + "\n")
