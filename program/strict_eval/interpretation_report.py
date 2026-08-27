from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from program.model.interpret import load_dti_targets, load_gene_order

from .faithfulness import jaccard_stability, target_retrieval_metrics, target_vs_control_test
from .io import atomic_write_text
from .statistics import benjamini_hochberg


def collect_cases(input_dirs: list[str | Path]) -> pd.DataFrame:
    gene_order = load_gene_order()
    gene_to_idx = {gene: idx for idx, gene in enumerate(gene_order)}
    targets_by_cid = load_dti_targets()
    rows = []
    rankings: dict[tuple[int, int], list[list[int]]] = {}
    for seed_index, directory in enumerate(input_dirs):
        for path in Path(directory).glob("**/drug*_cell*.json"):
            case = json.loads(path.read_text(encoding="utf-8"))
            if "ig_per_gene" not in case:
                continue
            importance = np.abs(np.asarray(case["ig_per_gene"], dtype=np.float64))
            cid = int(case["cid"])
            target_genes = targets_by_cid.get(cid, set())
            target_mask = np.asarray([gene in target_genes for gene in gene_order], dtype=bool)
            metrics = target_retrieval_metrics(importance, target_mask)
            control = ~target_mask
            test = target_vs_control_test(importance, target_mask, control)
            curve = case.get("deletion_curve") or {}
            fractions = np.asarray(curve.get("fractions", []), dtype=float)
            top_drop = np.asarray(curve.get("top_drop", []), dtype=float)
            random_drop = np.asarray(curve.get("random_drop", []), dtype=float)
            matched_drop = np.asarray(curve.get("matched_drop", []), dtype=float)
            top_vs_random = (
                float(np.trapezoid(top_drop - random_drop, fractions)) if len(fractions) else float("nan")
            )
            top_vs_matched = (
                float(np.trapezoid(top_drop - matched_drop, fractions)) if len(fractions) else float("nan")
            )
            key = (int(case["drug_idx"]), int(case["cell_idx"]))
            rankings.setdefault(key, []).append(np.argsort(-importance).tolist())
            rows.append({
                "seed_index": seed_index,
                "drug_idx": key[0],
                "cell_idx": key[1],
                "cid": cid,
                **metrics,
                "target_vs_control_p": case.get("target_vs_matched_p", test["p_value"]),
                "ig_convergence_delta": case.get("ig_convergence_delta"),
                "ig_baseline_spearman": case.get("ig_baseline_spearman"),
                "comprehensiveness": case.get("comprehensiveness"),
                "sufficiency": case.get("sufficiency"),
                "top_vs_random_deletion_auc": top_vs_random,
                "top_vs_matched_deletion_auc": top_vs_matched,
                "attention_mean_shuffle_change": (case.get("attention_randomization") or {}).get(
                    "mean_shuffle_change"
                ),
            })
    frame = pd.DataFrame(rows)
    if not frame.empty:
        valid = frame["target_vs_control_p"].fillna(1.0).to_numpy()
        frame["target_vs_control_q"] = benjamini_hochberg(valid)
        stability = {key: jaccard_stability(value)["mean_jaccard"] for key, value in rankings.items()}
        frame["seed_jaccard"] = [stability[(row.drug_idx, row.cell_idx)] for row in frame.itertuples()]
    return frame


def write_interpretation_report(input_dirs: list[str | Path], output_dir: str | Path) -> tuple[Path, Path]:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    frame = collect_cases(input_dirs)
    csv_path = output / "interpretation-metrics.csv"
    frame.to_csv(csv_path, index=False)
    lines = ["# Interpretation validation report", ""]
    if frame.empty:
        lines.append("No eligible attribution cases were found.")
    else:
        with_target = frame[frame["n_targets"] > 0]
        lines.extend([
            f"- Cases: {len(frame)}",
            f"- Cases with mapped targets: {len(with_target)}",
            f"- Mean target AUPRC: {with_target['auprc'].mean():.4f}",
            f"- Mean target NDCG: {with_target['ndcg'].mean():.4f}",
            f"- FDR-significant target-vs-control cases: {int((with_target['target_vs_control_q'] < 0.05).sum())}",
            f"- Mean cross-seed top-20 Jaccard: {frame['seed_jaccard'].mean():.4f}",
            f"- Cases where top deletion exceeds random control (AUC): {int((frame['top_vs_random_deletion_auc'] > 0).sum())}/{len(frame)}",
            f"- Cases where top deletion exceeds matched control (AUC): {int((frame['top_vs_matched_deletion_auc'] > 0).sum())}/{len(frame)}",
            f"- Mean IG baseline rank correlation: {frame['ig_baseline_spearman'].mean():.4f}",
            "",
            "Attention or attribution is not called a biological explanation unless intervention fields are present and outperform matched controls.",
        ])
    report_path = output / "可解释性验证报告.md"
    atomic_write_text(report_path, "\n".join(lines) + "\n")
    return csv_path, report_path
