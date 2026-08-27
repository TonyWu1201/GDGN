"""
Legacy grouped evaluation (not strict LODO/LOCO)
==================================================

This script applies a checkpoint trained with the random sample-pair split to
the complete response pool and groups its predictions. It does not retrain a
model after holding entities out, so its outputs are descriptive grouped
performance only. Use `program/run_strict_experiment.py` for Ver2 LDO/LCO.

职责
----
- **纯推理**: 用随机样本对训练的 ckpt 在完整池上预测，再按药物或癌种分组。
- 历史 `ldo_splits.pt` / `lco_splits.pt` 在这里仅充当分组索引；训练阶段没有留出实体。
- 因此所有输出只能称为 pooled grouped evaluation，不能衡量未见实体泛化。

机制
----
- 重建 Phase 1 的 `pairs` 全池 (66543 项, 顺序由 `ic50_matrix.csv` 行/列序确定)
  以匹配 `ldo_splits.pt`/`lco_splits.pt` 中的整数索引.
  Phase 1 仅持久化 sample_pairs_split.pt (stratified split 后的 train/val/test),
  未单独保存 pairs 全池, 故必须用同一 `collect_pairs()` 重建以保证索引对齐.
- **高效策略**: 全池推理一次得到 preds/trues (66543,), 各折用
  `fold['test']` 索引直接切片子集算指标, 不重复推理.
- 输出每折一行 + 汇总 (pooled / weighted-mean / median); 两种报告:
  `data/model/<model>/generalization_report.txt` (人读) + `.json` (机读).

不修改 Phase 1-5 任何代码 / 数据.

Usage
-----
全量评估 (单卡, 纯推理 ~分钟级别):
    uv run python program/model/eval_generalization.py --model gdgn \\
        --ckpt data/model/gdgn/best_model.pt \\
        --output_dir data/model/gdgn
    uv run python program/model/eval_generalization.py --model baseline_simple \\
        --ckpt data/model/baseline/best_model.pt \\
        --output_dir data/model/baseline

smoke 快速验证 (限 5 个 LODO + 5 个 LOCO 折 + 全池推理限 50 batch):
    uv run python program/model/eval_generalization.py --model gdgn \\
        --ckpt data/model/gdgn/best_model.pt \\
        --output_dir C:/temp/eval_gen_smoke --smoke

只跑 LODO 或 LOCO:
    uv run python program/model/eval_generalization.py --model gdgn \\
        --ckpt data/model/gdgn/best_model.pt --kinds lodo
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from tqdm import tqdm

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from program.model.dataset import inject_batch_omics, load_cell_line_features, load_hetero_graph
from program.model.train_gdgn import build_model, compute_metrics, bootstrap_ci, resolve_no_dti_drug_idx

PROC = _PROJECT_ROOT / "data" / "processed"
LDO_PT = PROC / "ldo_splits.pt"
LCO_PT = PROC / "lco_splits.pt"


def _rebuild_pairs_pool() -> list[dict]:
    """重建 Phase 1 的 `pairs` 全池 (顺序对 ldo/lco 索引至关重要).

    Phase 1 (program/preprocess/prepare_splits.py::main) 的 pairs 顺序由:
      1. pd.read_csv(IC50_PATH, index_col=0).iterrows()  -- 行序 = ic50_matrix.csv 行序
      2. row.items()  -- 列序 = ic50_matrix.csv 列序 (184)
      3. 跳过 NaN
    决定. 本函数直接调用同一 `collect_pairs` 函数 (同 import 路径), 保证顺序一致.

    返回 list[dict] 每项 {'cell_idx': int, 'drug_idx': int, 'ic50': float,
    'cancer_type': str}; 长度 66543.
    """
    from program.preprocess.prepare_splits import collect_pairs, get_cancer_types
    import pandas as pd

    IC50_PATH = PROC / "drug_sensitivity" / "ic50_matrix.csv"
    COMMON_CL_PATH = _PROJECT_ROOT / "data" / "common_cell_lines.csv"

    ic50 = pd.read_csv(IC50_PATH, index_col=0)
    common = pd.read_csv(COMMON_CL_PATH)
    cell_lines = common["Name"].astype(str).tolist()
    cell_idx = {name: i for i, name in enumerate(cell_lines)}
    drug_cids = [str(c) for c in ic50.columns]
    drug_idx = {cid: i for i, cid in enumerate(drug_cids)}
    cancer_of = get_cancer_types(cell_lines)

    pairs = collect_pairs(ic50, cell_idx, drug_idx, cancer_of)
    print(f"[gen] rebuilt pairs pool: len={len(pairs)} "
          f"(expected 66543 per Phase 1)")
    assert len(pairs) == 66543, f"pairs pool size {len(pairs)} != 66543 (Phase 1)"
    return pairs


class _PairIndexDataset(Dataset):
    """从 pairs 全池按 (cell_idx, drug_idx, ic50) 提供索引式样本.

    与 GDGNDataset 同 ABI (__getitem__ 返回 {cell_idx, drug_idx, y}),
    但不加载基图/分子图, 直接消费重建的 pairs list.
    """

    def __init__(self, pairs: list[dict]):
        self.pairs = pairs

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx: int) -> dict:
        p = self.pairs[idx]
        return {
            "cell_idx": int(p["cell_idx"]),
            "drug_idx": int(p["drug_idx"]),
            "y": torch.tensor(float(p["ic50"]), dtype=torch.float32),
        }


def _collate(batch: list[dict]) -> dict:
    cell_idx = torch.tensor([b["cell_idx"] for b in batch], dtype=torch.long)
    drug_idx = torch.tensor([b["drug_idx"] for b in batch], dtype=torch.long)
    y = torch.stack([b["y"] for b in batch], dim=0)
    return {"cell_idx": cell_idx, "drug_idx": drug_idx, "y": y}


@torch.no_grad()
def _predict_full_pool(
    model: torch.nn.Module,
    pairs: list[dict],
    clf: dict,
    device: torch.device,
    batch_size: int,
    max_batches: int | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """一次性在全池上推理得到 preds/trues/cell_idx/drug_idx (均长度 N=len(pairs)).

    max_batches=None 跑全池; smoke 用限批.
    """
    ds = _PairIndexDataset(pairs)
    loader = DataLoader(
        ds, batch_size=batch_size, shuffle=False, num_workers=0,
        collate_fn=_collate, pin_memory=torch.cuda.is_available(),
    )
    model.eval()
    pred_list, true_list, cell_list, drug_list = [], [], [], []
    for i, batch in enumerate(tqdm(loader)):
        if max_batches is not None and i >= max_batches:
            break
        cell_idx = batch["cell_idx"].to(device)
        drug_idx = batch["drug_idx"].to(device)
        y = batch["y"].to(device)
        omics = inject_batch_omics(clf, cell_idx)
        ic50_pred, _ = model(cell_idx, drug_idx, omics)
        pred_list.append(ic50_pred.detach().cpu().squeeze(-1))
        true_list.append(y.detach().cpu())
        cell_list.append(cell_idx.detach().cpu())
        drug_list.append(drug_idx.detach().cpu())
    return (
        torch.cat(pred_list).numpy(),
        torch.cat(true_list).numpy(),
        torch.cat(cell_list).numpy(),
        torch.cat(drug_list).numpy(),
    )


def _summarize_folds(
    folds: list[dict],
    preds: np.ndarray,
    trues: np.ndarray,
    drug_idx_full: np.ndarray,
    held_key: str,
    no_dti_set: set[int],
) -> tuple[list[dict], dict]:
    """对每个 fold 切片 test 索引, 计算 5 项指标.

    held_key: 'held_out_drug_idx' (LODO) 或 'held_out_cancer_type' (LOCO).
    返回 (per_fold_records, summary_dict).
    """
    per_fold = []
    pooled_preds = []
    pooled_trues = []
    pooled_no_dti_preds = []
    pooled_no_dti_trues = []

    for fi, fold in enumerate(folds):
        test_idx = fold["test"]
        if len(test_idx) == 0:
            continue
        idx = np.array(test_idx, dtype=np.int64)
        # 越界保护 (smoke 模式 max_batches 限制了全池推理时, 部分 fold 的 test_idx
        # 可能越界; 截断到 valid 索引, 空 fold 跳过)
        if idx.max() >= len(preds):
            valid = idx < len(preds)
            idx = idx[valid]
            if len(idx) == 0:
                continue
        p = preds[idx]
        t = trues[idx]
        d = drug_idx_full[idx]
        m = compute_metrics(p, t)

        # 7 无 DTI 子集 (仅 LODO 有意义: 若留出的药物恰在 no_dti_set 中,
        # 该折的 test 集全部样本即"未见 DTI 药物"的真实泛化)
        is_no_dti_fold = False
        no_dti_n = 0
        no_dti_pcc = float("nan")
        if held_key == "held_out_drug_idx":
            held_drug = int(fold["held_out_drug_idx"])
            if held_drug in no_dti_set:
                is_no_dti_fold = True
                no_dti_n = len(p)
                if no_dti_n >= 5:
                    no_dti_pcc = float(np.corrcoef(p, t)[0, 1]) if len(set(p)) > 1 else float("nan")

        per_fold.append({
            "fold_idx": fi,
            held_key: fold[held_key],
            "n_test": m["n"],
            "pcc": m["pcc"],
            "spearman": m["spearman"],
            "rmse": m["rmse"],
            "mae": m["mae"],
            "mse": m["mse"],
            "is_no_dti_drug": is_no_dti_fold,
            "no_dti_n": no_dti_n,
            "no_dti_pcc": no_dti_pcc,
        })

        pooled_preds.append(p)
        pooled_trues.append(t)
        # 无 DTI 药物的 fold 单独入 pooled_no_dti
        if is_no_dti_fold:
            pooled_no_dti_preds.append(p)
            pooled_no_dti_trues.append(t)

    pooled_preds_arr = np.concatenate(pooled_preds) if pooled_preds else np.zeros(0)
    pooled_trues_arr = np.concatenate(pooled_trues) if pooled_trues else np.zeros(0)
    pooled_m = compute_metrics(pooled_preds_arr, pooled_trues_arr) if len(pooled_preds_arr) > 5 else {}
    pooled_m["n_pooled"] = int(len(pooled_preds_arr))

    n_folds = len(per_fold)
    pccs = [r["pcc"] for r in per_fold if not np.isnan(r["pcc"])]
    rmses = [r["rmse"] for r in per_fold if not np.isnan(r["rmse"])]
    weighted_pcc_num = sum(r["pcc"] * r["n_test"] for r in per_fold if not np.isnan(r["pcc"]))
    weighted_pcc_den = sum(r["n_test"] for r in per_fold if not np.isnan(r["pcc"]))
    weighted_rmse_num = sum(r["rmse"] * r["n_test"] for r in per_fold if not np.isnan(r["rmse"]))
    weighted_rmse_den = sum(r["n_test"] for r in per_fold if not np.isnan(r["rmse"]))

    summary = {
        "n_folds_total": n_folds,
        "n_folds_with_metric": len(pccs),
        "pooled": pooled_m,
        "pcc_mean": float(np.mean(pccs)) if pccs else float("nan"),
        "pcc_median": float(np.median(pccs)) if pccs else float("nan"),
        "pcc_weighted_mean": float(weighted_pcc_num / weighted_pcc_den) if weighted_pcc_den else float("nan"),
        "pcc_std": float(np.std(pccs)) if pccs else float("nan"),
        "rmse_mean": float(np.mean(rmses)) if rmses else float("nan"),
        "rmse_median": float(np.median(rmses)) if rmses else float("nan"),
        "rmse_weighted_mean": float(weighted_rmse_num / weighted_rmse_den) if weighted_rmse_den else float("nan"),
    }

    # 无 DTI 药物子集跨折汇总
    if pooled_no_dti_preds:
        ndp = np.concatenate(pooled_no_dti_preds)
        ndt = np.concatenate(pooled_no_dti_trues)
        summary["no_dti_pooled"] = compute_metrics(ndp, ndt)
        summary["no_dti_n_folds"] = len(pooled_no_dti_preds)
    else:
        summary["no_dti_pooled"] = None
        summary["no_dti_n_folds"] = 0

    return per_fold, summary


def _render_report(
    model_name: str,
    ckpt_path: str,
    kinds_done: list[str],
    per_kind: dict[str, dict],
    total_infer_sec: float,
) -> str:
    rows = []
    rows.append("=" * 80)
    rows.append(f"Phase 6 Generalization Report | model={model_name}")
    rows.append(f"ckpt: {ckpt_path}")
    rows.append(f"kinds evaluated: {kinds_done}")
    rows.append(f"total inference time (full-pool): {total_infer_sec:.1f}s")
    rows.append("=" * 80)

    rows.append("")
    rows.append("[Note] 数据泄漏口径说明:")
    rows.append("  - 这是历史全样本池分组口径，不是严格 LODO/LOCO；模型训练阶段见过这些实体。")
    rows.append("  - 与 Phase 4 final_eval_report.txt 中 paired random split 不同:")
    rows.append("    Phase 4 的 cell/drug 同时出现在 train/val/test, 评估已见 cell/drug 新组合.")
    rows.append("  - 本评估**纯推理**, 不重训; 用 Phase 4 已训 ckpt 直接在留出集上算指标.")
    rows.append("  - 这些分组差异只描述实体条件下的表现，不能作为冷启动证据。")

    for kind, payload in per_kind.items():
        fold_label = "Leave-One-Drug-Out (LODO)" if kind == "lodo" else "Leave-One-Cancer-Out (LOCO)"
        held_key = "held_out_drug_idx" if kind == "lodo" else "held_out_cancer_type"

        rows.append("")
        rows.append("=" * 80)
        rows.append(f"[{fold_label}] 共 {payload['summary']['n_folds_total']} 折")
        rows.append("=" * 80)

        s = payload["summary"]
        rows.append("--- 汇总 ---")
        rows.append(f"  pooled (全 test 拼起来评估): "
                    f"PCC={s['pooled'].get('pcc', float('nan')):.4f} "
                    f"RMSE={s['pooled'].get('rmse', float('nan')):.4f} "
                    f"Spearman={s['pooled'].get('spearman', float('nan')):.4f} "
                    f"N={s['pooled'].get('n_pooled', 0)}")
        rows.append(f"  PCC  | mean={s['pcc_mean']:.4f}  median={s['pcc_median']:.4f}  "
                    f"weighted_mean={s['pcc_weighted_mean']:.4f}  std={s['pcc_std']:.4f}")
        rows.append(f"  RMSE | mean={s['rmse_mean']:.4f}  median={s['rmse_median']:.4f}  "
                    f"weighted_mean={s['rmse_weighted_mean']:.4f}")
        if s.get("no_dti_pooled"):
            nd = s["no_dti_pooled"]
            rows.append(f"  7 无 DTI 药物 folds (跨折 pooled): "
                        f"PCC={nd['pcc']:.4f}  RMSE={nd['rmse']:.4f}  "
                        f"N={nd['n']}  across {s['no_dti_n_folds']} folds")

        rows.append("")
        rows.append("--- 每折明细 ---")
        if kind == "lodo":
            rows.append(f"{'fold':<5}{'held_drug_idx':<16}{'N':>7}{'PCC':>9}{'Spearman':>10}"
                        f"{'RMSE':>9}{'MAE':>9}{'no_dti':>10}")
            for r in payload["per_fold"]:
                no_dti_tag = f"{r['no_dti_n']} (no_dti)" if r["is_no_dti_drug"] else "-"
                rows.append(
                    f"{r['fold_idx']:<5}{r[held_key]:<16}{r['n_test']:>7}"
                    f"{r['pcc']:>9.4f}{r['spearman']:>10.4f}"
                    f"{r['rmse']:>9.4f}{r['mae']:>9.4f}{no_dti_tag:>10}"
                )
        else:
            rows.append(f"{'fold':<5}{'held_cancer_type':<28}{'N':>7}{'PCC':>9}{'Spearman':>10}"
                        f"{'RMSE':>9}{'MAE':>9}")
            for r in payload["per_fold"]:
                rows.append(
                    f"{r['fold_idx']:<5}{str(r[held_key]):<28}{r['n_test']:>7}"
                    f"{r['pcc']:>9.4f}{r['spearman']:>10.4f}"
                    f"{r['rmse']:>9.4f}{r['mae']:>9.4f}"
                )

        # 95% CI bootstrap for pooled PCC / RMSE
        if payload["pooled_preds"].size > 5:
            ci = bootstrap_ci(payload["pooled_preds"], payload["pooled_trues"], n_boot=1000, seed=42)
            rows.append("")
            rows.append(f"  Pooled 95% CI bootstrap (n=1000): "
                        f"PCC=[{ci['pcc_lo']:.4f}, {ci['pcc_hi']:.4f}]  "
                        f"RMSE=[{ci['rmse_lo']:.4f}, {ci['rmse_hi']:.4f}]")

    return "\n".join(rows)


def evaluate_generalization(
    model_name: str,
    ckpt_path: str,
    output_dir: str | Path,
    kinds: list[str],
    batch_size: int,
    smoke: bool = False,
) -> dict:
    """按历史 LODO/LOCO 索引做全池分组推理；不重训，不代表严格泛化。

    Parameters
    ----------
    model_name : 'gdgn' | 'baseline_simple'
    ckpt_path : best_model.pt 路径 (Phase 4 已训模型, 用户服务器全量训练产出)
    output_dir : 报告输出目录 (默认 data/model/<model>/)
    kinds : ['lodo', 'lco'] 子集, 默认两个都跑
    batch_size : 全池推理 batch (推理可设较大, 推荐 64/128)
    smoke : 限 50 batch + 5 个折调试 (CPU 15s 内跑完)
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if smoke:
        n_folds_limit = 5
        # smoke 限 200 batch (B=4 = 800 样本). GDGN forward 慢 (8412 节点 GNN per step),
        # 全池推理 (66543) 在 CPU 上要数十分钟. smoke 用 max_batches=200 ≈ 30-60s 即可
        # 验证 pipeline 完整路径 (重建 pairs 全池 -> 加载 ckpt -> 全池推理 -> 切 fold 算指标
        # -> 写报告). 部分 fold 的 test_idx 越界时自动截断或跳过 (已加保护).
        # 全量评估时 max_batches=None 跑全池推理即可.
        max_batches = 200
        batch_size = min(batch_size, 8)  # GDGN 上 B=8 才不触发 BN B=1 隐患
        print(f"[gen] SMOKE: 限 {max_batches} batch (B={batch_size} = {max_batches*batch_size} 样本) + 5 folds展示")
    else:
        max_batches = None
        n_folds_limit = None
    print(f"[gen] device={device} smoke={smoke} bs={batch_size} kinds={kinds}")

    print(f"[gen] loading ckpt: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    config = ckpt["config"]

    hetero = load_hetero_graph(device)
    clf = load_cell_line_features(device)
    model = build_model(model_name, hetero, config, device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    print(f"[gen] model loaded (epoch={ckpt.get('epoch')}, val_metrics={ckpt.get('val_metrics')})")

    # 解析 7 无 DTI 药物 idx (LODO 子集统计用, 仅 gdgn 有此文件)
    try:
        no_dti_idx = resolve_no_dti_drug_idx()
        no_dti_set = set(no_dti_idx)
        print(f"[gen] no_DTI drug idx: {no_dti_idx} (n={len(no_dti_idx)})")
    except Exception as e:
        print(f"[gen] WARN: resolve_no_dti_drug_idx failed: {e}; using empty set")
        no_dti_set = set()

    # ---- 重建 pairs 全池 -------------------------------------------------
    print(f"[gen] rebuilding pairs pool ...")
    t0 = time.time()
    pairs = _rebuild_pairs_pool()
    print(f"[gen] pairs pool rebuilt in {time.time()-t0:.1f}s; len={len(pairs)}")

    # ---- 全池推理一次 ---------------------------------------------------
    print(f"[gen] running full-pool inference (B={batch_size}) ...")
    t0 = time.time()
    preds, trues, cell_full, drug_full = _predict_full_pool(
        model, pairs, clf, device, batch_size=batch_size, max_batches=max_batches,
    )
    infer_sec = time.time() - t0
    print(f"[gen] inference done in {infer_sec:.1f}s; "
          f"preds.shape={preds.shape} (smoke-max-batches={max_batches})")

    full_pooled = compute_metrics(preds, trues)
    print(f"[gen] full-pool (N={full_pooled['n']}) PCC={full_pooled['pcc']:.4f} "
          f"RMSE={full_pooled['rmse']:.4f}")

    # ---- 对每种 kind 跑 fold 切片 ---------------------------------------
    per_kind = {}
    kinds_done = []
    for kind in kinds:
        pt = LDO_PT if kind == "lodo" else LCO_PT
        if not pt.exists():
            print(f"[gen] WARN: {pt.name} not found, skipping {kind}")
            continue
        folds = torch.load(pt, weights_only=False)
        print(f"[gen] {kind}: loaded {len(folds)} folds from {pt}")
        if n_folds_limit is not None:
            folds = folds[:n_folds_limit]
            print(f"[gen]   smoke: limiting to first {n_folds_limit} folds")

        held_key = "held_out_drug_idx" if kind == "lodo" else "held_out_cancer_type"
        per_fold, summary = _summarize_folds(folds, preds, trues, drug_full, held_key, no_dti_set)
        per_kind[kind] = {
            "per_fold": per_fold,
            "summary": summary,
            "pooled_preds": preds,   # 全池拼起来; 用于 bootstrap CI
            "pooled_trues": trues,
        }
        kinds_done.append(kind)
        s = summary
        print(f"[gen] {kind} SUMMARY: pooled PCC={s['pooled'].get('pcc', float('nan')):.4f} "
              f"(mean={s['pcc_mean']:.4f} median={s['pcc_median']:.4f}), "
              f"n_folds={s['n_folds_total']}")

    # ---- 写报告 ---------------------------------------------------------
    report_text = _render_report(
        model_name=model_name,
        ckpt_path=ckpt_path,
        kinds_done=kinds_done,
        per_kind=per_kind,
        total_infer_sec=infer_sec,
    )
    report_path = output_dir / "grouped_evaluation_report.txt"
    report_path.write_text(report_text, encoding="utf-8")
    print(f"[gen] report -> {report_path}")

    # 机读 JSON
    json_path = output_dir / "grouped_evaluation_report.json"
    json_path.write_text(json.dumps({
        "model": model_name,
        "ckpt": ckpt_path,
        "kinds_done": kinds_done,
        "total_infer_sec": infer_sec,
        "full_pool": full_pooled,
        "per_kind": {k: {"per_fold": v["per_fold"], "summary": v["summary"]}
                     for k, v in per_kind.items()},
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[gen] json -> {json_path}")

    print(report_text)
    return {"report_path": str(report_path), "json_path": str(json_path)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=["gdgn", "baseline_simple", "cdr_baseline"], default="gdgn")
    ap.add_argument("--ckpt", type=str, required=True, help="best_model.pt 路径 (Phase 4 训练产出)")
    ap.add_argument("--output_dir", type=str, default=None,
                    help="报告输出目录 (默认 data/model/<model>/)")
    ap.add_argument("--kinds", type=str, default="lodo,lco",
                    help="逗号分隔, 可选 lodo / lco (默认两个都跑)")
    ap.add_argument("--batch_size", type=int, default=64,
                    help="全池推理 batch_size (推理路径默认 64, smoke 限 4)")
    ap.add_argument("--smoke", action="store_true",
                    help="限 50 batch + 5 折调试 (不上 GPU, CPU 15s 跑完)")
    args = ap.parse_args()

    output_dir = args.output_dir or f"data/model/{args.model}"
    kinds = [k.strip() for k in args.kinds.split(",") if k.strip()]
    evaluate_generalization(
        model_name=args.model,
        ckpt_path=args.ckpt,
        output_dir=output_dir,
        kinds=kinds,
        batch_size=args.batch_size,
        smoke=args.smoke,
    )


if __name__ == "__main__":
    main()
