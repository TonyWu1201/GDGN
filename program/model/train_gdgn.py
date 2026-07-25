"""
Phase 4 Step 3: 主任务训练循环 train_gdgn.py
=============================================

设计依据: Phase 4 计划 §5.1-§5.3, §0.2.2 (param group lr 分组), §0.2.6
(5 指标 + validation 监控 PCC), §A.1 (默认配置), §0.5 checklist (MSE only).

职责
----
- 共用 trainer: gdgn (GDGNModel) 与 baseline_simple (BaselineSimpleModel) 通过
  ModelSpec 抽象走同一 train_loop, 两路都通过 (cell_idx, drug_idx, omics) ABI
  返回 (ic50_pred (B,1), attn_or_None)
- param group 分组: encoder_params (lr_encoder=1e-5) + head_params (lr_head=1e-3)
- 早停: patience=10, 监控 val_PCC (max)
- ReduceLROnPlateau: mode="max", patience=5, factor=0.5
- Checkpoint: best_model.pt (val_PCC 最高) + last_model.pt (每 epoch 末)
- 日志: train_log.json
- 95% CI bootstrap (n=1000, scipy.stats.pearsonr + np.percentile)
- 11 个无 DTI 药物 idx 解析: interactions_filtered.csv + drug_cid_to_idx.json
  缓存到 data/model/gdgn/no_dti_drug_idx.json
- final_eval_report.txt 生成函数 (训练完后用户手动调用 --post_eval 跑, 主流程不触发)

Phase 4 评估项落实
------------------
- M3 (aux_loss_weight): 仅留传 config 入口 (默认 0.0); 不实现 >0 分支. TODO 注释
  标明 Phase 6 启用需 import EdgeMaskSampler + pretrain_loss + 重新负采样.
- M5 (数据泄漏): write_final_report 内嵌一段注释说明配对随机分口径, LODO/LOCO 是
  泛化真口径留 Phase 6.
- S2 (self._device): GDGNModel / BaselineSimpleModel 已在 Phase 4 Step 2/4 处理.

不修改 Phase 1-3 任何代码 / 数据.

Usage
-----
smoke (限 5 batch):
    uv run python program/model/train_gdgn.py --model gdgn --smoke
全量训练 (服务器):
    uv run python program/model/train_gdgn.py --model gdgn --config data/model/gdgn/gdgn_config.json
最终评估 (训练后):
    uv run python program/model/train_gdgn.py --model gdgn --post_eval \\
        --ckpt data/model/gdgn/best_model.pt --output_dir data/model/gdgn
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn.functional as F
from scipy.stats import pearsonr, spearmanr
from tqdm import tqdm

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from program.model.baseline_simple import BaselineSimpleModel
from program.model.dataset import get_dataloaders, inject_batch_omics, load_cell_line_features, load_hetero_graph
from program.model.gdgn_model import GDGNModel

GDGN_DIR = _PROJECT_ROOT / "data" / "model" / "gdgn"
BASELINE_DIR = _PROJECT_ROOT / "data" / "model" / "baseline"
INTERACTIONS_CSV = _PROJECT_ROOT / "data" / "processed" / "drug_gene_interaction" / "interactions_filtered.csv"
DRUG_CID_TO_IDX_JSON = _PROJECT_ROOT / "data" / "processed" / "drug_cid_to_idx.json"
NO_DTI_IDX_JSON = GDGN_DIR / "no_dti_drug_idx.json"


def build_model(model_name: str, hetero, config: dict, device) -> torch.nn.Module:
    """根据 model_name 构造对应模型 (共用 ABI: forward(cell_idx, drug_idx, omics) -> (B,1), attn_None)."""
    if model_name == "gdgn":
        return GDGNModel(
            hetero=hetero,
            pretrain_ckpt=config.get("pretrain_ckpt", "data/model/pretrain/best_encoder.pt"),
            device=device,
            freeze_encoder=config.get("freeze_encoder", False),
            use_main_drug_emb=config.get("use_main_drug_emb", False),
            num_heads=config.get("num_heads", 4),
        ).to(device)
    elif model_name == "baseline_simple":
        return BaselineSimpleModel(hetero=hetero, device=device).to(device)
    raise ValueError(f"unknown model_name: {model_name}; expect 'gdgn' or 'baseline_simple'")


def resolve_no_dti_drug_idx(force_rebuild: bool = False) -> list[int]:
    """解析 11 个无 DTI 药物 drug_idx (评估子集统计用).

    数据来源:
        - data/processed/drug_gene_interaction/interactions_filtered.csv (unique cid)
        - data/processed/drug_cid_to_idx.json (cid -> drug_idx 映射, 184 项)
    返回: 184 全集 - 有 DTI 的 cid 对应 drug_idx = 11 个无 DTI idx 列表
    缓存到 data/model/gdgn/no_dti_drug_idx.json
    """
    if NO_DTI_IDX_JSON.exists() and not force_rebuild:
        return json.loads(NO_DTI_IDX_JSON.read_text())

    import pandas as pd
    df = pd.read_csv(INTERACTIONS_CSV)
    cid_with_dti = set(df["cid"].astype(int).unique())
    cid_to_idx = json.loads(DRUG_CID_TO_IDX_JSON.read_text())
    all_idx = set(cid_to_idx.values())
    idx_with_dti = {cid_to_idx[str(c)] for c in cid_with_dti if str(c) in cid_to_idx}
    no_dti_idx = sorted(all_idx - idx_with_dti)
    GDGN_DIR.mkdir(parents=True, exist_ok=True)
    NO_DTI_IDX_JSON.write_text(json.dumps(no_dti_idx))
    return no_dti_idx


def compute_metrics(preds: np.ndarray, trues: np.ndarray) -> dict:
    """PCC / Spearman / RMSE / MAE / MSE 5 项指标."""
    preds = np.asarray(preds, dtype=np.float64).ravel()
    trues = np.asarray(trues, dtype=np.float64).ravel()
    if len(preds) < 2:
        return {"pcc": float("nan"), "spearman": float("nan"),
                "rmse": float("nan"), "mae": float("nan"), "mse": float("nan"), "n": int(len(preds))}
    pcc = float(pearsonr(preds, trues)[0])
    sp = float(spearmanr(preds, trues)[0])
    mse = float(((preds - trues) ** 2).mean())
    return {
        "pcc": pcc,
        "spearman": sp,
        "rmse": float(np.sqrt(mse)),
        "mae": float(np.abs(preds - trues).mean()),
        "mse": mse,
        "n": int(len(preds)),
    }


def bootstrap_ci(preds: np.ndarray, trues: np.ndarray, n_boot: int = 1000, seed: int = 42) -> dict:
    """Bootstrap 95% CI for PCC / RMSE."""
    preds = np.asarray(preds, dtype=np.float64).ravel()
    trues = np.asarray(trues, dtype=np.float64).ravel()
    rng = np.random.RandomState(seed)
    n = len(preds)
    if n < 5:
        return {"pcc_lo": float("nan"), "pcc_hi": float("nan"),
                "rmse_lo": float("nan"), "rmse_hi": float("nan"), "n_boot": 0}
    pccs, rmses = [], []
    for _ in range(n_boot):
        idx = rng.randint(0, n, n)
        try:
            pccs.append(float(pearsonr(preds[idx], trues[idx])[0]))
        except ValueError:
            pass
        rmses.append(float(np.sqrt(((preds[idx] - trues[idx]) ** 2).mean())))
    return {
        "pcc_lo": float(np.percentile(pccs, 2.5)) if pccs else float("nan"),
        "pcc_hi": float(np.percentile(pccs, 97.5)) if pccs else float("nan"),
        "rmse_lo": float(np.percentile(rmses, 2.5)),
        "rmse_hi": float(np.percentile(rmses, 97.5)),
        "n_boot": len(pccs),
    }


def collect_preds_trues(model, loader, clf, device, max_batches: int | None = None) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """跑一遍 loader, 收集 preds/trues + 对应 cell_idx/drug_idx (供 11 无 DTI 子集统计).

    返回 (preds (N,), trues (N,), cell_idx (N,), drug_idx (N,))
    """
    model.eval()
    pred_list, true_list, cell_list, drug_list = [], [], [], []
    with torch.no_grad():
        for i, batch in enumerate(loader):
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
    return (torch.cat(pred_list).numpy(), torch.cat(true_list).numpy(),
            torch.cat(cell_list).numpy(), torch.cat(drug_list).numpy())


def evaluate(model, loader, clf, device, max_batches: int | None = None) -> dict:
    """跑一遍 loader, 计算 5 项指标. max_batches 限批 (smoke 用)."""
    preds, trues, _, _ = collect_preds_trues(model, loader, clf, device, max_batches=max_batches)
    return compute_metrics(preds, trues)


def train_gdgn(config: dict, smoke: bool = False) -> dict:
    """Phase 4 主任务训练循环 (共用 gdgn / baseline_simple).

    config keys (必填): model, pretrain_ckpt (gdgn only), output_dir
    config keys (可选填默认): lr_encoder, lr_head, batch_size, max_epochs,
        early_stopping_patience, lr_patience, lr_factor, weight_decay,
        grad_clip, freeze_encoder, use_main_drug_emb, num_heads, seed
    """
    torch.manual_seed(config.get("seed", 42))
    np.random.seed(config.get("seed", 42))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[train] device={device} smoke={smoke} model={config['model']}")

    if config["model"] == "gdgn" and smoke and config.get("pretrain_ckpt"):
        ckpt_path = Path(config["pretrain_ckpt"])
        if ckpt_path.exists():
            info = torch.load(ckpt_path, map_location="cpu", weights_only=False)
            kw = info.get("encoder_kwargs", {})
            if int(kw.get("hidden_dim", 0)) != 256 or int(info.get("epoch", -1)) < 10:
                print(f"[train] SMOKE NOTE: {ckpt_path} is smoke ckpt (hidden={kw.get('hidden_dim')}, "
                      f"epoch={info.get('epoch')}); fallback to pretrain_ckpt=None (random init). "
                      f"全量训练须先重跑 Phase 2 hidden=256 ckpt.")
                config["pretrain_ckpt"] = None
        else:
            print(f"[train] SMOKE NOTE: {ckpt_path} not found; fallback to pretrain_ckpt=None.")
            config["pretrain_ckpt"] = None

    hetero = load_hetero_graph(device)
    clf = load_cell_line_features(device)

    model = build_model(config["model"], hetero, config, device)
    n_enc = sum(p.numel() for p in model.encoder_parameters())
    n_head = sum(p.numel() for p in model.head_parameters())
    print(f"[train] encoder_params={n_enc:,} head_params={n_head:,}")

    optimizer = torch.optim.Adam(
        [
            {"params": model.encoder_parameters(), "lr": config["lr_encoder"]},
            {"params": model.head_parameters(), "lr": config["lr_head"]},
        ],
        weight_decay=config.get("weight_decay", 0.0),
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max",
        patience=config.get("lr_patience", 5),
        factor=config.get("lr_factor", 0.5),
    )

    loaders = get_dataloaders(
        batch_size=config["batch_size"], num_workers=0,
        pin_memory=torch.cuda.is_available(), load_hetero=False,
    )

    output_dir = Path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    best_path = output_dir / "best_model.pt"
    last_path = output_dir / "last_model.pt"
    log_path = output_dir / "train_log.json"
    cfg_dump = output_dir / "trained_config.json"

    max_epochs = config.get("max_epochs", 50)
    patience_left = config.get("early_stopping_patience", 10)
    grad_clip = config.get("grad_clip", 1.0)
    aux_weight = config.get("aux_loss_weight", 0.0)
    if aux_weight > 0:
        # TODO Phase 6 ablation: 启用需重 import EdgeMaskSampler + pretrain_loss + 重新负采样
        # 当前仅留接口位, 主流程 raise 避免静默错误
        raise NotImplementedError(
            "aux_loss_weight > 0 暂未实现 (Phase 4 计划 §0.2.1 方案 C 仅留接口); "
            "Phase 6 ablation 启用需 import EdgeMaskSampler + pretrain_loss 一并重做主图边预测.")

    max_steps = config.get("max_steps_per_epoch") if smoke else None
    if smoke and max_steps is None:
        max_steps = 5

    best_val_pcc = -np.inf
    log_records = []
    train_t0 = time.time()

    for epoch in range(max_epochs):
        model.train()
        epoch_t0 = time.time()
        train_preds, train_trues = [], []
        epoch_losses = []
        step_iter = tqdm(loaders["train"], desc=f"epoch {epoch} train", disable=smoke)
        for step, batch in enumerate(step_iter):
            if max_steps is not None and step >= max_steps:
                break
            cell_idx = batch["cell_idx"].to(device)
            drug_idx = batch["drug_idx"].to(device)
            y = batch["y"].to(device)
            omics = inject_batch_omics(clf, cell_idx)
            ic50_pred, _ = model(cell_idx, drug_idx, omics)
            loss = F.mse_loss(ic50_pred.squeeze(-1), y)

            optimizer.zero_grad()
            loss.backward()
            if grad_clip and grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()

            train_preds.append(ic50_pred.detach().cpu().squeeze(-1))
            train_trues.append(y.detach().cpu())
            epoch_losses.append(float(loss.item()))
            if smoke:
                print(f"  [epoch {epoch}] step {step+1}/{max_steps} loss={loss.item():.4f}")

        train_metrics = compute_metrics(torch.cat(train_preds).numpy(), torch.cat(train_trues).numpy())
        train_metrics["mse_running"] = float(np.mean(epoch_losses)) if epoch_losses else float("nan")

        val_metrics = evaluate(model, loaders["val"], clf, device,
                            max_batches=(max_steps if smoke else None))

        epoch_time = time.time() - epoch_t0
        record = {
            "epoch": epoch,
            "train": train_metrics,
            "val": val_metrics,
            "lr_encoder": optimizer.param_groups[0]["lr"],
            "lr_head": optimizer.param_groups[1]["lr"],
            "epoch_time": epoch_time,
        }
        log_records.append(record)
        print(f"[epoch {epoch}] train_pcc={train_metrics['pcc']:.4f} mse={train_metrics['mse_running']:.4f} | "
              f"val_pcc={val_metrics['pcc']:.4f} val_rmse={val_metrics['rmse']:.4f} | "
              f"dt={epoch_time:.1f}s")

        scheduler.step(val_metrics["pcc"])

        if val_metrics["pcc"] > best_val_pcc:
            best_val_pcc = val_metrics["pcc"]
            patience_left = config.get("early_stopping_patience", 10)
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "val_metrics": val_metrics,
                "config": config,
            }, best_path)
            print(f"  [ckpt] new best val_pcc={best_val_pcc:.4f} -> {best_path}")
        else:
            patience_left -= 1
            print(f"  [early-stop] patience left = {patience_left}")
            if patience_left <= 0:
                print("[early-stop] triggered; stopping")
                break

        torch.save({"epoch": epoch, "model_state_dict": model.state_dict(), "config": config}, last_path)
        log_path.write_text(json.dumps({"records": log_records, "config": config}, ensure_ascii=False, indent=2))
        cfg_dump.write_text(json.dumps(config, ensure_ascii=False, indent=2))

        if smoke:
            print(f"[smoke] hit max_epochs/smoke after epoch {epoch}")
            break

    total_time = time.time() - train_t0
    log_path.write_text(json.dumps({"records": log_records, "config": config, "total_time": total_time}, ensure_ascii=False, indent=2))
    print(f"[train] DONE best_val_pcc={best_val_pcc:.4f} total_time={total_time:.1f}s log={log_path}")
    return {"best_val_pcc": best_val_pcc, "n_epochs_run": len(log_records), "log_path": str(log_path)}


def write_final_report(
    model_name: str,
    ckpt_path: str | Path,
    output_dir: str | Path,
    bootstrap_n: int = 1000,
    max_batches_per_split: int | None = None,
) -> dict:
    """训练后调用: 加载 best ckpt, 在 train/val/test 评估 + 7 无 DTI 子集 + 95% CI.

    输出: <output_dir>/final_eval_report.txt

    max_batches_per_split: 每集合最多跑多少 batch (smoke 用 50); None=全量.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    config = ckpt["config"]
    hetero = load_hetero_graph(device)
    clf = load_cell_line_features(device)
    model = build_model(model_name, hetero, config, device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    loaders = get_dataloaders(batch_size=config["batch_size"], num_workers=0,
                              pin_memory=torch.cuda.is_available(), load_hetero=False)

    splits = {}
    no_dti_idx = resolve_no_dti_drug_idx()
    no_dti_set = set(no_dti_idx)
    for sp in ["train", "val", "test"]:
        preds, trues, _, drug_idx = collect_preds_trues(model, loaders[sp], clf, device,
                                                       max_batches=max_batches_per_split)
        splits[sp] = {"preds": preds, "trues": trues, "drug_idx": drug_idx}

    rows = []
    rows.append("=" * 80)
    rows.append(f"Phase 4 Final Evaluation Report | model={model_name}")
    rows.append(f"ckpt: {ckpt_path}")
    rows.append(f"best epoch: {ckpt.get('epoch')}")
    rows.append("=" * 80)
    rows.append("[Table 1] 主指标: train/val/test PCC / Spearman / RMSE / MAE / MSE")
    rows.append(f"{'split':<8}{'PCC':>10}{'Spearman':>10}{'RMSE':>10}{'MAE':>10}{'MSE':>10}{'N':>8}")
    for sp in ["train", "val", "test"]:
        m = compute_metrics(splits[sp]["preds"], splits[sp]["trues"])
        rows.append(f"{sp:<8}{m['pcc']:>10.4f}{m['spearman']:>10.4f}{m['rmse']:>10.4f}"
                    f"{m['mae']:>10.4f}{m['mse']:>10.4f}{m['n']:>8}")

    rows.append("")
    rows.append(f"[Table 2] {len(no_dti_idx)} 个无 DTI 药物子集 PCC (Phase 3 §10.4 tracking)")
    rows.append(f"{'split':<8}{'PCC':>10}{'RMSE':>10}{'N':>8}{'no_dti_n':>10}")
    for sp in ["train", "val", "test"]:
        d = splits[sp]["drug_idx"]
        mask = np.array([int(x) in no_dti_set for x in d], dtype=bool)
        n_no = int(mask.sum())
        if n_no > 5:
            m = compute_metrics(splits[sp]["preds"][mask], splits[sp]["trues"][mask])
            rows.append(f"{sp:<8}{m['pcc']:>10.4f}{m['rmse']:>10.4f}{m['n']:>8}{n_no:>10}")
        else:
            rows.append(f"{sp:<8}{'N/A':>10}{'N/A':>10}{int(sum(mask)):>8}{n_no:>10} (too few)")

    rows.append("")
    rows.append("[Table 3] 95% CI bootstrap (n=1000) for test PCC / RMSE")
    ci = bootstrap_ci(splits["test"]["preds"], splits["test"]["trues"], n_boot=bootstrap_n)
    rows.append(f"PCC  95% CI: [{ci['pcc_lo']:.4f}, {ci['pcc_hi']:.4f}]  (n_boot={ci['n_boot']})")
    rows.append(f"RMSE 95% CI: [{ci['rmse_lo']:.4f}, {ci['rmse_hi']:.4f}]")

    rows.append("")
    rows.append("[Table 4] 训练曲线摘要")
    log_path = output_dir / "train_log.json"
    if log_path.exists():
        log = json.loads(log_path.read_text())
        recs = log.get("records", [])
        if recs:
            best_rec = max(recs, key=lambda r: r["val"]["pcc"])
            rows.append(f"best epoch: {best_rec['epoch']}  val_pcc={best_rec['val']['pcc']:.4f}")
            rows.append(f"total epochs logged: {len(recs)}")
            if "total_time" in log:
                rows.append(f"total train time: {log['total_time']:.1f}s")
            rows.append(f"final lr_encoder={recs[-1]['lr_encoder']:.2e} lr_head={recs[-1]['lr_head']:.2e}")

    rows.append("")
    rows.append("[Note] 数据泄漏口径:")
    rows.append("  - 主指标基于配对随机分 80/10/10 (Phase 1 sample_pairs_split.pt).")
    rows.append("  - 同一 cell / drug 会同时出现在 train/val/test, 评估的是已见 cell/drug 的新组合.")
    rows.append("  - LODO / LOCO 是真正的泛化口径, Phase 6 单独跑 (Phase 1 已备 ldo_splits.pt / lco_splits.pt).")

    report_text = "\n".join(rows)
    report_path = output_dir / "final_eval_report.txt"
    report_path.write_text(report_text, encoding="utf-8")
    print(f"[final] report -> {report_path}")
    print(report_text)
    return {"report_path": str(report_path)}


def default_config(model_name: str) -> dict:
    if model_name == "gdgn":
        return {
            "model": "gdgn",
            "pretrain_ckpt": "data/model/pretrain/best_encoder.pt",
            "output_dir": "data/model/gdgn",
            "batch_size": 32, "max_epochs": 50,
            "early_stopping_patience": 10, "lr_patience": 5, "lr_factor": 0.5,
            "lr_encoder": 1e-5, "lr_head": 1e-3,
            "weight_decay": 0.0, "grad_clip": 1.0,
            "freeze_encoder": False, "use_main_drug_emb": False,
            "aux_loss_weight": 0.0, "num_heads": 4,
            "seed": 42,
        }
    elif model_name == "baseline_simple":
        return {
            "model": "baseline_simple",
            "output_dir": "data/model/baseline",
            "batch_size": 32, "max_epochs": 50,
            "early_stopping_patience": 10, "lr_patience": 5, "lr_factor": 0.5,
            "lr_encoder": 1e-5, "lr_head": 1e-3,
            "weight_decay": 0.0, "grad_clip": 1.0,
            "seed": 42,
        }
    raise ValueError(f"unknown model: {model_name}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=["gdgn", "baseline_simple"], default="gdgn")
    ap.add_argument("--config", type=str, default=None, help="path to JSON config (覆盖 default_config)")
    ap.add_argument("--output_dir", type=str, default=None, help="覆盖 config 里的 output_dir")
    ap.add_argument("--smoke", action="store_true", help="限 5 batch / 1 epoch 调试")
    ap.add_argument("--post_eval", action="store_true", help="训练后单独跑 final_eval_report")
    ap.add_argument("--ckpt", type=str, default=None, help="post_eval 时指定 ckpt 路径")
    ap.add_argument("--max_epochs", type=int, default=None)
    ap.add_argument("--batch_size", type=int, default=None)
    ap.add_argument("--lr_encoder", type=float, default=None)
    ap.add_argument("--lr_head", type=float, default=None)
    ap.add_argument("--freeze_encoder", action="store_true", default=None)
    ap.add_argument("--use_main_drug_emb", action="store_true", default=None)
    ap.add_argument("--num_heads", type=int, default=None)
    ap.add_argument("--aux_loss_weight", type=float, default=None)
    ap.add_argument("--build_no_dti_cache", action="store_true",
                    help="只解析并缓存 11 无 DTI idx 到 no_dti_drug_idx.json, 不训练")
    args = ap.parse_args()

    if args.build_no_dti_cache:
        idx = resolve_no_dti_drug_idx(force_rebuild=True)
        print(f"[no_dti] {len(idx)} drugs without DTI: {idx}")
        print(f"[no_dti] cached to {NO_DTI_IDX_JSON}")
        return

    if args.config:
        config = json.loads(Path(args.config).read_text())
        config.setdefault("model", args.model)
    else:
        config = default_config(args.model)

    if args.output_dir:
        config["output_dir"] = args.output_dir
    if args.max_epochs is not None:
        config["max_epochs"] = args.max_epochs
    if args.batch_size is not None:
        config["batch_size"] = args.batch_size
    if args.lr_encoder is not None:
        config["lr_encoder"] = args.lr_encoder
    if args.lr_head is not None:
        config["lr_head"] = args.lr_head
    if args.freeze_encoder is not None:
        config["freeze_encoder"] = args.freeze_encoder
    if args.use_main_drug_emb is not None:
        config["use_main_drug_emb"] = args.use_main_drug_emb
    if args.num_heads is not None:
        config["num_heads"] = args.num_heads
    if args.aux_loss_weight is not None:
        config["aux_loss_weight"] = args.aux_loss_weight

    if args.smoke:
        config["max_epochs"] = 1
        config["max_steps_per_epoch"] = 2
        config["batch_size"] = 4

    if args.post_eval:
        if args.ckpt is None:
            ckpt = Path(config["output_dir"]) / "best_model.pt"
        else:
            ckpt = Path(args.ckpt)
        if args.smoke:
            config["batch_size"] = 4
            max_batches = 10
        else:
            max_batches = None
        write_final_report(args.model, ckpt, config["output_dir"],
                            max_batches_per_split=max_batches)
        return

    train_gdgn(config, smoke=args.smoke)


if __name__ == "__main__":
    main()