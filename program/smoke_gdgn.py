"""
Phase 4 Step 5: 整合 smoke test (GDGN/BaselineSimple 两路端到端 + ckpt 加载链路)
==============================================================================

设计依据: Phase 4 计划 §8 输出清单项 `program/smoke_gdgn.py`.

验收口径 (计划 §8.4):
- 两路模型可同时实例化 + 设备迁移成功 (Phase 4 评估项 S2 self._device 已处理)
- forward 后 ic50_pred shape (B, 1) + gdgn 路 attn_weights (B, 1, 8412)
- loss.backward() 所有权重 head 部分有梯度 (encoder 因 M2 死分支部分无梯度是已知行为)
- ckpt save + reload state_dict 不报错

CPU B=4 单 step 约 5s 内完成 (gdgn per-cell loop × 4 + drug_encoder_batch + cross_attn).
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import torch

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from program.model.dataset import inject_batch_omics, load_cell_line_features, load_hetero_graph
from program.model.baseline_simple import BaselineSimpleModel
from program.model.gdgn_model import GDGNModel


def _forward_backward(model, cell_idx, drug_idx, omics, name, expect_attn: bool):
    model.train()
    ic50_pred, attn = model(cell_idx, drug_idx, omics)
    assert ic50_pred.shape == (cell_idx.shape[0], 1), f"{name}: ic50_pred shape {ic50_pred.shape}"
    assert torch.isfinite(ic50_pred).all(), f"{name}: ic50_pred has NaN/Inf"
    if expect_attn:
        assert attn is not None and attn.shape == (cell_idx.shape[0], 1, 8412), \
            f"{name}: attn shape {getattr(attn, 'shape', None)} (expect (B, 1, 8412))"
        attn_sum = attn.sum(dim=-1)
        assert torch.allclose(attn_sum, torch.ones_like(attn_sum), atol=1e-4), \
            f"{name}: attention row sum != 1"
    else:
        assert attn is None, f"{name}: expect None attn"

    loss = ic50_pred.sum()
    loss.backward()
    n_total_head = sum(1 for _ in model.head_parameters())
    n_grad_head = sum(1 for p in model.head_parameters() if p.grad is not None and p.requires_grad)
    n_total_enc = sum(1 for _ in model.encoder_parameters())
    n_grad_enc = sum(1 for p in model.encoder_parameters() if p.grad is not None and p.requires_grad)
    return ic50_pred, n_total_enc, n_grad_enc, n_total_head, n_grad_head


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[smoke] device={device}")
    t0 = time.time()

    hetero = load_hetero_graph(device)
    clf = load_cell_line_features(device)

    cell_idx = torch.tensor([0, 1, 2, 3], device=device)
    drug_idx = torch.tensor([10, 50, 100, 150], device=device)
    omics = inject_batch_omics(clf, cell_idx)

    print("[smoke] === Path A: GDGNModel (use_main_drug_emb=False) ===")
    model_g = GDGNModel(hetero=hetero, pretrain_ckpt=None, device=device, freeze_encoder=False,
                        use_main_drug_emb=False).to(device)
    n_enc_g = sum(p.numel() for p in model_g.encoder_parameters())
    n_head_g = sum(p.numel() for p in model_g.head_parameters())
    print(f"[smoke] gdgn encoder={n_enc_g:,} head={n_head_g:,}")
    ic50_g, ne_g, ng_e, nh_g, ng_h = _forward_backward(model_g, cell_idx, drug_idx, omics, "gdgn-A", True)
    print(f"[smoke] gdgn-A forward OK: ic50={tuple(ic50_g.shape)} mean={ic50_g.mean():.4f}")
    print(f"[smoke] gdgn-A grad: encoder {ng_e}/{ne_g} (M2 know), head {ng_h}/{nh_g}")
    assert ng_h == nh_g, f"gdgn-A head only {ng_h}/{nh_g} have grad"

    print("[smoke] === Path B: GDGNModel (use_main_drug_emb=True) ===")
    model_gB = GDGNModel(hetero=hetero, pretrain_ckpt=None, device=device, freeze_encoder=False,
                         use_main_drug_emb=True).to(device)
    ic50_gB, ne_gB, ng_eB, nh_gB, ng_hB = _forward_backward(model_gB, cell_idx, drug_idx, omics, "gdgn-B", True)
    print(f"[smoke] gdgn-B forward OK: ic50={tuple(ic50_gB.shape)}")
    print(f"[smoke] gdgn-B grad: encoder {ng_eB}/{ne_gB}, head {ng_hB}/{nh_gB}")
    assert ng_hB == nh_gB, f"gdgn-B head only {ng_hB}/{nh_gB} have grad"

    print("[smoke] === Path C: BaselineSimpleModel ===")
    model_b = BaselineSimpleModel(hetero=hetero, device=device).to(device)
    n_enc_b = sum(p.numel() for p in model_b.encoder_parameters())
    n_head_b = sum(p.numel() for p in model_b.head_parameters())
    print(f"[smoke] baseline encoder={n_enc_b:,} head={n_head_b:,}")
    ic50_b, ne_b, ng_e_b, nh_b, ng_h_b = _forward_backward(model_b, cell_idx, drug_idx, omics, "baseline", False)
    print(f"[smoke] baseline forward OK: ic50={tuple(ic50_b.shape)} mean={ic50_b.mean():.4f}")
    print(f"[smoke] baseline grad: encoder {ng_e_b}/{ne_b}, head {ng_h_b}/{nh_b}")
    assert ng_h_b == nh_b, f"baseline head only {ng_h_b}/{nh_b} have grad"
    assert ng_e_b == ne_b, f"baseline encoder only {ng_e_b}/{ne_b} have grad"

    print("[smoke] === ckpt save + reload ===")
    tmp_dir = Path("data/model/gdgn")
    tmp_dir.mkdir(parents=True, exist_ok=True)
    tmp_ckpt = tmp_dir / "_smoke_gdgn_tmp.pt"
    torch.save({
        "epoch": 0,
        "model_state_dict": model_g.state_dict(),
        "config": {"model": "gdgn", "pretrain_ckpt": None,
                   "freeze_encoder": False, "use_main_drug_emb": False,
                   "num_heads": 4, "batch_size": 4},
        "val_metrics": {"pcc": 0.0},
    }, tmp_ckpt)
    print(f"[smoke] saved tmp ckpt: size={tmp_ckpt.stat().st_size:,} bytes")

    model_g2 = GDGNModel(hetero=hetero, pretrain_ckpt=None, device=device,
                         freeze_encoder=False, use_main_drug_emb=False).to(device)
    saved = torch.load(tmp_ckpt, map_location=device, weights_only=False)
    missing, unexpected = model_g2.load_state_dict(saved["model_state_dict"], strict=False)
    assert not missing and not unexpected, f"reload mismatch: missing={missing}, unexpected={unexpected}"
    model_g2.eval()
    with torch.no_grad():
        ic50_reload, _ = model_g2(cell_idx, drug_idx, omics)
    assert ic50_reload.shape == (4, 1)
    assert torch.isfinite(ic50_reload).all()
    print(f"[smoke] reload forward OK: ic50={tuple(ic50_reload.shape)}, "
          f"load_state_dict strict check passed (no missing/unexpected keys)")
    tmp_ckpt.unlink()
    print("[smoke] tmp ckpt cleared")

    elapsed = time.time() - t0
    print(f"[smoke] Phase 4 integration OK | elapsed={elapsed:.1f}s (期望 < 60s CPU)")


if __name__ == "__main__":
    main()