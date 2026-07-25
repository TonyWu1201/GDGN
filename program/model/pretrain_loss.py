"""
Step 3c: Pretraining Loss (多任务 BCE)
========================================

分边类型 BCE + lambda_dti 加权补偿 PPI:DTI = 168153:1124 的极不平衡 (§0.2.2).
pos_weight 参数缓解 pos:neg 比例 (1:1 时 = 1.0).
"""
from __future__ import annotations

import torch
import torch.nn.functional as F


def _to_pos_weight(pw, ref: torch.Tensor) -> torch.Tensor:
    if torch.is_tensor(pw):
        return pw.to(dtype=ref.dtype, device=ref.device)
    return torch.tensor(float(pw), dtype=ref.dtype, device=ref.device)


def pretrain_loss(
    pred_ppi: torch.Tensor,
    label_ppi: torch.Tensor,
    pred_dti: torch.Tensor,
    label_dti: torch.Tensor,
    lambda_dti: float = 10.0,
    pos_weight_ppi: float | torch.Tensor = 1.0,
    pos_weight_dti: float | torch.Tensor = 1.0,
    dti_margin_alpha: float = 0.0,
    dti_margin_gamma: float = 2.0,
) -> tuple[torch.Tensor, dict]:
    """Compute multi-task BCE loss + optional DTI pairwise margin (BPR-style).

    Parameters
    ----------
    pred_ppi : (P+N,) raw logits for PPI edges (positive first then negative)
    label_ppi : (P+N,) float {1.0, 0.0}
    pred_dti : (Q+M,) raw logits for DTI edges
    label_dti : (Q+M,) float {1.0, 0.0}
    lambda_dti : float
        weight on DTI loss (recommended 5-10).
    pos_weight_ppi, pos_weight_dti : float | Tensor
        passed to F.binary_cross_entropy_with_logits; PyTorch 2.x requires Tensor.
        float is auto-wrapped to scalar tensor.
    dti_margin_alpha : float
        weight on DTI pairwise margin (hinge) loss; 0 disables.
    dti_margin_gamma : float
        target margin between pos and neg scores (only used when alpha>0).

    Returns
    -------
    total_loss : scalar tensor (backward-able)
    stats : dict with floats {loss_total, loss_ppi, loss_dti, loss_dti_margin}
    """
    pw_ppi = _to_pos_weight(pos_weight_ppi, pred_ppi)
    pw_dti = _to_pos_weight(pos_weight_dti, pred_dti)
    loss_ppi = F.binary_cross_entropy_with_logits(pred_ppi, label_ppi, pos_weight=pw_ppi)
    bce_dti = F.binary_cross_entropy_with_logits(pred_dti, label_dti, pos_weight=pw_dti)

    margin_val = torch.zeros((), device=pred_dti.device)
    if dti_margin_alpha > 0:
        n_pos = int(label_dti.sum().item())
        n_neg = pred_dti.numel() - n_pos
        if n_pos > 0 and n_neg > 0:
            n_pairs = min(n_pos, n_neg)
            pos_scores = pred_dti[:n_pos][:n_pairs]
            neg_scores = pred_dti[n_pos:][:n_pairs]
            margin_val = F.relu(dti_margin_gamma - (pos_scores - neg_scores)).mean()

    loss_dti = bce_dti + dti_margin_alpha * margin_val
    total = loss_ppi + lambda_dti * loss_dti
    stats = {
        "loss_total": float(total.detach().item()),
        "loss_ppi": float(loss_ppi.detach().item()),
        "loss_dti": float(loss_dti.detach().item()),
        "loss_dti_margin": float(margin_val.detach().item()),
    }
    return total, stats


def make_labels(n_pos: int, n_neg: int, device: torch.device | str = "cpu") -> torch.Tensor:
    """Helper: 1s for pos, 0s for neg, matches sampler concat order."""
    return torch.cat([torch.ones(n_pos, device=device), torch.zeros(n_neg, device=device)], dim=0)


def _smoke_test() -> None:
    n_pos, n_neg = 100, 100
    pred_ppi = torch.randn(n_pos + n_neg, requires_grad=True)
    pred_dti = torch.randn(n_pos + n_neg, requires_grad=True)
    labels = make_labels(n_pos, n_neg)
    total, stats = pretrain_loss(pred_ppi, labels, pred_dti, labels, lambda_dti=10.0)
    total.backward()
    assert pred_ppi.grad is not None and pred_dti.grad is not None
    print(f"[smoke] lambda=10 -> total={stats['loss_total']:.4f} ppi={stats['loss_ppi']:.4f} dti={stats['loss_dti']:.4f}")

    pred_ppi_g = torch.tensor([10.0] * n_pos + [-10.0] * n_neg, requires_grad=True)
    pred_dti_g = torch.tensor([10.0] * n_pos + [-10.0] * n_neg, requires_grad=True)
    total, stats = pretrain_loss(pred_ppi_g, labels, pred_dti_g, labels, lambda_dti=1.0)
    assert stats["loss_ppi"] < 0.01 and stats["loss_dti"] < 0.01, "well-classified both should be small"
    print(f"[smoke] well-classified losses small: ppi={stats['loss_ppi']:.4f} dti={stats['loss_dti']:.4f}")

    pred_bad = torch.tensor([-10.0] * n_pos + [10.0] * n_neg, requires_grad=True)
    _, stats_bad = pretrain_loss(pred_bad, labels, pred_bad, labels, lambda_dti=10.0)
    assert stats_bad["loss_ppi"] > 10, "wrongly-classified ppi loss should be huge"
    print(f"[smoke] wrongly-classified ppi loss huge: ppi={stats_bad['loss_ppi']:.4f} (lambda*{stats_bad['loss_dti']:.2f}={stats_bad['loss_total']:.2f})")

    pw = torch.tensor([2.0])
    total, stats = pretrain_loss(pred_ppi, labels, pred_dti, labels, pos_weight_ppi=pw)
    print(f"[smoke] pos_weight=2 backward OK -> total={stats['loss_total']:.4f}")

    print("[smoke] ALL OK | pretrain_loss.py")


if __name__ == "__main__":
    _smoke_test()