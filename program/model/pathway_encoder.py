"""
Phase 3 Step 3: PathwayEncoder (通路活性预处理头)
=================================================

设计依据: Phase 3 计划 §5.2 (Phase 3 Step 3, 归 Phase 3 而非 Phase 4), §0.2.3 (Phase 3
smoke 一并验证).

职责
----
- 输入 batch 级 pathway_activity (B, 186) (来自 `cell_line_features.pt` 的 ssGSEA Z-score),
  输出 (B, pathway_dim) 供 Phase 4 直接 cat 进 fusion.
- Architecture: Linear(186, 64) -> BatchNorm1d -> ReLU -> Dropout(0.2).
- 输入已 Z-score, 不再额外标准化; BN1d 处理 batch 内部尺度增强训练稳定性.

⚠️ BatchNorm1d B=1 隐患 (Phase 3 计划 §0.3.5): 此 BN1d 作用在 (B, pathway_dim) 上, train
模式若 B=1 必触发 `expected more than 1 value per channel when training`. Phase 4 trainer
必须 drop_last=True + 评估切 eval(). 备选升级: GroupNorm(8, 64) 或 LayerNorm(64).
Phase 3 smoke 用 B>=4 验证形状, 此隐患留给 Phase 4 trainer 处理.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class PathwayEncoder(nn.Module):
    """Pathway activity (B, 186) → (B, pathway_dim) 简单预处理头.

    Parameters
    ----------
    pathway_in_dim : int
        ssGSEA 通路数 (Phase 1 实际 186).
    pathway_dim : int
        输出维度 (计划 §5.2 默认 64).
    dropout : float
        Dropout 概率 (默认 0.2).
    """

    def __init__(self, pathway_in_dim: int = 186, pathway_dim: int = 64, dropout: float = 0.2):
        super().__init__()
        self.pathway_in_dim = pathway_in_dim
        self.pathway_dim = pathway_dim
        self.net = nn.Sequential(
            nn.Linear(pathway_in_dim, pathway_dim),
            nn.BatchNorm1d(pathway_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

    def forward(self, pathway: torch.Tensor) -> torch.Tensor:
        """Parameters
        ----------
        pathway : (B, pathway_in_dim)    typically (B, 186)

        Returns
        -------
        (B, pathway_dim)                 typically (B, 64)
        """
        return self.net(pathway)


def _smoke_test() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[smoke] device={device}")

    enc = PathwayEncoder(pathway_in_dim=186, pathway_dim=64).to(device)
    n_params = sum(p.numel() for p in enc.parameters())
    print(f"[smoke] PathwayEncoder params={n_params:,} (expect ~12K)")
    assert 5_000 < n_params < 30_000, f"params {n_params} out of expected range"

    # eval 模式 (BN 用 running stats, B=1 安全)
    enc.eval()
    with torch.no_grad():
        out1 = enc(torch.randn(1, 186, device=device))
    assert out1.shape == (1, 64), out1.shape
    assert torch.isfinite(out1).all()
    print(f"[smoke] forward (B=1 eval) OK: shape={tuple(out1.shape)} std={out1.std():.4f}")

    # train 模式 B=4 forward + backward (BN1d B>=2 安全)
    enc.train()
    out = enc(torch.randn(4, 186, device=device))
    assert out.shape == (4, 64), out.shape
    loss = out.sum()
    loss.backward()
    n_total = sum(1 for _ in enc.parameters())
    n_grad = sum(1 for p in enc.parameters() if p.grad is not None)
    print(f"[smoke] forward (B=4 train) OK: shape={tuple(out.shape)}")
    print(f"[smoke] backward: {n_grad}/{n_total} params have grad")
    assert n_grad == n_total, f"only {n_grad}/{n_total} params have grad"

    print("[smoke] ALL OK | pathway_encoder.py")


if __name__ == "__main__":
    _smoke_test()