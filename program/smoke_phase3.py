"""
Phase 3 Step 4: 整合 smoke test (三编码器协同 + ckpt 加载链路验证)
==================================================================

设计依据: Phase 3 计划 §6.2 (伪代码示意), §6.3 (omics 张量方向不转置, 直接
复用 Phase 2 omics_to_per_gene (B, 8412, 4)).

验收口径 (计划 §6.4, 不训练):
- 三编码器同时实例化 + 设备迁移成功
- GeneEncoder     -> gene_emb (2, 8412, 256) + main_drug_emb (2, 184, 256)
- DrugEncoder     -> drug_emb (2, 128)
- PathwayEncoder  -> pathway_emb (2, 64)
- 三者相加 `loss = .sum()` backward -> 所有非冻结权重 grad 非空
- runtime < 30s CPU, 显存 < 2GB GPU
- 额外: ckpt 加载版本 vs 随机初始化版本 std 对照 (不为训练, 仅为 sanity baseline)
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import torch

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from program.model.dataset import (
    inject_batch_omics,
    load_cell_line_features,
    load_hetero_graph,
)
from program.model.drug_encoder import (
    DrugEncoder,
    drug_encoder_forward_batch,
    load_drug_mol_graphs,
)
from program.model.gene_encoder import GeneEncoder
from program.model.pathway_encoder import PathwayEncoder
from program.model.pretrain import omics_to_per_gene


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[smoke] device={device}")
    t0 = time.time()

    # ===== 1. 加载 Phase 1/2 产出 =====
    print("[smoke] loading hetero_graph_base.pt + cell_line_features.pt + drug_mol_graphs.pkl ...")
    hetero = load_hetero_graph(device)
    clf = load_cell_line_features(device)
    drug_mol_graphs = load_drug_mol_graphs()
    n_genes = int(hetero["gene"].num_nodes)
    n_drugs = int(hetero["drug"].num_nodes)
    n_pathways = int(clf["pathway_activity"].shape[1])
    print(f"[smoke] hetero gene.x={tuple(hetero['gene'].x.shape)} drug.x={tuple(hetero['drug'].x.shape)}")
    print(f"[smoke] pathway in-dim={n_pathways} n_mol_graphs={len(drug_mol_graphs)}")

    gene_static = hetero["gene"].x
    drug_x = hetero["drug"].x

    # ===== 2. 实例化三编码器 =====
    print("[smoke] instantiating 3 encoders ...")
    gene_enc = GeneEncoder(
        hetero,
        pretrain_ckpt="data/model/pretrain/best_encoder.pt",
        device=device,
        freeze=False,
    ).to(device)
    drug_enc = DrugEncoder(atom_dim=75, hidden=256, out_dim=128).to(device)
    pathway_enc = PathwayEncoder(pathway_in_dim=n_pathways, pathway_dim=64).to(device)

    n_gene_params = sum(p.numel() for p in gene_enc.parameters())
    n_drug_params = sum(p.numel() for p in drug_enc.parameters())
    n_pathway_params = sum(p.numel() for p in pathway_enc.parameters())
    print(f"[smoke] params: gene={n_gene_params:,} drug={n_drug_params:,} pathway={n_pathway_params:,}")
    print(f"[smoke] gene buffers: ppi_fwd={tuple(gene_enc.ppi_fwd.shape)} ppi_rev={tuple(gene_enc.ppi_rev.shape)} "
          f"dti_fwd={tuple(gene_enc.dti_fwd.shape)} dti_rev={tuple(gene_enc.dti_rev.shape)}")

    # ===== 3. 假样本 batch=2 =====
    cell_idx = torch.tensor([0, 1], device=device)
    drug_idx = torch.tensor([10, 50], device=device)
    omics = inject_batch_omics(clf, cell_idx)            # {'expr','mut','cnv','meth','pathway'} 各 (2, 8412)+pathway (2, 186)
    omics_4 = omics_to_per_gene(omics).to(device)        # 复用 Phase 2 (B, 8412, 4); 不转置
    assert omics_4.shape == (2, n_genes, 4), f"omics_4 shape {omics_4.shape}"
    pathway_inputs = omics["pathway"].to(device)
    assert pathway_inputs.shape == (2, n_pathways)

    # ===== 4. 三编码器协同 forward =====
    print("[smoke] forward 3 encoders with B=2 ...")
    # 都用 train mode 前向 + immediate backward (单次相位, 远离 BN1d B=1 隐患)
    gene_enc.train()
    drug_enc.train()
    pathway_enc.train()

    gene_emb, main_drug_emb = gene_enc(gene_static, drug_x, omics_4)
    drug_emb = drug_encoder_forward_batch(drug_enc, drug_idx, drug_mol_graphs, device)
    pathway_emb = pathway_enc(pathway_inputs)

    print(f"[smoke] gene_emb={tuple(gene_emb.shape)} "
          f"main_drug_emb={tuple(main_drug_emb.shape)} "
          f"drug_emb={tuple(drug_emb.shape)} "
          f"pathway_emb={tuple(pathway_emb.shape)}")
    assert gene_emb.shape == (2, n_genes, 256), gene_emb.shape
    assert main_drug_emb.shape == (2, n_drugs, 256), main_drug_emb.shape
    assert drug_emb.shape == (2, 128), drug_emb.shape
    assert pathway_emb.shape == (2, 64), pathway_emb.shape
    assert torch.isfinite(gene_emb).all() and torch.isfinite(drug_emb).all() \
        and torch.isfinite(pathway_emb).all(), "forward produced NaN/Inf"

    # ===== 5. 链式 backward (sanity grad check) =====
    print("[smoke] chained backward (loss = gene.sum + drug.sum + pathway.sum) ...")
    loss = gene_emb.sum() + drug_emb.sum() + pathway_emb.sum()
    loss.backward()

    def _count_grads(m, name):
        n_total = sum(1 for _ in m.parameters())
        n_grad = sum(1 for p in m.parameters() if p.grad is not None and p.requires_grad)
        print(f"[smoke] {name}: {n_grad}/{n_total} params have grad (requires_grad truthy)")
        return n_grad == n_total

    ok_a = _count_grads(gene_enc, "gene_enc")
    ok_b = _count_grads(drug_enc, "drug_enc")
    ok_c = _count_grads(pathway_enc, "pathway_enc")
    assert ok_a and ok_b and ok_c, "some encoder missing grads"

    # ===== 6. Phase 6 ablation baseline: 随机初始化 GeneEncoder 对照 =====
    print("[smoke] Phase 6 baseline: GeneEncoder(hetero, pretrain_ckpt=None) ...")
    gene_enc_random = GeneEncoder(hetero, pretrain_ckpt=None, device=device, freeze=False).to(device)
    gene_enc_random.eval()
    with torch.no_grad():
        g_random, d_random = gene_enc_random(gene_static, drug_x, omics_4)
    print(f"[smoke] random-init gene_emb std={g_random.std():.4f} "
          f"(pretrained std={gene_emb.std():.4f})")
    assert g_random.shape == gene_emb.shape

    elapsed = time.time() - t0
    print(f"[smoke] Phase 3 integration OK | elapsed={elapsed:.1f}s (期望 < 30s CPU)")


if __name__ == "__main__":
    main()