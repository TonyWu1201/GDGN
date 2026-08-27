from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from Bio import SwissProt
from tqdm import tqdm
from transformers import EsmModel, EsmTokenizer

def parse_uniprot_dat(dat_path):
    """解析 UniProt DAT 文件，建立基因名 → 序列的映射"""
    gene_to_seq = {}

    with open(dat_path) as f:
        for record in tqdm(SwissProt.parse(f), desc = "parsing sequences"):
            gene_names = record.gene_name  # 可能是列表
            if gene_names:
                # 取第一个基因名
                primary_gene = gene_names[0].get('Name', '')
                if primary_gene:
                    gene_to_seq[primary_gene] = record.sequence

    return gene_to_seq

@torch.no_grad()
def get_protein_embedding(sequence, tokenizer, model, device, max_residues: int = 1022):
    """
    输入: 氨基酸序列字符串（如 'MRPSGTAGAA...'）
    输出: (1280,) 的 ESM-2 嵌入向量
    """
    # Tokenize：将氨基酸字母转为数字ID
    sequence = sequence[:max_residues]
    inputs = tokenizer(sequence, return_tensors="pt", add_special_tokens=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    hidden = model(**inputs).last_hidden_state[0]
    # ESM adds BOS/EOS tokens; only residue positions are pooled.
    return hidden[1:1 + len(sequence)].mean(dim=0).cpu().numpy()


def main() -> None:
    parser = argparse.ArgumentParser(description="Build mean-pooled ESM embeddings without special tokens")
    parser.add_argument("--uniprot", type=Path, default=Path("data/raw/protein_sequence/uniprot_sprot.dat"))
    parser.add_argument("--model", default="facebook/esm2_t33_650M_UR50D")
    parser.add_argument("--output-dir", type=Path, default=Path("data/processed/gene_embeddings"))
    args = parser.parse_args()
    print("loading model")
    model = EsmModel.from_pretrained(args.model)
    tokenizer = EsmTokenizer.from_pretrained(args.model)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device).eval()
    protein_sequences = parse_uniprot_dat(args.uniprot)
    gene_order = sorted(protein_sequences)
    embeddings = [
        get_protein_embedding(protein_sequences[gene], tokenizer, model, device)
        for gene in tqdm(gene_order, desc="processing proteins")
    ]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    np.save(args.output_dir / "esm2_gene_embeddings.npy", np.stack(embeddings))
    (args.output_dir / "gene_order.txt").write_text("\n".join(gene_order) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
