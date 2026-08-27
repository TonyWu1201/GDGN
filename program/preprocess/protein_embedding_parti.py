"""Build per-gene ESM chunk embeddings without truncating long proteins.

The output is a cache for the trainable `ChunkAttentionPooler`; it is not a
formal fixed gene embedding until a task-trained pooler checkpoint is applied.
Special tokens are explicitly excluded from every chunk mean.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from Bio import SwissProt
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer


def load_sequences(dat_path: Path) -> dict[str, str]:
    result: dict[str, str] = {}
    with dat_path.open(encoding="utf-8", errors="ignore") as handle:
        for record in SwissProt.parse(handle):
            for item in record.gene_name or []:
                name = item.get("Name") if isinstance(item, dict) else None
                if name:
                    result.setdefault(name, record.sequence)
    return result


@torch.no_grad()
def embed_chunks(sequence: str, tokenizer, model, device, chunk_size: int, overlap: int) -> torch.Tensor:
    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size")
    chunks = []
    step = chunk_size - overlap
    for start in range(0, len(sequence), step):
        subsequence = sequence[start:start + chunk_size]
        if not subsequence:
            continue
        encoded = tokenizer(subsequence, return_tensors="pt", add_special_tokens=True).to(device)
        hidden = model(**encoded).last_hidden_state[0]
        residue_hidden = hidden[1:1 + len(subsequence)]
        chunks.append(residue_hidden.mean(dim=0).cpu())
        if start + chunk_size >= len(sequence):
            break
    return torch.stack(chunks)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--uniprot", type=Path, default=Path("data/raw/protein_sequence/uniprot_sprot.dat"))
    parser.add_argument("--gene-order", type=Path, default=Path("data/model/hetero_graph/core_gene_order.txt"))
    parser.add_argument("--model", default="facebook/esm2_t33_650M_UR50D")
    parser.add_argument("--chunk-size", type=int, default=1000)
    parser.add_argument("--overlap", type=int, default=128)
    parser.add_argument("--output", type=Path, default=Path("data/processed/gene_embeddings/esm2_parti_chunks.pt"))
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModel.from_pretrained(args.model).eval().to(device)
    sequences = load_sequences(args.uniprot)
    genes = args.gene_order.read_text(encoding="utf-8").splitlines()
    embedded = []
    missing = []
    for gene in tqdm(genes, desc="ESM chunks"):
        sequence = sequences.get(gene)
        if sequence:
            embedded.append(embed_chunks(sequence, tokenizer, model, device, args.chunk_size, args.overlap))
        else:
            embedded.append(torch.zeros(1, model.config.hidden_size))
            missing.append(gene)
    max_chunks = max(len(item) for item in embedded)
    chunks = torch.zeros(len(embedded), max_chunks, model.config.hidden_size)
    mask = torch.zeros(len(embedded), max_chunks, dtype=torch.bool)
    for index, item in enumerate(embedded):
        chunks[index, :len(item)] = item
        if genes[index] not in missing:
            mask[index, :len(item)] = True
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"chunks": chunks, "valid_mask": mask, "gene_order": genes, "missing": missing}, args.output)
    print(f"saved {args.output} shape={tuple(chunks.shape)} missing={len(missing)}")


if __name__ == "__main__":
    main()
