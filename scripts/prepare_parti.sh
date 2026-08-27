#!/usr/bin/env bash
set -euo pipefail

uv run python program/preprocess/protein_embedding_parti.py \
  --output data/processed/gene_embeddings/esm2_parti_chunks.pt
uv run python program/train_parti_pooler.py \
  --chunks data/processed/gene_embeddings/esm2_parti_chunks.pt \
  --output-dir data/model/pretrain-parti
uv run python program/pool_parti_embeddings.py \
  --chunks data/processed/gene_embeddings/esm2_parti_chunks.pt \
  --checkpoint data/model/pretrain-parti/checkpoint-best.pt \
  --output data/processed/gene_embeddings/esm2_parti_gene_embeddings.pt

echo "[prepare] PaRTI embeddings complete"
