from __future__ import annotations

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
PROCESSED_DIR = DATA_DIR / "processed"
MODEL_DIR = DATA_DIR / "model"
SPLITS_DIR = MODEL_DIR / "splits"
EXPERIMENTS_DIR = MODEL_DIR / "experiments"
CONFIG_DIR = PROJECT_ROOT / "configs" / "experiments"

PROTOCOLS = (
    "eval-lpo",
    "eval-lco",
    "eval-ldo-kt",
    "eval-ldo-so",
    "eval-lto",
    "eval-db",
)

CONFIRMATION_SEEDS = (42, 3407, 8128)
DEFAULT_N_FOLDS = 5
DEFAULT_MIN_GROUP_SIZE = 5

IC50_PATH = PROCESSED_DIR / "drug_sensitivity" / "ic50_matrix.csv"
CELL_INDEX_PATH = PROCESSED_DIR / "cell_line_to_idx.json"
DRUG_INDEX_PATH = PROCESSED_DIR / "drug_cid_to_idx.json"
CANCER_TYPE_PATH = PROCESSED_DIR / "cell_line_cancer_types.json"
CELL_ORDER_PATH = PROCESSED_DIR / "cell_line_canonical_order.txt"
CELL_FEATURE_PATH = MODEL_DIR / "hetero_graph" / "cell_line_features.pt"
HETERO_GRAPH_PATH = MODEL_DIR / "hetero_graph" / "hetero_graph_base.pt"
DRUG_SMILES_PATH = DATA_DIR / "raw" / "drug_structures" / "compound_cid_smiles.csv"
PPI_PATH = PROCESSED_DIR / "protein_protein_interaction" / "ppi_dg_filtered.csv"
DTI_PATH = PROCESSED_DIR / "drug_gene_interaction" / "interactions_filtered.csv"
