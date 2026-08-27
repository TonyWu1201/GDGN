from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from .io import sha256_file, write_json


@dataclass(frozen=True)
class ExternalSchema:
    dataset: str
    cell_column: str
    drug_column: str
    response_column: str
    tissue_column: str | None = None
    smiles_column: str | None = None


SCHEMAS = {
    "gdsc1": ExternalSchema("gdsc1", "CELL_LINE_NAME", "DRUG_NAME", "LN_IC50", "TCGA_DESC"),
    "ctrp": ExternalSchema("ctrp", "cell_name", "compound_name", "area_under_curve", "tissue"),
    "prism": ExternalSchema("prism", "depmap_id", "name", "auc", "lineage", "smiles"),
    "pdx": ExternalSchema("pdx", "sample_id", "drug_name", "response", "tumor_type"),
    "tcga": ExternalSchema("tcga", "sample_id", "drug_name", "clinical_endpoint", "cancer_type"),
}


def validate_external_frame(frame: pd.DataFrame, schema: ExternalSchema) -> dict:
    required = {schema.cell_column, schema.drug_column, schema.response_column}
    if missing := required - set(frame.columns):
        raise KeyError(f"{schema.dataset}: missing required columns {sorted(missing)}")
    response = pd.to_numeric(frame[schema.response_column], errors="coerce")
    duplicate_mask = frame.duplicated([schema.cell_column, schema.drug_column], keep=False)
    return {
        "dataset": schema.dataset,
        "n_rows": int(len(frame)),
        "n_cells": int(frame[schema.cell_column].nunique()),
        "n_drugs": int(frame[schema.drug_column].nunique()),
        "n_missing_response": int(response.isna().sum()),
        "n_duplicate_pair_rows": int(duplicate_mask.sum()),
        "response_min": float(response.min()),
        "response_max": float(response.max()),
    }


def prepare_external_dataset(
    input_path: str | Path,
    dataset: str,
    output_dir: str | Path,
) -> tuple[Path, Path]:
    if dataset not in SCHEMAS:
        raise ValueError(f"unsupported dataset: {dataset}; choose {sorted(SCHEMAS)}")
    source = Path(input_path)
    schema = SCHEMAS[dataset]
    frame = pd.read_csv(source)
    audit = validate_external_frame(frame, schema)
    standardized = pd.DataFrame({
        "cell_external_id": frame[schema.cell_column].astype(str),
        "drug_external_id": frame[schema.drug_column].astype(str),
        "response": pd.to_numeric(frame[schema.response_column], errors="coerce"),
        "tissue": frame[schema.tissue_column].astype(str) if schema.tissue_column and schema.tissue_column in frame else "unknown",
    }).dropna(subset=["response"])
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    csv_path = output / f"{dataset}-standardized.csv"
    manifest_path = output / f"{dataset}-manifest.json"
    standardized.to_csv(csv_path, index=False)
    write_json(manifest_path, audit | {
        "source_sha256": sha256_file(source),
        "mapping_status": "identifiers standardized; project-specific crosswalk still required",
        "endpoint_warning": "External endpoints are not assumed equivalent to GDSC2 LN_IC50.",
    })
    return csv_path, manifest_path
