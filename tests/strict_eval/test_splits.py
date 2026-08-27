import numpy as np
import pandas as pd
import pytest

from program.strict_eval.constants import PROTOCOLS
from program.strict_eval.splits import SplitOptions, make_protocol_folds


def synthetic_pairs(n_cells: int = 20, n_drugs: int = 20) -> pd.DataFrame:
    rows = []
    for cell in range(n_cells):
        for drug in range(n_drugs):
            rows.append({
                "pair_id": len(rows), "cell_idx": cell, "drug_idx": drug,
                "cell_name": f"c{cell}", "drug_cid": str(drug),
                "cancer_type": f"t{cell % 10}", "ic50": float(cell + drug),
            })
    return pd.DataFrame(rows)


@pytest.mark.parametrize("protocol", PROTOCOLS)
def test_all_protocols_are_deterministic_and_pass_audit(protocol):
    pairs = synthetic_pairs()
    options = SplitOptions(n_folds=5, seed=42, similarity_control=False)
    drug_groups = {idx: f"dg{idx // 2}" for idx in range(20)}
    cell_groups = {idx: f"cg{idx // 2}" for idx in range(20)}
    first = make_protocol_folds(pairs, protocol, options, drug_groups, cell_groups)
    second = make_protocol_folds(pairs, protocol, options, drug_groups, cell_groups)
    assert first == second
    assert len(first) == 5
    assert all(fold["audit"]["passed"] for fold in first)


def test_double_blind_excludes_cross_block_pairs():
    pairs = synthetic_pairs()
    options = SplitOptions(n_folds=5, seed=7, similarity_control=False)
    folds = make_protocol_folds(pairs, "eval-db", options)
    assert all(fold["n_excluded_pairs"] > 0 for fold in folds)
    for fold in folds:
        train = fold["splits"]["train"]
        test = fold["splits"]["test"]
        assert set(train["cell_idx"]).isdisjoint(test["cell_idx"])
        assert set(train["drug_idx"]).isdisjoint(test["drug_idx"])
