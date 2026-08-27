from pathlib import Path

from program.strict_eval.matrix import expand_sweep
from program.strict_eval.registry import RunSpec
from program.strict_eval.training import resolve_config_templates


def test_variant_has_isolated_run_directory():
    base = RunSpec("exp", "model-m2", "eval-lco", 0, 42)
    variant = RunSpec("exp", "model-m2", "eval-lco", 0, 42, "graph-none")
    assert base.run_dir != variant.run_dir
    assert variant.run_dir.parts[-3] == "eval-lco--model-m2--graph-none"


def test_every_sweep_entry_has_unique_registry_key():
    runs = expand_sweep(Path("configs/sweeps/abl-graph.yaml"))
    paths = {
        RunSpec(
            run["experiment_id"], run["model_id"], run["split_id"],
            run["fold"], run["seed"], run["variant"],
        ).run_dir
        for run in runs
    }
    assert len(paths) == len(runs)


def test_benchmark_gdgn_is_the_explicit_current_main_configuration():
    runs = expand_sweep(Path("configs/sweeps/benchmark.yaml"))
    gdgn = next(run for run in runs if run["model_id"] == "gdgn")
    assert gdgn["pretrain_ckpt"] == (
        "data/model/pretrain/strict/{split_id}/fold-{fold:02d}/seed-{seed:04d}/best_encoder.pt"
    )
    assert gdgn["cell_bypass_mode"] == "residual"
    assert gdgn["dual_cell"] is True
    resolved = resolve_config_templates(gdgn)
    assert resolved["pretrain_ckpt"].endswith(
        "eval-lpo/fold-00/seed-0042/best_encoder.pt"
    )
