import json

from program.strict_eval.registry import ExperimentRun, RunSpec


def test_completed_run_is_reused_and_failed_attempt_is_archived(tmp_path):
    spec = RunSpec("eval-lco", "model-m0", "eval-lco", 0, 42)
    config = {
        "experiment_id": "eval-lco", "model_id": "model-m0", "split_id": "eval-lco",
        "fold": 0, "seed": 42,
    }
    run = ExperimentRun(spec, config, tmp_path / "run")
    assert run.start() is True
    run.complete()
    assert run.start() is False
    assert not (tmp_path / "run" / "attempts").exists()

    failed = ExperimentRun(spec, config, tmp_path / "failed")
    assert failed.start() is True
    failed.fail(ValueError("expected"))
    assert failed.start() is True
    archived = tmp_path / "failed" / "attempts" / "attempt-01" / "status.json"
    assert json.loads(archived.read_text(encoding="utf-8"))["state"] == "failed"
