from __future__ import annotations

import traceback
import json
import shutil
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .constants import EXPERIMENTS_DIR, PROJECT_ROOT, PROTOCOLS
from .io import git_commit, write_json


@dataclass(frozen=True)
class RunSpec:
    experiment_id: str
    model_id: str
    split_id: str
    fold: int
    seed: int
    variant: str = "default"

    def __post_init__(self) -> None:
        if self.fold < 0 or self.seed < 0:
            raise ValueError("fold and seed must be non-negative")
        if not self.experiment_id or not self.model_id or not self.split_id or not self.variant:
            raise ValueError("experiment_id, model_id, split_id and variant are required")

    @property
    def run_dir(self) -> Path:
        model_dir = self.model_id if self.variant == "default" else f"{self.model_id}--{self.variant}"
        if self.experiment_id != self.split_id:
            model_dir = f"{self.split_id}--{model_dir}"
        return (
            EXPERIMENTS_DIR / self.experiment_id / model_dir
            / f"fold-{self.fold:02d}" / f"seed-{self.seed:04d}"
        )


class ExperimentRun:
    """Small, explicit registry for one fold/seed run.

    The status file is always preserved, including failures. Test evaluation is
    intentionally separate from checkpoint selection in the training runner.
    """

    def __init__(self, spec: RunSpec, config: dict[str, Any], run_dir: Path | None = None):
        self.spec = spec
        self.config = config
        self.run_dir = Path(run_dir) if run_dir else spec.run_dir
        self.run_dir.mkdir(parents=True, exist_ok=True)

    def start(self, force: bool = False) -> bool:
        if self.spec.experiment_id in PROTOCOLS and self.spec.split_id != self.spec.experiment_id:
            raise ValueError("evaluation experiment_id and split_id must match")
        status_path = self.run_dir / "status.json"
        config_path = self.run_dir / "config.json"
        if status_path.exists():
            status = json.loads(status_path.read_text(encoding="utf-8"))
            if status.get("state") == "complete" and not force:
                previous = json.loads(config_path.read_text(encoding="utf-8")) if config_path.exists() else None
                if previous != self.config:
                    raise RuntimeError(f"completed run key has a different config: {self.run_dir}")
                return False
            self._archive_previous_attempt()
        write_json(self.run_dir / "config.json", self.config)
        self._status("running")
        return True

    def _archive_previous_attempt(self) -> None:
        attempts = self.run_dir / "attempts"
        attempts.mkdir(parents=True, exist_ok=True)
        index = 1
        while (attempts / f"attempt-{index:02d}").exists():
            index += 1
        target = attempts / f"attempt-{index:02d}"
        target.mkdir()
        for path in list(self.run_dir.iterdir()):
            if path.name == "attempts":
                continue
            shutil.move(str(path), str(target / path.name))

    def complete(self, extra: dict[str, Any] | None = None) -> None:
        self._status("complete", extra)

    def fail(self, exc: BaseException) -> None:
        self._status("failed", {
            "error_type": type(exc).__name__,
            "error": str(exc),
            "traceback": "".join(traceback.format_exception(exc)),
        })

    def _status(self, state: str, extra: dict[str, Any] | None = None) -> None:
        payload: dict[str, Any] = {
            "state": state,
            "updated_at": datetime.now(timezone.utc).isoformat(),
            "git_commit": git_commit(PROJECT_ROOT),
            "run": asdict(self.spec),
        }
        if extra:
            payload.update(extra)
        write_json(self.run_dir / "status.json", payload)
