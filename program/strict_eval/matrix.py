from __future__ import annotations

import itertools
import json
from pathlib import Path

import yaml

from .constants import CONFIG_DIR


def expand_sweep(path: str | Path) -> list[dict]:
    spec = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    variants = spec.get("variants") or {"default": {}}
    base_overrides = spec.get("base_overrides") or {}
    model_overrides = spec.get("model_overrides") or {}
    runs = []
    for protocol, model, fold, seed, (variant, overrides) in itertools.product(
        spec["protocols"], spec["models"], spec["folds"], spec["seeds"], variants.items()
    ):
        base = yaml.safe_load((CONFIG_DIR / f"{protocol}.yaml").read_text(encoding="utf-8"))
        base.update(base_overrides)
        base.update(model_overrides.get(model, {}))
        base.update(overrides)
        base.update({
            "experiment_id": protocol if variant == "default" else Path(path).stem,
            "split_id": protocol,
            "model_id": model,
            "fold": int(fold),
            "seed": int(seed),
            "variant": variant,
        })
        runs.append(base)
    return runs


def write_manifest(sweep_path: str | Path, output: str | Path) -> Path:
    target = Path(output)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8", newline="\n") as handle:
        for config in expand_sweep(sweep_path):
            handle.write(json.dumps(config, ensure_ascii=False, sort_keys=True) + "\n")
    return target
