#!/usr/bin/env python
"""Launch a beta-OPSD training recipe.

Recipes keep public experiment settings in small YAML files. Environment
variables with uppercase argument names override recipe values, e.g.
`RUN=my_run MAX_STEPS=10 CONFIG=... bash scripts/train/beta_opsd.sh`.
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

try:
    import yaml
except ImportError as exc:  # pragma: no cover - exercised only in incomplete envs
    raise SystemExit("PyYAML is required. Install the release environment from environment.yml.") from exc


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def load_recipe(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Recipe must be a YAML mapping: {path}")
    for key in ("method", "entrypoint", "args"):
        if key not in data:
            raise ValueError(f"Recipe is missing required key `{key}`: {path}")
    if not isinstance(data["args"], dict):
        raise ValueError(f"Recipe `args` must be a mapping: {path}")
    return data


def parse_bool(value: str) -> bool:
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"Cannot parse boolean value: {value!r}")


def coerce_env_value(raw: str, current: Any) -> Any:
    if isinstance(current, bool):
        return parse_bool(raw)
    if isinstance(current, int) and not isinstance(current, bool):
        return int(raw)
    if isinstance(current, float):
        return float(raw)
    if isinstance(current, list):
        return shlex.split(raw)
    return raw


def apply_env_overrides(recipe: dict[str, Any]) -> None:
    top_level_env = {
        "RUN": "run_name",
        "OUTPUT_ROOT": "output_root",
        "ACCELERATE_CONFIG": "accelerate_config",
        "NUM_PROCESSES": "num_processes",
        "MAIN_PROCESS_PORT": "main_process_port",
        "ACCELERATE_GRADIENT_ACCUMULATION_STEPS": "accelerate_gradient_accumulation_steps",
    }
    for env_name, key in top_level_env.items():
        if env_name in os.environ:
            current = recipe.get(key, os.environ[env_name])
            recipe[key] = coerce_env_value(os.environ[env_name], current)

    args = recipe["args"]
    for key, current in list(args.items()):
        env_name = key.upper()
        if env_name in os.environ:
            args[key] = coerce_env_value(os.environ[env_name], current)


def append_arg(cmd: list[str], key: str, value: Any) -> None:
    flag = f"--{key}"
    if isinstance(value, bool):
        if value:
            cmd.append(flag)
        return
    if value is None:
        return
    if isinstance(value, list):
        cmd.append(flag)
        cmd.extend(str(item) for item in value)
        return
    cmd.extend([flag, str(value)])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("recipe", type=Path, help="Path to a YAML recipe")
    parser.add_argument("--dry-run", action="store_true", help="Print the command without executing it")
    parsed = parser.parse_args()

    root = repo_root()
    recipe_path = parsed.recipe
    if not recipe_path.is_absolute():
        recipe_path = root / recipe_path

    recipe = load_recipe(recipe_path)
    apply_env_overrides(recipe)

    run_name = str(recipe.get("run_name") or f"{recipe['method']}_{datetime.now():%Y%m%d_%H%M}")
    output_root = Path(str(recipe.get("output_root", "runs")))
    if not output_root.is_absolute():
        output_root = root / output_root
    run_dir = output_root / run_name
    checkpoints_dir = run_dir / "checkpoints"
    logs_dir = run_dir / "logs"
    eval_dir = run_dir / "eval"

    for directory in (checkpoints_dir, logs_dir, eval_dir):
        directory.mkdir(parents=True, exist_ok=True)

    cache_root = Path(os.environ.get("CACHE_ROOT", str(root / "cache")))
    os.environ.setdefault("TORCHINDUCTOR_CACHE_DIR", str(cache_root / "inductor"))
    os.environ.setdefault("TRITON_CACHE_DIR", str(cache_root / "triton"))
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    for env_name in ("TORCHINDUCTOR_CACHE_DIR", "TRITON_CACHE_DIR"):
        Path(os.environ[env_name]).mkdir(parents=True, exist_ok=True)
    if os.environ.get("VLLM_SRC"):
        os.environ["PYTHONPATH"] = f"{os.environ['VLLM_SRC']}:{os.environ.get('PYTHONPATH', '')}"

    args = dict(recipe["args"])
    args.setdefault("output_dir", str(checkpoints_dir))
    args.setdefault("run_config", run_name)

    accelerate_config = str(recipe.get("accelerate_config", "accelerate.yaml"))
    accelerate_cmd = [
        "accelerate",
        "launch",
        "--config_file",
        accelerate_config,
        "--num_processes",
        str(recipe.get("num_processes", 1)),
    ]
    if "accelerate_gradient_accumulation_steps" in recipe:
        accelerate_cmd.extend(
            ["--gradient_accumulation_steps", str(recipe["accelerate_gradient_accumulation_steps"])]
        )
    if "main_process_port" in recipe:
        accelerate_cmd.extend(["--main_process_port", str(recipe["main_process_port"])])

    cmd = accelerate_cmd + [str(recipe["entrypoint"])]
    for key, value in args.items():
        append_arg(cmd, key, value)

    config = {
        "run": run_name,
        "recipe": str(recipe_path.relative_to(root)),
        "method": recipe["method"],
        "launched": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID", "local"),
        "metadata": recipe.get("metadata", {}),
        "args": args,
    }
    with (run_dir / "config.json").open("w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)
        f.write("\n")

    print(f"[recipe] run={run_name}")
    print(f"[recipe] dir={run_dir}")
    print("[recipe] command:")
    print(" ".join(shlex.quote(part) for part in cmd))

    if parsed.dry_run:
        return 0
    os.chdir(root)
    return subprocess.call(cmd, env=os.environ)


if __name__ == "__main__":
    raise SystemExit(main())
