"""Replicate the matched Sudoku stability pilot on the local CPU only."""

from __future__ import annotations

import hashlib
import json
import os
import pickle
import subprocess
import sys
import time
from pathlib import Path

import jax

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PYTHON = sys.executable
DATA_DIR = PROJECT_ROOT / "datasets" / "sudoku_sakana_stability_pilot"
OUTPUT_ROOT = PROJECT_ROOT / "outputs" / "sakana_stability_local_multiseed"
SEEDS = (123, 456)
ARMS = {
    "clue_k20": "sudoku_sheaf_9x9_clue_lock_clean",
    "sinkhorn_k20": "sudoku_sheaf_9x9_constraint_aware_clean",
}
HORIZONS = (20, 50)


def _run(command: list[str], log_path: Path) -> float:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    env = os.environ.copy()
    env["WANDB_MODE"] = "disabled"
    env["PYTHONUNBUFFERED"] = "1"
    env["JAX_COMPILATION_CACHE_DIR"] = str(OUTPUT_ROOT / "jax-cache")
    (OUTPUT_ROOT / "jax-cache").mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as stream:
        process = subprocess.Popen(
            command,
            cwd=PROJECT_ROOT,
            env=env,
            stdout=stream,
            stderr=subprocess.STDOUT,
            text=True,
        )
        next_heartbeat = started + 30
        while process.poll() is None:
            time.sleep(5)
            if time.monotonic() >= next_heartbeat:
                stream.flush()
                print(
                    json.dumps(
                        {
                            "status": "running",
                            "log": str(log_path.relative_to(PROJECT_ROOT)),
                            "elapsed_minutes": round((time.monotonic() - started) / 60, 1),
                        }
                    ),
                    flush=True,
                )
                next_heartbeat = time.monotonic() + 30
    elapsed = time.monotonic() - started
    if process.returncode != 0:
        tail = log_path.read_text(encoding="utf-8")[-5000:]
        raise RuntimeError(f"exit={process.returncode}: {tail}")
    return elapsed


def _train(seed: int, arm: str, experiment: str) -> Path:
    out_dir = OUTPUT_ROOT / f"seed_{seed}" / arm
    checkpoint = out_dir / "best_checkpoint.pkl"
    final_checkpoint = out_dir / "checkpoint.pkl"
    history_path = out_dir / "history.json"
    if final_checkpoint.is_file() and checkpoint.is_file():
        history = json.loads(history_path.read_text(encoding="utf-8"))
        if len(history) == 5:
            print(f"[reuse] {arm} seed={seed}", flush=True)
            return checkpoint

    command = [
        PYTHON,
        "scripts/train.py",
        f"+experiment={experiment}",
        f"data.dir={DATA_DIR}",
        f"training.seed={seed}",
        "training.epochs=5",
        "training.batch_size=16",
        "training.K_train=20",
        "training.K_eval=20",
        "training.train_iters_dist=fixed",
        f"training.checkpoint_dir={out_dir}",
        "training.checkpoint_interval=1",
        "wandb.mode=disabled",
        f"hydra.run.dir={out_dir / 'hydra'}",
    ]
    resume = out_dir / "training_checkpoint.pkl"
    if resume.is_file():
        command.append(f"training.resume_from={resume}")
    print(f"[train] {arm} seed={seed} backend=cpu", flush=True)
    seconds = _run(command, out_dir / "train.log")
    history = json.loads(history_path.read_text(encoding="utf-8"))
    if len(history) != 5 or not final_checkpoint.is_file() or not checkpoint.is_file():
        raise RuntimeError(f"incomplete training: {out_dir}")
    with checkpoint.open("rb") as stream:
        metadata = pickle.load(stream)["metadata"]
    print(
        json.dumps(
            {
                "status": "trained",
                "arm": arm,
                "seed": seed,
                "seconds": round(seconds, 1),
                "best": metadata,
            }
        ),
        flush=True,
    )
    return checkpoint


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _summary_is_current(summary_path: Path, checkpoint: Path, data_split: str, k: int) -> bool:
    """Reuse an evaluation only if it came from this checkpoint, dataset, split and K."""
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if summary.get("split") != data_split or summary.get("num_iters") != k:
        return False
    if Path(summary.get("checkpoint", "")).resolve() != checkpoint.resolve():
        return False
    if Path(summary.get("data_dir", "")).resolve() != Path(DATA_DIR).resolve():
        return False
    recorded_hash = summary.get("checkpoint_sha256")
    if recorded_hash is not None:
        return recorded_hash == _sha256(checkpoint)
    # Older summaries carry no hash: a checkpoint rewritten after the summary is stale.
    return checkpoint.stat().st_mtime <= summary_path.stat().st_mtime


def _evaluate(checkpoint: Path, data_split: str, k: int, out_dir: Path) -> None:
    summary_path = out_dir / "summary.json"
    if summary_path.is_file():
        if _summary_is_current(summary_path, checkpoint, data_split, k):
            print(f"[reuse] {summary_path.relative_to(PROJECT_ROOT)}", flush=True)
            return
        print(f"[stale] re-evaluating {summary_path.relative_to(PROJECT_ROOT)}", flush=True)
    command = [
        PYTHON,
        "scripts/evaluate_sudoku_constraints.py",
        "--checkpoint",
        str(checkpoint),
        "--data-dir",
        str(DATA_DIR),
        "--split",
        data_split,
        "--batch-size",
        "64",
        "--num-iters",
        str(k),
        "--out-dir",
        str(out_dir),
    ]
    print(f"[eval] {out_dir.relative_to(PROJECT_ROOT)}", flush=True)
    _run(command, out_dir / "evaluate.log")
    if not summary_path.is_file():
        raise RuntimeError(f"missing evaluation summary: {summary_path}")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    print(
        json.dumps(
            {
                "status": "evaluated",
                "split": data_split,
                "K": k,
                "exact": summary["exact_solution_rate"],
                "hidden": summary["hidden_accuracy"],
                "out": str(out_dir.relative_to(PROJECT_ROOT)),
            }
        ),
        flush=True,
    )


def main() -> None:
    if jax.default_backend() != "cpu":
        raise RuntimeError(f"expected local CPU only; got {jax.devices()}")
    if not (DATA_DIR / "pilot_manifest.json").is_file():
        raise FileNotFoundError(DATA_DIR / "pilot_manifest.json")
    print(
        json.dumps({"backend": "cpu", "devices": [str(x) for x in jax.devices()]}),
        flush=True,
    )
    for seed in SEEDS:
        checkpoints = {
            arm: _train(seed, arm, experiment) for arm, experiment in ARMS.items()
        }
        for arm, checkpoint in checkpoints.items():
            for k in HORIZONS:
                _evaluate(
                    checkpoint,
                    "validation",
                    k,
                    OUTPUT_ROOT / f"seed_{seed}" / arm / "validation_curve" / f"k_{k}",
                )
        for arm, checkpoint in checkpoints.items():
            for k in HORIZONS:
                _evaluate(
                    checkpoint,
                    "holdout",
                    k,
                    OUTPUT_ROOT / f"seed_{seed}" / arm / "holdout" / f"k_{k}",
                )

    seed42_clue = PROJECT_ROOT / "outputs" / "sakana_stability_pilot" / "clue_k20"
    _evaluate(
        seed42_clue / "best_checkpoint.pkl",
        "holdout",
        50,
        seed42_clue / "holdout" / "k_50",
    )
    print(json.dumps({"status": "complete", "seeds": list(SEEDS)}), flush=True)


if __name__ == "__main__":
    main()
