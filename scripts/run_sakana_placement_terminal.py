"""Train the terminal-placement arm for the official-pool placement confirmation.

The arm uses clue locking inside every ADMM iteration and applies the same
clue-preserving Sinkhorn only at readout, with 12 rounds (the internal arm's
per-application count). A 240-round readout was rejected before training because
its cost alone exceeded internal K=10 latency. Data, seeds, epochs, batch size,
horizon, and optimizer match the existing ``sinkhorn_k20`` (internal) and
``clue_k20`` checkpoints. Runs are resumable and CPU-only.
"""

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import benchmark_sakana_constraint_inference as bench
import jax
import run_sakana_stability_local_multiseed as pilot

ROOT = pilot.PROJECT_ROOT
SEEDS = (123, 456, 789, 1011, 1213, 20260927)
EPOCHS = 5
EXPECTED_STEPS = 1280
READOUT_ROUNDS = 12
DATA_DIR = ROOT / "datasets" / "sudoku_sakana_stability_4096"
OUT_ROOT = ROOT / "outputs" / "sakana_placement_confirmation"
ARM = f"terminal{READOUT_ROUNDS}_k20"
EXPERIMENT = "sudoku_sheaf_9x9_decoder_only_sinkhorn_clean"
PROJECTION_KEYS = {
    "sudoku_projection_mode",
    "sudoku_sinkhorn_iters",
    "sudoku_sinkhorn_temperature",
}
RUNTIME_TRAINING_KEYS = {"checkpoint_dir", "resume_from", "checkpoint_interval"}


def terminal_checkpoint_path(seed: int) -> Path:
    return OUT_ROOT / f"seed_{seed}" / "n4096_e5" / ARM / "checkpoint.pkl"


def _load_config(path: Path) -> dict:
    with path.open("rb") as stream:
        return pickle.load(stream)["config"]


def assert_matches_clue_arm(seed: int) -> None:
    """The terminal arm may differ from the clue-lock arm only in readout projection."""
    terminal = _load_config(terminal_checkpoint_path(seed))
    clue = _load_config(bench.checkpoint_path(seed, "clue_k20"))
    model_diff = {
        key
        for key in set(terminal["model"]) | set(clue["model"])
        if terminal["model"].get(key) != clue["model"].get(key)
    }
    if not model_diff <= PROJECTION_KEYS:
        raise ValueError(f"seed {seed}: unexpected model differences {sorted(model_diff)}")
    if terminal["model"]["sudoku_sinkhorn_iters"] != READOUT_ROUNDS:
        raise ValueError(f"seed {seed}: readout rounds differ from {READOUT_ROUNDS}")
    training_diff = {
        key
        for key in set(terminal["training"]) | set(clue["training"])
        if key not in RUNTIME_TRAINING_KEYS
        and terminal["training"].get(key) != clue["training"].get(key)
    }
    if training_diff:
        raise ValueError(f"seed {seed}: unexpected training differences {sorted(training_diff)}")
    if Path(terminal["data"]["dir"]).resolve() != Path(clue["data"]["dir"]).resolve():
        raise ValueError(f"seed {seed}: training data differ from the clue-lock arm")


def train(seed: int) -> Path:
    out = terminal_checkpoint_path(seed).parent
    final = out / "checkpoint.pkl"
    restart = out / "training_checkpoint.pkl"
    history_path = out / "history.json"
    if final.is_file() and history_path.is_file():
        history = json.loads(history_path.read_text(encoding="utf-8"))
        with final.open("rb") as stream:
            metadata = pickle.load(stream)["metadata"]
        if len(history) == EPOCHS and metadata.get("global_step") == EXPECTED_STEPS:
            print(f"[reuse] {ARM} seed={seed}", flush=True)
            return final
    out.mkdir(parents=True, exist_ok=True)
    command = [
        pilot.PYTHON,
        "scripts/train.py",
        f"+experiment={EXPERIMENT}",
        f"data.dir={DATA_DIR}",
        f"training.seed={seed}",
        f"training.epochs={EPOCHS}",
        "training.batch_size=16",
        "training.K_train=20",
        "training.K_eval=20",
        "training.train_iters_dist=fixed",
        f"model.sudoku_sinkhorn_iters={READOUT_ROUNDS}",
        f"training.checkpoint_dir={out}",
        "training.checkpoint_interval=1",
        "wandb.mode=disabled",
        f"hydra.run.dir={out / 'hydra'}",
    ]
    if restart.is_file():
        command.append(f"training.resume_from={restart}")
    print(f"[train] {ARM} seed={seed} backend=cpu", flush=True)
    seconds = pilot._run(command, out / "train.log")
    history = json.loads(history_path.read_text(encoding="utf-8"))
    with final.open("rb") as stream:
        metadata = pickle.load(stream)["metadata"]
    if len(history) != EPOCHS or metadata.get("global_step") != EXPECTED_STEPS:
        raise RuntimeError(f"{ARM} seed={seed} did not finish at step {EXPECTED_STEPS}")
    print(json.dumps({"trained": ARM, "seed": seed, "minutes": round(seconds / 60, 1)}), flush=True)
    return final


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", default=",".join(map(str, SEEDS)))
    args = parser.parse_args()
    if jax.default_backend() != "cpu":
        raise RuntimeError(f"expected local CPU only; got {jax.devices()}")
    seeds = tuple(int(value) for value in args.seeds.split(",") if value.strip())
    for seed in seeds:
        train(seed)
        assert_matches_clue_arm(seed)
    print(json.dumps({"status": "complete", "arm": ARM, "seeds": list(seeds)}), flush=True)


if __name__ == "__main__":
    main()
