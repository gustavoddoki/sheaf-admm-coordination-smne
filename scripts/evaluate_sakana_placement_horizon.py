"""Placement x horizon evaluator for the official-pool 9x9 regime (session 2026-09-25).

Protocol: experiments/constraint_aware_size_equivariant_sudoku/
OFFICIAL_PLACEMENT_HORIZON_MAP_PROTOCOL.md. Trained (EMA) or untrained parameters of
the internal (``sinkhorn_k20``), terminal (``terminal12_k20``) and clue-lock
(``clue_k20``) arms are evaluated under any projection placement:

* ``P_int``: clue-preserving Sinkhorn after every x-update, no readout projection;
* ``P_term``: clue locking inside the iterations, Sinkhorn only at readout;
* ``P_clue``: clue locking inside the iterations, no readout projection.

The three arms share one parameter tree (projections have no parameters), so a
placement can be swapped at inference. A job is either a dedicated fixed-K pass or a
single K_max trajectory: ``model(..., num_iters=K_max, loss_window=K_max)`` runs the
same ADMM step as a fixed pass and decodes every iterate, so slice K is the
prediction after K rounds. The dev gate checks this equivalence against the 24/09
per-puzzle results. Every holdout must carry a fresh-holdout manifest with
``official_test_used=false``; the official test split is never read.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import pickle
import statistics
import time
from pathlib import Path

import benchmark_sakana_constraint_inference as bench
import jax
import jax.numpy as jnp
import numpy as np
import psutil
import run_sakana_placement_terminal as terminal_runner

from sheaf_admm.data import reassemble_sudoku_logits
from sheaf_admm.data.sudoku_metrics import analyze_sudoku_predictions
from sheaf_admm.models import model_config_from_dict
from sheaf_admm.training import build_model, create_train_state, make_task

ROOT = bench.ROOT
OUT_ROOT = ROOT / "outputs" / "sakana_placement_horizon_map"
STATE = OUT_ROOT / "session_state.json"
PROTOCOL = (
    ROOT
    / "experiments"
    / "constraint_aware_size_equivariant_sudoku"
    / "OFFICIAL_PLACEMENT_HORIZON_MAP_PROTOCOL.md"
)
DEV_HOLDOUT = ROOT / "datasets" / "sudoku_sakana_placement_confirmation_holdout"
DEV_SELECTION_SEED = 20261004
NEW_HOLDOUT = ROOT / "datasets" / "sudoku_sakana_horizon_map_holdout"
NEW_SELECTION_SEED = 20261005
DEV_CSV_ROOT = ROOT / "outputs" / "sakana_placement_confirmation" / "confirmation" / "per_puzzle"
EXTENSION_ROOT = ROOT / "outputs" / "sakana_placement_extension"
SEEDS = terminal_runner.SEEDS
MECHANISM_SEEDS = (123, 456, 789)
BATCH_SIZE = 64
NATIVE_ROUNDS = 12
ARM_DIRS = {"internal": "sinkhorn_k20", "terminal": "terminal12_k20", "clue": "clue_k20"}
NATIVE = {"internal": "P_int", "terminal": "P_term", "clue": "P_clue"}
PLACEMENTS = {
    "P_int": {"sudoku_internal_projection_mode": "sinkhorn"},
    "P_term": {
        "sudoku_internal_projection_mode": "clue_lock",
        "sudoku_projection_mode": "sinkhorn",
    },
    "P_clue": {"sudoku_internal_projection_mode": "clue_lock"},
}
# Native internal/clue checkpoints omit the readout keys (ModelConfig default "none"),
# so placements without a readout projection remove them instead of storing None.
READOUT_KEYS = ("sudoku_projection_mode", "sudoku_sinkhorn_iters", "sudoku_sinkhorn_temperature")
TRAJECTORY_CSV_KS = (10, 16, 20, 30, 40)


# ---------------------------------------------------------------------------
# provenance helpers


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def checkpoint_path(arm: str, seed: int, variant: str = "e5") -> Path:
    if variant == "e5":
        if arm == "terminal":
            return terminal_runner.terminal_checkpoint_path(seed)
        return bench.checkpoint_path(seed, ARM_DIRS[arm])
    if variant not in ("e8", "e10"):
        raise ValueError(f"unknown checkpoint variant {variant!r}")
    return EXTENSION_ROOT / f"seed_{seed}" / f"n4096_{variant}" / ARM_DIRS[arm] / "checkpoint.pkl"


def load_checkpoint_config(path: Path) -> dict:
    with path.open("rb") as stream:
        return pickle.load(stream)["config"]


def audit_holdout(holdout_dir: Path, selection_seed: int) -> dict:
    manifest_path = holdout_dir / "fresh_holdout_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if (
        manifest.get("official_test_used") is not False
        or manifest.get("count") != 1024
        or manifest.get("selection_seed") != selection_seed
        or not manifest.get("no_overlap_with_current_experiment_splits")
    ):
        raise ValueError(f"holdout audit failed for {holdout_dir}")
    return {
        "dir": str(holdout_dir.relative_to(ROOT)),
        "selection_seed": selection_seed,
        "count": manifest["count"],
        "manifest_sha256": sha256(manifest_path),
    }


def verify_frozen_protocol() -> str:
    """The confirmatory phase refuses to run unless the protocol hash matches the state."""
    state = json.loads(STATE.read_text(encoding="utf-8"))
    expected = (state.get("protocol_sha256") or {}).get("final")
    actual = sha256(PROTOCOL)
    if not expected or expected != actual:
        raise RuntimeError(
            f"protocol hash mismatch or not frozen: state={expected} actual={actual}"
        )
    return actual


# ---------------------------------------------------------------------------
# model configuration


def effective_model_config(
    base_model_cfg: dict, placement: str, internal_rounds: int, readout_rounds: int
) -> dict:
    cfg = dict(base_model_cfg)
    cfg.update(PLACEMENTS[placement])
    cfg["sudoku_internal_sinkhorn_iters"] = internal_rounds
    cfg["sudoku_internal_sinkhorn_temperature"] = 1.0
    if placement == "P_term":
        cfg["sudoku_sinkhorn_iters"] = readout_rounds
        cfg["sudoku_sinkhorn_temperature"] = 1.0
    else:
        for key in READOUT_KEYS:
            cfg.pop(key, None)
    return cfg


def check_effective_config(
    base_model_cfg: dict,
    cfg: dict,
    arm: str,
    placement: str,
    internal_rounds: int,
    readout_rounds: int,
) -> None:
    """Fail loudly if an override was not applied or a native job is not native."""
    changed = {
        key for key in set(cfg) | set(base_model_cfg) if cfg.get(key) != base_model_cfg.get(key)
    }
    allowed = {
        "sudoku_internal_projection_mode",
        "sudoku_projection_mode",
        "sudoku_internal_sinkhorn_iters",
        "sudoku_internal_sinkhorn_temperature",
        "sudoku_sinkhorn_iters",
        "sudoku_sinkhorn_temperature",
    }
    if not changed <= allowed:
        raise ValueError(f"override touched non-projection keys: {sorted(changed - allowed)}")
    native = (
        placement == NATIVE[arm]
        and internal_rounds == NATIVE_ROUNDS
        and readout_rounds == NATIVE_ROUNDS
    )
    # Exact dict equality (key presence included) for native jobs.
    if native and cfg != base_model_cfg:
        raise ValueError(f"native {arm} job differs from its checkpoint config: {sorted(changed)}")
    model_cfg = model_config_from_dict(dict(cfg))
    expected_internal = PLACEMENTS[placement]["sudoku_internal_projection_mode"]
    if model_cfg.sudoku_internal_projection_mode != expected_internal:
        raise ValueError("internal projection override not applied")
    if placement == "P_int" and model_cfg.sudoku_internal_sinkhorn_iters != internal_rounds:
        raise ValueError("internal round override not applied")
    if placement == "P_term" and (
        model_cfg.sudoku_projection_mode != "sinkhorn"
        or model_cfg.sudoku_sinkhorn_iters != readout_rounds
    ):
        raise ValueError("readout projection override not applied")
    if placement != "P_term" and model_cfg.sudoku_projection_mode != "none":
        raise ValueError("readout projection should be off for this placement")


def job_name(job: dict) -> str:
    name = job["arm"]
    if job.get("variant", "e5") != "e5":
        name += f"_{job['variant']}"
    if job.get("params", "ema") != "ema":
        name += f"_{job['params']}"
    name += f"_{job['placement']}"
    if job["placement"] == "P_int" and job.get("internal_rounds", NATIVE_ROUNDS) != NATIVE_ROUNDS:
        name += f"_r{job['internal_rounds']}"
    if job["placement"] == "P_term" and job.get("readout_rounds", NATIVE_ROUNDS) != NATIVE_ROUNDS:
        name += f"_ro{job['readout_rounds']}"
    return name


def job_key(job: dict) -> str:
    return f"seed_{job['seed']}/{job_name(job)}/{job['mode']}/K{job['k']}"


# ---------------------------------------------------------------------------
# predictors and metrics


def make_trajectory_predictor(model, task, k_max: int):
    @jax.jit
    def predict(params, fwd):
        logits, _state, _geometry = model.apply(
            params,
            fwd["patches"],
            fwd["edge_indices"],
            num_iters=k_max,
            loss_window=k_max,
            **fwd["model_kwargs"],
            training=False,
        )

        def decode(window_logits):
            recon = reassemble_sudoku_logits(
                jnp.transpose(window_logits, (1, 0, 2, 3)), use_boxes=task.use_boxes
            )
            return jnp.argmax(recon, axis=-1).astype(jnp.uint8)

        return jax.lax.map(decode, logits)  # [K_max, B, n, n]

    return predict


def grid_metrics(inputs: np.ndarray, labels: np.ndarray, predictions: np.ndarray) -> dict:
    """Vectorized exact/valid/hidden accuracy for predictions [P, n, n].

    Same definitions as ``analyze_sudoku_predictions``: a grid is valid when every
    row, column and box holds each symbol 1..n once and every clue is kept.
    """
    puzzles, n, _ = predictions.shape
    box = int(round(n**0.5))
    predictions = predictions.astype(np.int64)
    exact = np.all(predictions == labels, axis=(1, 2))
    hidden = inputs == 0
    correct_hidden = ((predictions == labels) & hidden).sum(axis=(1, 2))
    hidden_accuracy = correct_hidden / np.maximum(hidden.sum(axis=(1, 2)), 1)
    onehot = np.eye(n + 1, dtype=np.int16)[predictions][..., 1:]  # [P, n, n, n]
    rows_ok = np.all(onehot.sum(axis=2) == 1, axis=(1, 2))
    cols_ok = np.all(onehot.sum(axis=1) == 1, axis=(1, 2))
    boxes = onehot.reshape(puzzles, box, box, box, box, n).sum(axis=(2, 4))
    boxes_ok = np.all(boxes == 1, axis=(1, 2, 3))
    clue_ok = np.all((predictions == inputs) | hidden, axis=(1, 2))
    valid = rows_ok & cols_ok & boxes_ok & clue_ok
    return {"exact": exact, "valid": valid, "hidden_accuracy": hidden_accuracy.astype(np.float32)}


def trajectory_metrics(inputs: np.ndarray, labels: np.ndarray, predictions: np.ndarray) -> dict:
    """Per-K metrics for predictions [K, P, n, n], one K at a time to bound memory."""
    per_k = [grid_metrics(inputs, labels, predictions[k]) for k in range(predictions.shape[0])]
    return {key: np.stack([item[key] for item in per_k]) for key in per_k[0]}


def infer_trajectory(predictor, params, prepared) -> np.ndarray:
    parts = [np.asarray(jax.device_get(predictor(params, fwd))) for fwd in prepared]
    return np.concatenate(parts, axis=1)


# ---------------------------------------------------------------------------
# evaluation engine


class Evaluator:
    def __init__(self, holdout_dir: Path, selection_seed: int, out_dir: Path, phase: str):
        if jax.default_backend() != "cpu":
            raise RuntimeError(f"expected local CPU, got {jax.devices()}")
        self.holdout = audit_holdout(holdout_dir, selection_seed)
        self.out_dir = out_dir
        self.phase = phase
        reference_cfg = load_checkpoint_config(checkpoint_path("internal", SEEDS[0]))
        self.task = make_task("sudoku", **dict(reference_cfg.get("task_cfg", {})))
        self.prepared, self.inputs, self.labels = bench.prepare_batches(
            holdout_dir, self.task, BATCH_SIZE
        )
        if len(self.inputs) != self.holdout["count"]:
            raise ValueError(f"holdout count mismatch: {len(self.inputs)}")
        self.models: dict[str, object] = {}
        self.predictors: dict[tuple, object] = {}
        self.params_cache: dict[tuple, object] = {}
        self.base_configs: dict[tuple, dict] = {}
        out_dir.mkdir(parents=True, exist_ok=True)
        self.progress_path = out_dir / "progress.json"
        if self.progress_path.is_file():
            self.progress = json.loads(self.progress_path.read_text(encoding="utf-8"))
            if self.progress.get("holdout") != self.holdout or self.progress.get("phase") != phase:
                raise ValueError("existing progress belongs to another phase or holdout")
        else:
            self.progress = {"phase": phase, "holdout": self.holdout, "completed": {}}

    # -- models and parameters
    def base_config(self, arm: str, variant: str) -> dict:
        key = (arm, variant)
        if key not in self.base_configs:
            reference = load_checkpoint_config(
                checkpoint_path(arm, SEEDS[0] if variant == "e5" else 123, variant)
            )
            self.base_configs[key] = reference
        return self.base_configs[key]

    def assert_seed_config(self, arm: str, seed: int, variant: str) -> None:
        cfg = load_checkpoint_config(checkpoint_path(arm, seed, variant))
        reference = self.base_config(arm, variant)
        if (cfg["model"], cfg.get("task_cfg", {})) != (
            reference["model"],
            reference.get("task_cfg", {}),
        ):
            raise ValueError(f"{arm}/{variant} seed {seed} config differs from the reference seed")

    def model_for(self, job: dict):
        base = self.base_config(job["arm"], job.get("variant", "e5"))["model"]
        cfg = effective_model_config(
            base,
            job["placement"],
            job.get("internal_rounds", NATIVE_ROUNDS),
            job.get("readout_rounds", NATIVE_ROUNDS),
        )
        check_effective_config(
            base,
            cfg,
            job["arm"],
            job["placement"],
            job.get("internal_rounds", NATIVE_ROUNDS),
            job.get("readout_rounds", NATIVE_ROUNDS),
        )
        key = json.dumps(cfg, sort_keys=True, default=str)
        if key not in self.models:
            self.models[key] = build_model(model_config_from_dict(dict(cfg)), "sheaf")
        return key, self.models[key], cfg

    def params_for(self, job: dict):
        variant = job.get("variant", "e5")
        kind = job.get("params", "ema")
        key = (job["arm"], job["seed"], variant, kind)
        if key in self.params_cache:
            return self.params_cache[key]
        path = checkpoint_path(job["arm"], job["seed"], variant)
        if kind == "ema":
            params = bench.load_params(path)
        elif kind == "init":
            params = self._initial_params(path)
        else:
            raise ValueError(f"unknown params kind {kind!r}")
        # Keep at most a few parameter sets alive (RAM is tight on this machine).
        if len(self.params_cache) >= 6:
            self.params_cache.pop(next(iter(self.params_cache)))
        self.params_cache[key] = params
        return params

    def _initial_params(self, trained_checkpoint: Path):
        with trained_checkpoint.open("rb") as stream:
            cfg = pickle.load(stream)["config"]
        train_dir = Path(cfg["data"]["dir"]) / cfg["data"]["train_split"]
        inputs = np.load(train_dir / "all__inputs.npy", mmap_mode="r")[:1]
        labels = np.load(train_dir / "all__labels.npy", mmap_mode="r")[:1]
        sample_fwd, _, _ = self.task.prepare(
            {"inputs": np.asarray(inputs), "labels": np.asarray(labels)}
        )
        model = build_model(model_config_from_dict(dict(cfg["model"])), "sheaf")
        training = cfg["training"]
        state = create_train_state(
            model,
            sample_fwd,
            model_type="sheaf",
            lr=training["lr"],
            weight_decay=training["weight_decay"],
            warmup_steps=training["warmup_steps"],
            grad_clip=training["grad_clip"],
            ema_decay=training["ema_decay"],
            k_init=training["K_train"],
            loss_window=training["loss_window"],
            seed=training["seed"],
        )
        return jax.tree_util.tree_map(jnp.asarray, state.params)

    def predictor_for(self, model_key: str, model, mode: str, k: int):
        key = (model_key, mode, k)
        if key not in self.predictors:
            if mode == "fixed":
                self.predictors[key] = bench.make_predictor(model, self.task, k)
            elif mode == "trajectory":
                self.predictors[key] = make_trajectory_predictor(model, self.task, k)
            else:
                raise ValueError(f"unknown mode {mode!r}")
        return self.predictors[key]

    # -- output helpers
    def _csv_path(self, job: dict, k: int) -> Path:
        suffix = "" if job["mode"] == "fixed" else "__traj"
        return (
            self.out_dir
            / "per_puzzle"
            / f"seed_{job['seed']}"
            / f"{job_name(job)}{suffix}"
            / f"k_{k}.csv"
        )

    def _npz_path(self, job: dict) -> Path:
        return (
            self.out_dir
            / "trajectories"
            / f"seed_{job['seed']}"
            / f"{job_name(job)}_kmax{job['k']}.npz"
        )

    def _save_rows(self, job: dict, k: int, rows: list[dict]) -> None:
        path = self._csv_path(job, k)
        if path.exists():
            raise FileExistsError(f"refusing to overwrite {path}")
        path.parent.mkdir(parents=True, exist_ok=True)
        fieldnames = ["seed", "condition", "K", *rows[0].keys()]
        with path.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=fieldnames)
            writer.writeheader()
            for row in rows:
                writer.writerow({"seed": job["seed"], "condition": job_name(job), "K": k, **row})

    def _save_progress(self) -> None:
        tmp = self.progress_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self.progress, indent=2), encoding="utf-8")
        tmp.replace(self.progress_path)

    # -- main entry
    def run(self, job: dict) -> dict:
        key = job_key(job)
        if key in self.progress["completed"]:
            return self.progress["completed"][key]
        if job["mode"] == "fixed" and self._csv_path(job, job["k"]).exists():
            raise FileExistsError(f"output exists but job not in progress: {key}")
        if job["mode"] == "trajectory" and self._npz_path(job).exists():
            raise FileExistsError(f"output exists but job not in progress: {key}")
        variant = job.get("variant", "e5")
        self.assert_seed_config(job["arm"], job["seed"], variant)
        model_key, model, cfg = self.model_for(job)
        params = self.params_for(job)
        predictor = self.predictor_for(model_key, model, job["mode"], job["k"])
        load_before = psutil.cpu_percent(interval=None)
        started = time.perf_counter()
        record = {
            "job": {k: v for k, v in job.items()},
            "condition": job_name(job),
            "checkpoint": str(checkpoint_path(job["arm"], job["seed"], variant).relative_to(ROOT)),
            "checkpoint_sha256": sha256(checkpoint_path(job["arm"], job["seed"], variant)),
            "effective_projection": {
                key_: cfg.get(key_)
                for key_ in (
                    "sudoku_internal_projection_mode",
                    "sudoku_internal_sinkhorn_iters",
                    "sudoku_projection_mode",
                    "sudoku_sinkhorn_iters",
                )
            },
        }
        if job["mode"] == "fixed":
            predictions = bench.infer_once(predictor, params, self.prepared)
            summary, rows = analyze_sudoku_predictions(
                self.inputs, self.labels, predictions, use_boxes=self.task.use_boxes
            )
            self._save_rows(job, job["k"], rows)
            record["metrics"] = {
                key_: float(summary[key_])
                for key_ in ("exact_solution_rate", "valid_grid_rate", "hidden_accuracy")
            }
        else:
            predictions = infer_trajectory(predictor, params, self.prepared)
            if predictions.shape[0] != job["k"]:
                raise ValueError("trajectory length mismatch")
            metrics = trajectory_metrics(self.inputs, self.labels, predictions)
            path = self._npz_path(job)
            path.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(
                path,
                predictions=predictions.reshape(predictions.shape[0], predictions.shape[1], -1),
                exact=metrics["exact"],
                valid=metrics["valid"],
                hidden_accuracy=metrics["hidden_accuracy"],
            )
            for k in job.get("csv_ks", TRAJECTORY_CSV_KS):
                if k <= job["k"]:
                    _summary, rows = analyze_sudoku_predictions(
                        self.inputs, self.labels, predictions[k - 1], use_boxes=self.task.use_boxes
                    )
                    self._save_rows(job, k, rows)
            record["exact_by_k"] = [float(v) for v in metrics["exact"].mean(axis=1)]
            record["valid_by_k"] = [float(v) for v in metrics["valid"].mean(axis=1)]
            record["hidden_accuracy_by_k"] = [
                float(v) for v in metrics["hidden_accuracy"].mean(axis=1)
            ]
            record["finite"] = True
        record["seconds_including_compile"] = time.perf_counter() - started
        record["system_load_percent"] = load_before
        self.progress["completed"][key] = record
        self._save_progress()
        summary_text = (
            record["metrics"]["exact_solution_rate"]
            if job["mode"] == "fixed"
            else {k: round(record["exact_by_k"][k - 1], 4) for k in (10, 20) if k <= job["k"]}
        )
        print(
            json.dumps(
                {
                    "done": key,
                    "exact": summary_text,
                    "seconds": round(record["seconds_including_compile"], 1),
                }
            ),
            flush=True,
        )
        return record

    def timing_pair(
        self, left: dict, right: dict, repeats: int = 3, max_load: float = 60.0
    ) -> dict:
        """Steady-state paired latency (compiled, warmed-up, alternating order)."""
        entries = {}
        for job in (left, right):
            model_key, model, _cfg = self.model_for(job)
            predictor = self.predictor_for(model_key, model, "fixed", job["k"])
            params = self.params_for(job)
            bench.infer_once(predictor, params, self.prepared)  # compile + warm-up
            entries[job_key(job)] = (predictor, params)
        psutil.cpu_percent(interval=None)
        load = psutil.cpu_percent(interval=5.0)
        result = {"load_percent_before": load, "repeats": repeats, "runs": {k: [] for k in entries}}
        if load > max_load:
            result["skipped"] = f"system load {load:.0f}% above {max_load:.0f}%"
            return result
        names = list(entries)
        for repeat in range(repeats):
            order = names if repeat % 2 == 0 else list(reversed(names))
            for name in order:
                predictor, params = entries[name]
                psutil.cpu_percent(interval=None)
                cpu0 = time.process_time()
                started = time.perf_counter()
                bench.infer_once(predictor, params, self.prepared)
                wall = time.perf_counter() - started
                own = (time.process_time() - cpu0) / wall
                system = psutil.cpu_percent(interval=None)
                result["runs"][name].append(
                    {
                        "seconds": wall,
                        "process_cpu_over_wall": own,
                        "system_load_percent": system,
                        "external_load_percent_estimate": max(
                            0.0, system - 100.0 * own / psutil.cpu_count()
                        ),
                    }
                )
        medians = {k: statistics.median(r["seconds"] for r in v) for k, v in result["runs"].items()}
        result["median_seconds"] = medians
        result["ratio_left_over_right"] = medians[names[0]] / medians[names[1]]
        return result


# ---------------------------------------------------------------------------
# phases


def fixed(arm, seed, placement, k, **extra) -> dict:
    return {"arm": arm, "seed": seed, "placement": placement, "mode": "fixed", "k": k, **extra}


def trajectory(arm, seed, placement, k_max, **extra) -> dict:
    return {
        "arm": arm,
        "seed": seed,
        "placement": placement,
        "mode": "trajectory",
        "k": k_max,
        **extra,
    }


GATE_OLD = {
    ("internal", 10): "internal_K10",
    ("internal", 20): "internal_K20",
    ("terminal", 10): "terminal_K10",
    ("terminal", 14): "terminal_K14",
    ("terminal", 16): "terminal_K16",
    ("terminal", 20): "terminal_K20",
    ("clue", 15): "clue_K15",
}


def _read_rows(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as stream:
        return sorted(csv.DictReader(stream), key=lambda row: int(row["puzzle_index"]))


def run_gate(args) -> None:
    out = OUT_ROOT / "dev_gate"
    ev = Evaluator(DEV_HOLDOUT, DEV_SELECTION_SEED, out, "dev_gate")
    seeds = (123, 456)
    csv_ks = {"internal": (10, 20), "terminal": (10, 14, 16, 20), "clue": (15,)}
    for seed in seeds:
        for arm in ("internal", "terminal", "clue"):
            ev.run(trajectory(arm, seed, NATIVE[arm], args.k_max, csv_ks=csv_ks[arm] + (30, 40)))
    fixed_jobs = [
        fixed("internal", 123, "P_int", 20),
        fixed("terminal", 123, "P_term", 20),
        fixed("clue", 123, "P_clue", 15),
    ]
    for job in fixed_jobs:
        ev.run(job)
    comparisons = []
    for seed in seeds:
        for (arm, k), old_name in GATE_OLD.items():
            old = _read_rows(DEV_CSV_ROOT / f"seed_{seed}" / old_name / f"k_{k}.csv")
            job = trajectory(arm, seed, NATIVE[arm], args.k_max)
            new = _read_rows(ev._csv_path(job, k))
            comparisons.append(_compare(f"traj seed {seed} {old_name}", old, new))
    for job in fixed_jobs:
        old_name = GATE_OLD[(job["arm"], job["k"])]
        old = _read_rows(DEV_CSV_ROOT / f"seed_{job['seed']}" / old_name / f"k_{job['k']}.csv")
        new = _read_rows(ev._csv_path(job, job["k"]))
        comparisons.append(_compare(f"fixed seed {job['seed']} {old_name}", old, new))
    worst = min(item["wrong_cells_identical_fraction"] for item in comparisons)
    verdict = "pass_identical" if worst == 1.0 else "pass_near" if worst >= 0.999 else "fail"
    payload = {
        "verdict": verdict,
        "worst_wrong_cells_identical_fraction": worst,
        "comparisons": comparisons,
    }
    (out / "gate_result.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps({"gate": verdict, "worst": worst}), flush=True)


def _compare(label: str, old: list[dict], new: list[dict]) -> dict:
    if len(old) != len(new):
        raise ValueError(f"{label}: row count differs")
    same_exact = sum(
        a["exact_solution"] == b["exact_solution"] for a, b in zip(old, new, strict=True)
    )
    same_wrong = sum(a["wrong_cells"] == b["wrong_cells"] for a, b in zip(old, new, strict=True))
    return {
        "label": label,
        "puzzles": len(old),
        "exact_identical_fraction": same_exact / len(old),
        "wrong_cells_identical_fraction": same_wrong / len(old),
        "old_exact_rate": sum(a["exact_solution"] == "True" for a in old) / len(old),
        "new_exact_rate": sum(b["exact_solution"] == "True" for b in new) / len(new),
    }


def run_confirm(args) -> None:
    protocol_hash = verify_frozen_protocol()
    state = json.loads(STATE.read_text(encoding="utf-8"))
    k_b20 = int(state["K_B20"])
    out = OUT_ROOT / "confirmation"
    ev = Evaluator(NEW_HOLDOUT, NEW_SELECTION_SEED, out, "confirmation")
    ev.progress["protocol_sha256"] = protocol_hash
    # Latency is measured separately (--phase timing) in a quiet window.
    for seed in SEEDS:
        # Primary family C1-C3 (dedicated fixed passes).
        ev.run(fixed("internal", seed, "P_int", 20))
        ev.run(fixed("terminal", seed, "P_term", 20))
        ev.run(fixed("internal", seed, "P_int", 30))
        ev.run(fixed("terminal", seed, "P_term", 30))
        ev.run(fixed("internal", seed, "P_term", 20))
        # Secondary: readout quantity and the upper time-matched pair.
        ev.run(fixed("terminal", seed, "P_term", 20, readout_rounds=240))
        ev.run(fixed("terminal", seed, "P_term", k_b20))
    for seed in SEEDS:
        for arm in ("internal", "terminal", "clue"):
            ev.run(trajectory(arm, seed, NATIVE[arm], args.k_max))
    print(json.dumps({"status": "confirmation_complete"}), flush=True)


def run_timing(args) -> None:
    """H7 latency pair on the confirmation holdout (seed 123); no accuracy is computed."""
    verify_frozen_protocol()
    k_b20 = int(json.loads(STATE.read_text(encoding="utf-8"))["K_B20"])
    tag = f"_{args.tag}" if args.tag else ""
    path = OUT_ROOT / "confirmation" / f"timing_internal_K20_vs_terminal_K{k_b20}{tag}.json"
    if path.exists():
        raise FileExistsError(f"refusing to overwrite {path}")
    ev = Evaluator(NEW_HOLDOUT, NEW_SELECTION_SEED, OUT_ROOT / "timing", "timing")
    timing = ev.timing_pair(
        fixed("internal", 123, "P_int", 20),
        fixed("terminal", 123, "P_term", k_b20),
        repeats=args.repeats,
        max_load=args.max_load,
    )
    timing["K_B20"] = k_b20
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(timing, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {"timing_ratio": timing.get("ratio_left_over_right"), "skipped": timing.get("skipped")}
        ),
        flush=True,
    )


def run_mechanism(args) -> None:
    out = OUT_ROOT / "mechanism"
    ev = Evaluator(NEW_HOLDOUT, NEW_SELECTION_SEED, out, "mechanism")
    k_max = 40
    for seed in SEEDS:
        ev.run(trajectory("clue", seed, "P_term", k_max))
    for seed in MECHANISM_SEEDS:
        ev.run(trajectory("terminal", seed, "P_int", k_max))
        ev.run(trajectory("clue", seed, "P_int", k_max))
        ev.run(trajectory("internal", seed, "P_term", k_max))
        for rounds in (1, 2, 4):
            ev.run(trajectory("internal", seed, "P_int", k_max, internal_rounds=rounds))
    # Untrained controls: the three arms share one initialization per seed (the
    # projections have no parameters), so the internal arm's initialization is
    # evaluated under all three placements after checking that identity.
    for seed in MECHANISM_SEEDS:
        reference = ev.params_for(fixed("internal", seed, "P_int", 20, params="init"))
        for arm in ("terminal", "clue"):
            other = ev.params_for(fixed(arm, seed, NATIVE[arm], 20, params="init"))
            same = jax.tree_util.tree_all(
                jax.tree_util.tree_map(lambda a, b: bool(jnp.array_equal(a, b)), reference, other)
            )
            if not same:
                raise ValueError(f"seed {seed}: {arm} initialization differs from internal")
        for placement in ("P_int", "P_term", "P_clue"):
            ev.run(fixed("internal", seed, placement, 20, params="init"))
    print(json.dumps({"status": "mechanism_complete"}), flush=True)


EXTENSION_STEPS = {"e8": 2048, "e10": 2560}


def _assert_extension_checkpoint(arm: str, variant: str) -> None:
    """An extension checkpoint must be final at the right step, carry EMA and keep e5's model."""
    with checkpoint_path(arm, 123, variant).open("rb") as stream:
        payload = pickle.load(stream)
    step = payload.get("metadata", {}).get("global_step")
    if step != EXTENSION_STEPS[variant]:
        raise ValueError(f"{arm}/{variant}: global_step {step} != {EXTENSION_STEPS[variant]}")
    if payload.get("ema_params") is None:
        raise ValueError(f"{arm}/{variant}: checkpoint has no EMA parameters")
    source = load_checkpoint_config(checkpoint_path(arm, 123, "e5"))
    if payload["config"]["model"] != source["model"]:
        raise ValueError(f"{arm}/{variant}: model config differs from the epoch-5 checkpoint")


def run_extension_eval(args) -> None:
    out = OUT_ROOT / "extension_eval"
    ev = Evaluator(NEW_HOLDOUT, NEW_SELECTION_SEED, out, "extension_eval")
    for variant in ("e8", "e10"):
        for arm in ("internal", "terminal"):
            if not checkpoint_path(arm, 123, variant).is_file():
                print(json.dumps({"missing": f"{arm}/{variant}"}), flush=True)
                continue
            _assert_extension_checkpoint(arm, variant)
            for k in (10, 20, 30):
                ev.run(fixed(arm, 123, NATIVE[arm], k, variant=variant))
    for arm in ("internal", "terminal"):
        if checkpoint_path(arm, 123, "e10").is_file():
            ev.run(trajectory(arm, 123, NATIVE[arm], 40, variant="e10"))
    print(json.dumps({"status": "extension_eval_complete"}), flush=True)


def make_finite_checker(model, k: int, window: int):
    """Per-puzzle finiteness of every decoded window and of the final ADMM state."""

    @jax.jit
    def check(params, fwd):
        logits, state, _geometry = model.apply(
            params,
            fwd["patches"],
            fwd["edge_indices"],
            num_iters=k,
            loss_window=window,
            **fwd["model_kwargs"],
            training=False,
        )
        finite = jnp.all(jnp.isfinite(logits), axis=(0, 1, 3, 4))  # [B]
        for value in (state.x, state.z, state.y):
            finite &= jnp.all(jnp.isfinite(value), axis=(0, 2))
        return finite

    return check


def run_finite_audit(args) -> None:
    """Finiteness audit of transplanted placements and the long native terminal horizon.

    The evaluator decodes by argmax, which maps NaN logits to a digit; this audit
    records, per puzzle, whether every decoded window and the final state are finite,
    so a numerically broken condition is reported as not evaluable, never as a score.
    """
    out = OUT_ROOT / "finite_audit.json"
    if out.exists():
        raise FileExistsError(f"refusing to overwrite {out}")
    ev = Evaluator(NEW_HOLDOUT, NEW_SELECTION_SEED, OUT_ROOT / "finite_audit", "finite_audit")
    checks = [(fixed("internal", seed, "P_term", 20), 1) for seed in SEEDS]
    for seed in MECHANISM_SEEDS:
        checks += [
            (fixed("terminal", seed, "P_int", 20), 1),
            (fixed("clue", seed, "P_int", 20), 1),
            (fixed("clue", seed, "P_term", 20), 1),
        ]
    checks += [(fixed(arm, 123, NATIVE[arm], 48), 48) for arm in ("terminal", "internal", "clue")]
    report = {}
    for job, window in checks:
        model_key, model, _cfg = ev.model_for(job)
        params = ev.params_for(job)
        checker = make_finite_checker(model, job["k"], window)
        finite = np.concatenate(
            [np.asarray(jax.device_get(checker(params, fwd))) for fwd in ev.prepared]
        )
        key = f"seed_{job['seed']}/{job_name(job)}/K{job['k']}/windows{window}"
        report[key] = {"puzzles": int(finite.size), "non_finite_puzzles": int((~finite).sum())}
        print(json.dumps({key: report[key]}), flush=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")


def run_smoke(args) -> None:
    """Compile-time and agreement smoke test on 64 dev puzzles (no outputs kept)."""
    ev = Evaluator(DEV_HOLDOUT, DEV_SELECTION_SEED, OUT_ROOT / "smoke", "smoke")
    ev.prepared = ev.prepared[:1]
    ev.inputs = ev.inputs[:BATCH_SIZE]
    ev.labels = ev.labels[:BATCH_SIZE]
    report = {}
    for arm in ("internal", "terminal"):
        job = trajectory(arm, 123, NATIVE[arm], args.k_max)
        model_key, model, _cfg = ev.model_for(job)
        params = ev.params_for(job)
        predictor = ev.predictor_for(model_key, model, "trajectory", args.k_max)
        started = time.perf_counter()
        traj = infer_trajectory(predictor, params, ev.prepared)
        first = time.perf_counter() - started
        started = time.perf_counter()
        infer_trajectory(predictor, params, ev.prepared)
        second = time.perf_counter() - started
        fixed_job = fixed(arm, 123, NATIVE[arm], 20)
        fixed_pred = ev.predictor_for(model_key, model, "fixed", 20)
        fixed_out = bench.infer_once(fixed_pred, params, ev.prepared)
        agree = float(np.mean(np.all(fixed_out == traj[19], axis=(1, 2))))
        report[arm] = {
            "compile_plus_first_seconds": first,
            "steady_seconds_64_puzzles": second,
            "fixed_vs_trajectory_K20_identical_puzzles": agree,
            "fixed_job": job_name(fixed_job),
        }
        print(json.dumps({arm: report[arm]}), flush=True)
    (OUT_ROOT / "smoke_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--phase",
        required=True,
        choices=(
            "smoke",
            "gate",
            "confirm",
            "timing",
            "mechanism",
            "extension_eval",
            "finite_audit",
        ),
    )
    parser.add_argument("--k-max", type=int, default=48)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--max-load", type=float, default=60.0)
    parser.add_argument("--tag", default="")
    args = parser.parse_args()
    {
        "smoke": run_smoke,
        "gate": run_gate,
        "confirm": run_confirm,
        "timing": run_timing,
        "mechanism": run_mechanism,
        "extension_eval": run_extension_eval,
        "finite_audit": run_finite_audit,
    }[args.phase](args)


if __name__ == "__main__":
    main()
