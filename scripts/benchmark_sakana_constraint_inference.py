"""Benchmark local inference latency for matched Sudoku constraint checkpoints."""

from __future__ import annotations

import argparse
import csv
import json
import os
import pickle
import platform
import statistics
import sys
import time
from pathlib import Path

# Keep compiled executables between benchmark invocations, using the same
# cache already populated by the local Sakana evaluations when available.
ROOT = Path(__file__).resolve().parents[1]
CONFIRM_ROOT = ROOT / "outputs" / "sakana_stability_constraint_confirmation"
os.environ.setdefault("JAX_COMPILATION_CACHE_DIR", str(CONFIRM_ROOT / "jax-cache"))

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402

if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

import sheaf_admm as _sheaf_admm  # noqa: E402,F401
from sheaf_admm.data import PuzzleDataset, reassemble_sudoku_logits  # noqa: E402
from sheaf_admm.data.sudoku_metrics import analyze_sudoku_predictions  # noqa: E402
from sheaf_admm.models import model_config_from_dict  # noqa: E402
from sheaf_admm.training import build_model, make_task  # noqa: E402

SEEDS = (123, 456, 789, 1011, 1213)
ARMS = ("clue_k20", "sinkhorn_k20")
SCREEN_ROOT = ROOT / "outputs" / "sakana_stability_compute_matched"
DEFAULT_HOLDOUT = ROOT / "datasets" / "sudoku_sakana_compute_matched_fresh_holdout_k5_independent"
DEFAULT_OUT = CONFIRM_ROOT / "inference_runtime_independent_k5"


def checkpoint_path(seed: int, arm: str) -> Path:
    if seed == 123:
        return SCREEN_ROOT / "n4096_e5" / "seed_123" / arm / "checkpoint.pkl"
    return CONFIRM_ROOT / f"seed_{seed}" / "n4096_e5" / arm / "checkpoint.pkl"


def load_arm(checkpoint_path_value: Path) -> dict:
    with checkpoint_path_value.open("rb") as stream:
        checkpoint = pickle.load(stream)
    cfg = checkpoint["config"]
    if cfg.get("task") != "sudoku" or cfg.get("model_type") != "sheaf":
        raise ValueError(f"unexpected checkpoint configuration: {checkpoint_path_value}")
    task_cfg = dict(cfg.get("task_cfg", {}))
    model_cfg_dict = dict(cfg["model"])
    task = make_task("sudoku", **task_cfg)
    model_cfg = model_config_from_dict(model_cfg_dict)
    model = build_model(model_cfg, "sheaf")
    return {"model_cfg": model_cfg_dict, "task": task, "model": model}


def load_params(checkpoint_path_value: Path):
    with checkpoint_path_value.open("rb") as stream:
        checkpoint = pickle.load(stream)
    params = checkpoint.get("ema_params")
    if params is None:
        params = checkpoint["params"]
    return jax.tree_util.tree_map(jnp.asarray, params)


def assert_shared_model_config(seeds, arms) -> None:
    """Require every seed to match the first seed's model and task config.

    The benchmarks build one model per arm from the first seed's checkpoint and
    then apply every seed's parameters to it. A seed trained with a different
    config (for example another Sinkhorn iteration count) would otherwise be
    evaluated silently under the first seed's settings.
    """
    for arm in arms:
        reference = None
        for seed in seeds:
            with checkpoint_path(seed, arm).open("rb") as stream:
                cfg = pickle.load(stream)["config"]
            signature = (cfg["model"], cfg.get("task_cfg", {}))
            if reference is None:
                reference = signature
            elif signature != reference:
                raise ValueError(
                    f"{arm} seed {seed} model/task config differs from seed {seeds[0]}"
                )


def prepare_batches(data_dir: Path, task, batch_size: int):
    prepared = []
    all_inputs = []
    all_labels = []
    dataset = PuzzleDataset(data_dir, "test_hard")
    for _name, source in dataset.iter_test_batches(batch_size=batch_size):
        inputs = np.asarray(source["inputs"])
        labels = np.asarray(source["labels"])
        fwd, targets, _aux = task.prepare({"inputs": inputs, "labels": labels})
        prepared.append(fwd)
        all_inputs.append(inputs.reshape(-1, task.grid_size, task.grid_size))
        all_labels.append(np.asarray(targets["labels_grid"]))
    return prepared, np.concatenate(all_inputs), np.concatenate(all_labels)


def make_predictor(model, task, k: int):
    @jax.jit
    def predict(params, fwd):
        logits, _state, _geometry = model.apply(
            params,
            fwd["patches"],
            fwd["edge_indices"],
            num_iters=k,
            loss_window=1,
            **fwd["model_kwargs"],
            training=False,
        )
        recon = reassemble_sudoku_logits(
            jnp.transpose(logits[-1], (1, 0, 2, 3)),
            use_boxes=task.use_boxes,
        )
        return jnp.argmax(recon, axis=-1)

    return predict


def infer_once(predictor, params, prepared_batches) -> np.ndarray:
    # device_get forces completion of JAX's asynchronous CPU work before timing ends.
    predictions = [np.asarray(jax.device_get(predictor(params, fwd))) for fwd in prepared_batches]
    return np.concatenate(predictions, axis=0)


def describe_latency(samples: list[float], puzzle_count: int, exact_rate: float) -> dict:
    median_seconds = statistics.median(samples)
    return {
        "runs_seconds": [round(value, 6) for value in samples],
        "median_seconds": median_seconds,
        "min_seconds": min(samples),
        "max_seconds": max(samples),
        "milliseconds_per_puzzle": 1000.0 * median_seconds / puzzle_count,
        "puzzles_per_second": puzzle_count / median_seconds,
        "expected_exact_solutions_per_second": puzzle_count * exact_rate / median_seconds,
        "seconds_per_exact_solution": (
            median_seconds / (puzzle_count * exact_rate) if exact_rate > 0 else None
        ),
    }


def save_per_puzzle_rows(
    out_dir: Path,
    seed: int,
    arm: str,
    k: int,
    rows: list[dict[str, float | int | bool]],
) -> Path:
    """Persist paired puzzle-level outcomes so later strata remain auditable."""
    path = out_dir / "per_puzzle" / f"seed_{seed}" / arm / f"k_{k}.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["seed", "arm", "K", *rows[0].keys()]
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({"seed": seed, "arm": arm, "K": k, **row})
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--holdout-dir", type=Path, default=DEFAULT_HOLDOUT)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--seeds", default=",".join(map(str, SEEDS)))
    parser.add_argument("--k-values", default="5,20")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()
    holdout_dir = args.holdout_dir if args.holdout_dir.is_absolute() else ROOT / args.holdout_dir
    out_dir = args.out_dir if args.out_dir.is_absolute() else ROOT / args.out_dir
    seeds = tuple(int(value.strip()) for value in args.seeds.split(",") if value.strip())
    k_values = tuple(int(value.strip()) for value in args.k_values.split(",") if value.strip())
    if not seeds or not k_values or args.repeats < 2:
        raise ValueError("provide seeds, K values, and at least two timed repeats")
    if jax.default_backend() != "cpu":
        raise RuntimeError(f"expected local CPU inference, got {jax.devices()}")

    manifest_path = holdout_dir / "fresh_holdout_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("official_test_used") or manifest.get("count") != 1024:
        raise ValueError("independent holdout audit failed")

    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    summaries = []
    assert_shared_model_config(seeds, ARMS)
    base_models = {arm: load_arm(checkpoint_path(seeds[0], arm)) for arm in ARMS}
    base_task = base_models[ARMS[0]]["task"]
    for arm in ARMS[1:]:
        if (
            base_models[arm]["task"].grid_size != base_task.grid_size
            or base_models[arm]["task"].use_boxes != base_task.use_boxes
        ):
            raise ValueError("paired checkpoints use different Sudoku task geometry")
    predictors_by_k = {
        k: {
            arm: make_predictor(base_models[arm]["model"], base_models[arm]["task"], k)
            for arm in ARMS
        }
        for k in k_values
    }

    for seed in seeds:
        task = base_task
        params_by_arm = {arm: load_params(checkpoint_path(seed, arm)) for arm in ARMS}
        prepared_batches, inputs, labels = prepare_batches(holdout_dir, task, args.batch_size)
        if len(inputs) != manifest["count"]:
            raise ValueError(
                f"holdout count mismatch: manifest={manifest['count']}, loaded={len(inputs)}"
            )

        for k in k_values:
            predictors = {}
            warmups = {}
            metrics_by_arm = {}
            puzzle_rows_by_arm = {}
            for arm in ARMS:
                predictor = predictors_by_k[k][arm]
                predictors[arm] = predictor
                started = time.perf_counter()
                warm_prediction = infer_once(predictor, params_by_arm[arm], prepared_batches)
                warmups[arm] = time.perf_counter() - started
                metrics, puzzle_rows = analyze_sudoku_predictions(
                    inputs, labels, warm_prediction, use_boxes=task.use_boxes
                )
                metrics_by_arm[arm] = metrics
                puzzle_rows_by_arm[arm] = puzzle_rows
                save_per_puzzle_rows(out_dir, seed, arm, k, puzzle_rows)

            # Alternating pair order reduces bias from thermal/load drift.
            timed = {arm: [] for arm in ARMS}
            for repeat in range(args.repeats):
                order = ARMS if repeat % 2 == 0 else tuple(reversed(ARMS))
                for arm in order:
                    started = time.perf_counter()
                    _ = infer_once(predictors[arm], params_by_arm[arm], prepared_batches)
                    timed[arm].append(time.perf_counter() - started)

            arm_results = {}
            for arm in ARMS:
                latency = describe_latency(
                    timed[arm], len(inputs), float(metrics_by_arm[arm]["exact_solution_rate"])
                )
                result = {
                    "seed": seed,
                    "arm": arm,
                    "K": k,
                    "puzzles": len(inputs),
                    "first_pass_seconds": warmups[arm],
                    "metrics": metrics_by_arm[arm],
                    "steady_state": latency,
                }
                rows.append(result)
                arm_results[arm] = result

            clue = arm_results["clue_k20"]
            sink = arm_results["sinkhorn_k20"]
            pair = {
                "seed": seed,
                "K": k,
                "clue_exact_rate": clue["metrics"]["exact_solution_rate"],
                "sinkhorn_exact_rate": sink["metrics"]["exact_solution_rate"],
                "exact_rate_delta_percentage_points": 100.0
                * (sink["metrics"]["exact_solution_rate"] - clue["metrics"]["exact_solution_rate"]),
                "clue_median_seconds": clue["steady_state"]["median_seconds"],
                "sinkhorn_median_seconds": sink["steady_state"]["median_seconds"],
                "sinkhorn_over_clue_runtime_ratio": (
                    sink["steady_state"]["median_seconds"] / clue["steady_state"]["median_seconds"]
                ),
                "clue_seconds_per_exact_solution": clue["steady_state"][
                    "seconds_per_exact_solution"
                ],
                "sinkhorn_seconds_per_exact_solution": sink["steady_state"][
                    "seconds_per_exact_solution"
                ],
            }
            summaries.append(pair)
            progress = {
                "status": "running",
                "benchmark": (
                    "CPU steady-state JAX forward inference; task preparation and dataset "
                    "loading excluded; output device synchronization included"
                ),
                "device": str(jax.devices()[0]),
                "jax_version": jax.__version__,
                "holdout_dir": str(holdout_dir),
                "official_test_used": manifest.get("official_test_used", False),
                "puzzles": manifest["count"],
                "batch_size": args.batch_size,
                "repeats": args.repeats,
                "seeds": list(seeds),
                "k_values": list(k_values),
                "results": rows,
                "paired_summary": summaries,
            }
            (out_dir / "inference_runtime_benchmark.json").write_text(
                json.dumps(progress, indent=2), encoding="utf-8"
            )
            print(json.dumps({"status": "pair_complete", **pair}), flush=True)

    payload = {
        "status": "complete",
        "benchmark": (
            "CPU steady-state JAX forward inference; task preparation and dataset "
            "loading excluded; output device synchronization included"
        ),
        "device": str(jax.devices()[0]),
        "jax_version": jax.__version__,
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "processor": platform.processor(),
        "holdout_dir": str(holdout_dir),
        "holdout_selection_seed": manifest.get("selection_seed"),
        "official_test_used": manifest.get("official_test_used", False),
        "puzzles": manifest["count"],
        "batch_size": args.batch_size,
        "repeats": args.repeats,
        "seeds": list(seeds),
        "k_values": list(k_values),
        "results": rows,
        "paired_summary": summaries,
    }
    (out_dir / "inference_runtime_benchmark.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )
    with (out_dir / "inference_runtime_benchmark.csv").open(
        "w", newline="", encoding="utf-8"
    ) as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=[
                "seed",
                "K",
                "arm",
                "puzzles",
                "exact_solution_rate",
                "hidden_accuracy",
                "first_pass_seconds",
                "median_seconds",
                "min_seconds",
                "max_seconds",
                "milliseconds_per_puzzle",
                "puzzles_per_second",
                "expected_exact_solutions_per_second",
                "seconds_per_exact_solution",
            ],
        )
        writer.writeheader()
        for result in rows:
            writer.writerow(
                {
                    "seed": result["seed"],
                    "K": result["K"],
                    "arm": result["arm"],
                    "puzzles": result["puzzles"],
                    "exact_solution_rate": result["metrics"]["exact_solution_rate"],
                    "hidden_accuracy": result["metrics"]["hidden_accuracy"],
                    "first_pass_seconds": result["first_pass_seconds"],
                    **{
                        key: result["steady_state"][key]
                        for key in (
                            "median_seconds",
                            "min_seconds",
                            "max_seconds",
                            "milliseconds_per_puzzle",
                            "puzzles_per_second",
                            "expected_exact_solutions_per_second",
                            "seconds_per_exact_solution",
                        )
                    },
                }
            )

    report_lines = [
        "# CPU inference runtime: clue-lock vs Sinkhorn",
        "",
        (
            f"- Independent holdout: {manifest['count']} puzzles; official test used: "
            f"{manifest.get('official_test_used', False)}."
        ),
        (
            f"- Hardware: {platform.processor() or platform.platform()} "
            f"({jax.devices()[0]}), JAX {jax.__version__}."
        ),
        (
            f"- Batch size: {args.batch_size}; timed repeats per arm: {args.repeats}; "
            f"seeds: {', '.join(map(str, seeds))}."
        ),
        (
            "- Timings include synchronized model inference and host output transfer; "
            "dataset loading/task preparation are excluded."
        ),
        (
            "- The first full pass is reported separately and excluded from steady-state "
            "medians; it may include JIT compilation/cache effects."
        ),
        "",
        (
            "| K | Seeds | Clue exact | Sinkhorn exact | Accuracy delta | "
            "Sinkhorn/clue time | Clue→Sinkhorn ms/exact | Positive seeds |"
        ),
        "|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for k in k_values:
        group = [item for item in summaries if item["K"] == k]

        def mean(key, group=group):
            return statistics.mean(float(item[key]) for item in group)

        positive_seeds = sum(item["exact_rate_delta_percentage_points"] > 0 for item in group)
        report_lines.append(
            f"| {k} | {len(group)} | {100 * mean('clue_exact_rate'):.2f}% | "
            f"{100 * mean('sinkhorn_exact_rate'):.2f}% | "
            f"{mean('exact_rate_delta_percentage_points'):+.2f} pp | "
            f"{mean('sinkhorn_over_clue_runtime_ratio'):.2f}x | "
            f"{1000 * mean('clue_seconds_per_exact_solution'):.1f}→"
            f"{1000 * mean('sinkhorn_seconds_per_exact_solution'):.1f} ms | "
            f"{positive_seeds}/{len(group)} |"
        )
    report_lines.extend(
        [
            "",
            (
                "Per-seed measurements and repeat ranges are in "
                "`inference_runtime_benchmark.csv`; full provenance is in the JSON."
            ),
            "",
            (
                "Interpretation caution: this is single-process CPU inference, not a "
                "distributed network/message-count benchmark. Sinkhorn adds local "
                "computation and is not expected to reduce communications."
            ),
        ]
    )
    (out_dir / "inference_runtime_benchmark.md").write_text(
        "\n".join(report_lines) + "\n", encoding="utf-8"
    )
    print(
        json.dumps({"status": "complete", "out_dir": str(out_dir), "pairs": len(summaries)}),
        flush=True,
    )


if __name__ == "__main__":
    main()
