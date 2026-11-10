"""Resumable P3 factorial: message budget × iterative clue commitment.

Development only. The calibration families are 0–31; evaluation families are
32–63. Each model pass is checkpointed atomically so an interrupted run can
resume at the next complete pass without repeating prior work.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import pickle
import sys
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))
import evaluate_sakana_placement_horizon as horizon  # noqa: E402
from sheaf_admm.data import reassemble_sudoku_logits  # noqa: E402
from sheaf_admm.models import SheafADMMModel, model_config_from_dict  # noqa: E402
from sheaf_admm.training import build_model, make_task  # noqa: E402

DATA = ROOT / "datasets/sudoku_coordination_pilot_20261007"
OUT = ROOT / "outputs/coordination_pilot_20261007/p3_communication_decimation"
TRAIN_SEEDS = (123, 456, 789, 1011, 1213, 20260927)
ARMS = ("internal", "terminal")
K = 20
CALIBRATION_FAMILIES = 32
BASE_EVAL_FAMILY = 32
MASK_SEEDS = (391, 773, 1237)
MAX_FRACTION = 0.25
STOP_NAMES = {0: "active", 1: "valid", 2: "stuck", 3: "contradiction", 4: "max_passes"}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def atomic_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("wb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(tmp, path)


def atomic_npz(path: Path, arrays: dict[str, np.ndarray], provenance: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    path.parent.mkdir(parents=True, exist_ok=True)
    with tmp.open("wb") as stream:
        np.savez_compressed(stream, **arrays)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(tmp, path)
    sidecar = dict(provenance, output_sha256=sha256(path))
    atomic_bytes(path.with_suffix(".json"), json.dumps(sidecar, indent=2).encode("utf-8"))


def verified_npz(path: Path, provenance: dict) -> dict[str, np.ndarray] | None:
    sidecar_path = path.with_suffix(".json")
    if not path.is_file() or not sidecar_path.is_file():
        return None
    saved = json.loads(sidecar_path.read_text(encoding="utf-8"))
    expected = dict(provenance, output_sha256=saved.get("output_sha256"))
    if saved != expected or sha256(path) != saved["output_sha256"]:
        raise RuntimeError(f"P3 pass provenance or SHA-256 mismatch: {path}")
    with np.load(path, allow_pickle=False) as loaded:
        return {key: loaded[key] for key in loaded.files}


def checkpoint_path(arm: str, seed: int) -> Path:
    if arm == "terminal":
        return ROOT / f"outputs/sakana_placement_confirmation/seed_{seed}/n4096_e5/terminal12_k20/best_checkpoint.pkl"
    if seed == 123:
        return ROOT / f"outputs/sakana_stability_compute_matched/n4096_e5/seed_{seed}/sinkhorn_k20/best_checkpoint.pkl"
    return ROOT / f"outputs/sakana_stability_constraint_confirmation/seed_{seed}/n4096_e5/sinkhorn_k20/best_checkpoint.pkl"


def make_predictor(model, policy: str):
    """Compile one static message policy; masks and random seed remain inputs."""

    @jax.jit
    def predict(params, fwd, active_count_schedule, mask_seed):
        summary, logits, state, geometry, _rho = model.apply(
            params,
            fwd["patches"],
            fwd["edge_indices"],
            num_iters=K,
            threshold=0.0,
            gate_mode="absolute",
            mask_policy=policy,
            active_count_schedule=active_count_schedule,
            mask_seed=mask_seed,
            **fwd["model_kwargs"],
            training=False,
            method=SheafADMMModel.coordinate_triggered,
        )
        views = jnp.transpose(logits[-1], (1, 0, 2, 3))  # [B,27,9,C]
        mean_logits = reassemble_sudoku_logits(views)
        probs = jax.nn.softmax(mean_logits, axis=-1)
        family_logits = []
        for family in range(3):
            family_mask = jnp.zeros((27,)).at[9 * family : 9 * (family + 1)].set(1.0)
            family_logits.append(
                3.0 * reassemble_sudoku_logits(views * family_mask[None, :, None, None])
            )
        family_probs = jax.nn.softmax(jnp.stack(family_logits, axis=1), axis=-1)
        local_votes = jnp.argmax(family_probs, axis=-1)
        consensus = jnp.argmax(probs, axis=-1)
        agreement = jnp.all(local_votes == consensus[:, None], axis=1)
        return (
            probs,
            agreement,
            summary.active_edge_count,
            summary.edge_disagreement_mean[-1],
            jnp.asarray(geometry.restriction_maps.shape[-2], dtype=jnp.int32),
            jnp.mean(jnp.linalg.norm(state.x - state.z, axis=-1)),
        )

    return predict


def load_model(arm: str, seed: int):
    path = checkpoint_path(arm, seed)
    with path.open("rb") as stream:
        saved = pickle.load(stream)
    cfg = saved["config"]
    params = saved.get("ema_params")
    if params is None:
        params = saved["params"]
    params = jax.tree_util.tree_map(jnp.asarray, params)
    model = build_model(model_config_from_dict(dict(cfg["model"])), "sheaf")
    task = make_task("sudoku", **dict(cfg.get("task_cfg", {})))
    return path, sha256(path), cfg, params, model, task


def prepare_inputs(grids: np.ndarray, task):
    flat = np.asarray(grids, dtype=np.int64).reshape(len(grids), 81)
    fwd, _targets, _aux = task.prepare({"inputs": flat, "labels": np.zeros_like(flat)})
    return fwd


def run_prediction(predictor, params, task, grids, policy, fraction, mask_seed):
    fwd = prepare_inputs(grids, task)
    edge_count = int(fwd["edge_indices"].shape[0])
    batch_size = int(len(grids))
    if policy == "none":
        schedule = jnp.zeros((K,), dtype=jnp.int32)
    else:
        slots = edge_count * batch_size
        schedule = jnp.full((K,), int(round(fraction * slots)), dtype=jnp.int32)
    probs, agreement, active, edge_disagreement, stalk_dim, xz_gap = jax.device_get(
        predictor(params, fwd, schedule, jnp.asarray(mask_seed, dtype=jnp.int32))
    )
    probs = np.asarray(probs)
    predictions = probs.argmax(axis=-1).astype(np.uint8)
    confidence = probs.max(axis=-1).astype(np.float32)
    return {
        "predictions": predictions,
        "confidence": confidence,
        "agreement": np.asarray(agreement, dtype=np.uint8),
        "active_edge_slots_by_iteration": np.asarray(active, dtype=np.int64),
        "edge_disagreement": np.asarray(edge_disagreement, dtype=np.float32),
        "stalk_dim": int(stalk_dim),
        "xz_gap": float(xz_gap),
        "edge_count": edge_count,
    }


def select_threshold(predictions, confidence, inputs, labels, target_precision=0.995):
    empty = inputs == 0
    values = confidence[empty]
    correct = predictions[empty] == labels[empty]
    candidates = np.unique(values)
    acceptable = []
    for tau in candidates:
        selected = values >= tau
        if selected.any() and float(correct[selected].mean()) >= target_precision:
            acceptable.append((float(tau), float(selected.mean()), float(correct[selected].mean())))
    if not acceptable:
        return {"threshold": 1.000001, "coverage": 0.0, "precision": None,
                "target_precision": target_precision, "status": "no_threshold_reached_target"}
    # Lowest acceptable threshold gives the largest observed coverage; use the
    # inclusive mask actually used at inference so tied confidences are handled.
    tau, coverage, precision = min(acceptable, key=lambda item: item[0])
    return {"threshold": tau, "coverage": coverage, "precision": precision,
            "target_precision": target_precision, "status": "calibrated"}


def calibrate(model_ctx, dev_inputs, dev_labels, code_hash, dataset_hash, out_dir):
    path, checkpoint_sha, cfg, params, _model, task = model_ctx
    predictor = make_predictor(model_ctx[4], "random")
    fwd = prepare_inputs(dev_inputs, task)
    edge_count, batch_size = int(fwd["edge_indices"].shape[0]), len(dev_inputs)
    schedule = jnp.full((K,), edge_count * batch_size, dtype=jnp.int32)
    probs, _agreement, _active, _dis, _d, _gap = jax.device_get(
        predictor(params, fwd, schedule, jnp.asarray(7, dtype=jnp.int32))
    )
    probs = np.asarray(probs)
    result = select_threshold(probs.argmax(-1), probs.max(-1), dev_inputs, dev_labels)
    result.update({
        "checkpoint": str(path.relative_to(ROOT)),
        "checkpoint_sha256": checkpoint_sha,
        "dataset_manifest_sha256": dataset_hash,
        "code_sha256": code_hash,
        "calibration_families": CALIBRATION_FAMILIES,
        "clues": 40,
        "distribution": "concentrated",
        "K": K,
        "input_labels_not_provided_to_model": True,
    })
    return result


def is_valid_grid(grid: np.ndarray, original_clues: np.ndarray) -> bool:
    if np.any((grid < 1) | (grid > 9)) or np.any((original_clues > 0) & (grid != original_clues)):
        return False
    units = list(grid) + list(grid.T) + [
        grid[r : r + 3, c : c + 3].ravel()
        for r in (0, 3, 6) for c in (0, 3, 6)
    ]
    return all(len(set(unit.tolist())) == 9 for unit in units)


def legal(grid: np.ndarray, row: int, col: int, digit: int) -> bool:
    if grid[row, col] != 0 or digit < 1 or digit > 9:
        return False
    br, bc = 3 * (row // 3), 3 * (col // 3)
    return digit not in grid[row] and digit not in grid[:, col] and digit not in grid[br:br+3, bc:bc+3]


def has_dead_cell(grid: np.ndarray) -> bool:
    for row, col in zip(*np.nonzero(grid == 0), strict=True):
        if not any(legal(grid, int(row), int(col), digit) for digit in range(1, 10)):
            return True
    return False


def result_metrics(predictions, original_inputs, labels, common):
    metrics = horizon.grid_metrics(original_inputs, labels, predictions)
    common_correct = ((predictions.reshape(len(predictions), 81) == labels.reshape(len(labels), 81)) & common).sum(1)
    common_n = common.sum(1)
    common_accuracy = common_correct / np.maximum(common_n, 1)
    return metrics, common_accuracy


def run_branch(
    *, out_dir, provenance_base, method, initial_inputs, labels, common, predictor,
    params, task, policy, fraction, mask_seed_base, tau, max_passes,
):
    branch_dir = out_dir / method
    branch_dir.mkdir(parents=True, exist_ok=True)
    batch_size = len(initial_inputs)
    grids = initial_inputs.copy()
    stopped = np.zeros(batch_size, dtype=bool)
    stop_reason = np.zeros(batch_size, dtype=np.uint8)
    passes_used = np.zeros(batch_size, dtype=np.uint8)
    cumulative_commits = np.zeros(batch_size, dtype=np.uint16)
    cumulative_wrong = np.zeros(batch_size, dtype=np.uint16)
    last_predictions = np.zeros((batch_size, 9, 9), dtype=np.uint8)
    history = []

    for pass_index in range(max_passes):
        pass_path = branch_dir / f"pass_{pass_index + 1:02d}.npz"
        mask_seed = int((mask_seed_base + (pass_index + 1) * 104729) % 2_000_000_000)
        provenance = dict(
            provenance_base,
            method=method,
            pass_index=pass_index + 1,
            mask_seed=mask_seed,
            threshold=float(tau),
        )
        saved = verified_npz(pass_path, provenance)
        if saved is not None:
            grids = saved["grid_after"].astype(np.uint8)
            stopped = saved["stopped"].astype(bool)
            stop_reason = saved["stop_reason"].astype(np.uint8)
            passes_used = saved["passes_used"].astype(np.uint8)
            cumulative_commits = saved["cumulative_commits"].astype(np.uint16)
            cumulative_wrong = saved["cumulative_wrong"].astype(np.uint16)
            last_predictions = saved["last_predictions"].astype(np.uint8)
            history.append(saved)
            continue

        grid_before = grids.copy()
        pass_started = time.perf_counter()
        prediction = run_prediction(predictor, params, task, grid_before, policy, fraction, mask_seed)
        pass_seconds = time.perf_counter() - pass_started
        raw_predictions = prediction["predictions"]
        scored_predictions = raw_predictions.copy()
        if method == "decimation":
            for i in range(batch_size):
                if stopped[i]:
                    scored_predictions[i] = last_predictions[i]
                    continue
                passes_used[i] = pass_index + 1
                last_predictions[i] = raw_predictions[i]
                if is_valid_grid(raw_predictions[i], initial_inputs[i]):
                    stopped[i] = True
                    stop_reason[i] = 1
                    continue
                if pass_index == max_passes - 1:
                    stopped[i] = True
                    stop_reason[i] = 4
                    continue
                candidates = []
                for row, col in zip(*np.nonzero(grids[i] == 0), strict=True):
                    score = float(prediction["confidence"][i, row, col])
                    if score >= tau:
                        candidates.append((score, int(row), int(col)))
                candidates.sort(key=lambda item: (-item[0], item[1], item[2]))
                empty_count = int(np.count_nonzero(grids[i] == 0))
                limit = max(1, int(round(MAX_FRACTION * empty_count)))
                added = 0
                for _score, row, col in candidates:
                    if added >= limit:
                        break
                    digit = int(raw_predictions[i, row, col])
                    if legal(grids[i], row, col, digit):
                        grids[i, row, col] = digit
                        added += 1
                        cumulative_commits[i] += 1
                        cumulative_wrong[i] += int(digit != labels[i, row, col])
                if added == 0:
                    stopped[i] = True
                    stop_reason[i] = 2
                elif has_dead_cell(grids[i]):
                    stopped[i] = True
                    stop_reason[i] = 3
        else:
            passes_used[:] = pass_index + 1
            last_predictions = raw_predictions.copy()
            grids = initial_inputs.copy()

        metrics, common_accuracy = result_metrics(
            scored_predictions, initial_inputs, labels, common
        )
        agreement_map = prediction["agreement"].astype(bool)
        agreement_score = (agreement_map.reshape(batch_size, 81) * common).sum(1) / np.maximum(common.sum(1), 1)
        arrays = {
            "predictions": raw_predictions,
            "predictions_scored": scored_predictions,
            "grid_before": grid_before,
            "grid_after": grids,
            "last_predictions": last_predictions,
            "stopped": stopped.astype(np.uint8),
            "stop_reason": stop_reason,
            "passes_used": passes_used,
            "cumulative_commits": cumulative_commits,
            "cumulative_wrong": cumulative_wrong,
            "exact": np.asarray(metrics["exact"], dtype=np.uint8),
            "valid": np.asarray(metrics["valid"], dtype=np.uint8),
            "hidden_accuracy": np.asarray(metrics["hidden_accuracy"], dtype=np.float32),
            "common_accuracy": np.asarray(common_accuracy, dtype=np.float32),
            "agent_agreement_common": np.asarray(agreement_score, dtype=np.float32),
            "active_edge_slots_by_iteration": prediction["active_edge_slots_by_iteration"],
            "edge_disagreement": np.asarray([prediction["edge_disagreement"]], dtype=np.float32),
            "xz_gap": np.asarray([prediction["xz_gap"]], dtype=np.float32),
            "stalk_dim": np.asarray([prediction["stalk_dim"]], dtype=np.int32),
            "edge_count": np.asarray([prediction["edge_count"]], dtype=np.int32),
            "confidence": prediction["confidence"],
            "agreement_map": agreement_map.astype(np.uint8),
            "mask_seed": np.asarray([mask_seed], dtype=np.int64),
            "pass_seconds": np.asarray([pass_seconds], dtype=np.float32),
        }
        atomic_npz(pass_path, arrays, provenance)
        history.append(arrays)
        print(json.dumps({
            "pass_saved": str(pass_path.relative_to(ROOT)),
            "method": method,
            "pass": pass_index + 1,
            "mean_exact": float(np.mean(arrays["exact"])),
            "mean_commits": int(np.sum(cumulative_commits)),
            "elapsed_seconds": round(pass_seconds, 2),
        }), flush=True)
    return history


def condition_list(mask_reps: int):
    result = [{"name": "full_100", "policy": "random", "fraction": 1.0, "rep": 0}]
    for policy in ("random", "periodic"):
        for fraction in (0.75, 0.50):
            for rep in range(mask_reps):
                result.append({
                    "name": f"{policy}_{int(fraction * 100)}_r{rep}",
                    "policy": policy,
                    "fraction": fraction,
                    "rep": rep,
                })
    result.append({"name": "none_0", "policy": "none", "fraction": 0.0, "rep": 0})
    return result


def write_csv(path: Path, rows: list[dict]):
    if not rows:
        return
    tmp = path.with_suffix(path.suffix + ".tmp")
    path.parent.mkdir(parents=True, exist_ok=True)
    with tmp.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(tmp, path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--families", type=int, default=32,
                        help="number of reserved families to run (1..32)")
    parser.add_argument("--seeds", default="123,456")
    parser.add_argument("--arms", default="internal,terminal")
    parser.add_argument("--max-passes", type=int, default=8)
    parser.add_argument("--mask-reps", type=int, default=3)
    args = parser.parse_args()
    if not 1 <= args.families <= 32 or not 1 <= args.max_passes <= 8 or not 1 <= args.mask_reps <= 3:
        parser.error("families must be 1..32, max-passes 1..8, mask-reps 1..3")
    seeds = tuple(int(value) for value in args.seeds.split(",") if value)
    arms = tuple(value.strip() for value in args.arms.split(",") if value.strip())
    if any(seed not in TRAIN_SEEDS for seed in seeds) or any(arm not in ARMS for arm in arms):
        parser.error(f"seeds must be drawn from {TRAIN_SEEDS}; arms from {ARMS}")
    code_hash = sha256(Path(__file__))
    arm_tag = "".join({"internal": "i", "terminal": "t"}[arm] for arm in arms)
    run_out = OUT / (
        f"n{args.families}_s{'-'.join(map(str, seeds))}_a{arm_tag}_"
        f"p{args.max_passes}m{args.mask_reps}_{code_hash[:8]}"
    )
    run_out.mkdir(parents=True, exist_ok=True)

    manifest_path = DATA / "paired_clue_manifest.json"
    manifest_hash = sha256(manifest_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    all_rows = manifest["families"]
    eval_rows = sorted(
        [row for row in all_rows if BASE_EVAL_FAMILY <= int(row["family_id"]) < BASE_EVAL_FAMILY + args.families
         and row["policy"] == "concentrated" and int(row["clues"]) == 40],
        key=lambda row: int(row["family_id"]),
    )
    dev_rows = sorted(
        [row for row in all_rows if int(row["family_id"]) < CALIBRATION_FAMILIES
         and row["policy"] == "concentrated" and int(row["clues"]) == 40],
        key=lambda row: int(row["family_id"]),
    )
    if len(eval_rows) != args.families or len(dev_rows) != CALIBRATION_FAMILIES:
        raise RuntimeError("P3 calibration/evaluation family selection is incomplete")
    x_all = np.load(DATA / "test_hard/all__inputs.npy", mmap_mode="r")
    y_all = np.load(DATA / "test_hard/all__labels.npy", mmap_mode="r")

    def select(rows):
        indices = [int(row["row_index"]) for row in rows]
        x = np.asarray(x_all[indices], dtype=np.uint8).reshape(-1, 9, 9)
        y = np.asarray(y_all[indices], dtype=np.uint8).reshape(-1, 9, 9)
        common = np.zeros((len(rows), 81), dtype=bool)
        for i, row in enumerate(rows):
            common[i, np.asarray(row["common_evaluation_cells"], dtype=np.int32)] = True
        return indices, x, y, common

    dev_indices, dev_inputs, dev_labels, _dev_common = select(dev_rows)
    eval_indices, inputs, labels, common = select(eval_rows)
    data_key = {
        "manifest_sha256": manifest_hash,
        "eval_input_sha256": hashlib.sha256(inputs.tobytes()).hexdigest(),
        "eval_label_sha256": hashlib.sha256(labels.tobytes()).hexdigest(),
        "eval_row_indices": eval_indices,
        "dev_row_indices": dev_indices,
    }
    run_config = {
        "phase": "P3_development_factorial",
        "families": args.families,
        "seeds": seeds,
        "arms": arms,
        "max_passes": args.max_passes,
        "K_per_pass": K,
        "clues": 40,
        "distribution": "concentrated",
        "mask_reps": args.mask_reps,
        "conditions": condition_list(args.mask_reps),
        "max_commit_fraction_per_pass": MAX_FRACTION,
        "code_sha256": code_hash,
        "data": data_key,
        "backend": jax.default_backend(),
    }
    atomic_bytes(run_out / "run_config.json", json.dumps(run_config, indent=2).encode("utf-8"))

    calibration_table = {}
    calibrations_path = run_out / "calibration.json"
    if calibrations_path.exists():
        saved = json.loads(calibrations_path.read_text(encoding="utf-8"))
        if saved.get("run_config_sha256") != hashlib.sha256(
            json.dumps(run_config, sort_keys=True, default=list).encode("utf-8")
        ).hexdigest():
            raise RuntimeError("existing threshold calibration belongs to another P3 run")
        calibration_table = saved["thresholds"]

    model_contexts = {}
    for seed in seeds:
        for arm in arms:
            path, checkpoint_sha, cfg, params, model, task = load_model(arm, seed)
            context = (path, checkpoint_sha, cfg, params, model, task)
            model_contexts[(seed, arm)] = context
            key = f"seed_{seed}/{arm}"
            if key not in calibration_table:
                cal = calibrate(context, dev_inputs, dev_labels, code_hash, manifest_hash, run_out)
                calibration_table[key] = cal
                atomic_bytes(calibrations_path, json.dumps({
                    "run_config_sha256": hashlib.sha256(
                        json.dumps(run_config, sort_keys=True, default=list).encode("utf-8")
                    ).hexdigest(),
                    "thresholds": calibration_table,
                }, indent=2).encode("utf-8"))
                print(json.dumps({"calibration": key, **cal}, default=str), flush=True)

    pass_rows = []
    paired_rows = []
    condition_records = condition_list(args.mask_reps)
    for (seed, arm), context in model_contexts.items():
        path, checkpoint_sha, cfg, params, model, task = context
        predictors = {
            policy: make_predictor(model, policy)
            for policy in ("random", "periodic", "none")
        }
        threshold = float(calibration_table[f"seed_{seed}/{arm}"]["threshold"])
        cg_iters = int(cfg["model"].get("cg_iters", 3))
        for condition in condition_records:
            comm_name = condition["name"]
            policy = condition["policy"]
            fraction = float(condition["fraction"])
            rep = int(condition["rep"])
            mask_seed_base = MASK_SEEDS[rep]
            condition_dir = run_out / f"s{seed}_{arm[0]}" / comm_name
            provenance_base = {
                "checkpoint_sha256": checkpoint_sha,
                "dataset_manifest_sha256": manifest_hash,
                "data_key": hashlib.sha256(json.dumps(data_key, sort_keys=True).encode()).hexdigest(),
                "code_sha256": code_hash,
                "seed": seed,
                "arm": arm,
                "condition": comm_name,
                "communication_policy": policy,
                "communication_fraction": fraction,
                "mask_rep": rep,
                "row_indices": eval_indices,
                "K": K,
                "max_passes": args.max_passes,
            }
            predictor = predictors[policy]
            branch_histories = {}
            for method in ("direct", "decimation"):
                history = run_branch(
                    out_dir=condition_dir,
                    provenance_base=provenance_base,
                    method=method,
                    initial_inputs=inputs,
                    labels=labels,
                    common=common,
                    predictor=predictor,
                    params=params,
                    task=task,
                    policy=policy,
                    fraction=fraction,
                    mask_seed_base=mask_seed_base,
                    tau=threshold,
                    max_passes=args.max_passes,
                )
                branch_histories[method] = history
            direct_history = branch_histories["direct"]
            decimation_history = branch_histories["decimation"]
            final_direct = direct_history[-1]
            final_decimation = decimation_history[-1]
            for method, history in branch_histories.items():
                for pass_index, item in enumerate(history):
                    active_slots = int(item["active_edge_slots_by_iteration"].sum())
                    edge_count = int(item["edge_count"][0])
                    stalk_dim = int(item["stalk_dim"][0])
                    bootstrap_slots = 0 if policy == "none" else edge_count * len(inputs)
                    bootstrap_bytes = 2 * bootstrap_slots * stalk_dim * 4
                    vector_bytes = 2 * active_slots * stalk_dim * 4
                    cg_partial_bytes = 0 if policy == "none" else (
                        27 * len(inputs) * K * (1 + 2 * cg_iters) * 4
                    )
                    cg_result_bytes = 0 if policy == "none" else (
                        len(inputs) * K * (1 + 2 * cg_iters) * 4
                    )
                    pred_scored = item["predictions_scored"].astype(np.uint8)
                    metric, common_acc = result_metrics(pred_scored, inputs, labels, common)
                    for i, meta in enumerate(eval_rows):
                        pass_rows.append({
                            "family_id": int(meta["family_id"]),
                            "seed": seed,
                            "arm": arm,
                            "condition": comm_name,
                            "mask_rep": rep,
                            "method": method,
                            "pass": pass_index + 1,
                            "exact": int(metric["exact"][i]),
                            "valid": int(metric["valid"][i]),
                            "hidden_accuracy": float(metric["hidden_accuracy"][i]),
                            "common_hidden_accuracy": float(common_acc[i]),
                            "agent_agreement_common": float(item["agent_agreement_common"][i]),
                            "cumulative_commits": int(item["cumulative_commits"][i]),
                            "cumulative_wrong_commits": int(item["cumulative_wrong"][i]),
                            "passes_used": int(item["passes_used"][i]),
                            "stop_reason": STOP_NAMES[int(item["stop_reason"][i])],
                            "active_edge_slots_batch": active_slots,
                            "edge_vector_payload_bytes_batch": vector_bytes,
                            "initial_bootstrap_payload_bytes_batch": bootstrap_bytes,
                            "cg_global_partial_payload_bytes_batch": cg_partial_bytes,
                            "cg_reduced_scalar_payload_bytes_batch": cg_result_bytes,
                            "final_edge_disagreement_mean": float(item["edge_disagreement"][0]),
                            "final_xz_gap_mean": float(item["xz_gap"][0]),
                            "pass_seconds_including_compile": float(item["pass_seconds"][0]),
                        })

            # The per-puzzle direct comparator is selected at the same number
            # of complete passes actually needed by the decimation trajectory.
            for i, meta in enumerate(eval_rows):
                matched_pass = int(final_decimation["passes_used"][i])
                matched_pass = max(1, min(matched_pass, args.max_passes))
                direct_match = direct_history[matched_pass - 1]["predictions_scored"][i]
                decim_pred = final_decimation["predictions_scored"][i]
                direct_metric = horizon.grid_metrics(inputs[i:i+1], labels[i:i+1], direct_match[None])
                decim_metric = horizon.grid_metrics(inputs[i:i+1], labels[i:i+1], decim_pred[None])
                direct_common = float(((direct_match.reshape(81) == labels[i].reshape(81)) & common[i]).sum() / common[i].sum())
                decim_common = float(((decim_pred.reshape(81) == labels[i].reshape(81)) & common[i]).sum() / common[i].sum())
                gain = int(decim_metric["exact"][0]) - int(direct_metric["exact"][0])
                paired_rows.append({
                    "family_id": int(meta["family_id"]),
                    "seed": seed,
                    "arm": arm,
                    "condition": comm_name,
                    "mask_rep": rep,
                    "matched_passes": matched_pass,
                    "direct_exact_matched_budget": int(direct_metric["exact"][0]),
                    "decimation_exact": int(decim_metric["exact"][0]),
                    "decimation_gain_exact": gain,
                    "direct_common_hidden_accuracy": direct_common,
                    "decimation_common_hidden_accuracy": decim_common,
                    "gain_common_hidden_accuracy": decim_common - direct_common,
                    "decimation_commits": int(final_decimation["cumulative_commits"][i]),
                    "decimation_wrong_commits": int(final_decimation["cumulative_wrong"][i]),
                    "decimation_stop_reason": STOP_NAMES[int(final_decimation["stop_reason"][i])],
                })
            print(json.dumps({
                "condition_complete": f"seed_{seed}/{arm}/{comm_name}",
                "families": len(eval_rows),
                "mean_decimation_commits": float(final_decimation["cumulative_commits"].mean()),
                "mean_decimation_gain_matched": float(np.mean([
                    row["decimation_gain_exact"] for row in paired_rows
                    if row["seed"] == seed and row["arm"] == arm and row["condition"] == comm_name
                ])),
            }), flush=True)

    write_csv(run_out / "per_family_pass_results.csv", pass_rows)
    write_csv(run_out / "paired_budget_results.csv", paired_rows)
    summary_groups = {}
    for row in paired_rows:
        key = (row["seed"], row["arm"], row["condition"], row["mask_rep"])
        summary_groups.setdefault(key, []).append(row)
    summary = []
    for key, group in sorted(summary_groups.items()):
        summary.append({
            "seed": key[0], "arm": key[1], "condition": key[2], "mask_rep": key[3],
            "families": len(group),
            "direct_exact_matched_budget": float(np.mean([r["direct_exact_matched_budget"] for r in group])),
            "decimation_exact": float(np.mean([r["decimation_exact"] for r in group])),
            "decimation_gain_exact": float(np.mean([r["decimation_gain_exact"] for r in group])),
            "direct_common_hidden_accuracy": float(np.mean([r["direct_common_hidden_accuracy"] for r in group])),
            "decimation_common_hidden_accuracy": float(np.mean([r["decimation_common_hidden_accuracy"] for r in group])),
            "gain_common_hidden_accuracy": float(np.mean([r["gain_common_hidden_accuracy"] for r in group])),
            "mean_commits": float(np.mean([r["decimation_commits"] for r in group])),
            "mean_wrong_commits": float(np.mean([r["decimation_wrong_commits"] for r in group])),
        })
    write_csv(run_out / "summary_by_seed_condition.csv", summary)

    # Difference-in-differences: decimation gain under complete communication
    # minus the gain under each reduced/no-exchange condition, paired by family.
    interaction_rows = []
    grouped_lookup = {}
    for row in paired_rows:
        grouped_lookup[(row["seed"], row["arm"], row["family_id"], row["condition"], row["mask_rep"])] = row
    for row in paired_rows:
        if row["condition"] == "full_100":
            continue
        full = grouped_lookup[(row["seed"], row["arm"], row["family_id"], "full_100", 0)]
        interaction_rows.append({
            "family_id": row["family_id"], "seed": row["seed"], "arm": row["arm"],
            "reduced_condition": row["condition"], "mask_rep": row["mask_rep"],
            "gain_full": full["decimation_gain_exact"],
            "gain_reduced": row["decimation_gain_exact"],
            "interaction_full_minus_reduced": full["decimation_gain_exact"] - row["decimation_gain_exact"],
            "gain_common_full": full["gain_common_hidden_accuracy"],
            "gain_common_reduced": row["gain_common_hidden_accuracy"],
            "interaction_common_full_minus_reduced": (
                full["gain_common_hidden_accuracy"] - row["gain_common_hidden_accuracy"]
            ),
        })
    write_csv(run_out / "paired_interactions_vs_full.csv", interaction_rows)
    print(json.dumps({"status": "complete", "run_dir": str(run_out.relative_to(ROOT)),
                      "paired_rows": len(paired_rows), "pass_rows": len(pass_rows),
                      "interaction_rows": len(interaction_rows)}, indent=2))


if __name__ == "__main__":
    main()
