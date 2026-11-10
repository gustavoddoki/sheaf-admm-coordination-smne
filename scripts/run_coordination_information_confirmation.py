"""Resumable P5 confirmation of local information transmission over ADMM."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from scripts.run_coordination_local_information import (  # noqa: E402
    checkpoint_path,
    graph_distances,
    load_model,
    local_clue_patches,
    make_predictor,
    stable_softmax,
)
from sheaf_admm.models import SheafADMMModel  # noqa: E402

PROTOCOL = ROOT / "experiments/sheaf_admm_reasoning/P5_PROTOCOLO_CONFIRMATORIO_2026-10-07.md"
CONFIRM_DATA = ROOT / "datasets/sudoku_coordination_confirmation_20261007"
PILOT_DATA = ROOT / "datasets/sudoku_coordination_pilot_20261007"
DEFAULT_SEEDS = (123, 456, 789, 1011, 1213, 20260927)
DEFAULT_ARMS = ("internal", "terminal")
DEFAULT_DEPTHS = (1, 2, 4)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("wb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(tmp, path)


def atomic_json(path: Path, payload: dict) -> None:
    atomic_bytes(path, json.dumps(payload, indent=2, sort_keys=True).encode("utf-8"))


def atomic_npz(path: Path, arrays: dict[str, np.ndarray], metadata: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("wb") as stream:
        np.savez_compressed(stream, **arrays)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(tmp, path)
    sidecar = dict(metadata, output_sha256=sha256(path))
    atomic_json(path.with_suffix(".json"), sidecar)


def verified_npz(path: Path, expected: dict) -> dict[str, np.ndarray] | None:
    sidecar_path = path.with_suffix(".json")
    if not path.is_file() or not sidecar_path.is_file():
        # Recover safely if an interruption occurred between the atomic NPZ
        # replacement and its small provenance sidecar.
        if path.is_file() and not sidecar_path.exists():
            path.unlink()
        if sidecar_path.is_file() and not path.exists():
            sidecar_path.unlink()
        return None
    saved = json.loads(sidecar_path.read_text(encoding="utf-8"))
    if any(saved.get(key) != value for key, value in expected.items()):
        raise RuntimeError(f"P5 provenance mismatch: {sidecar_path}")
    if saved.get("output_sha256") != sha256(path):
        raise RuntimeError(f"P5 output checksum mismatch: {path}")
    with np.load(path, allow_pickle=False) as archive:
        return {key: archive[key] for key in archive.files}


def write_csv(path: Path, rows: list[dict]) -> None:
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


def select_families(
    data_dir: Path, family_start: int, family_count: int, targets_per_family: int,
) -> tuple[dict, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    manifest_path = data_dir / "paired_clue_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    requested_ids = list(range(family_start, family_start + family_count))
    rows_by_id: dict[int, dict] = {}
    for row in manifest["families"]:
        if row["policy"] == "concentrated" and int(row["clues"]) == 40:
            family_id = int(row["family_id"])
            if family_id in requested_ids:
                if family_id in rows_by_id:
                    raise RuntimeError(f"duplicate 40-clue family row: {family_id}")
                rows_by_id[family_id] = row
    if sorted(rows_by_id) != requested_ids:
        raise RuntimeError("requested 40-clue concentrated families are incomplete")

    inputs_all = np.load(data_dir / "test_hard/all__inputs.npy", mmap_mode="r")
    labels_all = np.load(data_dir / "test_hard/all__labels.npy", mmap_mode="r")
    selected_rows = [rows_by_id[family_id] for family_id in requested_ids]
    row_indices = np.asarray([int(row["row_index"]) for row in selected_rows], dtype=np.int32)
    inputs = np.asarray(inputs_all[row_indices], dtype=np.uint8).reshape(-1, 81)
    labels = np.asarray(labels_all[row_indices], dtype=np.uint8).reshape(-1, 81)

    family_ids = np.repeat(np.asarray(requested_ids, dtype=np.int32), targets_per_family)
    base_rows = np.repeat(np.arange(family_count, dtype=np.int32), targets_per_family)
    target_cells = []
    for family_id, row in zip(requested_ids, selected_rows, strict=True):
        common = np.asarray(row["common_evaluation_cells"], dtype=np.int32)
        if common.size != 20:
            raise RuntimeError(f"family {family_id} does not have 20 common evaluation cells")
        rng = np.random.default_rng(20261008 + 211 * family_id + 19)
        chosen = np.sort(rng.choice(common, size=targets_per_family, replace=False))
        target_cells.extend(int(value) for value in chosen)
    target_cells_array = np.asarray(target_cells, dtype=np.int32)
    task_inputs = inputs[base_rows]
    task_labels = labels[base_rows]
    target_digits = task_labels[np.arange(len(target_cells_array)), target_cells_array]
    if np.any(task_inputs[np.arange(len(target_cells_array)), target_cells_array] != 0):
        raise AssertionError("a selected P5 oracle cell is visible in the original puzzle")
    return manifest, inputs, labels, family_ids, target_cells_array, target_digits


def true_label_metrics(
    logits_base: np.ndarray,
    logits_clue: np.ndarray,
    *,
    task_inputs: np.ndarray,
    task_labels: np.ndarray,
    family_ids: np.ndarray,
    target_cells: np.ndarray,
    target_digits: np.ndarray,
    cell_ids: np.ndarray,
    incident: np.ndarray,
    distances: np.ndarray,
    seed: int,
    arm: str,
    depth: int,
    policy: str,
) -> tuple[list[dict], list[dict]]:
    # Convert node-major logits to task × node × local-slot × class.
    base = np.transpose(logits_base, (1, 0, 2, 3)).astype(np.float64)
    clue = np.transpose(logits_clue, (1, 0, 2, 3)).astype(np.float64)
    base_logp = base - np.logaddexp.reduce(base, axis=-1, keepdims=True)
    clue_logp = clue - np.logaddexp.reduce(clue, axis=-1, keepdims=True)
    base_pred = np.argmax(base, axis=-1)
    clue_pred = np.argmax(clue, axis=-1)
    task_rows = []
    agent_rows = []
    for task_index in range(len(target_cells)):
        by_distance = {}
        for node in range(base.shape[1]):
            distance = int(distances[task_index, node])
            incident_flag = int(incident[task_index, node])
            ids = cell_ids[node]
            hidden = task_inputs[task_index, ids] == 0
            labels = task_labels[task_index, ids].astype(np.int64)
            slots = np.flatnonzero(hidden)
            if slots.size == 0:
                continue
            idx = (task_index, node, slots, labels[slots])
            true_base = base_logp[idx]
            true_clue = clue_logp[idx]
            accuracy_base = base_pred[task_index, node, slots] == labels[slots]
            accuracy_clue = clue_pred[task_index, node, slots] == labels[slots]
            js_p = stable_softmax(base[task_index, node, slots])
            js_q = stable_softmax(clue[task_index, node, slots])
            mid = 0.5 * (js_p + js_q)
            tiny = np.finfo(np.float64).tiny
            js = 0.5 * np.sum(
                js_p * (np.log(np.maximum(js_p, tiny)) - np.log(np.maximum(mid, tiny)))
                + js_q * (np.log(np.maximum(js_q, tiny)) - np.log(np.maximum(mid, tiny))),
                axis=-1,
            )
            agent_rows.append({
                "family_id": int(family_ids[task_index]),
                "target_cell": int(target_cells[task_index]),
                "target_digit": int(target_digits[task_index]),
                "seed": seed,
                "arm": arm,
                "depth": depth,
                "communication_policy": policy,
                "node_id": node,
                "incident": incident_flag,
                "distance_from_incident_set": distance,
                "hidden_slots": int(slots.size),
                "true_logprob_no_clue": float(true_base.mean()),
                "true_logprob_local_clue": float(true_clue.mean()),
                "hidden_accuracy_no_clue": float(accuracy_base.mean()),
                "hidden_accuracy_local_clue": float(accuracy_clue.mean()),
                "js_divergence": float(js.mean()),
                "top1_changed_fraction": float(np.mean(
                    base_pred[task_index, node, slots] != clue_pred[task_index, node, slots]
                )),
                "mean_abs_logit_change": float(np.mean(
                    np.abs(clue[task_index, node, slots] - base[task_index, node, slots])
                )),
            })
            if not incident_flag:
                metrics = by_distance.setdefault(distance, {
                    "logp_base": [], "logp_clue": [], "acc_base": [], "acc_clue": [],
                })
                metrics["logp_base"].extend(true_base.tolist())
                metrics["logp_clue"].extend(true_clue.tolist())
                metrics["acc_base"].extend(accuracy_base.astype(float).tolist())
                metrics["acc_clue"].extend(accuracy_clue.astype(float).tolist())
        for distance, values in sorted(by_distance.items()):
            task_rows.append({
                "family_id": int(family_ids[task_index]),
                "target_cell": int(target_cells[task_index]),
                "target_digit": int(target_digits[task_index]),
                "seed": seed,
                "arm": arm,
                "depth": depth,
                "communication_policy": policy,
                "distance_from_incident_set": int(distance),
                "remote_slots": len(values["logp_base"]),
                "true_logprob_no_clue": float(np.mean(values["logp_base"])),
                "true_logprob_local_clue": float(np.mean(values["logp_clue"])),
                "hidden_accuracy_no_clue": float(np.mean(values["acc_base"])),
                "hidden_accuracy_local_clue": float(np.mean(values["acc_clue"])),
            })
    return task_rows, agent_rows


def family_bootstrap(values: np.ndarray, draws: int = 20000, seed: int = 20261008) -> dict:
    values = np.asarray(values, dtype=np.float64)
    rng = np.random.default_rng(seed)
    sample_indices = rng.integers(0, len(values), size=(draws, len(values)))
    samples = values[sample_indices].mean(axis=1)
    return {
        "families": int(len(values)),
        "estimate": float(values.mean()),
        "ci95_low": float(np.quantile(samples, 0.025)),
        "ci95_high": float(np.quantile(samples, 0.975)),
        "bootstrap_draws": draws,
        "bootstrap_seed": seed,
    }


def summarize_primary(task_rows: list[dict], *, families: list[int]) -> dict:
    by_key = {
        (row["family_id"], row["target_cell"], row["seed"], row["arm"], row["communication_policy"]): row
        for row in task_rows
        if int(row["depth"]) == 2 and int(row["distance_from_incident_set"]) == 1
    }
    per_family = {family: [] for family in families}
    per_seed_arm = {}
    for (family_id, target_cell, seed, arm, policy), row in list(by_key.items()):
        if policy != "full_admm":
            continue
        none_row = by_key.get((family_id, target_cell, seed, arm, "none_local_cg"))
        if none_row is None:
            continue
        full_delta = row["true_logprob_local_clue"] - row["true_logprob_no_clue"]
        none_delta = none_row["true_logprob_local_clue"] - none_row["true_logprob_no_clue"]
        did = full_delta - none_delta
        per_family[family_id].append(did)
        per_seed_arm.setdefault((seed, arm), []).append(did)
    if any(not values for values in per_family.values()):
        raise RuntimeError("primary P5 contrast is incomplete for one or more families")
    family_values = np.asarray([np.mean(per_family[family]) for family in families])
    by_checkpoint = {
        f"seed_{seed}_{arm}": {
            "estimate": float(np.mean(values)),
            "interventions": len(values),
        }
        for (seed, arm), values in sorted(per_seed_arm.items())
    }
    return {
        "estimand": "(local clue effect with full ADMM) - (local clue effect with local CG), K=2, distance=1",
        "family_cluster_bootstrap": family_bootstrap(family_values),
        "family_level_values": {str(f): float(v) for f, v in zip(families, family_values, strict=True)},
        "checkpoint_strata": by_checkpoint,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("smoke", "confirm"), default="smoke")
    parser.add_argument("--data-dir", type=Path)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--family-start", type=int, default=0)
    parser.add_argument("--families", type=int, default=4)
    parser.add_argument("--targets-per-family", type=int, default=4)
    parser.add_argument("--seeds", default="123")
    parser.add_argument("--arms", default="internal")
    parser.add_argument("--depths", default="1,2,4")
    args = parser.parse_args()

    data_dir = (args.data_dir or (CONFIRM_DATA if args.phase == "confirm" else PILOT_DATA)).resolve()
    seeds = tuple(int(value) for value in args.seeds.split(",") if value)
    arms = tuple(value.strip() for value in args.arms.split(",") if value.strip())
    depths = tuple(int(value) for value in args.depths.split(",") if value)
    if args.targets_per_family != 4:
        parser.error("the frozen P5 protocol uses exactly four oracle cells per family")
    if not seeds or any(seed not in DEFAULT_SEEDS for seed in seeds):
        parser.error(f"seeds must be selected from {DEFAULT_SEEDS}")
    if not arms or any(arm not in DEFAULT_ARMS for arm in arms):
        parser.error("arms must be internal and/or terminal")
    if not depths or any(depth not in DEFAULT_DEPTHS for depth in depths):
        parser.error(f"depths must be selected from {DEFAULT_DEPTHS}")
    if args.phase == "confirm":
        if (
            data_dir != CONFIRM_DATA.resolve() or args.family_start != 0 or args.families != 64
            or seeds != DEFAULT_SEEDS or arms != DEFAULT_ARMS or depths != DEFAULT_DEPTHS
        ):
            parser.error("confirm phase parameters must match the frozen P5 protocol exactly")
        output_root = ROOT / "outputs/coordination_pilot_20261007/p5_confirmation"
    else:
        if args.families < 1 or args.families > 32:
            parser.error("smoke phase supports 1..32 development families")
        output_root = args.output_root or ROOT / "outputs/coordination_pilot_20261007/p5_smoke"
        output_root = output_root.resolve()

    manifest_path = data_dir / "paired_clue_manifest.json"
    manifest_hash = sha256(manifest_path)
    if args.phase == "confirm":
        audit_path = ROOT / "outputs/coordination_pilot_20261007/p5_confirmation/holdout_audit.json"
        if not audit_path.is_file() or json.loads(audit_path.read_text(encoding="utf-8")).get("status") != "PASS":
            parser.error("confirmation holdout must pass the independent audit before inference")
    manifest, base_inputs, base_labels, family_ids, target_cells, target_digits = select_families(
        data_dir, args.family_start, args.families, args.targets_per_family
    )
    base_index = np.repeat(np.arange(args.families, dtype=np.int32), args.targets_per_family)
    task_inputs = base_inputs[base_index]
    task_labels = base_labels[base_index]
    manifest_rows = [
        row for row in manifest["families"]
        if row["policy"] == "concentrated" and int(row["clues"]) == 40
        and args.family_start <= int(row["family_id"]) < args.family_start + args.families
    ]
    row_indices = [int(row["row_index"]) for row in sorted(manifest_rows, key=lambda x: int(x["family_id"]))]
    inputs_sha = hashlib.sha256(base_inputs.tobytes()).hexdigest()
    labels_sha = hashlib.sha256(base_labels.tobytes()).hexdigest()
    target_digit_sha = hashlib.sha256(target_digits.tobytes()).hexdigest()
    code_hash = sha256(Path(__file__))
    helper_hash = sha256(Path(__file__).with_name("run_coordination_local_information.py"))
    protocol_hash = sha256(PROTOCOL)
    args_repr = {
        "phase": args.phase,
        "family_ids": family_ids.tolist(),
        "seeds": seeds,
        "arms": arms,
        "depths": depths,
        "targets_per_family": args.targets_per_family,
        "target_cells": target_cells.tolist(),
        "target_selection": "default_rng(20261008 + 211*family_id + 19), sample 4 of 20 common hidden cells, sorted",
        "communication_policies": ["full_admm", "none_local_cg"],
        "clues": 40,
        "distribution": "concentrated",
        "K_primary": 2,
        "dataset_manifest_sha256": manifest_hash,
        "input_sha256": inputs_sha,
        "label_sha256_for_scoring_only": labels_sha,
        "target_digit_sha256": target_digit_sha,
        "protocol_sha256": protocol_hash,
        "runner_sha256": code_hash,
        "shared_helper_sha256": helper_hash,
        "labels_provided_to_model": False,
        "backend": jax.default_backend(),
    }
    run_hash = hashlib.sha256(json.dumps(args_repr, sort_keys=True).encode()).hexdigest()
    run_dir = output_root / f"{args.phase}_n{args.families}_s{'-'.join(map(str,seeds))}_4cells_{run_hash[:8]}"
    run_dir.mkdir(parents=True, exist_ok=True)
    atomic_json(run_dir / "run_config.json", args_repr | {
        "run_hash": run_hash,
        "dataset_dir": str(data_dir.relative_to(ROOT)),
        "dataset_row_indices": row_indices,
        "family_count": args.families,
        "paired_intervention_count": int(len(target_cells)),
    })

    all_task_rows = []
    all_agent_rows = []
    expected_total = len(seeds) * len(arms) * len(depths) * 2
    completed = 0
    for seed in seeds:
        for arm in arms:
            checkpoint, checkpoint_hash, config, params, model, task = load_model(arm, seed)
            batch = {"inputs": task_inputs, "labels": task_labels}
            fwd, _targets, _aux = task.prepare(batch)
            base_patches = np.asarray(fwd["patches"], dtype=np.float32)
            edge_indices = np.asarray(fwd["edge_indices"], dtype=np.int32)
            cell_ids = np.asarray(fwd["model_kwargs"]["cell_ids"], dtype=np.int32)
            clue_patches, incident, cell_ids = local_clue_patches(
                base_patches, cell_ids, target_cells, target_digits
            )
            distances = np.stack([
                graph_distances(edge_indices, np.flatnonzero(incident[index]), base_patches.shape[0])
                for index in range(len(target_cells))
            ])
            paired_patches = np.concatenate([base_patches, clue_patches], axis=1)
            paired_size = paired_patches.shape[1]
            kwargs = {key: jnp.asarray(value) for key, value in fwd["model_kwargs"].items()}
            edge_jax = jnp.asarray(edge_indices)
            model_context = {
                "seed": seed,
                "arm": arm,
                "checkpoint_sha256": checkpoint_hash,
                "checkpoint_path": str(checkpoint.relative_to(ROOT)),
                "dataset_manifest_sha256": manifest_hash,
                "runner_sha256": code_hash,
                "shared_helper_sha256": helper_hash,
                "protocol_sha256": protocol_hash,
                "family_ids": family_ids.tolist(),
                "target_cells": target_cells.tolist(),
                "target_digit_sha256": target_digit_sha,
                "labels_provided_to_model": False,
                "run_hash": run_hash,
            }
            for depth in depths:
                for policy, api_policy in (("full_admm", "random"), ("none_local_cg", "none")):
                    artifact = run_dir / f"s{seed}_{arm[0]}" / policy / f"k{depth:02d}.npz"
                    provenance = dict(model_context, depth=depth, communication_policy=policy)
                    cached = verified_npz(artifact, provenance)
                    if cached is None:
                        predictor = make_predictor(
                            model, depth, api_policy, len(edge_indices), paired_size
                        )
                        started = time.perf_counter()
                        logits = np.asarray(jax.device_get(predictor(
                            params, jnp.asarray(paired_patches), edge_jax, kwargs
                        )), dtype=np.float32)
                        elapsed = time.perf_counter() - started
                        logits_base = logits[:, :len(target_cells)]
                        logits_clue = logits[:, len(target_cells):]
                        remote = incident == 0
                        if policy == "none_local_cg" and not np.array_equal(
                            logits_base.transpose(1, 0, 2, 3)[remote],
                            logits_clue.transpose(1, 0, 2, 3)[remote],
                        ):
                            raise AssertionError(f"local CG remote leakage at K={depth}")
                        if depth == 1 and not np.array_equal(
                            logits_base.transpose(1, 0, 2, 3)[remote],
                            logits_clue.transpose(1, 0, 2, 3)[remote],
                        ):
                            raise AssertionError("remote proposals changed before the first ADMM message")
                        atomic_npz(artifact, {
                            "logits_no_clue": logits_base,
                            "logits_local_clue": logits_clue,
                            "incident_agents": incident,
                            "graph_distances": distances,
                            "target_cells": target_cells,
                            "target_digits": target_digits,
                            "family_ids": family_ids,
                            "task_inputs": task_inputs,
                            "task_labels_for_scoring_only": task_labels,
                            "cell_ids": cell_ids,
                        }, dict(provenance, elapsed_seconds=float(elapsed)))
                        cached = verified_npz(artifact, provenance)
                        print(json.dumps({
                            "saved": str(artifact.relative_to(ROOT)),
                            "seed": seed, "arm": arm, "policy": policy, "K": depth,
                            "elapsed_seconds": round(elapsed, 2),
                            "completed": completed + 1, "total": expected_total,
                        }), flush=True)
                    else:
                        print(json.dumps({"verified_checkpoint": str(artifact.relative_to(ROOT))}), flush=True)
                    if cached is None:
                        raise RuntimeError(f"failed to reload saved output: {artifact}")
                    base_logits = cached["logits_no_clue"]
                    clue_logits = cached["logits_local_clue"]
                    task_rows, agent_rows = true_label_metrics(
                        base_logits, clue_logits,
                        task_inputs=task_inputs,
                        task_labels=task_labels,
                        family_ids=family_ids,
                        target_cells=target_cells,
                        target_digits=target_digits,
                        cell_ids=cached["cell_ids"],
                        incident=cached["incident_agents"],
                        distances=cached["graph_distances"],
                        seed=seed,
                        arm=arm,
                        depth=depth,
                        policy=policy,
                    )
                    all_task_rows.extend(task_rows)
                    all_agent_rows.extend(agent_rows)
                    completed += 1
                    # Incremental tables are safe to rebuild from already verified NPZs.
                    write_csv(run_dir / "intervention_metrics.csv", all_task_rows)
                    write_csv(run_dir / "agent_metrics.csv", all_agent_rows)
                    atomic_json(run_dir / "progress.json", {
                        "status": "running",
                        "completed_model_conditions": completed,
                        "total_model_conditions": expected_total,
                        "last_verified_artifact": str(artifact.relative_to(ROOT)),
                    })

    family_list = sorted(set(int(value) for value in family_ids))
    primary = summarize_primary(all_task_rows, families=family_list)
    report = {
        "status": "complete",
        "run_hash": run_hash,
        "run_dir": str(run_dir.relative_to(ROOT)),
        "saved_model_conditions": expected_total,
        "verified_archives": len(list(run_dir.glob("s*_*/*/k*.npz"))),
        "family_clusters": len(family_list),
        "interventions_per_family": args.targets_per_family,
        "total_interventions": len(target_cells),
        "no_message_remote_invariance_checked": True,
        "K1_remote_invariance_checked": True,
        "backend": jax.default_backend(),
        "primary_result": primary,
        "scope_note": "family-clustered inference conditional on the six existing checkpoint seeds; no labels were provided to model inputs",
    }
    atomic_json(run_dir / "summary.json", report)
    atomic_json(run_dir / "progress.json", {
        "status": "complete",
        "completed_model_conditions": completed,
        "total_model_conditions": expected_total,
        "summary_sha256": sha256(run_dir / "summary.json"),
    })
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
