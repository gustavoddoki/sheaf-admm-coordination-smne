"""Resumable P4 diagnostic: can a clue local to three agents alter remote proposals?"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import pickle
import sys
import time
from collections import deque
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))
from sheaf_admm.data import build_sudoku_multigraph
from sheaf_admm.models import SheafADMMModel, model_config_from_dict
from sheaf_admm.training import build_model, make_task

DATA = ROOT / "datasets/sudoku_coordination_pilot_20261007"
OUT = ROOT / "outputs/coordination_pilot_20261007/p4_local_info"
DEFAULT_SEEDS = (123, 456)
DEFAULT_ARMS = ("internal", "terminal")
DEFAULT_DEPTHS = (1, 2, 4, 8)


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


def atomic_npz(path: Path, arrays: dict[str, np.ndarray], metadata: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("wb") as stream:
        np.savez_compressed(stream, **arrays)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(tmp, path)
    sidecar = dict(metadata, output_sha256=sha256(path))
    atomic_bytes(path.with_suffix(".json"), json.dumps(sidecar, indent=2).encode("utf-8"))


def verified_npz(path: Path, metadata: dict) -> dict[str, np.ndarray] | None:
    sidecar_path = path.with_suffix(".json")
    if not path.is_file() or not sidecar_path.is_file():
        return None
    saved = json.loads(sidecar_path.read_text(encoding="utf-8"))
    if any(saved.get(key) != value for key, value in metadata.items()):
        raise RuntimeError(f"P4 provenance mismatch: {sidecar_path}")
    if not saved.get("output_sha256") or sha256(path) != saved["output_sha256"]:
        raise RuntimeError(f"P4 provenance or SHA-256 mismatch: {path}")
    with np.load(path, allow_pickle=False) as loaded:
        return {key: loaded[key] for key in loaded.files}


def checkpoint_path(arm: str, seed: int) -> Path:
    if arm == "terminal":
        return ROOT / f"outputs/sakana_placement_confirmation/seed_{seed}/n4096_e5/terminal12_k20/best_checkpoint.pkl"
    if seed == 123:
        return ROOT / f"outputs/sakana_stability_compute_matched/n4096_e5/seed_{seed}/sinkhorn_k20/best_checkpoint.pkl"
    return ROOT / f"outputs/sakana_stability_constraint_confirmation/seed_{seed}/n4096_e5/sinkhorn_k20/best_checkpoint.pkl"


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


def graph_distances(edge_indices: np.ndarray, source_nodes: np.ndarray, num_nodes: int) -> np.ndarray:
    adjacency = [set() for _ in range(num_nodes)]
    for left, right in np.asarray(edge_indices, dtype=np.int32):
        adjacency[int(left)].add(int(right))
        adjacency[int(right)].add(int(left))
    distance = np.full((num_nodes,), -1, dtype=np.int16)
    queue = deque()
    for node in source_nodes.tolist():
        distance[node] = 0
        queue.append(node)
    while queue:
        node = queue.popleft()
        for neighbor in adjacency[node]:
            if distance[neighbor] < 0:
                distance[neighbor] = distance[node] + 1
                queue.append(neighbor)
    if np.any(distance < 0):
        raise AssertionError("Sudoku agent graph is disconnected")
    return distance


def stable_softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits.astype(np.float64) - logits.max(axis=-1, keepdims=True)
    exp = np.exp(shifted)
    return exp / exp.sum(axis=-1, keepdims=True)


def make_predictor(model, num_iters: int, policy: str, edge_count: int, batch_size: int):
    if policy == "none":
        schedule = jnp.zeros((num_iters,), dtype=jnp.int32)
    else:
        schedule = jnp.full((num_iters,), edge_count * batch_size, dtype=jnp.int32)

    @jax.jit
    def predict(params, patches, edge_indices, model_kwargs):
        _summary, logits, _state, _geometry, _rho = model.apply(
            params,
            patches,
            edge_indices,
            num_iters=num_iters,
            threshold=0.0,
            gate_mode="absolute",
            mask_policy=policy,
            active_count_schedule=schedule,
            mask_seed=0,
            **model_kwargs,
            training=False,
            method=SheafADMMModel.coordinate_triggered,
        )
        return logits[-1]

    return predict


def local_clue_patches(
    base_patches: np.ndarray,
    cell_ids: np.ndarray,
    target_cells: np.ndarray,
    target_digits: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    clue_patches = base_patches.copy()
    incident = np.zeros((len(target_cells), base_patches.shape[0]), dtype=np.uint8)
    for batch_index, (cell, digit) in enumerate(zip(target_cells, target_digits, strict=True)):
        nodes = np.flatnonzero(np.any(cell_ids == int(cell), axis=1))
        if nodes.size != 3:
            raise AssertionError(f"cell {cell} should be in exactly 3 Sudoku views, found {nodes}")
        incident[batch_index, nodes] = 1
        for node in nodes.tolist():
            slots = np.flatnonzero(cell_ids[node] == int(cell))
            if slots.size != 1:
                raise AssertionError("each incident agent must contain the clue cell once")
            slot = int(slots[0])
            clue_patches[node, batch_index, slot, :] = 0
            clue_patches[node, batch_index, slot, int(digit)] = 1
    # Only the three incident local views may differ from the baseline input.
    for batch_index in range(len(target_cells)):
        changed_nodes = np.flatnonzero(np.any(
            clue_patches[:, batch_index] != base_patches[:, batch_index], axis=(1, 2)
        ))
        if not np.array_equal(changed_nodes, np.flatnonzero(incident[batch_index])):
            raise AssertionError("localized clue changed a nonincident agent view")
    return clue_patches, incident, np.asarray(cell_ids, dtype=np.int32)


def analyze_logits(
    logits_no_clue: np.ndarray,
    logits_local_clue: np.ndarray,
    *,
    inputs: np.ndarray,
    labels: np.ndarray,
    cell_ids: np.ndarray,
    incident: np.ndarray,
    distances: np.ndarray,
) -> list[dict]:
    p = stable_softmax(logits_no_clue)
    q = stable_softmax(logits_local_clue)
    midpoint = 0.5 * (p + q)
    tiny = np.finfo(np.float64).tiny
    kl_p = np.sum(p * (np.log(np.maximum(p, tiny)) - np.log(np.maximum(midpoint, tiny))), axis=-1)
    kl_q = np.sum(q * (np.log(np.maximum(q, tiny)) - np.log(np.maximum(midpoint, tiny))), axis=-1)
    js_by_slot = 0.5 * (kl_p + kl_q)
    pred_base = logits_no_clue.argmax(axis=-1)
    pred_clue = logits_local_clue.argmax(axis=-1)
    rows = []
    for family in range(inputs.shape[0]):
        for node in range(logits_no_clue.shape[0]):
            cell_indices = cell_ids[node]
            hidden = inputs[family, cell_indices] == 0
            local_truth = labels[family, cell_indices]
            changed = pred_base[node, family] != pred_clue[node, family]
            rows.append({
                "family_index": family,
                "node_id": node,
                "incident": int(incident[family, node]),
                "distance_from_incident_set": int(distances[family, node]),
                "js_divergence_mean_local_slots": float(np.mean(js_by_slot[node, family])),
                "mean_abs_logit_change": float(np.mean(np.abs(
                    logits_local_clue[node, family] - logits_no_clue[node, family]
                ))),
                "top_class_changed_fraction_local_slots": float(np.mean(changed)),
                "hidden_cells_scored": int(np.count_nonzero(hidden)),
                "hidden_accuracy_no_clue": float(np.mean(pred_base[node, family, hidden] == local_truth[hidden])) if np.any(hidden) else float("nan"),
                "hidden_accuracy_local_clue": float(np.mean(pred_clue[node, family, hidden] == local_truth[hidden])) if np.any(hidden) else float("nan"),
            })
    return rows


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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--families", type=int, default=32)
    parser.add_argument("--seeds", default=",".join(map(str, DEFAULT_SEEDS)))
    parser.add_argument("--arms", default=",".join(DEFAULT_ARMS))
    parser.add_argument("--depths", default=",".join(map(str, DEFAULT_DEPTHS)))
    args = parser.parse_args()
    if not 1 <= args.families <= 32:
        parser.error("families must be in 1..32")
    seeds = tuple(int(value) for value in args.seeds.split(",") if value)
    arms = tuple(value.strip() for value in args.arms.split(",") if value.strip())
    depths = tuple(int(value) for value in args.depths.split(",") if value)
    if any(seed not in (123, 456) for seed in seeds):
        parser.error("P4 currently supports the two audited checkpoints 123 and 456")
    if any(arm not in ("internal", "terminal") for arm in arms):
        parser.error("arms must be internal and/or terminal")
    if not depths or any(depth not in DEFAULT_DEPTHS for depth in depths):
        parser.error(f"depths must be selected from {DEFAULT_DEPTHS}")

    manifest_path = DATA / "paired_clue_manifest.json"
    manifest_hash = sha256(manifest_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    rows = sorted([
        row for row in manifest["families"]
        if int(row["family_id"]) < 32 and row["policy"] == "concentrated" and int(row["clues"]) == 40
    ], key=lambda row: int(row["family_id"]))[:args.families]
    if len(rows) != args.families:
        raise RuntimeError("P4 development family selection is incomplete")
    all_inputs = np.load(DATA / "test_hard/all__inputs.npy", mmap_mode="r")
    all_labels = np.load(DATA / "test_hard/all__labels.npy", mmap_mode="r")
    row_indices = [int(row["row_index"]) for row in rows]
    inputs = np.asarray(all_inputs[row_indices], dtype=np.uint8).reshape(-1, 9, 9)
    labels = np.asarray(all_labels[row_indices], dtype=np.uint8).reshape(-1, 9, 9)
    target_cells = np.asarray([
        int(row["common_evaluation_cells"][0]) for row in rows
    ], dtype=np.int32)
    target_digits = labels.reshape(len(rows), 81)[np.arange(len(rows)), target_cells]
    if np.any(inputs.reshape(len(rows), 81)[np.arange(len(rows)), target_cells] != 0):
        raise AssertionError("chosen P4 oracle cells must be hidden in every input")

    code_hash = sha256(Path(__file__))
    arm_tag = "".join({"internal": "i", "terminal": "t"}[arm] for arm in arms)
    run_dir = OUT / f"n{args.families}_s{'-'.join(map(str, seeds))}_a{arm_tag}_d{'-'.join(map(str, depths))}_{code_hash[:8]}"
    run_dir.mkdir(parents=True, exist_ok=True)
    manifest_hash_inputs = hashlib.sha256(inputs.tobytes()).hexdigest()
    manifest_hash_labels = hashlib.sha256(labels.tobytes()).hexdigest()
    run_config = {
        "phase": "P4_local_information_development",
        "families": args.families,
        "family_ids": [int(row["family_id"]) for row in rows],
        "row_indices": row_indices,
        "seeds": seeds,
        "arms": arms,
        "depths": depths,
        "communication_policies": ["random_full", "none_local_cg"],
        "target_cell_indices": target_cells.tolist(),
        "target_digit_sha256": hashlib.sha256(target_digits.tobytes()).hexdigest(),
        "target_selection": "first common hidden evaluation cell, fixed before model inference",
        "labels_provided_to_model": False,
        "clue_visibility": "only the row/column/box agent views containing that cell",
        "dataset_manifest_sha256": manifest_hash,
        "input_sha256": manifest_hash_inputs,
        "label_sha256_for_scoring_only": manifest_hash_labels,
        "code_sha256": code_hash,
        "backend": jax.default_backend(),
    }
    atomic_bytes(run_dir / "run_config.json", json.dumps(run_config, indent=2).encode("utf-8"))

    base_fwd_by_model = {}
    for seed in seeds:
        for arm in arms:
            checkpoint, checkpoint_hash, cfg, params, model, task = load_model(arm, seed)
            batch = {"inputs": inputs.reshape(len(rows), 81), "labels": labels.reshape(len(rows), 81)}
            fwd, _targets, _aux = task.prepare(batch)
            base_patches = np.asarray(fwd["patches"], dtype=np.float32)
            edge_indices = np.asarray(fwd["edge_indices"], dtype=np.int32)
            cell_ids = np.asarray(fwd["model_kwargs"]["cell_ids"], dtype=np.int32)
            clue_patches, incident, cell_ids = local_clue_patches(
                base_patches, cell_ids, target_cells, target_digits
            )
            distances = np.stack([
                graph_distances(edge_indices, np.flatnonzero(incident[index]), base_patches.shape[0])
                for index in range(len(rows))
            ])
            paired_patches = np.concatenate([base_patches, clue_patches], axis=1)
            paired_size = paired_patches.shape[1]
            kwargs = {key: jnp.asarray(value) for key, value in fwd["model_kwargs"].items()}
            edge_jax = jnp.asarray(edge_indices)
            code_ctx = {
                "seed": seed,
                "arm": arm,
                "checkpoint_sha256": checkpoint_hash,
                "checkpoint_path": str(checkpoint.relative_to(ROOT)),
                "dataset_manifest_sha256": manifest_hash,
                "input_sha256": manifest_hash_inputs,
                "label_sha256_for_scoring_only": manifest_hash_labels,
                "code_sha256": code_hash,
                "family_ids": run_config["family_ids"],
                "target_cell_indices": target_cells.tolist(),
                "target_digit_sha256": run_config["target_digit_sha256"],
                "labels_provided_to_model": False,
            }
            for depth in depths:
                for policy in ("random", "none"):
                    out = run_dir / f"s{seed}_{arm[0]}" / policy / f"k{depth:02d}.npz"
                    provenance = dict(code_ctx, depth=depth, communication_policy=policy)
                    cached = verified_npz(out, provenance)
                    if cached is not None:
                        print(json.dumps({"pass_verified": str(out.relative_to(ROOT))}), flush=True)
                        continue
                    predictor = make_predictor(
                        model, depth, policy, len(edge_indices), paired_size
                    )
                    started = time.perf_counter()
                    logits = np.asarray(jax.device_get(predictor(
                        params, jnp.asarray(paired_patches), edge_jax, kwargs
                    )), dtype=np.float32)
                    elapsed = time.perf_counter() - started
                    logits_base = logits[:, :len(rows)]
                    logits_clue = logits[:, len(rows):]
                    if policy == "none":
                        remote = incident == 0
                        if not np.allclose(
                            logits_base.transpose(1, 0, 2, 3)[remote],
                            logits_clue.transpose(1, 0, 2, 3)[remote],
                            atol=1e-6, rtol=1e-6,
                        ):
                            raise AssertionError(
                                f"No-ADMM-message local CG leaked remote influence at K={depth}"
                            )
                    atomic_npz(out, {
                        "logits_no_clue": logits_base,
                        "logits_local_clue": logits_clue,
                        "incident_agents": incident,
                        "graph_distances": distances,
                        "target_cells": target_cells,
                        "target_digits": target_digits,
                        "inputs": inputs,
                        "labels_for_scoring_only": labels,
                    }, dict(provenance, elapsed_seconds=float(elapsed)))
                    print(json.dumps({
                        "pass_saved": str(out.relative_to(ROOT)),
                        "seed": seed, "arm": arm, "K": depth, "policy": policy,
                        "mean_logit_abs": float(np.mean(np.abs(logits_base))),
                        "elapsed_seconds": round(elapsed, 2),
                    }), flush=True)

            # Re-read verified outputs to make family/agent diagnostics resumable.
            for depth in depths:
                for policy in ("random", "none"):
                    out = run_dir / f"s{seed}_{arm[0]}" / policy / f"k{depth:02d}.npz"
                    cached = verified_npz(
                        out, dict(code_ctx, depth=depth, communication_policy=policy)
                    )
                    if cached is None:
                        raise RuntimeError(f"Missing P4 checkpoint {out}")
                    agent_rows = analyze_logits(
                        cached["logits_no_clue"], cached["logits_local_clue"],
                        inputs=inputs.reshape(len(rows), 81),
                        labels=labels.reshape(len(rows), 81),
                        cell_ids=cell_ids,
                        incident=incident,
                        distances=distances,
                    )
                    for item in agent_rows:
                        item.update({
                            "seed": seed,
                            "arm": arm,
                            "depth": depth,
                            "communication_policy": policy,
                            "family_id": int(rows[item["family_index"]]["family_id"]),
                            "target_cell": int(target_cells[item["family_index"]]),
                            "target_digit": int(target_digits[item["family_index"]]),
                        })
                        item.pop("family_index")
                    table_path = run_dir / f"s{seed}_{arm[0]}" / policy / f"k{depth:02d}_agent_changes.csv"
                    write_csv(table_path, agent_rows)
            print(json.dumps({"model_complete": f"seed_{seed}/{arm}"}), flush=True)

    all_agent_rows = []
    for seed in seeds:
        for arm in arms:
            for depth in depths:
                for policy in ("random", "none"):
                    path = run_dir / f"s{seed}_{arm[0]}" / policy / f"k{depth:02d}_agent_changes.csv"
                    all_agent_rows.extend(read_csv(path))
    write_csv(run_dir / "agent_changes.csv", all_agent_rows)
    distance_groups = {}
    for row in all_agent_rows:
        if int(row["incident"]):
            continue
        key = (row["arm"], row["depth"], row["communication_policy"], row["distance_from_incident_set"])
        distance_groups.setdefault(key, []).append(row)
    distance_summary = []
    for key, group in sorted(distance_groups.items()):
        distance_summary.append({
            "arm": key[0], "depth": int(key[1]), "communication_policy": key[2],
            "distance_from_incident_set": int(key[3]), "agent_family_observations": len(group),
            "mean_js_divergence": float(np.mean([float(row["js_divergence_mean_local_slots"]) for row in group])),
            "mean_abs_logit_change": float(np.mean([float(row["mean_abs_logit_change"]) for row in group])),
            "top_class_changed_fraction": float(np.mean([float(row["top_class_changed_fraction_local_slots"]) for row in group])),
            "hidden_accuracy_no_clue": float(np.mean([float(row["hidden_accuracy_no_clue"]) for row in group])),
            "hidden_accuracy_local_clue": float(np.mean([float(row["hidden_accuracy_local_clue"]) for row in group])),
        })
    write_csv(run_dir / "distance_summary.csv", distance_summary)
    final = {
        "status": "complete",
        "run_dir": str(run_dir.relative_to(ROOT)),
        "code_sha256": code_hash,
        "dataset_manifest_sha256": manifest_hash,
        "family_ids": run_config["family_ids"],
        "seeds": seeds,
        "arms": arms,
        "depths": depths,
        "paired_oracle_cells": len(rows),
        "saved_model_calls": len(seeds) * len(arms) * len(depths) * 2,
        "nonincident_local_cg_invariance_checked": True,
        "no_admm_message_remote_invariance_checked": True,
        "agent_rows": len(all_agent_rows),
        "distance_summary_rows": len(distance_summary),
        "clue_visibility": "only the three row/column/box views incident to each selected cell",
    }
    atomic_bytes(run_dir / "summary.json", json.dumps(final, indent=2).encode("utf-8"))
    print(json.dumps(final, indent=2), flush=True)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


if __name__ == "__main__":
    main()
