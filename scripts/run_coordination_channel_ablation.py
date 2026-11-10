"""Checkpointed P7/P8 study separating continuous messages and discrete commits."""
from __future__ import annotations

import argparse
import csv
import hashlib
import inspect
import json
import math
import os
import signal
import sys
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

from scripts import run_coordination_communication_decimation as core  # noqa: E402
from sheaf_admm.data import build_sudoku_cell_indices  # noqa: E402
from sheaf_admm.models import SheafADMMModel  # noqa: E402

FINAL_DATA = ROOT / "datasets/sudoku_coordination_final_20261008"
P5_DEV_DATA = ROOT / "datasets/sudoku_coordination_pilot_20261007"
P5_CONFIRM = ROOT / "outputs/p5_decim/n64_s6_a2_p8m3_ac4fee68"
PROTOCOL = ROOT / "experiments/sheaf_admm_reasoning/P7_P8_PROTOCOLO_CONGELADO_2026-10-07.md"
OUT = ROOT / "outputs/coordination_channel_ablation"
TRAIN_SEEDS = (123, 456, 789, 1011, 1213, 20260927)
ARMS = ("internal", "terminal")
MASK_SEEDS = (391, 773, 1237)
K = 20
MAX_PASSES = 8
MAX_FRACTION = 0.25
PAUSE_REQUESTED = False


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def calibration_logic_sha256() -> str:
    functions = (
        select_rows, initial_agent_grids, patches_from_agent_grids,
        make_predictor, run_logits, calibrate_one,
    )
    source = "\n".join(inspect.getsource(function) for function in functions)
    return hashlib.sha256(source.encode("utf-8")).hexdigest()


def replace_with_retry(temp: Path, path: Path) -> None:
    for attempt in range(20):
        try:
            os.replace(temp, path)
            return
        except PermissionError:
            if attempt == 19:
                raise
            time.sleep(min(0.05 * (attempt + 1), 0.5))


def atomic_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    with temp.open("wb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
    replace_with_retry(temp, path)


def atomic_json(path: Path, payload: dict) -> None:
    atomic_bytes(path, json.dumps(payload, indent=2, sort_keys=True, default=list).encode("utf-8"))


def atomic_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    temp = path.with_suffix(path.suffix + ".tmp")
    path.parent.mkdir(parents=True, exist_ok=True)
    with temp.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
        stream.flush()
        os.fsync(stream.fileno())
    replace_with_retry(temp, path)


def atomic_npz(path: Path, arrays: dict[str, np.ndarray], provenance: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    with temp.open("wb") as stream:
        np.savez_compressed(stream, **arrays)
        stream.flush()
        os.fsync(stream.fileno())
    replace_with_retry(temp, path)
    atomic_json(path.with_suffix(".json"), dict(provenance, output_sha256=sha256(path)))


def load_verified_npz(path: Path, expected: dict) -> dict[str, np.ndarray] | None:
    sidecar = path.with_suffix(".json")
    if not path.exists() and not sidecar.exists():
        return None
    if not path.exists() or not sidecar.exists():
        raise RuntimeError(f"incomplete pass artifact; preserve and inspect: {path}")
    saved = json.loads(sidecar.read_text(encoding="utf-8"))
    if any(saved.get(key) != value for key, value in expected.items()):
        raise RuntimeError(f"pass provenance mismatch: {sidecar}")
    if saved.get("output_sha256") != sha256(path):
        raise RuntimeError(f"pass SHA-256 mismatch: {path}")
    with np.load(path, allow_pickle=False) as archive:
        return {key: archive[key] for key in archive.files}


def update_progress(run_dir: Path, **updates) -> None:
    path = run_dir / "progress.json"
    state = json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}
    state.update(updates)
    state["updated_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    atomic_json(path, state)


def select_rows(data_dir: Path, family_ids: list[int], clues: int, distribution: str):
    manifest_path = data_dir / "paired_clue_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    by_family = {
        int(row["family_id"]): row for row in manifest["families"]
        if row["policy"] == distribution and int(row["clues"]) == clues
        and int(row["family_id"]) in family_ids
    }
    if sorted(by_family) != sorted(family_ids):
        raise RuntimeError(f"incomplete family rows for {clues}/{distribution}: {family_ids}")
    rows = [by_family[family] for family in family_ids]
    inputs_all = np.load(data_dir / "test_hard/all__inputs.npy", mmap_mode="r")
    labels_all = np.load(data_dir / "test_hard/all__labels.npy", mmap_mode="r")
    row_indices = [int(row["row_index"]) for row in rows]
    inputs = np.asarray(inputs_all[row_indices], dtype=np.uint8).reshape(-1, 9, 9)
    labels = np.asarray(labels_all[row_indices], dtype=np.uint8).reshape(-1, 9, 9)
    common = np.zeros((len(rows), 81), dtype=bool)
    for index, row in enumerate(rows):
        cells = np.asarray(row["common_evaluation_cells"], dtype=np.int32)
        if cells.size != 20 or np.any(inputs[index].reshape(-1)[cells] != 0):
            raise RuntimeError(f"invalid common hidden cells in family {family_ids[index]}")
        common[index, cells] = True
    if np.any((inputs > 0) & (inputs != labels)):
        raise RuntimeError("clues and score labels disagree")
    return manifest, rows, row_indices, inputs, labels, common


def initial_agent_grids(grids: np.ndarray, cell_ids: np.ndarray) -> np.ndarray:
    return np.asarray(grids, dtype=np.uint8).reshape(len(grids), 81)[:, cell_ids]


def patches_from_agent_grids(agent_grids: np.ndarray) -> np.ndarray:
    one_hot = np.eye(10, dtype=np.float32)[np.asarray(agent_grids, dtype=np.int64)]
    return np.transpose(one_hot, (1, 0, 2, 3)).copy()


def make_predictor(model, policy: str):
    @jax.jit
    def predict(params, patches, edge_indices, model_kwargs, active_count_schedule, mask_seed):
        summary, logits, _state, _geometry, _rho = model.apply(
            params,
            patches,
            edge_indices,
            num_iters=K,
            threshold=0.0,
            gate_mode="absolute",
            mask_policy=policy,
            active_count_schedule=active_count_schedule,
            mask_seed=mask_seed,
            **model_kwargs,
            training=False,
            method=SheafADMMModel.coordinate_triggered,
        )
        # coordinate_triggered returns node-major [N, B, slot, class].
        return logits[-1], summary.active_edge_count

    return predict


def run_logits(predictor, params, fwd, batch_size: int, fraction: float, mask_seed: int):
    edge_count = int(np.asarray(fwd["edge_indices"]).shape[0])
    schedule = jnp.full((K,), int(round(edge_count * batch_size * fraction)), dtype=jnp.int32)
    logits, active = jax.device_get(predictor(
        params, fwd["patches"], fwd["edge_indices"], fwd["model_kwargs"],
        schedule, jnp.asarray(mask_seed, dtype=jnp.int32),
    ))
    return np.asarray(logits, dtype=np.float32), np.asarray(active, dtype=np.int64)


def make_owner_lookup(cell_ids: np.ndarray):
    lookup = {}
    for node in range(27):
        for slot, cell in enumerate(cell_ids[node]):
            lookup.setdefault(int(cell), []).append((node, slot))
    if any(len(lookup[cell]) != 3 for cell in range(81)):
        raise AssertionError("every Sudoku cell must belong to exactly three agent views")
    return lookup


def calibrate_one(model_ctx, seed: int, arm: str, cell_ids: np.ndarray, logic_sha: str):
    _path, checkpoint_sha, _cfg, params, model, task = model_ctx
    _manifest, _rows, _indices, inputs, labels, _common = select_rows(
        P5_DEV_DATA, list(range(32)), 40, "concentrated"
    )
    fwd, _targets, _aux = task.prepare({
        "inputs": inputs.reshape(len(inputs), 81).astype(np.int64),
        "labels": np.zeros((len(inputs), 81), dtype=np.int64),
    })
    fwd = dict(fwd)
    owner_grids = initial_agent_grids(inputs, cell_ids)
    fwd["patches"] = jnp.asarray(patches_from_agent_grids(owner_grids))
    predictor = make_predictor(model, "random")
    logits, _active = run_logits(predictor, params, fwd, len(inputs), 1.0, 20261009)
    shifted = logits - logits.max(axis=-1, keepdims=True)
    probs = np.exp(shifted)
    probs /= probs.sum(axis=-1, keepdims=True)
    candidates, correct = [], []
    for family in range(len(inputs)):
        for row in range(9):
            for col in range(9):
                cell = row * 9 + col
                if inputs[family].reshape(-1)[cell] != 0:
                    continue
                node, slot = row, col
                digit = int(np.argmax(probs[node, family, slot]))
                if digit == 0:
                    continue
                candidates.append(float(probs[node, family, slot, digit]))
                correct.append(int(digit == int(labels[family].reshape(-1)[cell])))
    conf = np.asarray(candidates, dtype=np.float64)
    hit = np.asarray(correct, dtype=bool)
    accepted = []
    for threshold in np.unique(conf):
        select = conf >= threshold
        if np.any(select) and float(hit[select].mean()) >= 0.995:
            accepted.append((float(threshold), int(select.sum()), float(hit[select].mean())))
    if accepted:
        threshold, selected, precision = min(accepted, key=lambda item: item[0])
        status = "calibrated"
    else:
        threshold, selected, precision, status = 1.000001, 0, None, "no_threshold_reached_target"
    return {
        "seed": seed, "arm": arm, "checkpoint_sha256": checkpoint_sha,
        "threshold": threshold, "selected_dev_predictions": selected,
        "precision": precision, "target_precision": 0.995,
        "dev_prediction_count": len(candidates), "status": status,
        "calibration_data_manifest_sha256": sha256(P5_DEV_DATA / "paired_clue_manifest.json"),
        "calibration_family_ids": list(range(32)), "clues": 40,
        "distribution": "concentrated", "owner_view": "row agent",
        "calibration_logic_sha256": logic_sha,
    }


def get_calibrations(contexts: dict, cell_ids: np.ndarray, logic_sha: str) -> dict:
    path = OUT / "calibration" / f"owner_thresholds_{logic_sha[:12]}.json"
    if path.is_file():
        saved = json.loads(path.read_text(encoding="utf-8"))
        expected_keys = {f"{seed}/{arm}" for seed, arm in contexts}
        if set(saved.get("thresholds", {})) == expected_keys:
            for key, row in saved["thresholds"].items():
                if row.get("calibration_logic_sha256") != logic_sha:
                    raise RuntimeError("calibration provenance does not match this runner")
            return saved["thresholds"]
    thresholds = {}
    for (seed, arm), model_ctx in contexts.items():
        key = f"{seed}/{arm}"
        thresholds[key] = calibrate_one(model_ctx, seed, arm, cell_ids, logic_sha)
        atomic_json(path, {
            "status": "calibrating",
            "thresholds": thresholds,
        })
        print(json.dumps({"calibration_saved": key, **thresholds[key]}), flush=True)
    atomic_json(path, {"status": "complete", "thresholds": thresholds})
    return thresholds


def p7_specs():
    specs = [
        {"condition": "direct_admm_on", "method": "direct", "admm": True,
         "visibility": "none", "policy": "random", "fraction": 1.0},
        {"condition": "direct_admm_off", "method": "direct", "admm": False,
         "visibility": "none", "policy": "none", "fraction": 0.0},
        {"condition": "decim_admm_on_global", "method": "decimation", "admm": True,
         "visibility": "global", "policy": "random", "fraction": 1.0},
        {"condition": "decim_admm_off_global", "method": "decimation", "admm": False,
         "visibility": "global", "policy": "none", "fraction": 0.0},
        {"condition": "decim_admm_on_local", "method": "decimation", "admm": True,
         "visibility": "local", "policy": "random", "fraction": 1.0},
        {"condition": "decim_admm_off_local", "method": "decimation", "admm": False,
         "visibility": "local", "policy": "none", "fraction": 0.0},
    ]
    for spec in specs:
        spec.update({"clues": 40, "distribution": "concentrated"})
    return specs


def p8_specs(family_start: int):
    specs = []
    for clues in (32, 40, 48):
        for distribution in ("balanced", "concentrated"):
            for fraction, reps in ((1.0, (0,)), (0.5, (0, 1, 2))):
                for rep in reps:
                    name = f"c{clues}_{distribution}_comm{int(fraction*100)}_r{rep}"
                    specs.append({
                        "condition": name, "family_start": family_start,
                        "method": "direct", "admm": True, "visibility": "none",
                        "policy": "random", "fraction": fraction,
                        "mask_rep": rep, "clues": clues,
                        "distribution": distribution,
                    })
                    specs.append({
                        "condition": name, "family_start": family_start,
                        "method": "decimation", "admm": True, "visibility": "global",
                        "policy": "random", "fraction": fraction,
                        "mask_rep": rep, "clues": clues,
                        "distribution": distribution,
                    })
    return specs


def legal_in_row(row_values: np.ndarray, digit: int) -> bool:
    return digit not in set(int(value) for value in row_values if value > 0)


def propose_row_commits(
    logits: np.ndarray, agent_grids: np.ndarray, labels: np.ndarray,
    threshold: float, cell_ids: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    batch = len(agent_grids)
    proposals = np.zeros((batch, 81), dtype=np.uint8)
    correct = np.zeros((batch, 81), dtype=np.uint8)
    confidences = np.zeros((batch, 81), dtype=np.float32)
    shifted = logits - logits.max(axis=-1, keepdims=True)
    probs = np.exp(shifted)
    probs /= probs.sum(axis=-1, keepdims=True)
    for b in range(batch):
        for row in range(9):
            node = row
            empty_slots = np.flatnonzero(agent_grids[b, node] == 0)
            if empty_slots.size == 0:
                continue
            candidates = []
            for slot in empty_slots.tolist():
                cell = int(cell_ids[node, slot])
                digit = int(np.argmax(probs[node, b, slot]))
                if digit == 0:
                    continue
                confidence = float(probs[node, b, slot, digit])
                if confidence >= threshold:
                    candidates.append((confidence, slot, digit, cell))
            candidates.sort(key=lambda item: (-item[0], item[1]))
            limit = max(1, int(math.floor(MAX_FRACTION * int(empty_slots.size))))
            row_state = agent_grids[b, node].copy()
            accepted = 0
            for confidence, slot, digit, cell in candidates:
                if accepted >= limit:
                    break
                if not legal_in_row(row_state, digit):
                    continue
                row_state[slot] = digit
                proposals[b, cell] = digit
                correct[b, cell] = int(digit == int(labels[b].reshape(-1)[cell]))
                confidences[b, cell] = confidence
                accepted += 1
    return proposals, correct, confidences


def target_scores(logits, labels, common, cell_ids):
    shifted = logits.astype(np.float64) - logits.astype(np.float64).max(axis=-1, keepdims=True)
    log_probs = shifted - np.log(np.exp(shifted).sum(axis=-1, keepdims=True))
    batch = len(labels)
    true_logp = np.zeros((batch, 27, 9), dtype=np.float32)
    true_correct = np.zeros((batch, 27, 9), dtype=np.uint8)
    target_mask = common[:, cell_ids]
    flat_labels = labels.reshape(batch, 81)
    for node in range(27):
        for slot, cell in enumerate(cell_ids[node]):
            digit = flat_labels[:, int(cell)].astype(np.int64)
            true_logp[:, node, slot] = log_probs[node, np.arange(batch), slot, digit]
            pred = logits[node, :, slot].argmax(axis=-1)
            true_correct[:, node, slot] = (pred == digit).astype(np.uint8)
    return true_logp, true_correct, target_mask


def route_commits(
    agent_grids: np.ndarray, proposals: np.ndarray, visibility: str,
    cell_ids: np.ndarray, owner_lookup: dict, frozen: np.ndarray,
    target_mask: np.ndarray, shadow_grid: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    batch = len(agent_grids)
    recipients = np.full((batch, 81, 3), 255, dtype=np.uint8)
    conflicts = np.zeros((batch, 81), dtype=np.uint8)
    sent_updates = np.zeros(batch, dtype=np.uint16)
    for b in range(batch):
        for cell in np.flatnonzero(proposals[b]).tolist():
            digit = int(proposals[b, cell])
            views = owner_lookup[int(cell)]
            owner = int(cell // 9)
            if (owner, int(cell % 9)) not in views:
                raise AssertionError("row proposer is not incident to its cell")
            if core.legal(shadow_grid[b].reshape(9, 9), cell // 9, cell % 9, digit):
                pass
            else:
                conflicts[b, cell] = 1
            shadow_grid[b, cell // 9, cell % 9] = digit
            accepted_views = views if visibility == "global" else [(owner, int(cell % 9))]
            for recipient, slot in accepted_views:
                if int(cell_ids[recipient, slot]) != int(cell):
                    raise AssertionError("commit destination does not contain its cell")
                agent_grids[b, recipient, slot] = digit
                recipients[b, cell, views.index((recipient, slot))] = recipient
                frozen[b, recipient, slot] = target_mask[b, recipient, slot]
                sent_updates[b] += 1
    return agent_grids, recipients, conflicts, sent_updates


def family_scores(score_logp, score_correct, target_mask):
    out_logp, out_acc = [], []
    for b in range(len(score_logp)):
        selected = target_mask[b]
        if not np.all(np.isfinite(score_logp[b][selected])):
            raise AssertionError("a common target view lacks a pre-commit score")
        out_logp.append(float(score_logp[b][selected].mean()))
        out_acc.append(float(score_correct[b][selected].mean()))
    return np.asarray(out_logp), np.asarray(out_acc)


def block_provenance(run_config, config_sha, model_ctx, seed, arm, spec, method, family_ids):
    return {
        "config_sha256": config_sha,
        "run_config_sha256": hashlib.sha256(json.dumps(
            run_config, sort_keys=True, default=list
        ).encode()).hexdigest(),
        "checkpoint_sha256": model_ctx[1],
        "runner_sha256": run_config["runner_sha256"],
        "core_sha256": run_config["core_sha256"],
        "protocol_sha256": run_config["protocol_sha256"],
        "dataset_manifest_sha256": run_config["dataset_manifest_sha256"],
        "seed": seed, "arm": arm, "condition": spec["condition"],
        "method": method, "admm": bool(spec["admm"]),
        "visibility": spec["visibility"], "policy": spec["policy"],
        "fraction": float(spec["fraction"]), "mask_rep": int(spec.get("mask_rep", 0)),
        "mask_seed_base": int(MASK_SEEDS[int(spec.get("mask_rep", 0))]),
        "family_ids": [int(value) for value in family_ids],
        "clues": int(spec["clues"]), "distribution": spec["distribution"],
        "K": K, "max_passes": int(run_config["max_passes"]),
    }


def run_trajectory(
    *, run_dir: Path, run_config: dict, config_sha: str, model_ctx, seed: int,
    arm: str, spec: dict, method: str, family_ids: list[int], inputs: np.ndarray,
    labels: np.ndarray, common: np.ndarray, task, cell_ids: np.ndarray,
    owner_lookup: dict, threshold: float,
) -> tuple[dict, bool]:
    _path, _checkpoint_sha, _cfg, params, model, _task = model_ctx
    batch = len(inputs)
    provenance = block_provenance(
        run_config, config_sha, model_ctx, seed, arm, spec, method, family_ids
    )
    trajectory = run_dir / f"seed_{seed}" / arm / spec["condition"] / method
    trajectory.mkdir(parents=True, exist_ok=True)
    fwd, _targets, _aux = task.prepare({
        "inputs": inputs.reshape(batch, 81).astype(np.int64),
        "labels": np.zeros((batch, 81), dtype=np.int64),
    })
    fwd = dict(fwd)
    base_grids = initial_agent_grids(inputs, cell_ids)
    agent_grids = base_grids.copy()
    score_logp = np.full((batch, 27, 9), np.nan, dtype=np.float32)
    score_correct = np.zeros((batch, 27, 9), dtype=np.uint8)
    frozen = np.zeros((batch, 27, 9), dtype=bool)
    target_mask = common[:, cell_ids]
    shadow_grid = inputs.copy()
    total_commits = np.zeros(batch, dtype=np.uint16)
    correct_commits = np.zeros(batch, dtype=np.uint16)
    wrong_commits = np.zeros(batch, dtype=np.uint16)
    conflicts = np.zeros(batch, dtype=np.uint16)
    routed_updates = np.zeros(batch, dtype=np.uint16)
    predictor = make_predictor(model, spec["policy"])
    fraction = float(spec["fraction"])
    max_passes = int(run_config["max_passes"])
    last_active = np.zeros(K, dtype=np.int64)
    for pass_index in range(max_passes):
        pass_path = trajectory / f"pass_{pass_index + 1:02d}.npz"
        update_progress(
            run_dir, status="running",
            current={"trajectory": f"{seed}/{arm}/{spec['condition']}/{method}",
                     "next_pass": pass_index + 1, "total_passes": max_passes},
        )
        pass_prov = dict(provenance, pass_index=pass_index + 1,
                         mask_seed=int((MASK_SEEDS[int(spec.get("mask_rep", 0))]
                                        + (pass_index + 1) * 104729) % 2_000_000_000))
        saved = load_verified_npz(pass_path, pass_prov)
        if saved is not None:
            # Resume from the latest consecutive verified pass. A gap followed
            # by an existing pass is corruption, not permission to skip work.
            agent_grids = saved["agent_grids"].astype(np.uint8)
            score_logp = saved["score_logp"].astype(np.float32)
            score_correct = saved["score_correct"].astype(np.uint8)
            frozen = saved["frozen"].astype(bool)
            shadow_grid = saved["shadow_grid"].astype(np.uint8)
            total_commits = saved["total_commits"].astype(np.uint16)
            correct_commits = saved["correct_commits"].astype(np.uint16)
            wrong_commits = saved["wrong_commits"].astype(np.uint16)
            conflicts = saved["conflicts"].astype(np.uint16)
            routed_updates = saved["routed_updates"].astype(np.uint16)
            last_active = saved["active_edges"].astype(np.int64)
            continue
        for later in range(pass_index + 2, max_passes + 1):
            if (trajectory / f"pass_{later:02d}.npz").exists():
                raise RuntimeError(f"nonconsecutive pass artifacts under {trajectory}")
        started = time.perf_counter()
        fwd["patches"] = jnp.asarray(patches_from_agent_grids(agent_grids))
        mask_seed = int(pass_prov["mask_seed"])
        logits, active = run_logits(predictor, params, fwd, batch, fraction, mask_seed)
        logp_now, correct_now, target_mask = target_scores(logits, labels, common, cell_ids)
        update_scores = target_mask & ~frozen
        score_logp[update_scores] = logp_now[update_scores]
        score_correct[update_scores] = correct_now[update_scores]
        pass_proposals = np.zeros((batch, 81), dtype=np.uint8)
        pass_correct = np.zeros((batch, 81), dtype=np.uint8)
        pass_confidence = np.zeros((batch, 81), dtype=np.float32)
        recipients = np.full((batch, 81, 3), 255, dtype=np.uint8)
        pass_conflicts = np.zeros((batch, 81), dtype=np.uint8)
        pass_updates = np.zeros(batch, dtype=np.uint16)
        if method == "decimation" and pass_index + 1 < max_passes:
            pass_proposals, pass_correct, pass_confidence = propose_row_commits(
                logits, agent_grids, labels, threshold, cell_ids
            )
            before = agent_grids.copy()
            agent_grids, recipients, pass_conflicts, pass_updates = route_commits(
                agent_grids, pass_proposals, spec["visibility"], cell_ids,
                owner_lookup, frozen, target_mask, shadow_grid,
            )
            committed = pass_proposals > 0
            total_commits += committed.sum(axis=1).astype(np.uint16)
            correct_commits += (committed & (pass_correct > 0)).sum(axis=1).astype(np.uint16)
            wrong_commits += (committed & (pass_correct == 0)).sum(axis=1).astype(np.uint16)
            conflicts += pass_conflicts.sum(axis=1).astype(np.uint16)
            routed_updates += pass_updates
            if not np.any(committed) and np.any(before != agent_grids):
                raise AssertionError("views changed without an emitted commit")
        frozen_scores_logp, frozen_scores_acc = family_scores(
            score_logp, score_correct, target_mask
        )
        arrays = {
            "agent_grids": agent_grids,
            "score_logp": score_logp,
            "score_correct": score_correct,
            "frozen": frozen.astype(np.uint8),
            "target_mask": target_mask.astype(np.uint8),
            "shadow_grid": shadow_grid,
            "proposals": pass_proposals,
            "proposal_correct": pass_correct,
            "proposal_confidence": pass_confidence,
            "recipients": recipients,
            "pass_conflicts": pass_conflicts,
            "total_commits": total_commits,
            "correct_commits": correct_commits,
            "wrong_commits": wrong_commits,
            "conflicts": conflicts,
            "routed_updates": routed_updates,
            "primary_logprob_by_family": frozen_scores_logp,
            "primary_accuracy_by_family": frozen_scores_acc,
            "active_edges": np.asarray(active, dtype=np.int64),
            "pass_seconds": np.asarray([time.perf_counter() - started], dtype=np.float64),
            "mask_seed": np.asarray([mask_seed], dtype=np.int64),
        }
        atomic_npz(pass_path, arrays, pass_prov)
        update_progress(
            run_dir, status="running",
            current={"trajectory": f"{seed}/{arm}/{spec['condition']}/{method}",
                     "pass_checkpointed": pass_index + 1, "total_passes": max_passes},
        )
        last_active = np.asarray(active, dtype=np.int64)
        print(json.dumps({
            "trajectory": f"{seed}/{arm}/{spec['condition']}/{method}",
            "pass_saved": pass_index + 1,
            "passes": max_passes,
            "mean_primary_logprob": float(frozen_scores_logp.mean()),
            "mean_commits": float(total_commits.mean()),
            "elapsed_seconds": round(float(arrays["pass_seconds"][0]), 2),
        }), flush=True)
        if PAUSE_REQUESTED or (run_dir / "PAUSE").exists():
            return {}, True
    final_path = trajectory / f"pass_{max_passes:02d}.npz"
    final = load_verified_npz(final_path, dict(
        provenance, pass_index=max_passes,
        mask_seed=int((MASK_SEEDS[int(spec.get("mask_rep", 0))]
                       + max_passes * 104729) % 2_000_000_000),
    ))
    row_logp = final["primary_logprob_by_family"].astype(float)
    row_acc = final["primary_accuracy_by_family"].astype(float)
    rows = []
    for index, family_id in enumerate(family_ids):
        rows.append({
            "family_id": int(family_id), "seed": seed, "arm": arm,
            "condition": spec["condition"], "method": method,
            "admm": int(spec["admm"]), "visibility": spec["visibility"],
            "communication_fraction": fraction, "mask_rep": int(spec.get("mask_rep", 0)),
            "clues": int(spec["clues"]), "distribution": spec["distribution"],
            "primary_logprob": float(row_logp[index]),
            "primary_accuracy": float(row_acc[index]),
            "commits": int(final["total_commits"][index]),
            "correct_commits": int(final["correct_commits"][index]),
            "wrong_commits": int(final["wrong_commits"][index]),
            "cross_view_conflicts": int(final["conflicts"][index]),
            "routed_view_updates": int(final["routed_updates"][index]),
            "active_edge_updates_per_pass": int(final["active_edges"].sum()),
            "threshold": float(threshold),
            "passes": max_passes,
        })
    return {"rows": rows, "last_active": last_active}, False


def technical_control(model_ctx, inputs, cell_ids):
    """Confirm local commit routing and no-ADMM remote invariance."""
    _path, _sha, _cfg, params, model, task = model_ctx
    owner_lookup = make_owner_lookup(cell_ids)
    one = inputs[:1].copy()
    fwd, _targets, _aux = task.prepare({
        "inputs": one.reshape(1, 81).astype(np.int64),
        "labels": np.zeros((1, 81), dtype=np.int64),
    })
    fwd = dict(fwd)
    grids = initial_agent_grids(one, cell_ids)
    base_patches = patches_from_agent_grids(grids)
    if not np.array_equal(base_patches, np.asarray(fwd["patches"])):
        raise AssertionError("agent-view reconstruction differs from the canonical Sudoku slicing")
    changed_node = 0
    changed_slot = int(np.flatnonzero(grids[0, changed_node] == 0)[0])
    changed_cell = int(cell_ids[changed_node, changed_slot])
    changed_digit = int((np.argmax(grids[0, changed_node]) + 1) % 9 + 1)
    p2 = base_patches.copy()
    p2[changed_node, 0, changed_slot] = 0
    p2[changed_node, 0, changed_slot, changed_digit] = 1
    changed_nodes = np.flatnonzero(np.any(p2[:, 0] != base_patches[:, 0], axis=(1, 2)))
    if not np.array_equal(changed_nodes, np.asarray([changed_node])):
        raise AssertionError("single local perturbation leaked into another input view")
    remote = np.asarray([node for node in range(27) if node != changed_node], dtype=np.int32)
    predictor = make_predictor(model, "none")
    edges = np.asarray(fwd["edge_indices"])
    schedule = jnp.zeros((K,), dtype=jnp.int32)
    l1, _ = jax.device_get(predictor(
        params, jnp.asarray(base_patches), fwd["edge_indices"], fwd["model_kwargs"],
        schedule, jnp.asarray(17, dtype=jnp.int32),
    ))
    l2, _ = jax.device_get(predictor(
        params, jnp.asarray(p2), fwd["edge_indices"], fwd["model_kwargs"],
        schedule, jnp.asarray(17, dtype=jnp.int32),
    ))
    max_remote_delta = float(np.max(np.abs(np.asarray(l1)[remote] - np.asarray(l2)[remote])))
    if max_remote_delta > 1e-6:
        raise AssertionError(f"no-ADMM remote invariance failed: delta={max_remote_delta}")
    recipients = [tuple(owner_lookup[changed_cell])]
    owner = (changed_cell // 9, changed_cell % 9)
    if owner not in recipients[0]:
        raise AssertionError("row owner is not an incident view")
    # Exercise the actual route implementation with one synthetic event.
    digit = next(value for value in range(1, 10)
                 if legal_in_row(grids[0, changed_node], value))
    one_proposal = np.zeros((1, 81), dtype=np.uint8)
    one_proposal[0, changed_cell] = digit
    local_grids = grids.copy()
    global_grids = grids.copy()
    local_frozen = np.zeros((1, 27, 9), dtype=bool)
    global_frozen = np.zeros_like(local_frozen)
    empty_target_mask = np.zeros_like(local_frozen)
    local_shadow = one.copy()
    global_shadow = one.copy()
    local_grids, local_dest, *_ = route_commits(
        local_grids, one_proposal, "local", cell_ids, owner_lookup,
        local_frozen, empty_target_mask, local_shadow,
    )
    global_grids, global_dest, *_ = route_commits(
        global_grids, one_proposal, "global", cell_ids, owner_lookup,
        global_frozen, empty_target_mask, global_shadow,
    )
    local_nodes = [int(x) for x in local_dest[0, changed_cell] if int(x) != 255]
    global_nodes = [int(x) for x in global_dest[0, changed_cell] if int(x) != 255]
    expected_global = [int(node) for node, _slot in recipients[0]]
    if local_nodes != [owner[0]] or set(global_nodes) != set(expected_global):
        raise AssertionError("local/global commit routing does not match the frozen semantics")
    if int(local_grids[0, owner[0], owner[1]]) != digit:
        raise AssertionError("local commit did not update the proposer view")
    for node, slot in recipients[0]:
        if int(global_grids[0, node, slot]) != digit:
            raise AssertionError("global commit did not update all incident views")
    return {
        "status": "PASS", "changed_node": changed_node,
        "changed_cell": changed_cell, "max_remote_logit_delta_no_admm": max_remote_delta,
        "global_recipients": expected_global,
        "local_recipients": local_nodes,
        "view_reconstruction_matches_canonical": True,
        "control": "single-view perturbation; all remote logits invariant with mask_policy=none",
    }


def phase_spec(phase: str, families_arg: int, start_arg: int, seeds_arg: tuple, arms_arg: tuple,
               max_passes: int):
    if phase == "pilot":
        if families_arg != 8 or start_arg != 0 or seeds_arg != (123, 456) or arms_arg != ARMS or max_passes != 8:
            raise ValueError("P7 pilot is frozen at families 0..7, seeds 123/456, both arms, eight passes")
        return list(range(0, 8)), p7_specs(), 40, "concentrated"
    if phase == "p7":
        if families_arg != 64 or start_arg != 8 or seeds_arg != TRAIN_SEEDS or arms_arg != ARMS or max_passes != 8:
            raise ValueError("P7 confirmation is frozen at families 8..71, six seeds, both arms, eight passes")
        return list(range(8, 72)), p7_specs(), 40, "concentrated"
    if phase == "p8_pilot":
        if families_arg != 8 or start_arg != 0 or seeds_arg != (123, 456) or arms_arg != ARMS or max_passes != 8:
            raise ValueError("P8 pilot uses development families 0..7, seeds 123/456, both arms, eight passes")
        return list(range(0, 8)), p8_specs(0), 40, "concentrated"
    if phase == "p8":
        if families_arg != 64 or start_arg != 72 or seeds_arg != TRAIN_SEEDS or arms_arg != ARMS or max_passes != 8:
            raise ValueError("P8 confirmation is frozen at families 72..135, six seeds, both arms, eight passes")
        return list(range(72, 136)), p8_specs(72), 32, "balanced"
    raise ValueError(f"unsupported phase: {phase}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("pilot", "p7", "p8_pilot", "p8"), required=True)
    parser.add_argument("--families", type=int, default=8)
    parser.add_argument("--family-start", type=int, default=0)
    parser.add_argument("--seeds", default="123,456")
    parser.add_argument("--arms", default="internal,terminal")
    parser.add_argument("--max-passes", type=int, default=8)
    parser.add_argument("--out-root", type=Path, default=OUT)
    parser.add_argument("--stop-after-trajectories", type=int, default=0)
    args = parser.parse_args()
    seeds = tuple(int(value) for value in args.seeds.split(",") if value)
    arms = tuple(value.strip() for value in args.arms.split(",") if value.strip())
    try:
        family_ids, specs, _base_clues, _base_distribution = phase_spec(
            args.phase, args.families, args.family_start, seeds, arms, args.max_passes
        )
    except ValueError as exc:
        parser.error(str(exc))
    if any(seed not in TRAIN_SEEDS for seed in seeds) or any(arm not in ARMS for arm in arms):
        parser.error("seeds or model arms are outside the frozen P7/P8 set")
    if args.phase in {"p7", "p8"}:
        p7_audit = OUT / "p7" / "audit.json"
        if args.phase == "p8" and not (
            p7_audit.is_file()
            and json.loads(p7_audit.read_text(encoding="utf-8")).get("status") == "PASS"
            and (OUT / "p7" / "analysis.json").is_file()
        ):
            parser.error("P8 holdout is gated on complete P7 audit and analysis")

    data_dir = FINAL_DATA
    manifest_path = data_dir / "paired_clue_manifest.json"
    if not manifest_path.is_file():
        parser.error(f"final disjoint family dataset is missing: {data_dir}")
    manifest_sha = sha256(manifest_path)
    script_sha = sha256(Path(__file__))
    core_sha = sha256(Path(core.__file__))
    protocol_sha = sha256(PROTOCOL)
    build_sha = sha256(data_dir / "frozen_build_config.json")
    cell_ids = np.asarray(build_sudoku_cell_indices(9), dtype=np.int32)
    owner_lookup = make_owner_lookup(cell_ids)
    calibration_contexts = {
        (seed, arm): core.load_model(arm, seed)
        for seed in TRAIN_SEEDS for arm in ARMS
    }
    contexts = {(seed, arm): calibration_contexts[(seed, arm)] for seed in seeds for arm in arms}
    calibration_sha = calibration_logic_sha256()
    calibrations = get_calibrations(calibration_contexts, cell_ids, calibration_sha)

    run_config = {
        "phase": args.phase, "family_ids": family_ids, "seeds": seeds, "arms": arms,
        "max_passes": args.max_passes, "K": K,
        "conditions": specs, "max_commit_fraction_per_row_per_pass": MAX_FRACTION,
        "threshold_calibration_sha256": sha256(OUT / "calibration" / f"owner_thresholds_{calibration_sha[:12]}.json"),
        "thresholds": {key: value["threshold"] for key, value in calibrations.items()},
        "dataset_manifest_sha256": manifest_sha, "dataset_build_config_sha256": build_sha,
        "runner_sha256": script_sha, "core_sha256": core_sha,
        "protocol_sha256": protocol_sha,
        "checkpoint_sha256": {f"{seed}/{arm}": ctx[1] for (seed, arm), ctx in contexts.items()},
        "backend": jax.default_backend(), "labels_provided_to_model": False,
        "primary_scoring": "equal agent-view cells; freeze each score immediately before that view receives a commit",
    }
    config_sha = hashlib.sha256(json.dumps(run_config, sort_keys=True, default=list).encode()).hexdigest()
    phase_folder = {"pilot": "p7_pilot", "p7": "p7", "p8_pilot": "p8_pilot", "p8": "p8"}[args.phase]
    run_dir = args.out_root.resolve() / phase_folder
    if args.phase in {"pilot", "p8_pilot"}:
        run_dir = run_dir / config_sha[:12]
    run_dir.mkdir(parents=True, exist_ok=True)
    run_config_path = run_dir / "run_config.json"
    if run_config_path.is_file():
        saved_config = json.loads(run_config_path.read_text(encoding="utf-8"))
        if saved_config != json.loads(json.dumps(run_config, default=list)):
            raise RuntimeError("run directory exists with a different frozen configuration")
    else:
        atomic_json(run_config_path, run_config)
    atomic_json(run_dir / "technical_control.json", technical_control(
        next(iter(contexts.values())),
        select_rows(data_dir, [family_ids[0]], 40, "concentrated")[3], cell_ids,
    ))

    total_trajectories = len(contexts) * len(specs)
    progress_path = run_dir / "progress.json"
    progress = json.loads(progress_path.read_text(encoding="utf-8")) if progress_path.is_file() else {
        "status": "running", "total_trajectories": total_trajectories,
        "completed_trajectories": [], "current": None,
    }
    expected_complete = set()
    family_rows = []
    done_count = 0
    for (seed, arm), model_ctx in contexts.items():
        for spec in specs:
            method = spec["method"]
            condition = spec["condition"]
            trajectory_key = f"{seed}/{arm}/{condition}/{method}"
            expected_complete.add(trajectory_key)
            if trajectory_key in progress["completed_trajectories"]:
                summary = read_trajectory_summary(
                    run_dir, seed, arm, spec, method, family_ids, model_ctx, run_config, config_sha
                )
                family_rows.extend(summary)
                done_count += 1
                continue
            if args.phase == "p8" or (args.phase == "p8_pilot"):
                clues, distribution = int(spec["clues"]), str(spec["distribution"])
            else:
                clues, distribution = 40, "concentrated"
            _manifest, _rows, _idx, inputs, labels, common = select_rows(
                data_dir, family_ids, clues, distribution
            )
            key = f"{seed}/{arm}"
            trajectory_result, paused = run_trajectory(
                run_dir=run_dir, run_config=run_config, config_sha=config_sha,
                model_ctx=model_ctx, seed=seed, arm=arm, spec=spec, method=method,
                family_ids=family_ids, inputs=inputs, labels=labels, common=common,
                task=model_ctx[-1], cell_ids=cell_ids, owner_lookup=owner_lookup,
                threshold=float(calibrations[key]["threshold"]),
            )
            if paused:
                progress.update({
                    "status": "paused", "current": trajectory_key,
                    "completed_count": len(progress["completed_trajectories"]),
                    "updated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                })
                atomic_json(progress_path, progress)
                print(json.dumps({"status": "paused", "run_dir": str(run_dir.relative_to(ROOT)),
                                  "completed": len(progress["completed_trajectories"]),
                                  "total": total_trajectories}), flush=True)
                return
            family_rows.extend(trajectory_result["rows"])
            progress["completed_trajectories"].append(trajectory_key)
            progress.update({
                "status": "running", "current": None,
                "completed_count": len(progress["completed_trajectories"]),
                "updated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            })
            atomic_json(progress_path, progress)
            done_count += 1
            if args.stop_after_trajectories and done_count >= args.stop_after_trajectories:
                progress.update({
                    "status": "checkpointed", "current": None,
                    "completed_count": len(progress["completed_trajectories"]),
                    "updated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                })
                atomic_json(progress_path, progress)
                print(json.dumps({"status": "stopped_at_checkpoint", "completed": done_count,
                                  "run_dir": str(run_dir.relative_to(ROOT))}), flush=True)
                return
    if set(progress["completed_trajectories"]) != expected_complete:
        raise RuntimeError("progress ledger and expected trajectories disagree")
    if not family_rows:
        for (seed, arm), model_ctx in contexts.items():
            for spec in specs:
                family_rows.extend(read_trajectory_summary(
                    run_dir, seed, arm, spec, spec["method"], family_ids,
                    model_ctx, run_config, config_sha,
                ))
    atomic_csv(run_dir / "family_results.csv", family_rows)
    progress.update({"status": "complete", "completed_count": len(expected_complete),
                     "current": None, "updated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())})
    atomic_json(progress_path, progress)
    print(json.dumps({"status": "complete", "run_dir": str(run_dir.relative_to(ROOT)),
                      "trajectories": len(expected_complete), "family_result_rows": len(family_rows),
                      "config_sha256": config_sha}), flush=True)


def read_trajectory_summary(run_dir, seed, arm, spec, method, family_ids,
                            model_ctx, run_config, config_sha):
    path = run_dir / f"seed_{seed}" / arm / spec["condition"] / method / f"pass_{run_config['max_passes']:02d}.npz"
    provenance = block_provenance(run_config, config_sha, model_ctx, seed, arm, spec, method, family_ids)
    expected = dict(provenance, pass_index=run_config["max_passes"],
                    mask_seed=int((MASK_SEEDS[int(spec.get("mask_rep", 0))]
                                   + run_config["max_passes"] * 104729) % 2_000_000_000))
    final = load_verified_npz(path, expected)
    rows = []
    fraction = float(spec["fraction"])
    for index, family_id in enumerate(family_ids):
        rows.append({
            "family_id": int(family_id), "seed": seed, "arm": arm,
            "condition": spec["condition"], "method": method,
            "admm": int(spec["admm"]), "visibility": spec["visibility"],
            "communication_fraction": fraction, "mask_rep": int(spec.get("mask_rep", 0)),
            "clues": int(spec["clues"]), "distribution": spec["distribution"],
            "primary_logprob": float(final["primary_logprob_by_family"][index]),
            "primary_accuracy": float(final["primary_accuracy_by_family"][index]),
            "commits": int(final["total_commits"][index]),
            "correct_commits": int(final["correct_commits"][index]),
            "wrong_commits": int(final["wrong_commits"][index]),
            "cross_view_conflicts": int(final["conflicts"][index]),
            "routed_view_updates": int(final["routed_updates"][index]),
            "active_edge_updates_per_pass": int(final["active_edges"].sum()),
            "threshold": float(run_config["thresholds"][f"{seed}/{arm}"]),
            "passes": run_config["max_passes"],
        })
    return rows


def install_pause_handler():
    def request_pause(_signum, _frame):
        global PAUSE_REQUESTED
        PAUSE_REQUESTED = True
    signal.signal(signal.SIGINT, request_pause)


if __name__ == "__main__":
    install_pause_handler()
    main()
