"""Checkpointed P5 communication × decimation follow-up on the new holdout."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import run_coordination_communication_decimation as core  # noqa: E402

CONFIRM_DATA = ROOT / "datasets/sudoku_coordination_confirmation_20261007"
DEV_DATA = ROOT / "datasets/sudoku_coordination_pilot_20261007"
OUT = ROOT / "outputs/p5_decim"
PROTOCOL = ROOT / "experiments/sheaf_admm_reasoning/P5_PROTOCOLO_CONFIRMATORIO_2026-10-07.md"
TRAIN_SEEDS = (123, 456, 789, 1011, 1213, 20260927)
ARMS = ("internal", "terminal")
MASK_SEEDS = (391, 773, 1237)
MAX_PASSES = 8
MASK_REPS = 3


def sha256(path: Path) -> str:
    return core.sha256(path)


def atomic_json(path: Path, payload: dict) -> None:
    core.atomic_bytes(path, json.dumps(payload, indent=2).encode("utf-8"))


def condition_records(mask_reps: int = 3) -> list[dict]:
    rows = [{"name": "full_100", "policy": "random", "fraction": 1.0, "rep": 0}]
    for fraction in (0.75, 0.50):
        for rep in range(mask_reps):
            rows.append({
                "name": f"random_{int(fraction * 100)}_r{rep}",
                "policy": "random", "fraction": fraction, "rep": rep,
            })
    rows.append({"name": "none_0", "policy": "none", "fraction": 0.0, "rep": 0})
    return rows


def select_rows(data_dir: Path, family_ids: list[int]):
    manifest_path = data_dir / "paired_clue_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    by_family = {
        int(row["family_id"]): row
        for row in manifest["families"]
        if row["policy"] == "concentrated" and int(row["clues"]) == 40
        and int(row["family_id"]) in family_ids
    }
    if sorted(by_family) != sorted(family_ids):
        raise RuntimeError(f"incomplete P5 40-clue rows in {data_dir}")
    rows = [by_family[family] for family in family_ids]
    all_inputs = np.load(data_dir / "test_hard/all__inputs.npy", mmap_mode="r")
    all_labels = np.load(data_dir / "test_hard/all__labels.npy", mmap_mode="r")
    indices = [int(row["row_index"]) for row in rows]
    inputs = np.asarray(all_inputs[indices], dtype=np.uint8).reshape(-1, 9, 9)
    labels = np.asarray(all_labels[indices], dtype=np.uint8).reshape(-1, 9, 9)
    common = np.zeros((len(rows), 81), dtype=bool)
    for i, row in enumerate(rows):
        common[i, np.asarray(row["common_evaluation_cells"], dtype=np.int32)] = True
    return manifest, rows, indices, inputs, labels, common


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("smoke", "confirm"), default="smoke")
    parser.add_argument("--families", type=int, default=2)
    parser.add_argument("--eval-family-start", type=int, default=32)
    parser.add_argument("--seeds", default="123")
    parser.add_argument("--arms", default="internal")
    parser.add_argument("--max-passes", type=int, default=2)
    parser.add_argument("--mask-reps", type=int, default=1)
    parser.add_argument("--eval-data-dir", type=Path)
    parser.add_argument("--dev-data-dir", type=Path, default=DEV_DATA)
    args = parser.parse_args()

    eval_dir = (args.eval_data_dir or (CONFIRM_DATA if args.phase == "confirm" else DEV_DATA)).resolve()
    dev_dir = args.dev_data_dir.resolve()
    seeds = tuple(int(value) for value in args.seeds.split(",") if value)
    arms = tuple(value.strip() for value in args.arms.split(",") if value.strip())
    if args.phase == "confirm":
        if (
            eval_dir != CONFIRM_DATA.resolve() or dev_dir != DEV_DATA.resolve()
            or args.families != 64 or args.eval_family_start != 0
            or seeds != TRAIN_SEEDS or arms != ARMS
            or args.max_passes != MAX_PASSES or args.mask_reps != MASK_REPS
        ):
            parser.error("confirm parameters must match the frozen P5 protocol")
        audit_path = ROOT / "outputs/coordination_pilot_20261007/p5_confirmation/holdout_audit.json"
        if not audit_path.is_file() or json.loads(audit_path.read_text(encoding="utf-8")).get("status") != "PASS":
            parser.error("confirmation holdout must pass its independent audit first")
        out_root = OUT
    else:
        if not 1 <= args.families <= 8 or not 1 <= args.max_passes <= 8 or not 1 <= args.mask_reps <= 3:
            parser.error("smoke supports 1..8 families, 1..8 passes and 1..3 mask seeds")
        if seeds == TRAIN_SEEDS or not seeds or any(seed not in TRAIN_SEEDS for seed in seeds):
            parser.error("smoke seeds must be selected from the audited checkpoint list")
        if not arms or any(arm not in ARMS for arm in arms):
            parser.error(f"arms must be drawn from {ARMS}")
        out_root = ROOT / "outputs/p5smoke/decim"

    if not seeds or any(seed not in TRAIN_SEEDS for seed in seeds):
        parser.error(f"seeds must be drawn from {TRAIN_SEEDS}")
    if not arms or any(arm not in ARMS for arm in arms):
        parser.error(f"arms must be drawn from {ARMS}")
    if not 1 <= args.max_passes <= 8 or not 1 <= args.mask_reps <= 3:
        parser.error("max-passes must be 1..8 and mask-reps 1..3")

    eval_ids = list(range(args.eval_family_start, args.eval_family_start + args.families))
    dev_ids = list(range(32))
    eval_manifest, eval_rows, eval_indices, inputs, labels, common = select_rows(eval_dir, eval_ids)
    dev_manifest, dev_rows, dev_indices, dev_inputs, dev_labels, _dev_common = select_rows(dev_dir, dev_ids)
    eval_manifest_path = eval_dir / "paired_clue_manifest.json"
    dev_manifest_path = dev_dir / "paired_clue_manifest.json"
    eval_manifest_sha = sha256(eval_manifest_path)
    dev_manifest_sha = sha256(dev_manifest_path)
    code_sha = sha256(Path(__file__))
    core_sha = sha256(Path(core.__file__))
    protocol_sha = sha256(PROTOCOL)
    data_key = {
        "eval_manifest_sha256": eval_manifest_sha,
        "dev_manifest_sha256": dev_manifest_sha,
        "eval_input_sha256": hashlib.sha256(inputs.tobytes()).hexdigest(),
        "eval_label_sha256_for_scoring_only": hashlib.sha256(labels.tobytes()).hexdigest(),
        "dev_input_sha256": hashlib.sha256(dev_inputs.tobytes()).hexdigest(),
        "dev_label_sha256_for_calibration_only": hashlib.sha256(dev_labels.tobytes()).hexdigest(),
        "eval_row_indices": eval_indices,
        "dev_row_indices": dev_indices,
    }
    conditions = condition_records(args.mask_reps)
    config = {
        "phase": "P5_secondary_communication_decimation" if args.phase == "confirm" else "P5_smoke_communication_decimation",
        "families": len(eval_ids),
        "family_ids": eval_ids,
        "seeds": seeds,
        "arms": arms,
        "max_passes": args.max_passes,
        "K_per_pass": core.K,
        "clues": 40,
        "distribution": "concentrated",
        "mask_reps": args.mask_reps,
        "conditions": conditions,
        "methods": ["direct", "decimation"],
        "max_commit_fraction_per_pass": core.MAX_FRACTION,
        "threshold_target_precision": 0.995,
        "threshold_calibration_family_ids": dev_ids,
        "code_sha256": code_sha,
        "communication_core_sha256": core_sha,
        "protocol_sha256": protocol_sha,
        "data": data_key,
        "backend": core.jax.default_backend(),
        "labels_provided_to_model": False,
    }
    config_hash = hashlib.sha256(json.dumps(config, sort_keys=True, default=list).encode()).hexdigest()
    tag = "".join({"internal": "i", "terminal": "t"}[arm] for arm in arms)
    out_dir = out_root / (
        f"n{len(eval_ids)}_s{len(seeds)}_a{len(arms)}_p{args.max_passes}m{args.mask_reps}_{config_hash[:8]}"
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    run_config = json.loads(json.dumps(
        dict(config, config_sha256=config_hash, run_dir=str(out_dir.relative_to(ROOT))),
        default=list,
    ))
    config_path = out_dir / "run_config.json"
    if config_path.exists():
        saved = json.loads(config_path.read_text(encoding="utf-8"))
        if saved != run_config:
            raise RuntimeError("P5 decimation run directory belongs to another frozen configuration")
    else:
        atomic_json(config_path, run_config)

    calibration_path = out_dir / "calibration.json"
    calibrations = {}
    if calibration_path.exists():
        saved = json.loads(calibration_path.read_text(encoding="utf-8"))
        if saved.get("config_sha256") != config_hash:
            raise RuntimeError("saved threshold calibration belongs to another P5 configuration")
        calibrations = saved["thresholds"]

    contexts = {}
    for seed in seeds:
        for arm in arms:
            model_ctx = core.load_model(arm, seed)
            contexts[(seed, arm)] = model_ctx
            key = f"seed_{seed}/{arm}"
            if key not in calibrations:
                calibration = core.calibrate(
                    model_ctx, dev_inputs, dev_labels, code_sha, dev_manifest_sha, out_dir
                )
                calibrations[key] = calibration
                atomic_json(calibration_path, {"config_sha256": config_hash, "thresholds": calibrations})
                print(json.dumps({"calibration_complete": key, "threshold": calibration["threshold"]}), flush=True)

    pass_rows: list[dict] = []
    paired_rows: list[dict] = []
    completed_conditions = 0
    total_conditions = len(contexts) * len(conditions)
    for (seed, arm), model_ctx in contexts.items():
        checkpoint, checkpoint_sha, cfg, params, model, task = model_ctx
        predictors = {policy: core.make_predictor(model, policy) for policy in ("random", "none")}
        tau = float(calibrations[f"seed_{seed}/{arm}"]["threshold"])
        cg_iters = int(cfg["model"].get("cg_iters", 3))
        for condition in conditions:
            name = condition["name"]
            policy = condition["policy"]
            fraction = float(condition["fraction"])
            rep = int(condition["rep"])
            mask_seed = MASK_SEEDS[rep]
            condition_dir = out_dir / f"s{seed}_{arm[0]}" / name
            provenance_base = {
                "checkpoint_sha256": checkpoint_sha,
                "eval_manifest_sha256": eval_manifest_sha,
                "dev_manifest_sha256": dev_manifest_sha,
                "data_key_sha256": hashlib.sha256(json.dumps(data_key, sort_keys=True).encode()).hexdigest(),
                "code_sha256": code_sha,
                "communication_core_sha256": core_sha,
                "protocol_sha256": protocol_sha,
                "seed": seed,
                "arm": arm,
                "condition": name,
                "communication_policy": policy,
                "communication_fraction": fraction,
                "mask_rep": rep,
                "row_indices": eval_indices,
                "K": core.K,
                "max_passes": args.max_passes,
                "threshold": tau,
            }
            branch_histories = {}
            for method in ("direct", "decimation"):
                history = core.run_branch(
                    out_dir=condition_dir,
                    provenance_base=provenance_base,
                    method=method,
                    initial_inputs=inputs,
                    labels=labels,
                    common=common,
                    predictor=predictors[policy],
                    params=params,
                    task=task,
                    policy=policy,
                    fraction=fraction,
                    mask_seed_base=mask_seed,
                    tau=tau,
                    max_passes=args.max_passes,
                )
                branch_histories[method] = history
            direct_history = branch_histories["direct"]
            decimation_history = branch_histories["decimation"]
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
                        27 * len(inputs) * core.K * (1 + 2 * cg_iters) * 4
                    )
                    cg_result_bytes = 0 if policy == "none" else (
                        len(inputs) * core.K * (1 + 2 * cg_iters) * 4
                    )
                    scored = item["predictions_scored"].astype(np.uint8)
                    metrics, common_acc = core.result_metrics(scored, inputs, labels, common)
                    agreement = item["agreement_map"].astype(bool).reshape(len(inputs), 81)
                    agreement_score = (agreement * common).sum(1) / np.maximum(common.sum(1), 1)
                    for index, meta in enumerate(eval_rows):
                        pass_rows.append({
                            "family_id": int(meta["family_id"]),
                            "seed": seed, "arm": arm, "condition": name, "mask_rep": rep,
                            "method": method, "pass": pass_index + 1,
                            "exact": int(metrics["exact"][index]),
                            "valid": int(metrics["valid"][index]),
                            "hidden_accuracy": float(metrics["hidden_accuracy"][index]),
                            "common_hidden_accuracy": float(common_acc[index]),
                            "agent_agreement_common": float(agreement_score[index]),
                            "cumulative_commits": int(item["cumulative_commits"][index]),
                            "cumulative_wrong_commits": int(item["cumulative_wrong"][index]),
                            "passes_used": int(item["passes_used"][index]),
                            "stop_reason": core.STOP_NAMES[int(item["stop_reason"][index])],
                            "active_edge_slots_batch": active_slots,
                            "edge_vector_payload_bytes_batch": vector_bytes,
                            "initial_bootstrap_payload_bytes_batch": bootstrap_bytes,
                            "cg_global_partial_payload_bytes_batch": cg_partial_bytes,
                            "cg_reduced_scalar_payload_bytes_batch": cg_result_bytes,
                            "final_edge_disagreement_mean": float(item["edge_disagreement"][0]),
                            "final_xz_gap_mean": float(item["xz_gap"][0]),
                            "pass_seconds_including_compile": float(item["pass_seconds"][0]),
                        })
            for index, meta in enumerate(eval_rows):
                matched_pass = max(1, min(int(final_decimation["passes_used"][index]), args.max_passes))
                direct_pred = direct_history[matched_pass - 1]["predictions_scored"][index]
                decim_pred = final_decimation["predictions_scored"][index]
                direct_metric = core.horizon.grid_metrics(inputs[index:index+1], labels[index:index+1], direct_pred[None])
                decim_metric = core.horizon.grid_metrics(inputs[index:index+1], labels[index:index+1], decim_pred[None])
                direct_common = float(((direct_pred.reshape(81) == labels[index].reshape(81)) & common[index]).sum() / common[index].sum())
                decim_common = float(((decim_pred.reshape(81) == labels[index].reshape(81)) & common[index]).sum() / common[index].sum())
                paired_rows.append({
                    "family_id": int(meta["family_id"]), "seed": seed, "arm": arm,
                    "condition": name, "mask_rep": rep, "matched_passes": matched_pass,
                    "direct_exact_matched_budget": int(direct_metric["exact"][0]),
                    "decimation_exact": int(decim_metric["exact"][0]),
                    "decimation_gain_exact": int(decim_metric["exact"][0]) - int(direct_metric["exact"][0]),
                    "direct_common_hidden_accuracy": direct_common,
                    "decimation_common_hidden_accuracy": decim_common,
                    "gain_common_hidden_accuracy": decim_common - direct_common,
                    "decimation_commits": int(final_decimation["cumulative_commits"][index]),
                    "decimation_wrong_commits": int(final_decimation["cumulative_wrong"][index]),
                    "decimation_stop_reason": core.STOP_NAMES[int(final_decimation["stop_reason"][index])],
                })
            completed_conditions += 1
            atomic_json(out_dir / "progress.json", {
                "status": "running",
                "completed_seed_condition_blocks": completed_conditions,
                "total_seed_condition_blocks": total_conditions,
                "last_block": f"seed_{seed}/{arm}/{name}",
                "completed_pass_files": len(list(out_dir.glob("s*/*/*/pass_*.npz"))),
            })
            print(json.dumps({
                "condition_complete": f"seed_{seed}/{arm}/{name}",
                "completed_blocks": completed_conditions,
                "total_blocks": total_conditions,
                "mean_commits": float(final_decimation["cumulative_commits"].mean()),
            }), flush=True)

    core.write_csv(out_dir / "per_family_pass_results.csv", pass_rows)
    core.write_csv(out_dir / "paired_budget_results.csv", paired_rows)
    summary_groups = {}
    for row in paired_rows:
        key = (row["seed"], row["arm"], row["condition"], row["mask_rep"])
        summary_groups.setdefault(key, []).append(row)
    summary_rows = []
    for key, group in sorted(summary_groups.items()):
        summary_rows.append({
            "seed": key[0], "arm": key[1], "condition": key[2], "mask_rep": key[3],
            "families": len(group),
            "direct_exact_matched_budget": float(np.mean([row["direct_exact_matched_budget"] for row in group])),
            "decimation_exact": float(np.mean([row["decimation_exact"] for row in group])),
            "decimation_gain_exact": float(np.mean([row["decimation_gain_exact"] for row in group])),
            "direct_common_hidden_accuracy": float(np.mean([row["direct_common_hidden_accuracy"] for row in group])),
            "decimation_common_hidden_accuracy": float(np.mean([row["decimation_common_hidden_accuracy"] for row in group])),
            "gain_common_hidden_accuracy": float(np.mean([row["gain_common_hidden_accuracy"] for row in group])),
            "mean_commits": float(np.mean([row["decimation_commits"] for row in group])),
            "mean_wrong_commits": float(np.mean([row["decimation_wrong_commits"] for row in group])),
        })
    core.write_csv(out_dir / "summary_by_seed_condition.csv", summary_rows)

    interaction_rows = []
    lookup = {
        (row["seed"], row["arm"], row["family_id"], row["condition"], row["mask_rep"]): row
        for row in paired_rows
    }
    for row in paired_rows:
        if row["condition"] == "full_100":
            continue
        full = lookup[(row["seed"], row["arm"], row["family_id"], "full_100", 0)]
        interaction_rows.append({
            "family_id": row["family_id"], "seed": row["seed"], "arm": row["arm"],
            "reduced_condition": row["condition"], "mask_rep": row["mask_rep"],
            "gain_full": full["decimation_gain_exact"],
            "gain_reduced": row["decimation_gain_exact"],
            "interaction_full_minus_reduced": full["decimation_gain_exact"] - row["decimation_gain_exact"],
            "gain_common_full": full["gain_common_hidden_accuracy"],
            "gain_common_reduced": row["gain_common_hidden_accuracy"],
            "interaction_common_full_minus_reduced": full["gain_common_hidden_accuracy"] - row["gain_common_hidden_accuracy"],
        })
    core.write_csv(out_dir / "interaction_by_family.csv", interaction_rows)
    atomic_json(out_dir / "summary.json", {
        "status": "complete",
        "run_config_sha256": config_hash,
        "run_dir": str(out_dir.relative_to(ROOT)),
        "family_count": len(eval_ids),
        "trained_seeds": list(seeds),
        "arms": list(arms),
        "condition_count": len(conditions),
        "paired_budget_rows": len(paired_rows),
        "pass_result_rows": len(pass_rows),
        "interaction_rows": len(interaction_rows),
        "calibration_thresholds": calibrations,
        "backend": core.jax.default_backend(),
        "resumable_pass_sidecars": len(list(out_dir.rglob("pass_*.json"))),
        "scope_note": "secondary functional confirmation conditional on existing checkpoints; shared commit broadcasts remain active in the no-ADMM-message arm",
    })
    atomic_json(out_dir / "progress.json", {
        "status": "complete",
        "completed_seed_condition_blocks": completed_conditions,
        "total_seed_condition_blocks": total_conditions,
        "summary_sha256": sha256(out_dir / "summary.json"),
    })
    print(json.dumps({"status": "complete", "run_dir": str(out_dir.relative_to(ROOT)),
                      "blocks": completed_conditions, "rows": len(paired_rows)}, indent=2), flush=True)


if __name__ == "__main__":
    main()
