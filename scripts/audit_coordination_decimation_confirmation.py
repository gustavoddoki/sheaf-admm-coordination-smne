"""Independent integrity and completeness audit for the P5 functional run."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
EXPECTED_SEEDS = [123, 456, 789, 1011, 1213, 20260927]
EXPECTED_ARMS = ["internal", "terminal"]
MASK_SEEDS = [391, 773, 1237]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def digest_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def data_selection(data_dir: Path, family_ids: list[int]) -> tuple[list[int], np.ndarray, np.ndarray]:
    manifest = json.loads((data_dir / "paired_clue_manifest.json").read_text(encoding="utf-8"))
    rows = {
        int(row["family_id"]): row
        for row in manifest["families"]
        if row["policy"] == "concentrated" and int(row["clues"]) == 40
        and int(row["family_id"]) in family_ids
    }
    if sorted(rows) != sorted(family_ids):
        raise AssertionError(f"Missing 40-clue concentrated families in {data_dir}")
    selected = [rows[family] for family in family_ids]
    indices = [int(row["row_index"]) for row in selected]
    inputs = np.asarray(
        np.load(data_dir / "test_hard/all__inputs.npy", mmap_mode="r")[indices], dtype=np.uint8
    ).reshape(-1, 9, 9)
    labels = np.asarray(
        np.load(data_dir / "test_hard/all__labels.npy", mmap_mode="r")[indices], dtype=np.uint8
    ).reshape(-1, 9, 9)
    return indices, inputs, labels


def valid_partial_grid(grid: np.ndarray) -> bool:
    units = list(grid) + list(grid.T) + [
        grid[r:r + 3, c:c + 3].ravel()
        for r in (0, 3, 6) for c in (0, 3, 6)
    ]
    return all(len(values := unit[unit > 0]) == len(set(values.tolist())) for unit in units)


def audit(run_dir: Path, completed_prefix: bool = False) -> dict:
    run_dir = run_dir.resolve()
    config_path = run_dir / "run_config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    saved_config_hash = config["config_sha256"]
    unhashed_config = {key: value for key, value in config.items() if key not in {"config_sha256", "run_dir"}}
    config_hash = digest_bytes(json.dumps(unhashed_config, sort_keys=True, default=list).encode())
    if config_hash != saved_config_hash:
        raise AssertionError("Frozen configuration hash mismatch")
    if config["phase"] != "P5_secondary_communication_decimation":
        raise AssertionError("Unexpected run phase")
    if config["seeds"] != EXPECTED_SEEDS or config["arms"] != EXPECTED_ARMS:
        raise AssertionError("Seed/checkpoint arms differ from the frozen protocol")
    if config["family_ids"] != list(range(64)) or config["families"] != 64:
        raise AssertionError("Confirmation must contain all 64 frozen families")
    if config["max_passes"] != 8 or config["K_per_pass"] != 20:
        raise AssertionError("Unexpected horizon or pass cap")
    if config["clues"] != 40 or config["distribution"] != "concentrated":
        raise AssertionError("Unexpected evaluation regime")
    if config["labels_provided_to_model"] is not False:
        raise AssertionError("Labels are marked as model inputs")

    conditions = config["conditions"]
    condition_map = {item["name"]: item for item in conditions}
    expected_names = ["full_100"] + [
        f"random_{fraction}_r{rep}"
        for fraction in (75, 50) for rep in range(3)
    ] + ["none_0"]
    if list(condition_map) != expected_names:
        raise AssertionError(f"Unexpected condition set/order: {list(condition_map)}")
    if config["methods"] != ["direct", "decimation"]:
        raise AssertionError("Unexpected inference methods")

    progress_path = run_dir / "progress.json"
    progress = json.loads(progress_path.read_text(encoding="utf-8"))
    partial_run = progress.get("status") != "complete"
    if partial_run and not completed_prefix:
        raise AssertionError("Run is incomplete; use --completed-prefix to audit only closed blocks")
    all_condition_blocks = [
        (seed, arm, condition["name"])
        for seed in EXPECTED_SEEDS for arm in EXPECTED_ARMS for condition in conditions
    ]
    if partial_run:
        completed_block_count = int(progress.get("completed_seed_condition_blocks", 0))
        if not 0 < completed_block_count < len(all_condition_blocks):
            raise AssertionError("Unexpected completed-block count in progress file")
        completed_conditions = set(all_condition_blocks[:completed_block_count])
        last_block = progress.get("last_block", "").split("/")
        parsed_last_block = (
            int(last_block[0].removeprefix("seed_")), last_block[1], last_block[2]
        ) if len(last_block) == 3 else None
        if parsed_last_block != all_condition_blocks[completed_block_count - 1]:
            raise AssertionError("Progress file does not identify a valid completed prefix")
    else:
        completed_block_count = len(all_condition_blocks)
        completed_conditions = set(all_condition_blocks)

    eval_dir = ROOT / "datasets/sudoku_coordination_confirmation_20261007"
    dev_dir = ROOT / "datasets/sudoku_coordination_pilot_20261007"
    eval_manifest = eval_dir / "paired_clue_manifest.json"
    dev_manifest = dev_dir / "paired_clue_manifest.json"
    if sha256(eval_manifest) != config["data"]["eval_manifest_sha256"]:
        raise AssertionError("Evaluation manifest hash mismatch")
    if sha256(dev_manifest) != config["data"]["dev_manifest_sha256"]:
        raise AssertionError("Development manifest hash mismatch")
    eval_indices, eval_inputs, eval_labels = data_selection(eval_dir, config["family_ids"])
    dev_indices, dev_inputs, dev_labels = data_selection(dev_dir, list(range(32)))
    data = config["data"]
    if eval_indices != data["eval_row_indices"] or dev_indices != data["dev_row_indices"]:
        raise AssertionError("Dataset row selection differs from frozen configuration")
    for key, value in (
        ("eval_input_sha256", digest_bytes(eval_inputs.tobytes())),
        ("eval_label_sha256_for_scoring_only", digest_bytes(eval_labels.tobytes())),
        ("dev_input_sha256", digest_bytes(dev_inputs.tobytes())),
        ("dev_label_sha256_for_calibration_only", digest_bytes(dev_labels.tobytes())),
    ):
        if data[key] != value:
            raise AssertionError(f"Frozen data hash mismatch: {key}")

    runner_path = ROOT / "scripts/run_coordination_decimation_confirmation.py"
    core_path = ROOT / "scripts/run_coordination_communication_decimation.py"
    protocol_path = ROOT / "experiments/sheaf_admm_reasoning/P5_PROTOCOLO_CONFIRMATORIO_2026-10-07.md"
    if sha256(runner_path) != config["code_sha256"]:
        raise AssertionError("Confirmation runner changed after the run was frozen")
    if sha256(core_path) != config["communication_core_sha256"]:
        raise AssertionError("Communication core changed after the run was frozen")
    if sha256(protocol_path) != config["protocol_sha256"]:
        raise AssertionError("Frozen P5 protocol hash mismatch")

    holdout_audit_path = ROOT / "outputs/coordination_pilot_20261007/p5_confirmation/holdout_audit.json"
    holdout_audit = json.loads(holdout_audit_path.read_text(encoding="utf-8"))
    if holdout_audit.get("status") != "PASS":
        raise AssertionError("The independent holdout audit did not pass")

    calibration_path = run_dir / "calibration.json"
    calibration = json.loads(calibration_path.read_text(encoding="utf-8"))
    if calibration.get("config_sha256") != saved_config_hash:
        raise AssertionError("Calibration belongs to a different configuration")
    thresholds = calibration["thresholds"]
    checkpoint_checks = 0
    for seed in EXPECTED_SEEDS:
        for arm in EXPECTED_ARMS:
            key = f"seed_{seed}/{arm}"
            entry = thresholds.get(key)
            if not entry or entry.get("status") != "calibrated":
                raise AssertionError(f"Missing calibration for {key}")
            if entry.get("calibration_families") != 32 or entry.get("target_precision") != 0.995:
                raise AssertionError(f"Calibration split/target mismatch for {key}")
            if entry.get("dataset_manifest_sha256") != data["dev_manifest_sha256"]:
                raise AssertionError(f"Calibration dataset mismatch for {key}")
            if entry.get("code_sha256") != config["code_sha256"]:
                raise AssertionError(f"Calibration runner hash mismatch for {key}")
            if entry.get("clues") != 40 or entry.get("distribution") != "concentrated" or entry.get("K") != 20:
                raise AssertionError(f"Calibration regime mismatch for {key}")
            if not entry.get("input_labels_not_provided_to_model"):
                raise AssertionError(f"Calibration labels were provided as model inputs for {key}")
            if float(entry["precision"]) < 0.995:
                raise AssertionError(f"Calibration target not met for {key}")
            checkpoint_path = ROOT / Path(entry["checkpoint"].replace("\\", "/"))
            if not checkpoint_path.is_file() or sha256(checkpoint_path) != entry["checkpoint_sha256"]:
                raise AssertionError(f"Checkpoint hash mismatch for {key}")
            checkpoint_checks += 1

    config_data_hash = digest_bytes(json.dumps(data, sort_keys=True).encode())
    archives_all = sorted(run_dir.rglob("pass_*.npz"))
    archives = []
    for archive in archives_all:
        parts = archive.relative_to(run_dir).parts
        if len(parts) != 4:
            raise AssertionError(f"Unexpected artifact path: {archive}")
        seed_text, arm_initial = parts[0][1:].split("_", 1)
        seed = int(seed_text)
        arm = next((value for value in EXPECTED_ARMS if value.startswith(arm_initial)), None)
        if arm is None:
            raise AssertionError(f"Unexpected artifact arm: {archive}")
        if (seed, arm, parts[1]) in completed_conditions:
            archives.append(archive)
    sidecar_files = [
        sidecar for sidecar in run_dir.rglob("pass_*.json")
        if len(sidecar.relative_to(run_dir).parts) == 4
        and (int(sidecar.relative_to(run_dir).parts[0][1:].split("_", 1)[0]),
             next((value for value in EXPECTED_ARMS if value.startswith(
                 sidecar.relative_to(run_dir).parts[0][1:].split("_", 1)[1]
             )), ""),
             sidecar.relative_to(run_dir).parts[1]) in completed_conditions
    ]
    expected_count = completed_block_count * 2 * 8
    observed: dict[tuple[int, str, str, str], list[int]] = {}
    digest_checks = 0
    no_message_passes = 0
    pass_arrays: dict[tuple[int, str, str, str, int], dict[str, np.ndarray]] = {}
    required_arrays = {
        "predictions", "predictions_scored", "grid_before", "grid_after", "last_predictions",
        "stopped", "stop_reason", "passes_used", "cumulative_commits", "cumulative_wrong",
        "exact", "valid", "hidden_accuracy", "common_accuracy", "active_edge_slots_by_iteration",
        "confidence", "agreement_map", "mask_seed", "pass_seconds",
    }
    for archive in archives:
        parts = archive.relative_to(run_dir).parts
        if len(parts) != 4:
            raise AssertionError(f"Unexpected artifact path: {archive}")
        seed_arm, condition_name, method, filename = parts
        seed_text, arm_initial = seed_arm[1:].split("_", 1)
        seed = int(seed_text)
        arm = next((value for value in EXPECTED_ARMS if value.startswith(arm_initial)), None)
        if arm is None or condition_name not in condition_map or method not in {"direct", "decimation"}:
            raise AssertionError(f"Unexpected artifact branch: {archive}")
        pass_index = int(Path(filename).stem.split("_")[-1])
        if pass_index not in range(1, 9):
            raise AssertionError(f"Unexpected pass index: {archive}")
        condition = condition_map[condition_name]
        mask_rep = int(condition["rep"])
        metadata_path = archive.with_suffix(".json")
        if not metadata_path.is_file():
            raise AssertionError(f"Missing sidecar for {archive}")
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if sha256(archive) != metadata.get("output_sha256"):
            raise AssertionError(f"Artifact checksum mismatch: {archive}")
        digest_checks += 1
        expected_fields = {
            "checkpoint_sha256": thresholds[f"seed_{seed}/{arm}"]["checkpoint_sha256"],
            "eval_manifest_sha256": data["eval_manifest_sha256"],
            "dev_manifest_sha256": data["dev_manifest_sha256"],
            "data_key_sha256": config_data_hash,
            "code_sha256": config["code_sha256"],
            "communication_core_sha256": config["communication_core_sha256"],
            "protocol_sha256": config["protocol_sha256"],
            "seed": seed, "arm": arm, "condition": condition_name,
            "communication_policy": condition["policy"],
            "communication_fraction": condition["fraction"],
            "mask_rep": mask_rep, "row_indices": eval_indices,
            "K": 20, "max_passes": 8, "method": method, "pass_index": pass_index,
            "threshold": thresholds[f"seed_{seed}/{arm}"]["threshold"],
            "mask_seed": (MASK_SEEDS[mask_rep] + pass_index * 104729) % 2_000_000_000,
        }
        for field, expected in expected_fields.items():
            if metadata.get(field) != expected:
                raise AssertionError(f"{field} mismatch in {metadata_path}")

        with np.load(archive, allow_pickle=False) as stored:
            missing = required_arrays.difference(stored.files)
            if missing:
                raise AssertionError(f"Missing arrays {sorted(missing)} in {archive}")
            arrays = {name: stored[name] for name in required_arrays}
        if arrays["predictions"].shape != (64, 9, 9) or arrays["grid_after"].shape != (64, 9, 9):
            raise AssertionError(f"Unexpected grid shape in {archive}")
        for name in ("cumulative_commits", "cumulative_wrong", "passes_used", "exact", "valid"):
            if arrays[name].shape != (64,):
                raise AssertionError(f"Unexpected family-vector shape for {name} in {archive}")
        if np.any(arrays["cumulative_wrong"] > arrays["cumulative_commits"]):
            raise AssertionError(f"Wrong commits exceed total commits in {archive}")
        if not np.array_equal(arrays["mask_seed"], np.asarray([expected_fields["mask_seed"]])):
            raise AssertionError(f"Mask seed array/sidecar mismatch in {archive}")
        if method == "direct" and (
            np.any(arrays["cumulative_commits"] != 0)
            or not np.array_equal(arrays["grid_after"], arrays["grid_before"])
        ):
            raise AssertionError(f"Direct inference unexpectedly committed cells in {archive}")
        if condition_name == "none_0":
            if np.any(arrays["active_edge_slots_by_iteration"] != 0):
                raise AssertionError(f"No-message condition used ADMM vector edges in {archive}")
            no_message_passes += 1
        grid_after = arrays["grid_after"]
        if np.any((eval_inputs > 0) & (grid_after != eval_inputs)):
            raise AssertionError(f"A clue changed in {archive}")
        if any(not valid_partial_grid(grid) for grid in grid_after):
            raise AssertionError(f"Committed grid violates Sudoku unit constraints in {archive}")
        added_cells = (grid_after != eval_inputs).sum(axis=(1, 2))
        if not np.array_equal(added_cells, arrays["cumulative_commits"].astype(added_cells.dtype)):
            raise AssertionError(f"Commit count does not match grid in {archive}")
        wrong_cells = ((grid_after != eval_inputs) & (grid_after != eval_labels)).sum(axis=(1, 2))
        if not np.array_equal(wrong_cells, arrays["cumulative_wrong"].astype(wrong_cells.dtype)):
            raise AssertionError(f"Wrong-commit count does not match labels in {archive}")

        branch = (seed, arm, condition_name, method)
        observed.setdefault(branch, []).append(pass_index)
        pass_arrays[(seed, arm, condition_name, method, pass_index)] = {
            name: arrays[name].copy()
            for name in ("grid_before", "grid_after", "cumulative_commits", "cumulative_wrong")
        }

    expected_sidecars = {archive.with_suffix(".json") for archive in archives}
    if set(sidecar_files) != expected_sidecars:
        raise AssertionError("Orphan, missing, or unexpected pass sidecars exist")
    if not partial_run and list(run_dir.rglob("*.tmp")):
        raise AssertionError("Temporary/incomplete files remain in the run directory")
    if len(archives) != expected_count:
        raise AssertionError(f"Expected {expected_count} pass artifacts, found {len(archives)}")
    expected_branches = {
        (seed, arm, condition["name"], method)
        for seed, arm, condition_name in completed_conditions
        for condition in conditions if condition["name"] == condition_name
        for method in ("direct", "decimation")
    }
    if set(observed) != expected_branches:
        raise AssertionError("The observed branch set is incomplete or contains extras")
    for branch, indices in observed.items():
        if sorted(indices) != list(range(1, 9)):
            raise AssertionError(f"Missing/duplicate passes in {branch}: {sorted(indices)}")
        seed, arm, condition_name, method = branch
        previous = None
        for pass_index in range(1, 9):
            arrays = pass_arrays[(seed, arm, condition_name, method, pass_index)]
            if previous is not None:
                if method == "decimation" and not np.array_equal(arrays["grid_before"], previous["grid_after"]):
                    raise AssertionError(f"Decimation pass boundary mismatch in {branch}, pass {pass_index}")
                if np.any(arrays["cumulative_commits"] < previous["cumulative_commits"]):
                    raise AssertionError(f"Commit count decreases in {branch}, pass {pass_index}")
                if np.any(arrays["cumulative_wrong"] < previous["cumulative_wrong"]):
                    raise AssertionError(f"Wrong-commit count decreases in {branch}, pass {pass_index}")
            previous = arrays

    progress = json.loads((run_dir / "progress.json").read_text(encoding="utf-8"))
    if partial_run:
        expected_no_message = sum(1 for _seed, _arm, condition in completed_conditions if condition == "none_0") * 2 * 8
        if no_message_passes != expected_no_message:
            raise AssertionError("No-message pass count is inconsistent in completed prefix")
        csv_counts = None
        report_status = "PASS_COMPLETED_PREFIX"
    else:
        if progress.get("completed_seed_condition_blocks") != 96:
            raise AssertionError("Run progress does not mark all 96 blocks complete")
        summary_path = run_dir / "summary.json"
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        if summary.get("status") != "complete" or summary.get("run_config_sha256") != saved_config_hash:
            raise AssertionError("Run summary is missing or belongs to another configuration")
        if progress.get("summary_sha256") != sha256(summary_path):
            raise AssertionError("Final summary checksum mismatch")

        paired = read_csv(run_dir / "paired_budget_results.csv")
        pass_rows = read_csv(run_dir / "per_family_pass_results.csv")
        summary_rows = read_csv(run_dir / "summary_by_seed_condition.csv")
        interaction_rows = read_csv(run_dir / "interaction_by_family.csv")
        expected_paired = 64 * len(EXPECTED_SEEDS) * len(EXPECTED_ARMS) * len(conditions)
        expected_pass_rows = expected_paired * 2 * 8
        expected_interactions = 64 * len(EXPECTED_SEEDS) * len(EXPECTED_ARMS) * (len(conditions) - 1)
        if (len(paired), len(pass_rows), len(summary_rows), len(interaction_rows)) != (
            expected_paired, expected_pass_rows, 96, expected_interactions
        ):
            raise AssertionError("Summary CSV row counts do not match the frozen factorial")
        paired_keys = [(r["family_id"], r["seed"], r["arm"], r["condition"], r["mask_rep"]) for r in paired]
        pass_keys = [(r["family_id"], r["seed"], r["arm"], r["condition"], r["mask_rep"], r["method"], r["pass"]) for r in pass_rows]
        if len(set(paired_keys)) != expected_paired or len(set(pass_keys)) != expected_pass_rows:
            raise AssertionError("Duplicate rows in paired or pass-level CSV")
        for row in pass_rows:
            if row["condition"] == "none_0":
                for field in (
                    "active_edge_slots_batch", "edge_vector_payload_bytes_batch",
                    "initial_bootstrap_payload_bytes_batch", "cg_global_partial_payload_bytes_batch",
                    "cg_reduced_scalar_payload_bytes_batch",
                ):
                    if float(row[field]) != 0:
                        raise AssertionError(f"No-message payload is nonzero in {field}")
        csv_counts = {
            "paired_budget_rows": len(paired),
            "pass_level_rows": len(pass_rows),
            "seed_condition_summary_rows": len(summary_rows),
            "interaction_rows": len(interaction_rows),
        }
        report_status = "PASS"
    report = {
        "status": report_status,
        "run_dir": str(run_dir),
        "config_sha256": saved_config_hash,
        "runner_sha256": config["code_sha256"],
        "communication_core_sha256": config["communication_core_sha256"],
        "protocol_sha256": config["protocol_sha256"],
        "evaluation_manifest_sha256": data["eval_manifest_sha256"],
        "families": 64,
        "seeds": EXPECTED_SEEDS,
        "arms": EXPECTED_ARMS,
        "conditions": expected_names,
        "expected_and_verified_pass_artifacts": len(archives),
        "completed_condition_blocks": completed_block_count,
        "verified_pass_sidecar_hashes": digest_checks,
        "verified_checkpoint_hashes": checkpoint_checks,
        "verified_no_message_passes": no_message_passes,
        "csv_counts": csv_counts,
        "scope_note": "No-ADMM-message arms still share discrete commits through the board.",
    }
    report_path = run_dir / ("partial_audit.json" if partial_run else "audit.json")
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--completed-prefix", action="store_true")
    args = parser.parse_args()
    print(json.dumps(audit(args.run_dir, completed_prefix=args.completed_prefix), indent=2))


if __name__ == "__main__":
    main()
