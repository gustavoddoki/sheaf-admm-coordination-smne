"""Independent pass, pairing, hash, and routing audit for P7/P8."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import run_coordination_channel_ablation as runner  # noqa: E402
from scripts import run_coordination_communication_decimation as core  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--resume-reference", type=Path)
    args = parser.parse_args()
    run_dir = args.run_dir.resolve()
    config = json.loads((run_dir / "run_config.json").read_text(encoding="utf-8"))
    progress = json.loads((run_dir / "progress.json").read_text(encoding="utf-8"))
    if progress.get("status") != "complete":
        raise RuntimeError(f"run is not complete: {progress.get('status')}")
    family_ids = [int(value) for value in config["family_ids"]]
    seeds = [int(value) for value in config["seeds"]]
    arms = list(config["arms"])
    specs = config["conditions"]
    expected_keys = {
        f"{seed}/{arm}/{spec['condition']}/{spec['method']}"
        for seed in seeds for arm in arms for spec in specs
    }
    completed_list = progress.get("completed_trajectories", [])
    if len(completed_list) != len(expected_keys) or set(completed_list) != expected_keys:
        raise RuntimeError("progress does not cover the complete frozen design")
    config_sha = hashlib.sha256(json.dumps(config, sort_keys=True, default=list).encode()).hexdigest()
    cell_ids = np.asarray(runner.build_sudoku_cell_indices(9), dtype=np.int32)
    owner_lookup = runner.make_owner_lookup(cell_ids)
    manifest_sha = runner.sha256(runner.FINAL_DATA / "paired_clue_manifest.json")
    if manifest_sha != config["dataset_manifest_sha256"]:
        raise RuntimeError("dataset manifest changed after execution")
    calibration_sha = runner.sha256(
        runner.OUT / "calibration" / f"owner_thresholds_{runner.calibration_logic_sha256()[:12]}.json"
    )
    if calibration_sha != config["threshold_calibration_sha256"]:
        raise RuntimeError("threshold calibration changed after execution")
    control = json.loads((run_dir / "technical_control.json").read_text(encoding="utf-8"))
    if control.get("status") != "PASS" or control.get("max_remote_logit_delta_no_admm", 1.0) > 1e-6:
        raise RuntimeError("technical channel-isolation control did not pass")

    pass_count = sidecar_count = committed_events = routed_events = 0
    expected_rows = []
    for seed in seeds:
        for arm in arms:
            checkpoint = core.checkpoint_path(arm, seed)
            checkpoint_sha = runner.sha256(checkpoint)
            model_stub = (checkpoint, checkpoint_sha)
            for spec in specs:
                method = str(spec["method"])
                condition = str(spec["condition"])
                if config["phase"] in {"p8", "p8_pilot"}:
                    clues, distribution = int(spec["clues"]), str(spec["distribution"])
                else:
                    clues, distribution = 40, "concentrated"
                _manifest, _rows, _idx, inputs, labels, common = runner.select_rows(
                    runner.FINAL_DATA, family_ids, clues, distribution
                )
                initial = runner.initial_agent_grids(inputs, cell_ids)
                target_mask = common[:, cell_ids]
                trajectory = run_dir / f"seed_{seed}" / arm / condition / method
                previous_grids = initial.copy()
                previous_frozen = np.zeros((len(family_ids), 27, 9), dtype=bool)
                previous_logp = np.full((len(family_ids), 27, 9), np.nan, dtype=np.float32)
                shadow = inputs.copy()
                cumulative_commits = np.zeros(len(family_ids), dtype=np.uint16)
                cumulative_correct = np.zeros(len(family_ids), dtype=np.uint16)
                cumulative_wrong = np.zeros(len(family_ids), dtype=np.uint16)
                cumulative_conflicts = np.zeros(len(family_ids), dtype=np.uint16)
                cumulative_updates = np.zeros(len(family_ids), dtype=np.uint16)
                last = None
                provenance = runner.block_provenance(
                    config, config_sha, model_stub, seed, arm, spec, method, family_ids
                )
                for pass_index in range(1, int(config["max_passes"]) + 1):
                    path = trajectory / f"pass_{pass_index:02d}.npz"
                    expected = dict(
                        provenance, pass_index=pass_index,
                        mask_seed=int((runner.MASK_SEEDS[int(spec.get("mask_rep", 0))]
                                       + pass_index * 104729) % 2_000_000_000),
                    )
                    arrays = runner.load_verified_npz(path, expected)
                    if arrays is None:
                        raise RuntimeError(f"missing pass artifact: {path}")
                    sidecar_count += 1
                    pass_count += 1
                    grids = arrays["agent_grids"].astype(np.uint8)
                    proposals = arrays["proposals"].astype(np.uint8)
                    recipients = arrays["recipients"].astype(np.uint8)
                    if grids.shape != (len(family_ids), 27, 9):
                        raise RuntimeError(f"agent state shape mismatch: {path}")
                    if arrays["target_mask"].astype(bool).shape != target_mask.shape or not np.array_equal(
                        arrays["target_mask"].astype(bool), target_mask
                    ):
                        raise RuntimeError(f"evaluation mask mismatch: {path}")
                    if not np.all(np.isfinite(arrays["score_logp"][target_mask])):
                        raise RuntimeError(f"nonfinite or missing pre-commit score: {path}")
                    frozen = arrays["frozen"].astype(bool)
                    if np.any(previous_frozen & (arrays["score_logp"] != previous_logp)):
                        raise RuntimeError(f"frozen score changed after its commit: {path}")

                    expected_grids = previous_grids.copy()
                    expected_recipients = np.full_like(recipients, 255)
                    expected_pass_conflict = np.zeros_like(arrays["pass_conflicts"], dtype=np.uint8)
                    expected_shadow = shadow.copy()
                    if method == "direct":
                        if np.any(proposals) or np.any(recipients != 255):
                            raise RuntimeError(f"direct inference emitted a commit: {path}")
                    else:
                        for b in range(len(family_ids)):
                            for cell in np.flatnonzero(proposals[b]).tolist():
                                digit = int(proposals[b, cell])
                                if pass_index == int(config["max_passes"]):
                                    raise RuntimeError("the final pass emitted a post-measurement commit")
                                owner = int(cell // 9)
                                views = owner_lookup[int(cell)]
                                if (owner, int(cell % 9)) not in views:
                                    raise RuntimeError("proposal origin is not the row owner")
                                if not core.legal(expected_shadow[b].reshape(9, 9), cell // 9, cell % 9, digit):
                                    expected_pass_conflict[b, cell] = 1
                                expected_shadow[b, cell // 9, cell % 9] = digit
                                expected_views = views if spec["visibility"] == "global" else [
                                    (owner, int(cell % 9))
                                ]
                                for node, slot in expected_views:
                                    expected_grids[b, node, slot] = digit
                                    expected_recipients[b, cell, views.index((node, slot))] = node
                        if not np.array_equal(expected_recipients, recipients):
                            raise RuntimeError(f"commit destinations differ from condition: {path}")
                        if not np.array_equal(expected_pass_conflict, arrays["pass_conflicts"]):
                            raise RuntimeError(f"shadow conflict counts disagree: {path}")
                        if not np.array_equal(expected_shadow, arrays["shadow_grid"]):
                            raise RuntimeError(f"shadow board update mismatch: {path}")
                    if not np.array_equal(expected_grids, grids):
                        raise RuntimeError(f"agent view changed outside authorized commit routing: {path}")

                    event_mask = proposals > 0
                    event_count = event_mask.sum(axis=1).astype(np.uint16)
                    event_correct = (event_mask & (arrays["proposal_correct"] > 0)).sum(axis=1).astype(np.uint16)
                    event_wrong = (event_mask & (arrays["proposal_correct"] == 0)).sum(axis=1).astype(np.uint16)
                    cumulative_commits += event_count
                    cumulative_correct += event_correct
                    cumulative_wrong += event_wrong
                    cumulative_conflicts += expected_pass_conflict.sum(axis=1).astype(np.uint16)
                    cumulative_updates += (recipients != 255).sum(axis=(1, 2)).astype(np.uint16)
                    for b in range(len(family_ids)):
                        if int(arrays["total_commits"][b]) != int(cumulative_commits[b]):
                            raise RuntimeError(f"cumulative commit count mismatch: {path}")
                        if int(arrays["correct_commits"][b]) != int(cumulative_correct[b]):
                            raise RuntimeError(f"correct commit count mismatch: {path}")
                        if int(arrays["wrong_commits"][b]) != int(cumulative_wrong[b]):
                            raise RuntimeError(f"wrong commit count mismatch: {path}")
                        if int(arrays["conflicts"][b]) != int(cumulative_conflicts[b]):
                            raise RuntimeError(f"conflict count mismatch: {path}")
                        if int(arrays["routed_updates"][b]) != int(cumulative_updates[b]):
                            raise RuntimeError(f"routed update count mismatch: {path}")
                    recomputed_logp = arrays["score_logp"].astype(np.float64)
                    recomputed_acc = arrays["score_correct"].astype(np.float64)
                    for b in range(len(family_ids)):
                        if not np.isclose(
                            arrays["primary_logprob_by_family"][b],
                            recomputed_logp[b][target_mask[b]].mean(), atol=1e-6,
                        ):
                            raise RuntimeError(f"family log-probability summary mismatch: {path}")
                        if not np.isclose(
                            arrays["primary_accuracy_by_family"][b],
                            recomputed_acc[b][target_mask[b]].mean(), atol=1e-6,
                        ):
                            raise RuntimeError(f"family accuracy summary mismatch: {path}")
                    previous_grids = grids.copy()
                    previous_frozen = frozen.copy()
                    previous_logp = arrays["score_logp"].astype(np.float32)
                    shadow = arrays["shadow_grid"].astype(np.uint8)
                    last = arrays
                if last is None:
                    raise RuntimeError("trajectory has no pass results")
                committed_events += int(cumulative_commits.sum())
                routed_events += int(cumulative_updates.sum())
                for index, family_id in enumerate(family_ids):
                    expected_rows.append((int(family_id), seed, arm, condition, method))
    csv_path = run_dir / "family_results.csv"
    if not csv_path.is_file():
        raise RuntimeError("final family-level results table is missing")
    import pandas as pd
    table = pd.read_csv(csv_path)
    if len(table) != len(expected_rows):
        raise RuntimeError(f"family result row count mismatch: {len(table)} != {len(expected_rows)}")
    observed = set(zip(table.family_id.astype(int), table.seed.astype(int), table.arm,
                       table.condition, table.method, strict=True))
    if observed != set(expected_rows):
        raise RuntimeError("family result table has missing or duplicate paired records")
    stray_temps = [p for p in run_dir.rglob("*.tmp") if p.is_file()]
    if stray_temps:
        raise RuntimeError(f"temporary files remain under completed run: {stray_temps[:4]}")
    resume_equivalence = None
    if args.resume_reference is not None:
        reference_dir = args.resume_reference.resolve()
        reference_config = json.loads((reference_dir / "run_config.json").read_text(encoding="utf-8"))
        if reference_config != config:
            raise RuntimeError("continuous reference and resumed run do not share an identical configuration")
        if config["phase"] != "pilot":
            raise RuntimeError("the continuity check is reserved for the excluded technical pilot")
        condition = "decim_admm_on_global"
        method = "decimation"
        spec = next(item for item in config["conditions"]
                    if item["condition"] == condition and item["method"] == method)
        seed, arm = 123, "internal"
        checkpoint = core.checkpoint_path(arm, seed)
        model_stub = (checkpoint, runner.sha256(checkpoint))
        provenance = runner.block_provenance(
            config, config_sha, model_stub, seed, arm, spec, method, family_ids
        )
        current_dir = run_dir / f"seed_{seed}" / arm / condition / method
        reference_path = reference_dir / f"seed_{seed}" / arm / condition / method
        compared = []
        for pass_index in range(1, int(config["max_passes"]) + 1):
            mask_seed = int((runner.MASK_SEEDS[0] + pass_index * 104729) % 2_000_000_000)
            expected = dict(provenance, pass_index=pass_index, mask_seed=mask_seed)
            current = runner.load_verified_npz(current_dir / f"pass_{pass_index:02d}.npz", expected)
            reference = runner.load_verified_npz(reference_path / f"pass_{pass_index:02d}.npz", expected)
            mismatches = [
                key for key in current
                if key != "pass_seconds"
                and not np.array_equal(current[key], reference[key], equal_nan=True)
            ]
            if mismatches:
                raise RuntimeError(f"pause/resume diverged from uninterrupted execution at pass {pass_index}: {mismatches}")
            compared.append(pass_index)
        resume_equivalence = {
            "status": "PASS", "trajectory": f"{seed}/{arm}/{condition}/{method}",
            "passes_compared": compared,
            "all_non_timing_arrays_bitwise_equal": True,
            "timing_field_excluded": "pass_seconds",
            "continuous_reference_run_dir": str(reference_dir.relative_to(ROOT)),
        }
    report = {
        "status": "PASS", "phase": config["phase"],
        "families": len(family_ids), "trajectories": len(expected_keys),
        "pass_artifacts": pass_count, "verified_sidecars": sidecar_count,
        "family_result_rows": len(table), "commit_events": committed_events,
        "routed_view_updates": routed_events,
        "technical_channel_control": control,
        "pause_resume_equivalence": resume_equivalence,
        "config_sha256": config_sha, "dataset_manifest_sha256": manifest_sha,
        "audit_scope": "complete pass hashes, exact local/global routing, no unauthorized view updates, frozen pre-commit scores, paired family rows",
    }
    (run_dir / "audit.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
