"""Independent integrity and primary-estimand audit for P5 transmission results."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def bootstrap(values: np.ndarray, draws: int = 20000, seed: int = 20261008) -> dict:
    values = np.asarray(values, dtype=np.float64)
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(values), size=(draws, len(values)))
    samples = values[indices].mean(axis=1)
    return {
        "families": int(len(values)),
        "estimate": float(values.mean()),
        "ci95_low": float(np.quantile(samples, 0.025)),
        "ci95_high": float(np.quantile(samples, 0.975)),
        "bootstrap_draws": draws,
        "bootstrap_seed": seed,
    }


def score_distance_one(
    data: dict[str, np.ndarray], *, equal_agent_weight: bool
) -> tuple[np.ndarray, np.ndarray]:
    base = np.transpose(data["logits_no_clue"], (1, 0, 2, 3)).astype(np.float64)
    clue = np.transpose(data["logits_local_clue"], (1, 0, 2, 3)).astype(np.float64)
    base_logp = base - np.logaddexp.reduce(base, axis=-1, keepdims=True)
    clue_logp = clue - np.logaddexp.reduce(clue, axis=-1, keepdims=True)
    task_inputs = data["task_inputs"]
    task_labels = data["task_labels_for_scoring_only"]
    cell_ids = data["cell_ids"]
    incident = data["incident_agents"].astype(bool)
    distances = data["graph_distances"]
    base_scores = np.full((len(task_inputs),), np.nan, dtype=np.float64)
    clue_scores = np.full((len(task_inputs),), np.nan, dtype=np.float64)
    for task in range(len(task_inputs)):
        base_slots, clue_slots = [], []
        base_agent_means, clue_agent_means = [], []
        for node in range(base.shape[1]):
            if incident[task, node] or int(distances[task, node]) != 1:
                continue
            slots = np.flatnonzero(task_inputs[task, cell_ids[node]] == 0)
            if slots.size == 0:
                continue
            truth = task_labels[task, cell_ids[node, slots]].astype(np.int64)
            agent_base = base_logp[task, node, slots, truth]
            agent_clue = clue_logp[task, node, slots, truth]
            base_slots.extend(agent_base.tolist())
            clue_slots.extend(agent_clue.tolist())
            base_agent_means.append(float(agent_base.mean()))
            clue_agent_means.append(float(agent_clue.mean()))
        if equal_agent_weight and base_agent_means:
            base_scores[task] = float(np.mean(base_agent_means))
            clue_scores[task] = float(np.mean(clue_agent_means))
        elif base_slots:
            base_scores[task] = float(np.mean(base_slots))
            clue_scores[task] = float(np.mean(clue_slots))
    return base_scores, clue_scores


def audit(run_dir: Path, data_dir: Path) -> dict:
    config = json.loads((run_dir / "run_config.json").read_text(encoding="utf-8"))
    summary_path = run_dir / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if summary.get("status") != "complete":
        raise AssertionError("P5 run summary is incomplete")
    if config.get("phase") != "confirm" or config.get("labels_provided_to_model") is not False:
        raise AssertionError("run configuration does not match the frozen holdout protocol")
    if config.get("runner_sha256") != sha256(ROOT / "scripts/run_coordination_information_confirmation.py"):
        raise AssertionError("runner source hash differs from the frozen run configuration")
    if config.get("shared_helper_sha256") != sha256(ROOT / "scripts/run_coordination_local_information.py"):
        raise AssertionError("shared helper hash differs from the frozen run configuration")
    if config.get("protocol_sha256") != sha256(ROOT / "experiments/sheaf_admm_reasoning/P5_PROTOCOLO_CONFIRMATORIO_2026-10-07.md"):
        raise AssertionError("protocol hash differs from the frozen run configuration")

    expected_count = len(config["seeds"]) * len(config["arms"]) * len(config["depths"]) * 2
    archives = sorted(run_dir.glob("s*_*/*/k*.npz"))
    if len(archives) != expected_count or summary.get("verified_archives") != expected_count:
        raise AssertionError(f"expected {expected_count} NPZs; found {len(archives)}")

    dataset_manifest = json.loads((data_dir / "paired_clue_manifest.json").read_text(encoding="utf-8"))
    if config.get("dataset_manifest_sha256") != sha256(data_dir / "paired_clue_manifest.json"):
        raise AssertionError("holdout manifest hash differs from the frozen run configuration")
    rows_by_family = {
        int(row["family_id"]): row
        for row in dataset_manifest["families"]
        if row["policy"] == "concentrated" and int(row["clues"]) == 40
    }
    target_cells_expected = []
    family_ids = list(map(int, config["family_ids"]))
    unique_families = sorted(set(family_ids))
    targets_by_family = {}
    for family_id in unique_families:
        common = np.asarray(rows_by_family[family_id]["common_evaluation_cells"], dtype=np.int32)
        rng = np.random.default_rng(20261008 + 211 * family_id + 19)
        targets_by_family[family_id] = sorted(map(
            int, np.sort(rng.choice(common, size=4, replace=False))
        ))

    keyed_scores_equal_agent: dict[tuple, tuple[float, float]] = {}
    keyed_scores_slot_weighted: dict[tuple, tuple[float, float]] = {}
    hash_count = 0
    no_message_invariance = 0
    k1_invariance = 0
    total_elapsed = 0.0
    per_seed_arm = {}
    for archive in archives:
        sidecar_path = archive.with_suffix(".json")
        if not sidecar_path.is_file():
            raise AssertionError(f"missing provenance sidecar: {archive}")
        sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
        if sidecar.get("output_sha256") != sha256(archive):
            raise AssertionError(f"output SHA-256 mismatch: {archive}")
        hash_count += 1
        total_elapsed += float(sidecar.get("elapsed_seconds", 0.0))
        relative = archive.relative_to(run_dir).parts
        seed_arm, policy, filename = relative
        seed_text, arm_letter = seed_arm[1:].split("_", 1)
        seed = int(seed_text)
        arm = next(value for value in config["arms"] if value[0] == arm_letter)
        depth = int(filename[1:3])
        if (
            sidecar.get("seed") != seed
            or sidecar.get("arm") != arm
            or sidecar.get("depth") != depth
            or sidecar.get("communication_policy") != policy
            or sidecar.get("protocol_sha256") != config["protocol_sha256"]
            or sidecar.get("dataset_manifest_sha256") != config["dataset_manifest_sha256"]
            or sidecar.get("runner_sha256") != config["runner_sha256"]
            or sidecar.get("shared_helper_sha256") != config["shared_helper_sha256"]
            or sidecar.get("labels_provided_to_model") is not False
        ):
            raise AssertionError(f"sidecar provenance mismatch: {archive}")
        checkpoint_path = ROOT / sidecar["checkpoint_path"]
        if not checkpoint_path.is_file() or sha256(checkpoint_path) != sidecar.get("checkpoint_sha256"):
            raise AssertionError(f"checkpoint hash mismatch: {archive}")
        with np.load(archive, allow_pickle=False) as loaded:
            data = {key: loaded[key] for key in loaded.files}
        expected_tasks = len(config["family_ids"])
        shape = (27, expected_tasks, 9, 10)
        if data["logits_no_clue"].shape != shape or data["logits_local_clue"].shape != shape:
            raise AssertionError(f"unexpected logits shape: {archive}")
        if data["incident_agents"].shape != (expected_tasks, 27):
            raise AssertionError(f"unexpected incident-agent shape: {archive}")
        if not np.all(data["incident_agents"].sum(axis=1) == 3):
            raise AssertionError(f"a clue did not reach exactly three agents: {archive}")
        if not np.array_equal(data["family_ids"], np.asarray(config["family_ids"])):
            raise AssertionError(f"family provenance mismatch: {archive}")
        if not np.array_equal(data["target_cells"], np.asarray(config["target_cells"])):
            raise AssertionError(f"target-cell provenance mismatch: {archive}")
        task_index = np.arange(expected_tasks)
        if np.any(data["task_inputs"][task_index, data["target_cells"]] != 0):
            raise AssertionError(f"an oracle cell was already visible: {archive}")
        if not np.array_equal(
            data["task_labels_for_scoring_only"][task_index, data["target_cells"]],
            data["target_digits"],
        ):
            raise AssertionError(f"oracle digit differs from the scoring label: {archive}")

        remote = ~data["incident_agents"].astype(bool)
        base_logits = data["logits_no_clue"].transpose(1, 0, 2, 3)
        clue_logits = data["logits_local_clue"].transpose(1, 0, 2, 3)
        if policy == "none_local_cg":
            if not np.array_equal(base_logits[remote], clue_logits[remote]):
                raise AssertionError(f"remote output changed under local CG: {archive}")
            no_message_invariance += 1
        if depth == 1:
            if not np.array_equal(base_logits[remote], clue_logits[remote]):
                raise AssertionError(f"remote output changed at K=1: {archive}")
            k1_invariance += 1

        if depth == 2:
            equal_base, equal_clue = score_distance_one(data, equal_agent_weight=True)
            pooled_base, pooled_clue = score_distance_one(data, equal_agent_weight=False)
            for index, (family_id, target_cell) in enumerate(zip(
                data["family_ids"].astype(int), data["target_cells"].astype(int), strict=True
            )):
                key = (family_id, target_cell, seed, arm, policy)
                if np.isfinite(equal_base[index]) and np.isfinite(equal_clue[index]):
                    keyed_scores_equal_agent[key] = (float(equal_base[index]), float(equal_clue[index]))
                if np.isfinite(pooled_base[index]) and np.isfinite(pooled_clue[index]):
                    keyed_scores_slot_weighted[key] = (float(pooled_base[index]), float(pooled_clue[index]))
    if no_message_invariance != len(config["seeds"]) * len(config["arms"]) * len(config["depths"]):
        raise AssertionError("not all no-message artifacts were checked")
    if k1_invariance != len(config["seeds"]) * len(config["arms"]) * 2:
        raise AssertionError("not all K=1 artifacts were checked")

    def aggregate_primary(score_map: dict[tuple, tuple[float, float]]):
        family_effects = {family: [] for family in unique_families}
        checkpoint_effects = {}
        for family_id in unique_families:
            for target_cell in targets_by_family[family_id]:
                for seed in config["seeds"]:
                    for arm in config["arms"]:
                        full = score_map.get((family_id, target_cell, int(seed), arm, "full_admm"))
                        none = score_map.get((family_id, target_cell, int(seed), arm, "none_local_cg"))
                        if full is None or none is None:
                            continue
                        did = (full[1] - full[0]) - (none[1] - none[0])
                        family_effects[family_id].append(did)
                        checkpoint_effects.setdefault((int(seed), arm), []).append(did)
        expected_per_family = 4 * len(config["seeds"]) * len(config["arms"])
        if any(len(values) != expected_per_family for values in family_effects.values()):
            raise AssertionError("at least one family is missing a paired primary intervention")
        values = np.asarray([np.mean(family_effects[family]) for family in unique_families], dtype=np.float64)
        by_checkpoint = {
            f"seed_{seed}_{arm}": float(np.mean(effect_values))
            for (seed, arm), effect_values in sorted(checkpoint_effects.items())
        }
        return values, bootstrap(values), by_checkpoint

    protocol_family_values, protocol_primary, protocol_by_checkpoint = aggregate_primary(keyed_scores_equal_agent)
    pooled_family_values, pooled_primary, pooled_by_checkpoint = aggregate_primary(keyed_scores_slot_weighted)
    claimed = summary["primary_result"]["family_cluster_bootstrap"]
    for key in ("estimate", "ci95_low", "ci95_high"):
        if not np.isclose(pooled_primary[key], float(claimed[key]), rtol=0, atol=1e-12):
            raise AssertionError(f"runner slot-weighted result {key} does not match raw NPZ files")
    claimed_family_values = summary["primary_result"]["family_level_values"]
    for family_id, value in zip(unique_families, pooled_family_values, strict=True):
        if not np.isclose(value, float(claimed_family_values[str(family_id)]), rtol=0, atol=1e-12):
            raise AssertionError(f"runner slot-weighted family effect mismatch: {family_id}")

    report = {
        "status": "PASS",
        "run_hash": config["run_hash"],
        "summary_sha256": sha256(summary_path),
        "verified_output_hashes": hash_count,
        "expected_output_hashes": expected_count,
        "family_clusters": len(unique_families),
        "interventions_per_family": 4,
        "no_message_remote_invariance_artifacts": no_message_invariance,
        "K1_remote_invariance_artifacts": k1_invariance,
        "primary_recomputed_from_raw_npz_equal_agent_weighted": protocol_primary,
        "primary_equal_agent_weighted_per_seed_arm": protocol_by_checkpoint,
        "runner_slot_weighted_result_recomputed_from_raw_npz": pooled_primary,
        "runner_slot_weighted_per_seed_arm": pooled_by_checkpoint,
        "analysis_deviation": (
            "The runner summary pooled all remote agent-cell slots, whereas the frozen protocol specified "
            "equal weighting across remote agents after averaging their local slots. The preregistered "
            "equal-agent estimand is recomputed here from raw logits; the runner's pooled result is retained "
            "as a diagnostic and independently verified."
        ),
        "per_seed_arm_effects": protocol_by_checkpoint,
        "model_elapsed_seconds": round(total_elapsed, 2),
        "inference_backend": summary["backend"],
        "labels_provided_to_model": False,
    }
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--data-dir", type=Path, default=ROOT / "datasets/sudoku_coordination_confirmation_20261007")
    args = parser.parse_args()
    report = audit(args.run_dir.resolve(), args.data_dir.resolve())
    text = json.dumps(report, indent=2)
    print(text)
    (args.run_dir.resolve() / "audit.json").write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
