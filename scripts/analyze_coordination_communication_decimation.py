"""Family-clustered summaries for communication × decimation runs."""
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


CONDITIONS = ["full_100", "random_75", "periodic_75", "random_50", "periodic_50", "none_0"]
METRICS = [
    "direct_exact_matched_budget", "decimation_exact", "decimation_gain_exact",
    "direct_common_hidden_accuracy", "decimation_common_hidden_accuracy",
    "gain_common_hidden_accuracy", "decimation_commits", "decimation_wrong_commits",
]


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def mean(values: list[float]) -> float:
    return float(np.mean(values)) if values else float("nan")


def cluster_bootstrap(values: np.ndarray, rng: np.random.Generator, draws: int = 20000):
    """Percentile interval resampling independent family IDs, preserving repeats."""
    if values.ndim != 1 or len(values) < 2:
        raise ValueError("cluster bootstrap requires one value per at least two families")
    indices = rng.integers(0, len(values), size=(draws, len(values)))
    samples = values[indices].mean(axis=1)
    return float(values.mean()), float(np.quantile(samples, 0.025)), float(np.quantile(samples, 0.975))


def condition_group(name: str) -> str:
    if name in {"full_100", "none_0"}:
        return name
    for policy in ("random", "periodic"):
        for fraction in ("75", "50"):
            prefix = f"{policy}_{fraction}_r"
            if name.startswith(prefix):
                return f"{policy}_{fraction}"
    raise ValueError(f"Unknown communication condition: {name}")


def aggregate(run_dir: Path, bootstrap_draws: int = 20000, seed: int = 20261007) -> dict:
    run_config = json.loads((run_dir / "run_config.json").read_text(encoding="utf-8"))
    family_count = int(run_config["families"])
    rows = read_csv(run_dir / "paired_budget_results.csv")
    commit_updates = defaultdict(list)
    for row in rows:
        commit_updates[(row["arm"], condition_group(row["condition"]))].append(
            float(row["decimation_commits"])
        )
    # First average the three communication-mask seeds within each trained model,
    # then average the trained-model seeds within each family. Families remain
    # the bootstrap clusters; masks and checkpoint seeds are repeated measures.
    checkpoint_values: dict[tuple[str, str, int, int], list[dict[str, float]]] = defaultdict(list)
    for row in rows:
        arm = row["arm"]
        condition = condition_group(row["condition"])
        family = int(row["family_id"])
        train_seed = int(row["seed"])
        checkpoint_values[(arm, condition, family, train_seed)].append(
            {metric: float(row[metric]) for metric in METRICS}
        )

    # Use only conditions present in the frozen run. Confirmation runs omit the
    # periodic masks because their deterministic offsets were not varied.
    conditions = sorted({condition_group(row["condition"]) for row in rows})

    # Build a single metric value per family and condition after averaging
    # mask repetitions within each checkpoint, then checkpoint seeds equally.
    per_family: dict[tuple[str, str, str], dict[int, float]] = defaultdict(dict)
    arms = sorted({row["arm"] for row in rows})
    families = sorted({int(row["family_id"]) for row in rows})
    train_seeds = sorted({int(row["seed"]) for row in rows})
    for arm in arms:
        for condition in conditions:
            for metric in METRICS:
                for family in families:
                    per_seed = []
                    for train_seed in train_seeds:
                        repeated = [
                            record[metric]
                            for record in checkpoint_values[(arm, condition, family, train_seed)]
                        ]
                        per_seed.append(mean(repeated))
                    per_family[(arm, condition, metric)][family] = mean(per_seed)

    rng = np.random.default_rng(seed)
    summary_rows = []
    for arm in arms:
        for condition in conditions:
            for metric in METRICS:
                arr = np.asarray([per_family[(arm, condition, metric)][family] for family in families])
                estimate, low, high = cluster_bootstrap(arr, rng, bootstrap_draws)
                summary_rows.append({
                    "arm": arm, "condition": condition, "metric": metric,
                    "families": len(families), "estimate": estimate,
                    "cluster_bootstrap_95_low": low, "cluster_bootstrap_95_high": high,
                })

    interaction_rows = []
    for arm in arms:
        for condition in conditions:
            if condition == "full_100":
                continue
            for metric in ("decimation_gain_exact", "gain_common_hidden_accuracy"):
                paired = np.asarray([
                    per_family[(arm, "full_100", metric)][family]
                    - per_family[(arm, condition, metric)][family]
                    for family in families
                ])
                estimate, low, high = cluster_bootstrap(paired, rng, bootstrap_draws)
                interaction_rows.append({
                    "arm": arm, "reduced_condition": condition, "metric": metric,
                    "contrast": "full_100_decimation_gain_minus_reduced_gain",
                    "families": len(families), "estimate": estimate,
                    "cluster_bootstrap_95_low": low, "cluster_bootstrap_95_high": high,
                })

    # One family-level functional contrast, averaging the two rule-position arms
    # equally. Seeds and random masks remain repeated measurements within family.
    if {"full_100", "random_50"}.issubset(conditions) and len(arms) == 2:
        pooled_gain = {
            condition: np.asarray([
                mean([
                    per_family[(arm, condition, "decimation_gain_exact")][family]
                    for arm in arms
                ])
                for family in families
            ])
            for condition in ("full_100", "random_50")
        }
        primary_values = pooled_gain["full_100"] - pooled_gain["random_50"]
        estimate, low, high = cluster_bootstrap(primary_values, rng, bootstrap_draws)
        primary_functional = {
            "estimand": "(decimation gain, full_100) - (decimation gain, random_50)",
            "metric": "decimation_gain_exact",
            "arm_weighting": "equal internal and terminal weights within family",
            "repeated_measure_aggregation": "average mask replicates within checkpoint, then seeds, then arms",
            "bootstrap_unit": "solution-base family",
            "families": len(families),
            "estimate": estimate,
            "cluster_bootstrap_95_low": low,
            "cluster_bootstrap_95_high": high,
            "bootstrap_draws": bootstrap_draws,
        }
    else:
        raise ValueError("The frozen functional contrast requires both arms and full/50% conditions")

    # Derive correct commits after the frozen inferential metrics so adding this
    # secondary summary does not shift their seeded bootstrap streams.
    correct_commit_values: dict[tuple[str, str, int, int], list[float]] = defaultdict(list)
    for row in rows:
        total = float(row["decimation_commits"])
        wrong = float(row["decimation_wrong_commits"])
        if wrong > total:
            raise ValueError(
                f"Wrong commits exceed total commits for family {row['family_id']}"
            )
        key = (
            row["arm"], condition_group(row["condition"]),
            int(row["family_id"]), int(row["seed"]),
        )
        correct_commit_values[key].append(total - wrong)
    for arm in arms:
        for condition in conditions:
            family_values = []
            for family in families:
                seed_values = []
                for train_seed in train_seeds:
                    repeated = correct_commit_values[(arm, condition, family, train_seed)]
                    seed_values.append(mean(repeated))
                family_values.append(mean(seed_values))
            estimate, low, high = cluster_bootstrap(
                np.asarray(family_values), rng, bootstrap_draws
            )
            summary_rows.append({
                "arm": arm, "condition": condition,
                "metric": "decimation_correct_commits",
                "families": len(families), "estimate": estimate,
                "cluster_bootstrap_95_low": low,
                "cluster_bootstrap_95_high": high,
            })

    # Summarize communication burden once per model/condition/pass because the
    # batch-level values are repeated on all 32 family rows in the CSV.
    pass_rows = read_csv(run_dir / "per_family_pass_results.csv")
    seen: dict[tuple, dict[str, float]] = {}
    for row in pass_rows:
        key = (row["seed"], row["arm"], row["condition"], row["mask_rep"], row["method"], row["pass"])
        seen[key] = {
            name: float(row[name]) for name in (
                "active_edge_slots_batch", "edge_vector_payload_bytes_batch",
                "initial_bootstrap_payload_bytes_batch", "cg_global_partial_payload_bytes_batch",
                "cg_reduced_scalar_payload_bytes_batch", "pass_seconds_including_compile",
            )
        }
    traffic = defaultdict(lambda: defaultdict(list))
    branch_payloads = defaultdict(list)
    branch_bootstrap = defaultdict(list)
    for (train_seed, arm, condition, mask_rep, method, pass_id), record in seen.items():
        group = (arm, condition, method)
        for field, value in record.items():
            traffic[group][field].append(value)
        branch = (train_seed, arm, condition, mask_rep, method)
        branch_payloads[branch].append(sum(record[name] for name in (
            "edge_vector_payload_bytes_batch", "cg_global_partial_payload_bytes_batch",
            "cg_reduced_scalar_payload_bytes_batch",
        )))
        branch_bootstrap[branch].append(record["initial_bootstrap_payload_bytes_batch"])
    branch_totals = defaultdict(list)
    branch_bootstrap_means = defaultdict(list)
    for branch, payload_per_pass in branch_payloads.items():
        key = (branch[1], branch[2], branch[4])
        bootstrap_values = branch_bootstrap[branch]
        if len(set(bootstrap_values)) != 1:
            raise AssertionError(f"Bootstrap payload varies within branch {branch}")
        expected_passes = int(run_config.get("max_passes", 8))
        if len(bootstrap_values) != expected_passes:
            raise AssertionError(f"Expected {expected_passes} model-inference passes in branch {branch}")
        branch_bootstrap_means[key].append(bootstrap_values[0])
        # Each outer inference pass calls model.apply anew and resets the cache,
        # so every pass incurs its own initial synchronization payload.
        branch_totals[key].append(sum(payload_per_pass) + sum(bootstrap_values))
    traffic_rows = []
    for (arm, condition, method), fields in sorted(traffic.items()):
        grouped_condition = condition_group(condition)
        mean_commits = mean(commit_updates[(arm, grouped_condition)])
        traffic_rows.append({
            "arm": arm, "condition": condition, "method": method,
            "model_seed_mask_replicate_pass_records": len(fields["active_edge_slots_batch"]),
            **{f"mean_{name}": mean(values) for name, values in fields.items()},
            "mean_modeled_payload_bytes_per_batch_inference_pass_excluding_bootstrap": mean([
                sum(record[name] for name in (
                    "edge_vector_payload_bytes_batch", "cg_global_partial_payload_bytes_batch",
                    "cg_reduced_scalar_payload_bytes_batch",
                ))
                for key, record in seen.items()
                if (key[1], key[2], key[4]) == (arm, condition, method)
            ]),
            "mean_bootstrap_payload_bytes_per_batch_inference_pass": mean(
                branch_bootstrap_means[(arm, condition, method)]
            ),
            "mean_total_modeled_payload_bytes_per_batch_inference_pass_including_bootstrap": (
                mean([
                    sum(record[name] for name in (
                        "edge_vector_payload_bytes_batch", "initial_bootstrap_payload_bytes_batch",
                        "cg_global_partial_payload_bytes_batch", "cg_reduced_scalar_payload_bytes_batch",
                    ))
                    for key, record in seen.items()
                    if (key[1], key[2], key[4]) == (arm, condition, method)
                ])
            ),
            "mean_total_modeled_payload_bytes_per_batch_8_inference_branch": mean(
                branch_totals[(arm, condition, method)]
            ),
            "mean_decimation_commit_events_per_puzzle": (
                mean_commits if method == "decimation" else 0.0
            ),
            "mean_commit_broadcast_agent_view_updates_per_puzzle": (
                3.0 * mean_commits if method == "decimation" else 0.0
            ),
            f"mean_commit_broadcast_agent_view_updates_per_{family_count}_family_batch": (
                3.0 * mean_commits * family_count if method == "decimation" else 0.0
            ),
        })

    result = {
        "run_dir": str(run_dir),
        "scope_note": (
            "Intervals resample solution-base families and preserve repeated mask/checkpoint measurements; "
            f"inference is conditional on {len(train_seeds)} existing checkpoint seeds."
        ),
        "bootstrap_draws": bootstrap_draws,
        "bootstrap_seed": seed,
        "summary": summary_rows,
        "interactions": interaction_rows,
        "primary_functional_interaction": primary_functional,
        "communication_traffic": traffic_rows,
    }
    (run_dir / "analysis.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    write_table(run_dir / "summary_family_bootstrap.csv", summary_rows)
    write_table(run_dir / "interaction_family_bootstrap.csv", interaction_rows)
    write_table(run_dir / "primary_functional_interaction.csv", [primary_functional])
    write_table(run_dir / "communication_traffic_summary.csv", traffic_rows)
    return result


def write_table(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--bootstrap-draws", type=int, default=20000)
    parser.add_argument("--seed", type=int, default=20261007)
    args = parser.parse_args()
    result = aggregate(args.run_dir, args.bootstrap_draws, args.seed)
    print(json.dumps({
        "status": "complete",
        "summary_rows": len(result["summary"]),
        "interaction_rows": len(result["interactions"]),
        "traffic_rows": len(result["communication_traffic"]),
        "analysis_path": str(args.run_dir / "analysis.json"),
    }, indent=2))


if __name__ == "__main__":
    main()
