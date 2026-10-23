"""Family-clustered secondary summaries for P5 local information transmission."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


METRICS = {
    "hidden_accuracy_delta": lambda frame: frame["hidden_accuracy_local_clue"] - frame["hidden_accuracy_no_clue"],
    "true_logprob_delta": lambda frame: frame["true_logprob_local_clue"] - frame["true_logprob_no_clue"],
    "js_divergence": lambda frame: frame["js_divergence"],
    "top1_changed_fraction": lambda frame: frame["top1_changed_fraction"],
    "mean_abs_logit_change": lambda frame: frame["mean_abs_logit_change"],
}


def family_bootstrap(values: np.ndarray, draws: int = 20000, seed: int = 20261008) -> dict:
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


def analyze(run_dir: Path) -> dict:
    config = json.loads((run_dir / "run_config.json").read_text(encoding="utf-8"))
    audit = json.loads((run_dir / "audit.json").read_text(encoding="utf-8"))
    agents = pd.read_csv(run_dir / "agent_metrics.csv")
    agents = agents[
        (agents["depth"] == 2)
        & (agents["incident"] == 0)
        & agents["distance_from_incident_set"].isin([1, 2])
    ].copy()
    for name, make_metric in METRICS.items():
        if name in {"hidden_accuracy_delta", "true_logprob_delta"}:
            agents[name] = make_metric(agents)
        else:
            agents[name] = make_metric(agents)

    # Each row is an agent. Average agents equally within an intervention,
    # then average the four selected cells/checkpoints within a solution-base.
    group_keys = [
        "family_id", "target_cell", "seed", "arm", "communication_policy",
        "distance_from_incident_set",
    ]
    intervention = agents.groupby(group_keys, as_index=False)[list(METRICS)].mean()
    families = sorted(set(int(value) for value in config["family_ids"]))
    output_rows = []
    checkpoint_summaries = {}

    for distance in (1, 2):
        by_distance = intervention[intervention["distance_from_incident_set"] == distance]
        for metric in METRICS:
            wide = by_distance.pivot_table(
                index=["family_id", "target_cell", "seed", "arm"],
                columns="communication_policy", values=metric, aggfunc="mean",
            ).reset_index()
            if "full_admm" not in wide or "none_local_cg" not in wide:
                raise AssertionError("P5 communication controls are incomplete")
            wide["did"] = wide["full_admm"] - wide["none_local_cg"]
            family_vals = wide.groupby("family_id")["did"].mean()
            if set(map(int, family_vals.index)) != set(families):
                raise AssertionError(f"missing family clusters at graph distance {distance}")
            ordered = np.asarray([float(family_vals.loc[f]) for f in families], dtype=np.float64)
            output_rows.append({
                "distance": distance,
                "metric": metric,
                "communication_contrast": "full_admm_minus_none_local_cg",
                **family_bootstrap(ordered),
            })

            for arm in ("internal", "terminal"):
                arm_wide = by_distance[by_distance["arm"] == arm].pivot_table(
                    index=["family_id", "target_cell", "seed"],
                    columns="communication_policy", values=metric, aggfunc="mean",
                ).reset_index()
                arm_wide["did"] = arm_wide["full_admm"] - arm_wide["none_local_cg"]
                arm_fam = arm_wide.groupby("family_id")["did"].mean()
                arm_values = np.asarray([float(arm_fam.loc[f]) for f in families], dtype=np.float64)
                output_rows.append({
                    "distance": distance,
                    "metric": metric,
                    "communication_contrast": f"full_admm_minus_none_local_cg_{arm}",
                    **family_bootstrap(arm_values),
                })

        # Descriptive response under active communication by checkpoint seed.
        for (seed, arm), group in by_distance[by_distance["communication_policy"] == "full_admm"].groupby(["seed", "arm"]):
            checkpoint_summaries[f"distance_{distance}_seed_{int(seed)}_{arm}"] = {
                metric: float(group[metric].mean()) for metric in METRICS
            }

    family_rows = []
    distance1 = intervention[intervention["distance_from_incident_set"] == 1]
    for family, group in distance1.groupby("family_id"):
        family_rows.append({
            "family_id": int(family),
            **{
                f"distance1_{column}": float(group[column].mean())
                for column in METRICS
            },
        })
    analysis = {
        "status": "complete",
        "run_hash": config["run_hash"],
        "families": len(families),
        "targets_per_family": 4,
        "checkpoints": len(config["seeds"]) * len(config["arms"]),
        "primary_metric_source": "audit.json equal-agent recomputation directly from raw logits",
        "primary_confirmatory_result_equal_agent_weighted": audit["primary_recomputed_from_raw_npz_equal_agent_weighted"],
        "runner_slot_weighted_diagnostic": audit["runner_slot_weighted_result_recomputed_from_raw_npz"],
        "analysis_deviation": audit["analysis_deviation"],
        "secondary_metrics_are_equal_agent_weighted": True,
        "secondary_family_cluster_summaries": output_rows,
        "per_seed_arm_descriptives": checkpoint_summaries,
        "interpretation_scope": "effects conditional on the six existing training seeds; solution-base families are the bootstrap clusters",
    }
    (run_dir / "analysis.json").write_text(json.dumps(analysis, indent=2), encoding="utf-8")
    pd.DataFrame(output_rows).to_csv(run_dir / "secondary_family_bootstrap.csv", index=False)
    pd.DataFrame(family_rows).to_csv(run_dir / "secondary_family_effects_distance1.csv", index=False)
    return analysis


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    args = parser.parse_args()
    result = analyze(args.run_dir.resolve())
    print(json.dumps({
        "status": result["status"],
        "families": result["families"],
        "secondary_summary_rows": len(result["secondary_family_cluster_summaries"]),
        "analysis_path": str(args.run_dir / "analysis.json"),
    }, indent=2))


if __name__ == "__main__":
    main()
