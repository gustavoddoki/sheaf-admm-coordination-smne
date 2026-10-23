"""Exploratory family-bootstrap sensitivity analysis for P8 log-probability."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Recompute the P8 48-vs-32 clue communication interaction on "
            "pre-commit log-probability as an exploratory sensitivity analysis."
        )
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("outputs/coordination_channel_ablation/p8/family_results.csv"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "outputs/coordination_channel_ablation/p8/"
            "logprob_interaction_sensitivity.json"
        ),
    )
    parser.add_argument("--bootstrap-draws", type=int, default=20_000)
    parser.add_argument("--seed", type=int, default=20261008)
    args = parser.parse_args()

    if args.bootstrap_draws < 1:
        parser.error("--bootstrap-draws must be positive")

    raw = pd.read_csv(args.input)
    selected = raw.loc[
        raw["method"].eq("decimation")
        & raw["clues"].isin([32, 48])
        & raw["communication_fraction"].isin([0.5, 1.0]),
        [
            "family_id",
            "distribution",
            "clues",
            "communication_fraction",
            "primary_accuracy",
            "primary_logprob",
        ],
    ].copy()
    if selected.empty:
        raise ValueError("No P8 decimation records matched the requested conditions")

    # Average masks, seeds, and model positions inside each independent family
    # and experimental cell before calculating the interaction or its CI.
    cells = selected.groupby(
        ["family_id", "distribution", "clues", "communication_fraction"],
        observed=True,
    )[["primary_accuracy", "primary_logprob"]].mean()
    expected_distributions = {"balanced", "concentrated"}
    if set(selected["distribution"].unique()) != expected_distributions:
        raise ValueError("Expected balanced and concentrated clue distributions")

    confirmatory = json.loads(
        Path("outputs/coordination_channel_ablation/p8/analysis.json").read_text(
            encoding="utf-8"
        )
    )
    rng = np.random.default_rng(args.seed)
    output: dict[str, object] = {
        "analysis": "exploratory sensitivity; not a confirmatory primary test",
        "input_csv": str(args.input),
        "input_sha256": hashlib.sha256(args.input.read_bytes()).hexdigest(),
        "formula": "(48 clues, 100%-50% communication) - (32 clues, 100%-50%)",
        "independent_unit": "solution family",
        "aggregation": "average repeated masks, checkpoint seeds, and model positions within family and cell",
        "bootstrap": {
            "method": "percentile cluster bootstrap over families",
            "draws": args.bootstrap_draws,
            "seed": args.seed,
        },
        "results": {},
    }

    for distribution in sorted(expected_distributions):
        wide = cells.xs(distribution, level="distribution").unstack(
            ["clues", "communication_fraction"]
        )
        required = [
            (metric, clue, communication)
            for metric in ("primary_accuracy", "primary_logprob")
            for clue in (32, 48)
            for communication in (0.5, 1.0)
        ]
        missing = [key for key in required if key not in wide.columns]
        if missing or wide[required].isna().any().any():
            raise ValueError(f"Incomplete family cells for {distribution}: {missing}")
        if len(wide) != int(confirmatory["family_count"]):
            raise ValueError(
                f"Expected {confirmatory['family_count']} families, found {len(wide)}"
            )

        metrics: dict[str, object] = {}
        for metric, label in (
            ("primary_logprob", "pre-commit log-probability"),
            ("primary_accuracy", "pre-commit accuracy reconstruction"),
        ):
            effect = (
                wide[(metric, 48, 1.0)]
                - wide[(metric, 48, 0.5)]
                - wide[(metric, 32, 1.0)]
                + wide[(metric, 32, 0.5)]
            ).to_numpy(dtype=float)
            indices = rng.integers(
                0, len(effect), size=(args.bootstrap_draws, len(effect))
            )
            bootstrap = effect[indices].mean(axis=1)
            result: dict[str, object] = {
                "label": label,
                "estimate": float(effect.mean()),
                "ci95_family_bootstrap": [
                    float(v) for v in np.quantile(bootstrap, [0.025, 0.975])
                ],
                "families": int(len(effect)),
            }
            if metric == "primary_accuracy":
                frozen = confirmatory["primary_contrasts"][distribution]["estimate"]
                result["confirmatory_accuracy_estimate"] = float(frozen)
                result["absolute_difference_from_confirmatory"] = float(
                    abs(effect.mean() - frozen)
                )
            metrics[metric] = result
        output["results"][distribution] = metrics  # type: ignore[index]

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(output, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(output["results"], indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
