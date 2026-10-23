"""Family-level preregistered analyses for P7 and P8 channel ablations."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def bootstrap_mean_ci(values: np.ndarray, rng: np.random.Generator,
                      draws: int) -> tuple[float, float]:
    values = np.asarray(values, dtype=np.float64)
    n = len(values)
    samples = rng.integers(0, n, size=(draws, n))
    means = values[samples].mean(axis=1)
    return float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def holm_adjust(p_values: dict[str, float]) -> dict[str, float]:
    ordered = sorted(p_values.items(), key=lambda item: item[1])
    adjusted, running, count = {}, 0.0, len(ordered)
    for rank, (name, value) in enumerate(ordered):
        running = max(running, min(1.0, (count - rank) * float(value)))
        adjusted[name] = running
    return adjusted


def contrast_summary(name: str, values: np.ndarray, rng: np.random.Generator,
                     draws: int) -> dict:
    values = np.asarray(values, dtype=np.float64)
    low, high = bootstrap_mean_ci(values, rng, draws)
    p_value = float(stats.ttest_1samp(values, 0.0).pvalue)
    return {
        "name": name, "families": int(len(values)),
        "estimate": float(values.mean()), "ci95_family_bootstrap": [low, high],
        "sd_between_families": float(values.std(ddof=1)),
        "p_two_sided_paired_t": p_value,
    }


def analyze_p7(df: pd.DataFrame, rng: np.random.Generator, draws: int) -> dict:
    required = {
        "decim_admm_on_global", "decim_admm_off_global",
        "decim_admm_on_local", "decim_admm_off_local",
        "direct_admm_on", "direct_admm_off",
    }
    if not required.issubset(set(df["condition"].unique())):
        raise RuntimeError("P7 table lacks frozen conditions")
    aggregate = df.groupby(["family_id", "condition"], as_index=False).agg(
        primary_logprob=("primary_logprob", "mean"),
        primary_accuracy=("primary_accuracy", "mean"),
        commits=("commits", "mean"),
        correct_commits=("correct_commits", "mean"),
        wrong_commits=("wrong_commits", "mean"),
        cross_view_conflicts=("cross_view_conflicts", "mean"),
        routed_view_updates=("routed_view_updates", "mean"),
    )
    wide = aggregate.pivot(index="family_id", columns="condition")

    def difference(condition_a: str, condition_b: str, field: str = "primary_logprob"):
        return (wide[field][condition_a] - wide[field][condition_b]).to_numpy(dtype=float)

    contrasts = {
        "admm_effect_global_commits": difference("decim_admm_on_global", "decim_admm_off_global"),
        "admm_effect_local_commits": difference("decim_admm_on_local", "decim_admm_off_local"),
    }
    summaries = {name: contrast_summary(name, value, rng, draws)
                 for name, value in contrasts.items()}
    adjusted = holm_adjust({name: row["p_two_sided_paired_t"] for name, row in summaries.items()})
    for name, row in summaries.items():
        row["p_holm_two_primary"] = adjusted[name]
    interaction = contrasts["admm_effect_global_commits"] - contrasts["admm_effect_local_commits"]
    direct_effect = difference("direct_admm_on", "direct_admm_off")
    means = aggregate.groupby("condition").primary_logprob.mean().to_dict()
    commits = aggregate[aggregate.condition.str.startswith("decim_")].groupby("condition").agg(
        mean_commits=("commits", "mean"), mean_correct=("correct_commits", "mean"),
        mean_wrong=("wrong_commits", "mean"), mean_conflicts=("cross_view_conflicts", "mean"),
        mean_routed_view_updates=("routed_view_updates", "mean"),
    ).to_dict(orient="index")
    return {
        "primary_endpoint": "family mean pre-commit log-probability, equal agent-view weighting",
        "family_count": int(wide.shape[0]), "conditions_mean_logprob": means,
        "primary_contrasts": summaries,
        "secondary_interaction_global_minus_local_mean": float(interaction.mean()),
        "secondary_direct_admm_on_minus_off_mean": float(direct_effect.mean()),
        "secondary_commit_traffic": commits,
    }


def analyze_p8(df: pd.DataFrame, rng: np.random.Generator, draws: int) -> dict:
    decim = df[df.method == "decimation"]
    if set(decim.clues.astype(int).unique()) != {32, 40, 48}:
        raise RuntimeError("P8 table lacks one of the frozen clue levels")
    if set(decim.distribution.unique()) != {"balanced", "concentrated"}:
        raise RuntimeError("P8 table lacks one of the frozen distributions")
    # Average mask repeats, checkpoint seeds, and model positions inside each
    # independent family; none of those repeats inflate the sample size.
    aggregate = decim.groupby(
        ["family_id", "clues", "distribution", "communication_fraction"],
        as_index=False,
    ).agg(
        primary_accuracy=("primary_accuracy", "mean"),
        primary_logprob=("primary_logprob", "mean"),
        commits=("commits", "mean"),
        cross_view_conflicts=("cross_view_conflicts", "mean"),
    )
    estimates = {}
    interactions = {}
    for distribution in ("balanced", "concentrated"):
        part = aggregate[aggregate.distribution == distribution]
        wide_acc = part.pivot(index="family_id", columns=["clues", "communication_fraction"],
                              values="primary_accuracy")
        diff = (
            wide_acc[(48, 1.0)] - wide_acc[(48, 0.5)]
            - wide_acc[(32, 1.0)] + wide_acc[(32, 0.5)]
        ).to_numpy(dtype=float)
        interactions[distribution] = diff
        estimates[distribution] = contrast_summary(
            f"{distribution}_clue_by_communication_interaction", diff, rng, draws
        )
    adjusted = holm_adjust({name: row["p_two_sided_paired_t"] for name, row in estimates.items()})
    for name, row in estimates.items():
        row["p_holm_two_primary"] = adjusted[name]

    curves = {}
    for distribution in ("balanced", "concentrated"):
        for fraction, label in ((1.0, "communication_100"), (0.5, "communication_50")):
            values = []
            for clues in (32, 40, 48):
                selected = aggregate[
                    (aggregate.distribution == distribution)
                    & (aggregate.communication_fraction == fraction)
                    & (aggregate.clues == clues)
                ]
                family = selected.set_index("family_id").primary_accuracy
                lo, hi = bootstrap_mean_ci(family.to_numpy(), rng, draws)
                values.append({
                    "clues": clues, "mean_accuracy": float(family.mean()),
                    "ci95_family_bootstrap": [lo, hi],
                })
            curves[f"{distribution}/{label}"] = values
    direct = df[df.method == "direct"].groupby(
        ["clues", "distribution", "communication_fraction"], as_index=False
    ).agg(primary_accuracy=("primary_accuracy", "mean"), primary_logprob=("primary_logprob", "mean"))
    return {
        "primary_endpoint": "family mean pre-commit accuracy, equal agent-view weighting; decimation with global commits",
        "family_count": int(aggregate.family_id.nunique()),
        "primary_contrasts": estimates,
        "decimation_curves": curves,
        "secondary_direct_means": direct.to_dict(orient="records"),
        "interaction_formula": "(acc48_100 - acc48_50) - (acc32_100 - acc32_50), per distribution",
    }


def plot_p7(result: dict, out: Path, language: str = "pt") -> None:
    if language not in {"pt", "en"}:
        raise ValueError("language must be 'pt' or 'en'")
    names = list(result["primary_contrasts"])
    labels = (["Commits globais", "Commits locais"] if language == "pt"
              else ["Global commitments", "Local commitments"])
    estimates = [result["primary_contrasts"][name]["estimate"] for name in names]
    lows = [result["primary_contrasts"][name]["ci95_family_bootstrap"][0] for name in names]
    highs = [result["primary_contrasts"][name]["ci95_family_bootstrap"][1] for name in names]
    errors = np.asarray([[est - lo for est, lo in zip(estimates, lows)],
                         [hi - est for est, hi in zip(estimates, highs)]])
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    ax.axhline(0, color="#404040", lw=1)
    ax.errorbar(np.arange(2), estimates, yerr=errors, fmt="o", capsize=5,
                color="#165d91", markersize=7, linewidth=2)
    ax.set_xticks([0, 1], labels)
    ax.set_ylabel("Δ log-probabilidade do dígito correto" if language == "pt"
                  else "Δ correct-digit log probability")
    ax.set_title("P7 · efeito de ligar mensagens ADMM na decimação" if language == "pt"
                 else "P7 · Effect of enabling ADMM messages under decimation")
    ax.grid(axis="y", alpha=.22)
    fig.tight_layout()
    fig.savefig(out.with_suffix(".png"), dpi=180)
    fig.savefig(out.with_suffix(".pdf"))
    plt.close(fig)


def plot_p8(result: dict, out: Path, language: str = "pt") -> None:
    if language not in {"pt", "en"}:
        raise ValueError("language must be 'pt' or 'en'")
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.1), sharey=True)
    colors = {"communication_100": "#165d91", "communication_50": "#d97706"}
    distribution_labels = {
        "balanced": "equilibrada" if language == "pt" else "balanced",
        "concentrated": "concentrada" if language == "pt" else "concentrated",
    }
    for ax, distribution in zip(axes, ("balanced", "concentrated"), strict=True):
        for label in ("communication_100", "communication_50"):
            rows = result["decimation_curves"][f"{distribution}/{label}"]
            x = [row["clues"] for row in rows]
            y = [row["mean_accuracy"] for row in rows]
            lo = [row["ci95_family_bootstrap"][0] for row in rows]
            hi = [row["ci95_family_bootstrap"][1] for row in rows]
            yerr = np.asarray([[v - a for v, a in zip(y, lo)],
                               [b - v for v, b in zip(y, hi)]])
            ax.errorbar(x, y, yerr=yerr, marker="o", capsize=4,
                        label=label.replace("communication_", "Comunicação " if language == "pt" else "Communication ") + "%",
                        color=colors[label], linewidth=2)
        ax.set_title(("Distribuição " if language == "pt" else "") + distribution_labels[distribution].capitalize())
        ax.set_xlabel("Pistas iniciais" if language == "pt" else "Initial clues")
        ax.set_xticks([32, 40, 48])
        ax.grid(alpha=.22)
    axes[0].set_ylabel("Acurácia pré-commit nas células ocultas comuns" if language == "pt"
                       else "Pre-commit accuracy on shared hidden cells")
    axes[1].legend(frameon=False)
    fig.suptitle("P8 · informação inicial e canal de comunicação" if language == "pt"
                 else "P8 · Initial information and communication channel")
    fig.tight_layout()
    fig.savefig(out.with_suffix(".png"), dpi=180)
    fig.savefig(out.with_suffix(".pdf"))
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--bootstrap-draws", type=int, default=20_000)
    parser.add_argument("--seed", type=int, default=20261008)
    parser.add_argument("--language", choices=("pt", "en"), default="pt")
    parser.add_argument("--figure-dir", type=Path)
    parser.add_argument("--plot-only", action="store_true",
                        help="Render a figure from the saved analysis.json without recomputing results")
    args = parser.parse_args()
    run_dir = args.run_dir.resolve()
    figure_dir = (args.figure_dir.resolve() if args.figure_dir else run_dir)
    figure_dir.mkdir(parents=True, exist_ok=True)
    if args.plot_only:
        result = json.loads((run_dir / "analysis.json").read_text(encoding="utf-8"))
        if result.get("phase") == "P7":
            out = figure_dir / "P7_ADMM_CHANNEL_EFFECT"
            plot_p7(result, out, args.language)
        elif result.get("phase") == "P8":
            out = figure_dir / "P8_CLUES_COMMUNICATION"
            plot_p8(result, out, args.language)
        else:
            raise RuntimeError("plot-only expects a saved P7 or P8 analysis.json")
        print(json.dumps({"phase": result["phase"], "figure_dir": str(figure_dir),
                          "language": args.language}, indent=2))
        return
    config = json.loads((run_dir / "run_config.json").read_text(encoding="utf-8"))
    audit_path = run_dir / "audit.json"
    if not audit_path.is_file() or json.loads(audit_path.read_text(encoding="utf-8")).get("status") != "PASS":
        raise RuntimeError("analysis is gated on a complete independent PASS audit")
    df = pd.read_csv(run_dir / "family_results.csv")
    rng = np.random.default_rng(args.seed)
    if config["phase"] == "p7":
        result = {"phase": "P7", **analyze_p7(df, rng, args.bootstrap_draws)}
        plot_p7(result, figure_dir / "P7_ADMM_CHANNEL_EFFECT", args.language)
    elif config["phase"] == "p8":
        result = {"phase": "P8", **analyze_p8(df, rng, args.bootstrap_draws)}
        plot_p8(result, figure_dir / "P8_CLUES_COMMUNICATION", args.language)
    else:
        raise RuntimeError("confirmatory analysis accepts only P7/P8 holdout runs")
    result.update({
        "run_config_sha256": __import__("hashlib").sha256(
            json.dumps(config, sort_keys=True, default=list).encode()
        ).hexdigest(),
        "dataset_manifest_sha256": config["dataset_manifest_sha256"],
        "family_cluster_bootstrap_draws": args.bootstrap_draws,
        "bootstrap_seed": args.seed,
        "independent_unit": "solution family",
        "repeated_measures": "checkpoint seed, model position, and communication mask",
    })
    (run_dir / "analysis.json").write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    report_lines = [f"# {result['phase']} resultados confirmatórios", "",
                    f"Famílias independentes: {result['family_count']}.",
                    f"Desfecho primário: {result['primary_endpoint']}.", ""]
    for name, contrast in result["primary_contrasts"].items():
        lo, hi = contrast["ci95_family_bootstrap"]
        report_lines.append(
            f"- **{name}**: {contrast['estimate']:.6f} "
            f"[IC95% {lo:.6f}, {hi:.6f}], p={contrast['p_two_sided_paired_t']:.6g}, "
            f"p-Holm={contrast['p_holm_two_primary']:.6g}."
        )
    report_lines += ["", "A unidade de inferência é a família de solução; seeds, posições e máscaras são medidas repetidas.",
                     "Os resultados secundários e a execução exata estão em `analysis.json` e `family_results.csv`."]
    (run_dir / "analysis.md").write_text("\n".join(report_lines) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
