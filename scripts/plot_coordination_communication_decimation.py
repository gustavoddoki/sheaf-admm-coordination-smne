"""Plot the completed P5 communication × decimation confirmation."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ARMS = ("internal", "terminal")
ARM_LABELS = {"internal": "Interno", "terminal": "Terminal"}
ARM_COLORS = {"internal": "#0072B2", "terminal": "#D55E00"}
CONDITIONS = ("full_100", "random_75", "random_50", "none_0")
CONDITION_LABELS = {
    "full_100": "Completa",
    "random_75": "Aleatória 75%",
    "random_50": "Aleatória 50%",
    "none_0": "0% ADMM*",
}
REDUCED = ("random_75", "random_50", "none_0")


def interval(row: dict, metric: str) -> tuple[float, float, float]:
    estimate = 100.0 * float(row["estimate"])
    low = 100.0 * float(row["cluster_bootstrap_95_low"])
    high = 100.0 * float(row["cluster_bootstrap_95_high"])
    return estimate, estimate - low, high - estimate


def create_figure(analysis_path: Path, language: str = "pt", out_dir: Path | None = None) -> tuple[Path, Path]:
    if language not in {"pt", "en"}:
        raise ValueError("language must be 'pt' or 'en'")
    labels = {
        "pt": {
            "arms": {"internal": "Interno", "terminal": "Terminal"},
            "conditions": {
                "full_100": "Completa", "random_75": "Aleatória 75%",
                "random_50": "Aleatória 50%", "none_0": "0% ADMM*",
            },
            "direct": "Direta, orçamento pareado", "decimation": "Decimação",
            "exact": "Grades exatas (%)", "panel_a": "(a) Decimação e controle direto",
            "gain": "Ganho da decimação: completa − reduzida (pp)",
            "panel_b": "(b) Dependência da comunicação",
            "title": "Comunicação entre agentes e inferência por decimação",
            "contrast": "Contraste funcional pré-especificado (completa − 50%, braços com peso igual):",
            "bars": "Barras: IC95% por solução-base; checkpoints e máscaras são medidas repetidas. ",
            "no_admm": "* Sem mensagens ADMM: compromissos ainda compartilhados pelo tabuleiro.",
        },
        "en": {
            "arms": {"internal": "Internal", "terminal": "Terminal"},
            "conditions": {
                "full_100": "Full", "random_75": "Random 75%",
                "random_50": "Random 50%", "none_0": "0% ADMM*",
            },
            "direct": "Direct, matched budget", "decimation": "Decimation",
            "exact": "Exact boards (%)", "panel_a": "(a) Decimation and direct-inference control",
            "gain": "Change in decimation gain: full − reduced (pp)",
            "panel_b": "(b) Dependence on communication",
            "title": "Agent communication and inference by decimation",
            "contrast": "Pre-specified functional contrast (full − 50%, equally weighted arms):",
            "bars": "Bars: family-bootstrap 95% CIs; checkpoints and masks are repeated measures. ",
            "no_admm": "* No ADMM messages: commitments are still shared across the board.",
        },
    }[language]
    analysis = json.loads(analysis_path.read_text(encoding="utf-8"))
    primary = analysis.get("primary_functional_interaction")
    if not primary:
        raise ValueError("The completed analysis is missing the pooled primary functional contrast")
    summary = {
        (row["arm"], row["condition"], row["metric"]): row
        for row in analysis["summary"]
    }
    interactions = {
        (row["arm"], row["reduced_condition"], row["metric"]): row
        for row in analysis["interactions"]
    }
    required_summary = {
        (arm, condition, metric)
        for arm in ARMS for condition in CONDITIONS
        for metric in ("direct_exact_matched_budget", "decimation_exact")
    }
    required_interactions = {
        (arm, condition, "decimation_gain_exact")
        for arm in ARMS for condition in REDUCED
    }
    if not required_summary.issubset(summary) or not required_interactions.issubset(interactions):
        raise ValueError("The completed analysis is missing a pre-specified P5 functional row")

    plt.rcParams.update({"font.size": 9.5, "figure.dpi": 150, "savefig.dpi": 300})
    figure, axes = plt.subplots(1, 2, figsize=(11.2, 5.2), gridspec_kw={"width_ratios": [1.25, 1]})
    x = np.arange(len(CONDITIONS), dtype=float)
    styles = {
        "direct_exact_matched_budget": ("--", "o", labels["direct"]),
        "decimation_exact": ("-", "s", labels["decimation"]),
    }
    for arm in ARMS:
        for metric, (linestyle, marker, method_label) in styles.items():
            rows = [summary[(arm, condition, metric)] for condition in CONDITIONS]
            estimates, lower, upper = zip(*(interval(row, metric) for row in rows), strict=True)
            offset = -0.035 if arm == "internal" else 0.035
            axes[0].errorbar(
                x + offset, estimates, yerr=np.vstack((lower, upper)),
                color=ARM_COLORS[arm], linestyle=linestyle, marker=marker,
                markersize=5, linewidth=1.5, capsize=3,
                label=f"{labels['arms'][arm]} · {method_label}",
            )
    axes[0].set_xticks(x, [labels["conditions"][item] for item in CONDITIONS], rotation=10, ha="right")
    axes[0].set_ylim(-3, 103)
    axes[0].set_ylabel(labels["exact"])
    axes[0].set_title(labels["panel_a"], loc="left", fontweight="bold")
    axes[0].grid(axis="y", color="#D9DEE7", linewidth=0.7)
    axes[0].spines[["top", "right"]].set_visible(False)
    axes[0].legend(
        frameon=False, fontsize=7.4, ncol=2,
        loc="upper center", bbox_to_anchor=(0.5, -0.27),
    )

    x_reduced = np.arange(len(REDUCED), dtype=float)
    for arm in ARMS:
        rows = [interactions[(arm, condition, "decimation_gain_exact")] for condition in REDUCED]
        estimates, lower, upper = zip(*(interval(row, "decimation_gain_exact") for row in rows), strict=True)
        axes[1].errorbar(
            x_reduced + (-0.04 if arm == "internal" else 0.04),
            estimates, yerr=np.vstack((lower, upper)),
            color=ARM_COLORS[arm], marker="o", markersize=6,
            linewidth=1.5, capsize=4, label=labels["arms"][arm],
        )
    axes[1].axhline(0, color="#555555", linewidth=1.0, zorder=0)
    axes[1].set_xticks(x_reduced, [
        "75%", "50%", "0% ADMM*"
    ], rotation=10, ha="right")
    axes[1].set_ylabel(labels["gain"])
    axes[1].set_title(labels["panel_b"], loc="left", fontweight="bold")
    axes[1].grid(axis="y", color="#D9DEE7", linewidth=0.7)
    axes[1].spines[["top", "right"]].set_visible(False)
    axes[1].legend(
        frameon=False, ncol=2,
        loc="upper center", bbox_to_anchor=(0.5, -0.27),
    )

    figure.suptitle(labels["title"], fontsize=12.5, fontweight="bold")
    primary_ci = (
        100.0 * float(primary["estimate"]),
        100.0 * float(primary["cluster_bootstrap_95_low"]),
        100.0 * float(primary["cluster_bootstrap_95_high"]),
    )
    figure.text(
        0.5, 0.060,
        f"{labels['contrast']} {primary_ci[0]:+.2f} pp "
        f"[{primary_ci[1]:+.2f}, {primary_ci[2]:+.2f}].",
        ha="center", fontsize=8.1, color="#333333",
    )
    figure.text(
        0.5, 0.022,
        labels["bars"] + labels["no_admm"],
        ha="center", fontsize=8.1, color="#444444",
    )
    figure.tight_layout(rect=(0, 0.18, 1, 0.93))
    output = out_dir or analysis_path.parent
    output.mkdir(parents=True, exist_ok=True)
    png = output / "P5_FUNCTIONAL_COMMUNICATION.png"
    pdf = output / "P5_FUNCTIONAL_COMMUNICATION.pdf"
    figure.savefig(png, bbox_inches="tight")
    figure.savefig(pdf, bbox_inches="tight")
    plt.close(figure)
    return png, pdf


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--language", choices=("pt", "en"), default="pt")
    parser.add_argument("--out-dir", type=Path)
    args = parser.parse_args()
    png, pdf = create_figure(args.run_dir / "analysis.json", args.language, args.out_dir)
    print(json.dumps({"png": str(png), "pdf": str(pdf)}, indent=2))


if __name__ == "__main__":
    main()
