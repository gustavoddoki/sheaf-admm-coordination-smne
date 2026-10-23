"""Plot the confirmatory local-information propagation result by graph distance."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


CONTRASTS = {
    "pt": {
        "internal": ("full_admm_minus_none_local_cg_internal", "Interno", "#0072B2", -0.10),
        "pooled": ("full_admm_minus_none_local_cg", "Média equiponderada", "#222222", 0.0),
        "terminal": ("full_admm_minus_none_local_cg_terminal", "Terminal", "#D55E00", 0.10),
    },
    "en": {
        "internal": ("full_admm_minus_none_local_cg_internal", "Internal", "#0072B2", -0.10),
        "pooled": ("full_admm_minus_none_local_cg", "Agent-weighted mean", "#222222", 0.0),
        "terminal": ("full_admm_minus_none_local_cg_terminal", "Terminal", "#D55E00", 0.10),
    },
}


def interval(row: dict, multiplier: float = 1.0) -> tuple[float, float, float]:
    estimate = multiplier * float(row["estimate"])
    low = multiplier * float(row["ci95_low"])
    high = multiplier * float(row["ci95_high"])
    return estimate, estimate - low, high - estimate


def make_figure(run_dir: Path, language: str = "pt", out_dir: Path | None = None) -> tuple[Path, Path]:
    if language not in CONTRASTS:
        raise ValueError("language must be 'pt' or 'en'")
    labels = {
        "pt": {
            "edges": ("1 aresta", "2 arestas"),
            "logprob": "Δ log-probabilidade do valor correto",
            "accuracy": "Δ acurácia nas células ocultas (pp)",
            "remote": "(a) Propostas remotas",
            "predictive": "(b) Consequência preditiva",
            "title": "Propagação local de informação pelas mensagens ADMM",
            "footnote": "Ponto preto: estimando primário a uma aresta; demais estimativas são secundárias. "
                        "IC95% bootstrap por solução-base (64 famílias); checkpoints como medidas repetidas.",
        },
        "en": {
            "edges": ("1 edge", "2 edges"),
            "logprob": "Δ correct-value log probability",
            "accuracy": "Δ accuracy on hidden cells (pp)",
            "remote": "(a) Remote proposals",
            "predictive": "(b) Predictive consequence",
            "title": "Local information propagation through ADMM messages",
            "footnote": "Black diamond: primary estimate at distance one; other estimates are secondary. "
                        "Family-bootstrap 95% CIs (64 solution families); checkpoints are repeated measures.",
        },
    }[language]
    analysis_path = run_dir / "analysis.json"
    payload = json.loads(analysis_path.read_text(encoding="utf-8"))
    if payload.get("status") != "complete" or payload.get("families") != 64:
        raise ValueError("Expected the completed 64-family P5 mechanistic analysis")
    secondary = {
        (int(row["distance"]), row["metric"], row["communication_contrast"]): row
        for row in payload["secondary_family_cluster_summaries"]
    }
    primary = payload["primary_confirmatory_result_equal_agent_weighted"]

    plt.rcParams.update({"font.size": 9.2, "figure.dpi": 150, "savefig.dpi": 300})
    figure, axes = plt.subplots(1, 2, figsize=(10.7, 4.4), sharex=True)
    x = np.asarray([0.0, 1.0])
    for name, (contrast, label, color, offset) in CONTRASTS[language].items():
        logprob_rows = [
            secondary[(distance, "true_logprob_delta", contrast)]
            for distance in (1, 2)
        ]
        logprob = [interval(row) for row in logprob_rows]
        if name == "pooled":
            logprob[0] = interval({
                "estimate": primary["estimate"],
                "ci95_low": primary["ci95_low"],
                "ci95_high": primary["ci95_high"],
            })
        estimates, lower, upper = zip(*logprob, strict=True)
        axes[0].errorbar(
            x + offset, estimates, yerr=np.vstack((lower, upper)),
            color=color, marker="o" if name != "pooled" else "D",
            markersize=5.2, linewidth=1.4, capsize=3, label=label,
        )

        accuracy_rows = [
            secondary[(distance, "hidden_accuracy_delta", contrast)]
            for distance in (1, 2)
        ]
        estimates, lower, upper = zip(*(interval(row, 100.0) for row in accuracy_rows), strict=True)
        axes[1].errorbar(
            x + offset, estimates, yerr=np.vstack((lower, upper)),
            color=color, marker="o" if name != "pooled" else "D",
            markersize=5.2, linewidth=1.4, capsize=3, label=label,
        )

    for axis in axes:
        axis.axhline(0, color="#555555", linewidth=1.0, zorder=0)
        axis.set_xticks(x, labels["edges"])
        axis.grid(axis="y", color="#D9DEE7", linewidth=0.7)
        axis.spines[["top", "right"]].set_visible(False)
        axis.legend(frameon=False, fontsize=8, loc="best")
    axes[0].set_ylabel(labels["logprob"])
    axes[0].set_title(labels["remote"], loc="left", fontweight="bold")
    axes[1].set_ylabel(labels["accuracy"])
    axes[1].set_title(labels["predictive"], loc="left", fontweight="bold")
    figure.suptitle(labels["title"], fontsize=12, fontweight="bold")
    figure.text(
        0.5, 0.018,
        labels["footnote"],
        ha="center", fontsize=8, color="#444444",
    )
    figure.tight_layout(rect=(0, 0.07, 1, 0.92))
    output = out_dir or run_dir
    output.mkdir(parents=True, exist_ok=True)
    png = output / "P5_LOCAL_PROPAGATION.png"
    pdf = output / "P5_LOCAL_PROPAGATION.pdf"
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
    png, pdf = make_figure(args.run_dir.resolve(), args.language, args.out_dir)
    print(json.dumps({"png": str(png), "pdf": str(pdf)}, indent=2))


if __name__ == "__main__":
    main()
