#!/usr/bin/env python3
"""Generate PARA paper figures from checked-in source data."""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "source_data"
OUT = ROOT / "generated"


plt.rcParams.update(
    {
        "font.size": 9,
        "font.family": "DejaVu Sans",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.color": "#e6e6e6",
        "grid.linewidth": 0.7,
        "axes.axisbelow": True,
        "figure.dpi": 160,
        "savefig.bbox": "tight",
    }
)


def read_csv(name: str) -> list[dict[str, str]]:
    with (DATA / name).open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def save(fig: plt.Figure, stem: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / f"{stem}.svg")
    fig.savefig(OUT / f"{stem}.pdf")
    plt.close(fig)


def box(ax, xy, w, h, text, fc="#eef4fb", ec="#315a7d", fontsize=8.5):
    patch = FancyBboxPatch(
        xy,
        w,
        h,
        boxstyle="round,pad=0.018,rounding_size=0.025",
        linewidth=1.1,
        edgecolor=ec,
        facecolor=fc,
    )
    ax.add_patch(patch)
    ax.text(xy[0] + w / 2, xy[1] + h / 2, text, ha="center", va="center", fontsize=fontsize)


def arrow(ax, start, end, color="#4a4a4a", rad=0.0):
    ax.add_patch(
        FancyArrowPatch(
            start,
            end,
            arrowstyle="-|>",
            mutation_scale=11,
            linewidth=1.0,
            color=color,
            connectionstyle=f"arc3,rad={rad}",
        )
    )


def fig_protocol() -> None:
    fig, ax = plt.subplots(figsize=(11.2, 4.25))
    ax.set_axis_off()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    ax.text(0.02, 0.955, "PARA: Path Accountable Reasoning", fontsize=13, weight="bold")
    ax.text(
        0.02,
        0.905,
        "LLM plans executable path programs; symbolic layers verify, prove, and audit",
        fontsize=9,
        color="#555555",
    )

    # Inputs and learning lane.
    ax.text(0.03, 0.84, "Learning stage", fontsize=10, weight="bold", color="#3f7f5f")
    y_top = 0.69
    box(ax, (0.03, y_top), 0.12, 0.12, "Project graph\nEDB + schema", "#f5f1e8", "#8c6d31", 7.5)
    box(ax, (0.18, y_top), 0.12, 0.12, "Train split\npos. + neg.", "#f5f1e8", "#8c6d31", 7.5)
    box(ax, (0.33, y_top), 0.13, 0.12, "Witness\nobserver", "#edf7ef", "#3f7f5f", 7.7)
    box(ax, (0.50, y_top), 0.15, 0.12, "LLM Planner /\nRefiner", "#edf7ef", "#3f7f5f", 7.7)
    box(ax, (0.69, y_top), 0.15, 0.12, "Path-program\nportfolio", "#edf7ef", "#3f7f5f", 7.7)
    arrow(ax, (0.15, y_top + 0.06), (0.18, y_top + 0.06), color="#8c6d31")
    arrow(ax, (0.30, y_top + 0.06), (0.33, y_top + 0.06), color="#3f7f5f")
    arrow(ax, (0.46, y_top + 0.06), (0.50, y_top + 0.06), color="#3f7f5f")
    arrow(ax, (0.65, y_top + 0.06), (0.69, y_top + 0.06), color="#3f7f5f")
    arrow(ax, (0.09, y_top), (0.385, y_top + 0.015), color="#8c6d31", rad=-0.18)

    # Symbolic acceptance lane.
    y_mid = 0.48
    box(ax, (0.22, y_mid), 0.115, 0.12, "Type-chain\nvalidator", "#eef4fb", "#315a7d", 7.3)
    box(ax, (0.365, y_mid), 0.115, 0.12, "Indexed\nexecutor", "#eef4fb", "#315a7d", 7.3)
    box(ax, (0.51, y_mid), 0.115, 0.12, "Candidate\ncompiler", "#eef4fb", "#315a7d", 7.3)
    box(ax, (0.655, y_mid), 0.115, 0.12, "Symbolic\nverifier", "#eef4fb", "#315a7d", 7.3)
    box(ax, (0.80, y_mid), 0.13, 0.12, "Run-local\nIDB rules", "#e8f4ee", "#3f7f5f", 7.3)
    arrow(ax, (0.765, y_top), (0.278, y_mid + 0.12), color="#3f7f5f", rad=-0.22)
    arrow(ax, (0.335, y_mid + 0.06), (0.365, y_mid + 0.06), color="#315a7d")
    arrow(ax, (0.48, y_mid + 0.06), (0.51, y_mid + 0.06), color="#315a7d")
    arrow(ax, (0.625, y_mid + 0.06), (0.655, y_mid + 0.06), color="#315a7d")
    arrow(ax, (0.77, y_mid + 0.06), (0.80, y_mid + 0.06), color="#315a7d")
    ax.text(
        0.46,
        0.625,
        "strict contract: no direct-rule channel; no deterministic BFS fallback",
        fontsize=7.5,
        color="#7a3d3d",
        ha="center",
    )

    # Reasoning lane.
    ax.text(0.03, 0.36, "Reasoning stage", fontsize=10, weight="bold", color="#315a7d")
    y_low = 0.23
    box(ax, (0.03, y_low), 0.12, 0.12, "Held-out\nquery R(a,b)", "#f5f1e8", "#8c6d31", 7.5)
    box(ax, (0.20, y_low), 0.13, 0.12, "Indexed\nEDB facts", "#eef4fb", "#315a7d", 7.6)
    box(ax, (0.38, y_low), 0.14, 0.12, "Bounded\nproof search", "#eef4fb", "#315a7d", 7.6)
    box(ax, (0.57, y_low), 0.14, 0.12, "Proof trace\nEDB + IDB", "#eef4fb", "#315a7d", 7.6)
    box(ax, (0.77, y_low), 0.16, 0.12, "SUPPORTED\nor INCONCLUSIVE", "#f2edf7", "#6c4c8c", 7.6)
    arrow(ax, (0.15, y_low + 0.06), (0.20, y_low + 0.06), color="#315a7d")
    arrow(ax, (0.33, y_low + 0.06), (0.38, y_low + 0.06), color="#315a7d")
    arrow(ax, (0.52, y_low + 0.06), (0.57, y_low + 0.06), color="#315a7d")
    arrow(ax, (0.71, y_low + 0.06), (0.77, y_low + 0.06), color="#315a7d")
    arrow(ax, (0.865, y_mid), (0.45, y_low + 0.12), color="#3f7f5f", rad=-0.18)
    arrow(ax, (0.09, y_top), (0.265, y_low + 0.12), color="#8c6d31", rad=0.22)

    # Accountability lane.
    ax.text(0.37, 0.135, "Accountability checks", fontsize=10, weight="bold", color="#9a5b36")
    box(ax, (0.32, 0.02), 0.12, 0.10, "Fact audit\nin BK", "#fff4ef", "#9a5b36", 7.4)
    box(ax, (0.49, 0.02), 0.13, 0.10, "Counterfactual\nfact ablation", "#fff4ef", "#9a5b36", 7.2)
    box(ax, (0.67, 0.02), 0.13, 0.10, "Near-miss /\nintegrity audit", "#fff4ef", "#9a5b36", 7.2)
    arrow(ax, (0.64, y_low), (0.38, 0.12), color="#9a5b36", rad=0.13)
    arrow(ax, (0.64, y_low), (0.555, 0.12), color="#9a5b36", rad=0.03)
    arrow(ax, (0.64, y_low), (0.735, 0.12), color="#9a5b36", rad=-0.13)

    save(fig, "fig1_para_protocol_architecture")


def fig_reasoning_accountability() -> None:
    rows = read_csv("spring_multiseed_reasoning.csv")
    targets = [r["target"] for r in rows]
    accuracy = np.array([float(r["held_out_accuracy_mean"]) for r in rows])
    recall = np.array([float(r["supported_recall_mean"]) for r in rows])
    abstention = np.array([float(r["positive_abstention_mean"]) for r in rows])
    baseline = float(rows[0]["no_rule_accuracy"])

    audit_rows = read_csv("proof_audit_counterfactual.csv")
    critical = np.array([float(r["critical_fact_rate"]) for r in audit_rows])
    alternative = np.array([float(r["alternative_fact_rate"]) for r in audit_rows])
    fig, ax = plt.subplots(figsize=(9.6, 3.05))
    y = np.arange(len(targets))
    h = 0.12
    offsets = np.array([2, 1, 0, -1, -2]) * h
    metrics = [
        ("Held-out acc.", accuracy, "#2c7fb8"),
        ("Supported recall", recall, "#41ab5d"),
        ("Positive abstention", abstention, "#fdae6b"),
        ("Critical fact rate", critical, "#756bb1"),
        ("Alternative proof rate", alternative, "#bcbddc"),
    ]

    for offset, (label, values, color) in zip(offsets, metrics):
        ax.barh(y + offset, values, height=h * 0.88, color=color, label=label)

    ax.scatter(
        np.full(len(targets), baseline),
        y + offsets[0],
        marker="|",
        s=150,
        linewidths=1.3,
        color="#555555",
        zorder=4,
        label="No-rule accuracy = 0.667",
    )
    ax.set_xlim(0, 1.05)
    ax.set_yticks(y, targets)
    ax.set_xlabel("")
    ax.set_title("Spring xlarge3: reasoning quality and proof sensitivity", weight="bold", fontsize=10.2)

    for vals, yy in [(accuracy, y + offsets[0]), (recall, y + offsets[1]), (abstention, y + offsets[2])]:
        for value, yi in zip(vals, yy):
            ax.text(min(value + 0.012, 1.01), yi, f"{value:.3f}", va="center", fontsize=6.7)

    for vals, yy in [(critical, y + offsets[3]), (alternative, y + offsets[4])]:
        for value, yi in zip(vals, yy):
            if value >= 0.12:
                ax.text(
                    value / 2,
                    yi,
                    f"{value:.2f}",
                    va="center",
                    ha="center",
                    fontsize=6.6,
                    color="white" if value > 0.55 else "#333333",
                )

    ax.tick_params(axis="both", labelsize=7.2)
    ax.legend(frameon=False, fontsize=6.7, loc="lower center", bbox_to_anchor=(0.5, -0.24), ncol=6)
    fig.subplots_adjust(bottom=0.26, top=0.84, left=0.13, right=0.98)
    save(fig, "fig2_reasoning_accountability")


def fig_spring_multiseed() -> None:
    """Legacy standalone panel kept for appendix/debugging."""
    rows = read_csv("spring_multiseed_reasoning.csv")
    targets = [r["target"] for r in rows]
    accuracy = np.array([float(r["held_out_accuracy_mean"]) for r in rows])
    recall = np.array([float(r["supported_recall_mean"]) for r in rows])
    abstention = np.array([float(r["positive_abstention_mean"]) for r in rows])
    baseline = float(rows[0]["no_rule_accuracy"])

    fig, ax = plt.subplots(figsize=(7.8, 3.55))
    y = np.arange(len(targets))
    h = 0.23
    ax.barh(y + h, accuracy, height=h, color="#2c7fb8", label="Held-out acc.")
    ax.barh(y, recall, height=h, color="#41ab5d", label="Supported recall")
    ax.barh(y - h, abstention, height=h, color="#fdae6b", label="Positive abstention (lower is better)")
    ax.axvline(baseline, color="#555555", linestyle="--", linewidth=1.1, label="No-rule acc. = 0.667")
    ax.set_xlim(0, 1.05)
    ax.set_yticks(y, targets)
    ax.set_xlabel("Metric value")
    ax.set_title("Spring xlarge3 held-out reasoning", weight="bold")
    ax.legend(frameon=False, fontsize=7.6, loc="lower center", bbox_to_anchor=(0.5, -0.25), ncol=2)
    save(fig, "fig2_spring_multiseed_reasoning")


def fig_direct_ablation() -> None:
    rows = read_csv("direct_query_answering_matched.csv")
    rule_rows = read_csv("direct_rule_generation_ablation.csv")
    models = ["Qwen3.5-27B", "DeepSeek V4 Pro"]
    targets = ["canCallClass", "isAllowedToUse", "overridesMethod"]
    lookup = {(r["model"], r["target"], r["method"]): r for r in rows}

    records = []
    for target in targets:
        for model in models:
            d = float(lookup[(model, target, "Direct QA")]["supported_recall"])
            p = float(lookup[("Qwen3.5-27B", target, "Strict PARA path-program")]["supported_recall"])
            records.append((target, model, d, p, p - d))

    display_records = records[::-1]
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(10.6, 4.15), gridspec_kw={"width_ratios": [1.22, 1]})
    y = np.arange(len(display_records))
    target_colors = {
        "canCallClass": "#2c7fb8",
        "isAllowedToUse": "#41ab5d",
        "overridesMethod": "#756bb1",
    }

    for yi, (target, model, direct, para, gain) in zip(y, display_records):
        color = target_colors[target]
        ax.plot([direct, para], [yi, yi], color="#bdbdbd", linewidth=1.7, zorder=1)
        ax.scatter(direct, yi, marker="o", s=46, facecolor="white", edgecolor=color, linewidth=1.4, zorder=2)
        ax.scatter(para, yi, marker="s", s=48, color=color, zorder=3)
        if abs(gain) > 0.015:
            label_x = min(max((direct + para) / 2, 0.06), 0.94)
            ax.text(
                label_x,
                yi,
                f"{gain:+.2f}",
                va="center",
                ha="center",
                fontsize=7.4,
                color="#333333",
                bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.72, "pad": 0.6},
            )

    labels = [f"{target} | {model}" for target, model, _, _, _ in display_records]
    ax.set_yticks(y, labels, fontsize=7.2)
    ax.set_xlim(-0.02, 1.05)
    ax.set_xlabel("Held-out supported recall")
    ax.set_title("(a) Direct query answering", weight="bold", fontsize=10.2)
    ax.axvline(1.0, color="#555555", linestyle=":", linewidth=1)
    ax.scatter([], [], marker="o", s=46, facecolor="white", edgecolor="#555555", label="Direct QA")
    ax.scatter([], [], marker="s", s=48, color="#555555", label="PARA path program")
    ax.legend(frameon=False, ncol=2, loc="upper left", fontsize=7.4)

    rule_models = ["Qwen3.5-27B", "GPT-4o", "DeepSeek Chat", "Gemini 2.5 Pro", "PARA path program"]
    rule_targets = targets
    rule_lookup = {(r["model"], r["target"]): r for r in rule_rows}
    matrix = []
    for model in rule_models:
        row = []
        for target in rule_targets:
            if model == "PARA path program":
                row.append(float(rule_lookup[("Qwen3.5-27B", target)]["para_recall"]))
            else:
                row.append(float(rule_lookup[(model, target)]["direct_rule_recall"]))
        matrix.append(row)
    matrix = np.array(matrix)

    im = ax2.imshow(matrix, cmap="YlGnBu", vmin=0, vmax=1, aspect="auto")
    ax2.set_title("(b) Free-form rule generation", weight="bold", fontsize=10.2)
    ax2.set_xticks(np.arange(len(rule_targets)), ["canCall", "isAllowed", "override"], rotation=25, ha="right")
    ax2.set_yticks(np.arange(len(rule_models)), rule_models, fontsize=7.2)
    ax2.grid(False)
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            value = matrix[i, j]
            ax2.text(
                j,
                i,
                f"{value:.3f}" if value not in (0.0, 1.0) else f"{value:.1f}",
                ha="center",
                va="center",
                fontsize=7.1,
                color="white" if value > 0.62 else "#222222",
            )
    for spine in ax2.spines.values():
        spine.set_visible(False)
    cbar = fig.colorbar(im, ax=ax2, fraction=0.045, pad=0.025)
    cbar.ax.tick_params(labelsize=7)
    cbar.set_label("Held-out supported recall", fontsize=7.5)
    fig.suptitle("Direct LLM interfaces versus PARA path-program proofs", weight="bold", fontsize=11.2)
    fig.subplots_adjust(wspace=0.38, bottom=0.18, top=0.85)
    save(fig, "fig3_direct_vs_path_contract")


def fig_accountability() -> None:
    rows = read_csv("proof_audit_counterfactual.csv")
    targets = [r["target"] for r in rows]
    critical = [float(r["critical_fact_rate"]) for r in rows]
    alternative = [float(r["alternative_fact_rate"]) for r in rows]
    edb = [int(r["edb_fact_mentions"]) for r in rows]
    errors = [int(r["audit_errors"]) for r in rows]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(3.55, 4.75), gridspec_kw={"height_ratios": [1, 1]})
    y = np.arange(len(targets))
    ax1.barh(y, critical, label="Critical fact rate", color="#756bb1")
    ax1.barh(y, alternative, left=critical, label="Alternative proof rate", color="#bcbddc")
    ax1.set_xlim(0, 1.05)
    ax1.set_yticks(y, targets)
    ax1.set_xlabel("Outcome rate")
    ax1.set_title("(a) Counterfactual proof sensitivity", weight="bold", fontsize=9.2)

    ax2.barh(y, edb, color="#2c7fb8", label="EDB fact mentions")
    ax2.scatter(errors, y, color="#d73027", s=45, zorder=3, label="Audit errors")
    ax2.set_yticks(y, targets)
    ax2.set_xlabel("Count")
    ax2.set_title("(b) Fact-grounded proof audit", weight="bold", fontsize=9.2)
    for yi, value in zip(y, edb):
        ax2.text(value + 2, yi, str(value), va="center", fontsize=8)

    fig.suptitle("Accountability evidence", weight="bold", fontsize=9.5)
    fig.subplots_adjust(hspace=0.72)
    save(fig, "fig4_accountability_audit_counterfactual")


def main() -> None:
    fig_protocol()
    fig_reasoning_accountability()
    fig_spring_multiseed()
    fig_direct_ablation()
    fig_accountability()
    print(f"Wrote figures to {OUT}")


if __name__ == "__main__":
    main()
