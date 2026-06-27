from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch


REPO_ROOT = Path(__file__).resolve().parents[2]
FIGURE_ROOT = REPO_ROOT / "figures"
SOURCE_DATA = FIGURE_ROOT / "source_data"
TABLES = SOURCE_DATA
UNIFIED = REPO_ROOT / "results" / "final_icse2027" / "unified_rerun"
HCMS = REPO_ROOT / "results" / "final_icse2027" / "high_complexity_multiseed"
OUT = FIGURE_ROOT / "paper"


def read_rows(name: str) -> list[dict[str, str]]:
    with (TABLES / name).open(newline="") as f:
        return list(csv.DictReader(f))


def read_unified(name: str) -> list[dict[str, str]]:
    with (UNIFIED / name).open(newline="") as f:
        return list(csv.DictReader(f))


def as_float(value: str) -> float:
    try:
        if value == "":
            return 0.0
        return float(value)
    except ValueError:
        return 0.0


def box(ax, xy, wh, text, fc="#eef6ff", ec="#376092", fs=8, lw=1.1):
    x, y = xy
    w, h = wh
    p = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.018,rounding_size=0.018",
        linewidth=lw,
        edgecolor=ec,
        facecolor=fc,
    )
    ax.add_patch(p)
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs)


def arrow(ax, start, end, color="#333333", style="solid"):
    ax.add_patch(
        FancyArrowPatch(
            start,
            end,
            arrowstyle="-|>",
            mutation_scale=10,
            linewidth=1.0,
            color=color,
            linestyle=style,
        )
    )


def fig_architecture():
    fig, ax = plt.subplots(figsize=(7.2, 3.65))
    ax.set_axis_off()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    ax.text(
        0.50,
        0.965,
        "From strategy proposal to auditable proof contract",
        ha="center",
        va="center",
        fontsize=9,
        weight="bold",
    )

    box(ax, (0.03, 0.70), (0.94, 0.20), "", "#ffffff", "#d9d9d9", fs=1, lw=0.8)
    ax.text(0.06, 0.875, "replaceable strategy sources", ha="left", va="center", fontsize=7.5, weight="bold")
    box(ax, (0.06, 0.755), (0.14, 0.075), "typed\nsearch", "#fff7df", "#bf9000", fs=6.6)
    box(ax, (0.23, 0.755), (0.14, 0.075), "GraphRAG\nretrieval", "#fff7df", "#bf9000", fs=6.6)
    box(ax, (0.40, 0.755), (0.14, 0.075), "one-shot\nstrategy", "#fff7df", "#bf9000", fs=6.6)
    box(
        ax,
        (0.58, 0.735),
        (0.24, 0.115),
        "path-planning agent\nmemory + diagnostics + repair",
        "#e2f0d9",
        "#548235",
        fs=6.8,
    )
    box(ax, (0.35, 0.60), (0.30, 0.075), "typed ProofStrategy\npaths + bounds + audit", "#d9ead3", "#548235", fs=6.8)

    box(ax, (0.03, 0.40), (0.17, 0.13), "typed facts\nEDB + schema\nsplits", "#f7f7f7", "#555555", fs=6.8)
    box(ax, (0.25, 0.40), (0.16, 0.13), "native ingest\n+ type check\n+ compile", "#edf2f9", "#376092", fs=6.8)
    box(ax, (0.46, 0.38), (0.18, 0.17), "symbolic verifier\ntrain gate\nadmit / reject", "#fce4d6", "#c55a11", fs=6.9, lw=1.5)
    box(ax, (0.70, 0.40), (0.17, 0.13), "bounded\nproof search\nheld-out q", "#e2f0d9", "#548235", fs=6.8)

    box(ax, (0.50, 0.16), (0.21, 0.13), "SUPPORTED\nrule r + proof tree\nIDB + EDB leaves", "#eaf3ff", "#376092", fs=6.6)
    box(ax, (0.76, 0.16), (0.21, 0.13), "INCONCLUSIVE\nmissing-support\nreview context", "#f2f2f2", "#666666", fs=6.6)
    box(
        ax,
        (0.17, 0.025),
        (0.66, 0.075),
        "audit / replay: proof consistency  ·  fact removal  ·  near-miss stress  ·  artifacts",
        "#fafafa",
        "#777777",
        fs=6.5,
    )

    arrow(ax, (0.13, 0.755), (0.40, 0.675), color="#548235")
    arrow(ax, (0.30, 0.755), (0.44, 0.675), color="#548235")
    arrow(ax, (0.47, 0.755), (0.49, 0.675), color="#548235")
    arrow(ax, (0.70, 0.735), (0.58, 0.675), color="#548235")
    arrow(ax, (0.50, 0.60), (0.33, 0.53))
    arrow(ax, (0.20, 0.465), (0.25, 0.465))
    arrow(ax, (0.41, 0.465), (0.46, 0.465))
    arrow(ax, (0.64, 0.465), (0.70, 0.465))
    arrow(ax, (0.78, 0.40), (0.61, 0.29))
    arrow(ax, (0.80, 0.40), (0.86, 0.29))
    arrow(ax, (0.60, 0.16), (0.42, 0.10), color="#777777")
    arrow(ax, (0.86, 0.16), (0.58, 0.10), color="#777777")
    arrow(ax, (0.64, 0.535), (0.76, 0.735), color="#777777", style="dashed")
    ax.text(0.75, 0.615, "reject +\ndiagnostics", ha="center", va="center", fontsize=6.0, color="#666666")
    fig.tight_layout(pad=0.4)
    fig.savefig(OUT / "fig_architecture.pdf")
    plt.close(fig)


def fig_case_package_dependency():
    fig, ax = plt.subplots(figsize=(7.2, 3.45))
    ax.set_axis_off()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    ax.text(
        0.50,
        0.965,
        "End-to-end case: cross-level package dependency proof",
        ha="center",
        va="center",
        fontsize=9,
        weight="bold",
    )

    box(
        ax,
        (0.025, 0.72),
        (0.22, 0.16),
        "Query\npackageCallDependency(Pa, Pb)\nPa: web.reactive.socket...\nPb: web.server",
        "#edf2f9",
        "#376092",
        fs=6.8,
    )
    box(
        ax,
        (0.025, 0.47),
        (0.22, 0.16),
        "Agent ProofStrategy\npackage -> class -> method\ncall -> method -> class -> package\n+ import guard",
        "#e2f0d9",
        "#548235",
        fs=6.5,
    )
    box(
        ax,
        (0.025, 0.20),
        (0.22, 0.16),
        "Symbolic admission\nseed0-2: 3/3 admitted\ntrain F1 = 1.000\nheld-out F1 = 1.000",
        "#fce4d6",
        "#c55a11",
        fs=6.6,
        lw=1.3,
    )

    node_fc = "#ffffff"
    node_ec = "#548235"
    nodes = [
        ("Package Pa", 0.35, 0.80),
        ("Class\nHandshakeWebSocketService", 0.35, 0.62),
        ("Method\ncreateHandshakeInfo(...)", 0.35, 0.44),
        ("Method\nServerWebExchange.getLogPrefix", 0.62, 0.44),
        ("Class\nServerWebExchange", 0.62, 0.62),
        ("Package Pb", 0.62, 0.80),
    ]
    for text, x, y in nodes:
        box(ax, (x - 0.105, y - 0.045), (0.21, 0.09), text, node_fc, node_ec, fs=6.2)

    arrow(ax, (0.35, 0.755), (0.35, 0.665), color="#548235")
    arrow(ax, (0.35, 0.575), (0.35, 0.485), color="#548235")
    arrow(ax, (0.455, 0.44), (0.515, 0.44), color="#548235")
    arrow(ax, (0.62, 0.485), (0.62, 0.575), color="#548235")
    arrow(ax, (0.62, 0.665), (0.62, 0.755), color="#548235")
    ax.text(0.355, 0.705, "containsClass", ha="left", va="center", fontsize=5.6, color="#4f7f31")
    ax.text(0.355, 0.525, "containsMethod", ha="left", va="center", fontsize=5.6, color="#4f7f31")
    ax.text(0.485, 0.465, "callsMethod", ha="center", va="bottom", fontsize=5.6, color="#4f7f31")
    ax.text(0.625, 0.525, "containsMethod", ha="left", va="center", fontsize=5.6, color="#4f7f31")
    ax.text(0.625, 0.705, "containsClass", ha="left", va="center", fontsize=5.6, color="#4f7f31")
    ax.add_patch(
        FancyArrowPatch(
            (0.455, 0.62),
            (0.515, 0.62),
            arrowstyle="-|>",
            mutation_scale=10,
            linewidth=1.1,
            color="#8064a2",
            linestyle="dashed",
            connectionstyle="arc3,rad=0.0",
        )
    )
    ax.text(0.485, 0.645, "importsClass guard", ha="center", va="bottom", fontsize=5.8, color="#8064a2")

    box(
        ax,
        (0.29, 0.16),
        (0.40, 0.13),
        "Admitted rule combines\npackage/class containment + method call + import guard\nContract: 6 EDB leaves; 3 evidence chains",
        "#fafafa",
        "#777777",
        fs=6.4,
    )

    box(
        ax,
        (0.73, 0.64),
        (0.24, 0.21),
        "SUPPORTED output\nfinite proof tree\nroot: packageCallDependency\nleaves: concrete EDB facts",
        "#eaf3ff",
        "#376092",
        fs=6.7,
    )
    box(
        ax,
        (0.73, 0.39),
        (0.24, 0.16),
        "Negative control\nINCONCLUSIVE\n0 proofs found\nreview obligation",
        "#f2f2f2",
        "#666666",
        fs=6.7,
    )
    box(
        ax,
        (0.73, 0.10),
        (0.24, 0.18),
        "Same hard-negative protocol\nAgent: 3/3, F1=1.000\nGraphRAG: 1/3, F1=.855\nTyped: 0/3 admitted",
        "#fff7df",
        "#bf9000",
        fs=6.4,
    )

    arrow(ax, (0.245, 0.80), (0.245, 0.55), color="#777777")
    arrow(ax, (0.245, 0.55), (0.245, 0.28), color="#777777")
    arrow(ax, (0.245, 0.28), (0.30, 0.235), color="#777777")
    arrow(ax, (0.67, 0.235), (0.73, 0.745), color="#376092")
    arrow(ax, (0.67, 0.235), (0.73, 0.47), color="#666666", style="dashed")

    ax.text(0.50, 0.075, "The agent proposes a multi-predicate path; symbolic admission decides; proof search returns a contract.", ha="center", va="center", fontsize=7.0)
    fig.tight_layout(pad=0.35)
    fig.savefig(OUT / "fig_case_package_dependency.pdf")
    plt.close(fig)


def fig_agent_evidence():
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 4.3))
    ax = axes[0, 0]
    hc_rows = read_csv_path(HCMS / "high_complexity_multiseed_summary.csv")
    hc_targets = ["signatureSafeOverride", "packageCallDependency", "interfaceOverride"]
    hc_labels = ["signature", "package", "interface"]
    hc_arms = ["proof_strategy", "heuristic", "graphrag"]
    hc_arm_labels = ["agent", "typed", "GraphRAG"]
    hc = {(r["target"], r["arm"]): r for r in hc_rows}
    values = []
    for target in hc_targets:
        row_values = []
        for arm in hc_arms:
            item = hc.get((target, arm), {})
            row_values.append(as_float(item.get("admission_rate", "")))
        values.append(row_values)
    ax.imshow(values, cmap="YlGn", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(len(hc_arms)))
    ax.set_xticklabels(hc_arm_labels, fontsize=7.5)
    ax.set_yticks(range(len(hc_targets)))
    ax.set_yticklabels(hc_labels, fontsize=7.5)
    ax.set_title("(a) High-complexity stress matrix", fontsize=8.5)
    ax.tick_params(length=0)
    for y, target in enumerate(hc_targets):
        for x, arm in enumerate(hc_arms):
            item = hc.get((target, arm), {})
            admitted = item.get("admitted", "0")
            seeds = item.get("seeds", "3")
            f1_text = item.get("train_f1_mean", "")
            admission = as_float(item.get("admission_rate", ""))
            color = "white" if admission >= 0.72 else "#222222"
            label = f"{admitted}/{seeds}"
            if f1_text:
                label += f"\n{as_float(f1_text):.2f}"
            ax.text(x, y, label, ha="center", va="center", fontsize=7, color=color)
    for spine in ax.spines.values():
        spine.set_visible(False)

    agent_rows = read_rows("table3_agent_ablation_summary.csv")
    groups = {"cold_one_shot": 0, "cold_loop": 0, "memory_one_shot": 0, "memory_loop": 0}
    runs = {"cold_one_shot": 0, "cold_loop": 0, "memory_one_shot": 0, "memory_loop": 0}
    for r in agent_rows:
        arm = r["arm"]
        if arm in groups:
            groups[arm] += int(r["admitted"])
            runs[arm] += int(r["runs"])
    ax = axes[0, 1]
    matrix_keys = [
        ["cold_one_shot", "cold_loop"],
        ["memory_one_shot", "memory_loop"],
    ]
    values = [
        [groups[k] / runs[k] if runs[k] else 0 for k in row]
        for row in matrix_keys
    ]
    ax.imshow(values, cmap="Greens", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["one-shot", "loop"], fontsize=8)
    ax.set_yticks([0, 1])
    ax.set_yticklabels(["cold", "memory"], fontsize=8)
    ax.set_title("(b) Memory x control mode", fontsize=8.5)
    ax.tick_params(length=0)
    for y, row in enumerate(matrix_keys):
        for x, key in enumerate(row):
            rate = values[y][x]
            color = "white" if rate >= 0.72 else "#222222"
            ax.text(x, y - 0.08, f"{groups[key]}/{runs[key]}", ha="center", va="center", fontsize=9, color=color)
            ax.text(x, y + 0.16, f"{rate:.2f}", ha="center", va="center", fontsize=7.5, color=color)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_xlim(-0.5, 2.12)
    ax.text(
        1.78,
        0.50,
        "repair case\nreject -> admit\nF1=.936",
        ha="center",
        va="center",
        fontsize=7,
        bbox=dict(boxstyle="round,pad=0.18", facecolor="#ffffff", edgecolor="#888888", linewidth=0.7),
    )

    ax = axes[1, 0]
    models = ["Qwen", "GPT-4o", "DeepSeek", "Gemini", "PARA"]
    train_f1 = [0.0, 0.980, 0.980, 0.980, 1.000]
    held_recall = [0.0, 0.220, 0.220, 0.913, 1.000]
    xx = range(len(models))
    ax.bar([i - 0.17 for i in xx], train_f1, width=0.32, color="#9bbb59", label="train F1")
    ax.bar([i + 0.17 for i in xx], held_recall, width=0.32, color="#8064a2", label="held-out recall")
    ax.set_xticks(list(xx))
    ax.set_xticklabels(models, fontsize=7, rotation=15)
    ax.set_ylim(0, 1.10)
    ax.set_title("(c) Direct free-form rules on overridesMethod", fontsize=8.5)
    ax.legend(fontsize=7, loc="upper left")
    ax.grid(axis="y", alpha=0.25)
    for i, r in enumerate(held_recall):
        ax.text(i + 0.17, min(r + 0.035, 1.04), f"{r:.2f}", ha="center", fontsize=6.5)

    ax = axes[1, 1]
    ks = [1, 2, 3, 4, 5]
    recalls = [0.687, 0.687, 0.687, 1.000, 1.000]
    ax.plot(ks, recalls, marker="o", color="#4f81bd", linewidth=1.8)
    ax.fill_between(ks, recalls, [0] * len(ks), color="#4f81bd", alpha=0.12)
    ax.set_xticks(ks)
    ax.set_ylim(0, 1.05)
    ax.set_xlabel("portfolio size k", fontsize=8)
    ax.set_ylabel("canCallClass recall", fontsize=8)
    ax.set_title("(d) Portfolio effect under same verifier", fontsize=8.5)
    ax.grid(axis="y", alpha=0.25)
    for k, r in zip(ks, recalls):
        ax.text(k, r + 0.015, f"{r:.3f}", ha="center", fontsize=7)
    fig.tight_layout(pad=0.5)
    fig.savefig(OUT / "fig_agent_evidence.pdf")
    plt.close(fig)


def fig_unified_matrix():
    rows = read_unified("unified_core_summary_by_target_method.csv")
    methods = ["proof_strategy", "graphrag", "typed_heuristic"]
    targets = ["canCallClass", "isAllowedToUse", "overridesMethod", "packageCallDependency"]
    labels = ["canCall", "isAllowed", "overrides", "package"]
    data = {(r["target"], r["method"]): r for r in rows}
    metrics = [
        ("A", "admission_rate", "#4f81bd"),
        ("R", "mean_heldout_recall", "#8064a2"),
        ("F", "mean_best_candidate_f1", "#70ad47"),
    ]
    fig, ax = plt.subplots(figsize=(7.2, 3.0))
    ax.set_xlim(0, len(methods))
    ax.set_ylim(0, len(targets))
    ax.invert_yaxis()
    ax.set_xticks([i + 0.5 for i in range(len(methods))])
    ax.set_xticklabels(["agent", "GraphRAG", "typed"], fontsize=8)
    ax.set_yticks([i + 0.5 for i in range(len(targets))])
    ax.set_yticklabels(labels, fontsize=8)
    ax.tick_params(length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)

    for y, target in enumerate(targets):
        for x, method in enumerate(methods):
            ax.add_patch(plt.Rectangle((x, y), 1, 1, facecolor="#fafafa", edgecolor="#d9d9d9", linewidth=0.8))
            row = data.get((target, method), {})
            for m, (short, key, color) in enumerate(metrics):
                value_text = row.get(key, "")
                value = as_float(value_text)
                band_y = y + m / 3
                alpha = 0.12 + 0.72 * value if value_text != "" else 0.02
                ax.add_patch(
                    plt.Rectangle(
                        (x + 0.02, band_y + 0.02),
                        0.96,
                        1 / 3 - 0.035,
                        facecolor=color,
                        edgecolor="none",
                        alpha=alpha,
                    )
                )
                shown = "--" if value_text == "" else f"{value:.2f}"
                txt_color = "white" if value >= 0.72 and value_text != "" else "#222222"
                ax.text(x + 0.50, band_y + 0.17, f"{short} {shown}", ha="center", va="center", fontsize=7.2, color=txt_color)
    ax.set_title("Unified evidence matrix (three seeds per cell)", fontsize=9.5, pad=14)
    legend_x = 0.03
    for short, label, color in [("A", "admission", "#4f81bd"), ("R", "held-out recall", "#8064a2"), ("F", "candidate F1", "#70ad47")]:
        ax.text(legend_x, 1.055, f"{short}={label}", transform=ax.transAxes, ha="left", va="center", fontsize=7.4, color=color)
        legend_x += 0.25
    ax.set_xlabel("strategy source", fontsize=8)
    ax.set_ylabel("target", fontsize=8)
    fig.tight_layout(pad=0.55)
    fig.savefig(OUT / "fig_unified_matrix.pdf")
    plt.close(fig)


def read_csv_path(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def fig_audit_layers():
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.75), gridspec_kw={"width_ratios": [1.0, 1.05, 1.25]})

    ax = axes[0]
    labels = ["trees", "EDB", "exact\nneg.", "arg\nneg.", "rec.\ntrees", "rec.\nEDB"]
    counts = [50, 190, 554, 705, 250, 756]
    colors = ["#4f81bd", "#4f81bd", "#c0504d", "#c0504d", "#9bbb59", "#9bbb59"]
    ax.bar(range(len(labels)), counts, color=colors)
    ax.set_yscale("log")
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, fontsize=7)
    ax.set_title("(a) Audit coverage; failures=0\nnot one comparable metric scale", fontsize=8.0)
    ax.set_ylabel("count (log)", fontsize=8)
    ax.grid(axis="y", alpha=0.25)
    for i, c in enumerate(counts):
        ax.text(i, c * 1.08, str(c), ha="center", fontsize=7)

    ax = axes[1]
    targets = ["canCall", "isAllowed", "overrides"]
    critical = [0.667, 0.167, 1.000]
    alternative = [0.333, 0.833, 0.000]
    all_alt = [0.200, 0.625, 0.000]
    x = range(len(targets))
    ax.bar([i - 0.22 for i in x], critical, width=0.22, label="critical", color="#c0504d")
    ax.bar(x, alternative, width=0.22, label="alternative", color="#4f81bd")
    ax.bar([i + 0.22 for i in x], all_alt, width=0.22, label="all-alt", color="#9bbb59")
    ax.set_xticks(list(x))
    ax.set_xticklabels(targets, fontsize=7, rotation=20)
    ax.set_ylim(0, 1.10)
    ax.set_title("(b) Fact-removal sensitivity", fontsize=8.5)
    ax.set_ylabel("support lost", fontsize=8)
    ax.legend(fontsize=6.5, loc="upper center", ncol=1)
    ax.grid(axis="y", alpha=0.25)

    ax = axes[2]
    qtargets = ["package", "overrides", "canCall"]
    precision = [1.0, 1.0, 1.0]
    recall = [1.0, 1.0, 0.65]
    nn = [1.0, 1.0, 1.0]
    x = range(len(qtargets))
    ax.bar([i - 0.22 for i in x], precision, width=0.22, label="precision", color="#4f81bd")
    ax.bar(x, recall, width=0.22, label="recall", color="#8064a2")
    ax.bar([i + 0.22 for i in x], nn, width=0.22, label="neg. non-support", color="#9bbb59")
    ax.set_xticks(list(x))
    ax.set_xticklabels(qtargets, fontsize=7, rotation=20)
    ax.set_ylim(0, 1.10)
    ax.set_title("(c) Sample query quality", fontsize=8.5)
    ax.legend(fontsize=6.5, loc="lower left")
    ax.grid(axis="y", alpha=0.25)
    for i, r in enumerate(recall):
        ax.text(i, r + 0.025, f"{r:.2f}", ha="center", fontsize=7)
    fig.tight_layout(pad=0.45)
    fig.savefig(OUT / "fig_audit_layers.pdf")
    plt.close(fig)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    fig_architecture()
    fig_case_package_dependency()
    fig_unified_matrix()
    fig_agent_evidence()
    fig_audit_layers()


if __name__ == "__main__":
    main()
