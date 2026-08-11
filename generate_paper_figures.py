"""
Generate paper-ready tables and figures from ami_cells_v2.json (the
recalibrated, self-referential-threshold results).

Usage:
    python generate_paper_figures.py ./kaggle_output/ami_cells_v2.json ./paper_figures

Produces, in the output directory:
    table1_main_results.csv / .md / .tex   -- full per-cell results table
    table2_l2_vs_l3_partition.csv / .md    -- the "count isn't everything" evidence
    fig1_granularity_curve.png              -- THE signature figure
    fig2_partition_effect.png               -- L2 vs L3 bar comparison
    fig3_ami_vs_sigma.png                   -- exploratory covariate scatter
    fig4_methodology_pipeline.png           -- process/pipeline explainer diagram

All figures render at 300 DPI, sized for a 2-column ACM sigconf paper
(single-column width ~3.4in, double-column ~7.0in).
"""

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import pandas as pd
import numpy as np
from scipy import stats

# ---------------------------------------------------------------------------
# Style: clean, colorblind-safe, print-friendly
# ---------------------------------------------------------------------------
plt.rcParams.update({
    "font.size": 9,
    "font.family": "serif",
    "axes.edgecolor": "#333333",
    "axes.linewidth": 0.8,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "grid.linewidth": 0.5,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
})

MODEL_COLORS = {
    "llama-3.2-3b": "#1f77b4",
    "mistral-7b": "#d62728",
    "qwen2.5-7b": "#2ca02c",
}
MODEL_LABELS = {
    "llama-3.2-3b": "Llama-3.2-3B",
    "mistral-7b": "Mistral-7B",
    "qwen2.5-7b": "Qwen2.5-7B",
}

CASTE_LEVEL_ORDER = [
    ("l1_group", "Binary", 1),
    ("l2_group", "Varna", 2),
    ("l3_group", "Constitutional", 3),
    ("l4_group", "Jati\n(fine-grained)", 4),
]
SANSKRITI_LEVEL_ORDER = [
    ("l1_group", "Pooled\n(All-India)", 1),
    ("l2_group", "Macro-region", 2),
    ("l3_group", "State/UT", 3),
]


def load_cells(path):
    with open(path) as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# TABLE 1: Main results table
# ---------------------------------------------------------------------------

def make_table1(cells, outdir):
    level_names = {
        ("indicasa_caste", "l1_group"): "Binary", ("indicasa_caste", "l2_group"): "Varna",
        ("indicasa_caste", "l3_group"): "Constitutional", ("indicasa_caste", "l4_group"): "Jati",
        ("bhed_religion", "l1_group"): "Religion (mixed-n)",
        ("sanskriti", "l2_group"): "Macro-region", ("sanskriti", "l3_group"): "State/UT",
    }
    rows = []
    for c in cells:
        rows.append({
            "Dataset": c["dataset"],
            "Level": level_names.get((c["dataset"], c["level"]), c["level"]),
            "Model": MODEL_LABELS.get(c["model"], c["model"]),
            "n groups": c["n_groups"],
            "Pooled": round(c["pooled_rate"], 3),
            "Worst group": c["worst_group"],
            "Worst rate": round(c["worst_rate"], 3),
            "AMI": round(c["ami"], 3),
            "sigma": round(c["sigma_between"], 3),
            "Deviation flagged": "Yes" if c.get("flipped") else "No",
        })
    df = pd.DataFrame(rows)
    df.to_csv(outdir / "table1_main_results.csv", index=False)
    with open(outdir / "table1_main_results.md", "w") as f:
        f.write(df.to_markdown(index=False))
    with open(outdir / "table1_main_results.tex", "w") as f:
        f.write(df.to_latex(index=False, caption="Per-cell results: pooled vs. worst-group rate, "
                             "Aggregation Masking Index (AMI), and self-referential subgroup-deviation "
                             "flag across all (model, dataset, granularity level) cells.",
                             label="tab:main_results", escape=True))
    print(f"Table 1 saved ({len(df)} rows): table1_main_results.{{csv,md,tex}}")
    return df


# ---------------------------------------------------------------------------
# TABLE 2: The L2 vs L3 partition-effect evidence (headline supporting table)
# ---------------------------------------------------------------------------

def make_table2(cells, outdir):
    rows = []
    for model in MODEL_LABELS:
        l2 = next(c for c in cells if c["model"] == model and c["dataset"] == "indicasa_caste" and c["level"] == "l2_group")
        l3 = next(c for c in cells if c["model"] == model and c["dataset"] == "indicasa_caste" and c["level"] == "l3_group")
        rows.append({
            "Model": MODEL_LABELS[model],
            "Varna AMI (n=4)": round(l2["ami"], 3),
            "Constitutional AMI (n=4)": round(l3["ami"], 3),
            "Ratio": round(l2["ami"] / l3["ami"], 2) if l3["ami"] > 0 else float("nan"),
        })
    df = pd.DataFrame(rows)
    df.to_csv(outdir / "table2_l2_vs_l3_partition.csv", index=False)
    with open(outdir / "table2_l2_vs_l3_partition.md", "w") as f:
        f.write(df.to_markdown(index=False))
    print(f"Table 2 saved: table2_l2_vs_l3_partition.{{csv,md}}")
    return df


# ---------------------------------------------------------------------------
# FIGURE 1: Granularity Sensitivity Curve -- THE signature figure
# ---------------------------------------------------------------------------

def fig1_granularity_curve(cells, outdir):
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.8))

    # Panel A: IndiCASA caste
    ax = axes[0]
    for model in MODEL_LABELS:
        xs, ys = [], []
        for col, label, x in CASTE_LEVEL_ORDER:
            cell = next((c for c in cells if c["model"] == model and c["dataset"] == "indicasa_caste"
                         and c["level"] == col), None)
            if cell:
                xs.append(x)
                ys.append(cell["ami"])
        ax.plot(xs, ys, marker="o", markersize=5, linewidth=1.6,
                color=MODEL_COLORS[model], label=MODEL_LABELS[model])
    ax.set_xticks([x for _, _, x in CASTE_LEVEL_ORDER])
    ax.set_xticklabels([label for _, label, _ in CASTE_LEVEL_ORDER], fontsize=7.5)
    ax.set_ylabel("Aggregation Masking Index (AMI)")
    ax.set_title("(a) IndiCASA caste", fontsize=9)
    ax.set_ylim(-0.02, 0.65)
    # HONEST framing: Varna and Constitutional are ALTERNATIVE partitions at
    # similar cardinality (both n=4), not sequential steps of one nested
    # hierarchy -- unlike SANSKRITI's genuinely nested pooled->region->state.
    # Plotting them on one x-axis is visually convenient but the resulting
    # zigzag is real, not an error -- see Fig. 2 for the dedicated comparison.
    ax.annotate("Varna & Constitutional are\nALTERNATIVE partitions (both n=4),\nnot sequential steps -- see Fig. 2",
                xy=(2.5, 0.58), fontsize=6, ha="center", style="italic", color="#555555")

    # Panel B: SANSKRITI (add synthetic AMI=0 pooled point -- mathematically
    # certain when n_groups=1, not fabricated data)
    ax = axes[1]
    for model in MODEL_LABELS:
        xs, ys = [1], [0.0]  # pooled level, AMI=0 by definition
        for col, label, x in SANSKRITI_LEVEL_ORDER[1:]:
            cell = next((c for c in cells if c["model"] == model and c["dataset"] == "sanskriti"
                         and c["level"] == col), None)
            if cell:
                xs.append(x)
                ys.append(cell["ami"])
        ax.plot(xs, ys, marker="o", markersize=5, linewidth=1.6,
                color=MODEL_COLORS[model], label=MODEL_LABELS[model])
    ax.set_xticks([x for _, _, x in SANSKRITI_LEVEL_ORDER])
    ax.set_xticklabels([label for _, label, _ in SANSKRITI_LEVEL_ORDER], fontsize=7.5)
    ax.set_title("(b) SANSKRITI (genuinely nested: pooled\u2192region\u2192state)", fontsize=8.5)
    ax.set_ylim(-0.02, 0.65)
    ax.legend(fontsize=7, loc="upper left", frameon=True)

    # HONEST suptitle: claim only what SANSKRITI's genuinely nested hierarchy
    # actually shows (monotonic increase). Panel (a) demonstrates a related
    # but distinct point -- that partition choice, not sequence position,
    # drives AMI when levels are NOT nested -- stated in the annotation above.
    fig.suptitle("Aggregation Masking Index across granularity levels:\nmonotonic under a true nested hierarchy (b); partition-dependent otherwise (a)",
                  fontsize=8.5, y=1.08)
    fig.tight_layout()
    fig.savefig(outdir / "fig1_granularity_curve.png")
    plt.close(fig)
    print("Figure 1 saved: fig1_granularity_curve.png")


# ---------------------------------------------------------------------------
# FIGURE 2: The partition-effect evidence (L2 vs L3, same n, different AMI)
# ---------------------------------------------------------------------------

def fig2_partition_effect(cells, outdir):
    fig, ax = plt.subplots(figsize=(3.4, 2.6))
    models = list(MODEL_LABELS.keys())
    x = np.arange(len(models))
    width = 0.35

    l2_vals = [next(c for c in cells if c["model"] == m and c["dataset"] == "indicasa_caste"
                     and c["level"] == "l2_group")["ami"] for m in models]
    l3_vals = [next(c for c in cells if c["model"] == m and c["dataset"] == "indicasa_caste"
                     and c["level"] == "l3_group")["ami"] for m in models]

    ax.bar(x - width/2, l2_vals, width, label="Varna partition", color="#4c72b0")
    ax.bar(x + width/2, l3_vals, width, label="Constitutional partition", color="#dd8452")

    ax.set_xticks(x)
    ax.set_xticklabels([MODEL_LABELS[m] for m in models], fontsize=8, rotation=10)
    ax.set_ylabel("AMI")
    ax.set_title("Same group count (n=4), different partition,\ndifferent masking severity", fontsize=8.5)
    ax.legend(fontsize=7.5)
    ax.annotate("Group COUNT alone\ndoes not determine AMI", xy=(0.5, 0.5), xycoords="axes fraction",
                fontsize=6.5, ha="center", style="italic", color="#555555")

    fig.tight_layout()
    fig.savefig(outdir / "fig2_partition_effect.png")
    plt.close(fig)
    print("Figure 2 saved: fig2_partition_effect.png")


# ---------------------------------------------------------------------------
# FIGURE 3: AMI vs sigma_between (exploratory covariate, labeled honestly)
# ---------------------------------------------------------------------------

def fig3_ami_vs_sigma(cells, outdir):
    fig, ax = plt.subplots(figsize=(3.4, 2.8))
    amis = np.array([c["ami"] for c in cells])
    sigmas = np.array([c["sigma_between"] for c in cells])
    datasets = [c["dataset"] for c in cells]

    markers = {"indicasa_caste": "o", "bhed_religion": "s", "sanskriti": "^"}
    ds_labels = {"indicasa_caste": "IndiCASA caste", "bhed_religion": "BhED religion", "sanskriti": "SANSKRITI"}
    for ds in markers:
        mask = [d == ds for d in datasets]
        ax.scatter(sigmas[mask], amis[mask], marker=markers[ds], s=28,
                   alpha=0.8, label=ds_labels[ds], edgecolors="black", linewidths=0.4)

    # OLS trend line for reference (exploratory, not a validated bound -- label accordingly)
    slope, intercept, r, p, se = stats.linregress(sigmas, amis)
    xs = np.linspace(sigmas.min(), sigmas.max(), 50)
    ax.plot(xs, slope * xs + intercept, "--", color="gray", linewidth=1,
            label=f"OLS trend (r={r:.2f})")

    sr, sp = stats.spearmanr(sigmas, amis)
    ax.set_xlabel(r"$\sigma_{between}$ (between-group spread)")
    ax.set_ylabel("AMI")
    ax.set_title(f"Exploratory covariate relationship\n(Spearman r={sr:.2f}, p={sp:.4f}, n={len(cells)})",
                 fontsize=8.5)
    ax.legend(fontsize=6.5, loc="upper left")
    fig.tight_layout()
    fig.savefig(outdir / "fig3_ami_vs_sigma.png")
    plt.close(fig)
    print("Figure 3 saved: fig3_ami_vs_sigma.png")


# ---------------------------------------------------------------------------
# FIGURE 4: Methodology pipeline diagram -- explains the PROCESS
# ---------------------------------------------------------------------------

def fig4_pipeline_diagram(outdir):
    fig, ax = plt.subplots(figsize=(7.2, 2.6))
    ax.set_xlim(0, 10.4)
    ax.set_ylim(0, 3)
    ax.axis("off")

    boxes = [
        (0.2, "Benchmark\ndata\n(IndiCASA,\nBhED, SANSKRITI)", "#cfe2f3"),
        (2.25, "Granularity\nmapping\n(L1\u2192L4 /\nregion\u2192state)", "#d9ead3"),
        (4.3, "Model scoring\n(log-likelihood /\nMCQ accuracy)", "#fff2cc"),
        (6.35, "AMI\ncomputation\n(worst vs.\npooled rate)", "#f4cccc"),
        (8.4, "Self-referential\ndeviation flag\n(pooled + \u03c3)", "#e6d9f2"),
    ]
    box_w, box_h = 1.75, 1.6
    centers = []
    for x, text, color in boxes:
        rect = FancyBboxPatch((x, 0.7), box_w, box_h, boxstyle="round,pad=0.08",
                                linewidth=1, edgecolor="#555555", facecolor=color)
        ax.add_patch(rect)
        ax.text(x + box_w/2, 0.7 + box_h/2, text, ha="center", va="center", fontsize=6.6)
        centers.append(x + box_w)

    for i in range(len(centers) - 1):
        gap_start = centers[i]
        gap_end = boxes[i + 1][0]
        arrow = FancyArrowPatch((gap_start + 0.03, 0.7 + box_h/2), (gap_end - 0.03, 0.7 + box_h/2),
                                  arrowstyle="-|>", mutation_scale=12, color="#333333", linewidth=1.2)
        ax.add_patch(arrow)

    ax.text(5.2, 2.6, "Methodology Pipeline: from raw benchmark data to a flagged subgroup",
            ha="center", fontsize=9, weight="bold")

    fig.tight_layout()
    fig.savefig(outdir / "fig4_methodology_pipeline.png")
    plt.close(fig)
    print("Figure 4 saved: fig4_methodology_pipeline.png")


def main():
    if len(sys.argv) != 3:
        print("Usage: python generate_paper_figures.py <ami_cells_v2.json> <output_dir>")
        sys.exit(1)

    in_path = Path(sys.argv[1])
    outdir = Path(sys.argv[2])
    outdir.mkdir(parents=True, exist_ok=True)

    cells = load_cells(in_path)
    print(f"Loaded {len(cells)} cells from {in_path}\n")

    make_table1(cells, outdir)
    make_table2(cells, outdir)
    fig1_granularity_curve(cells, outdir)
    fig2_partition_effect(cells, outdir)
    fig3_ami_vs_sigma(cells, outdir)
    fig4_pipeline_diagram(outdir)

    print(f"\nAll outputs saved to {outdir}/")


if __name__ == "__main__":
    main()
