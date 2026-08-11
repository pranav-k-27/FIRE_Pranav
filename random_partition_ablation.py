"""
Random-partition ablation: answers the reviewer question "would different
four-group partitions of the same communities behave differently, or did
you get lucky with Varna/Constitutional?"

Given per-group (rate, n) detail at the finest level (from
l4_group_detail.json, produced by save_l4_detail.py on Kaggle), this
script:
  1. Computes the TRUE pooled rate (fixed, independent of partition).
  2. Generates K random partitions of the fine-grained groups into 4
     buckets each (matching Varna/Constitutional's cardinality).
  3. Computes AMI for each random partition.
  4. Reports where the ACTUAL Varna and Constitutional partitions fall
     within the resulting random-partition AMI distribution (percentile).
  5. Saves a histogram figure with the actual partitions marked.

Entirely local -- no GPU, no Kaggle, runs on the (rate, n) summary data.

Usage:
    python random_partition_ablation.py ./kaggle_output_l4detail/l4_group_detail.json ./paper_figures
"""

import json
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent))
from metrics import compute_ami

N_RANDOM_PARTITIONS = 1000
N_BUCKETS = 4  # matches Varna and Constitutional's cardinality
SEED = 42

# The actual Varna and Constitutional AMI values, computed from the real
# full pipeline run (ami_cells_v2.json) -- hardcoded here as the reference
# points to compare against the random-partition distribution, since they
# come from a different data file (ami_cells_v2.json) than this script's
# main input (l4_group_detail.json).
ACTUAL_PARTITION_AMI = {
    "llama-3.2-3b": {"Varna": 0.536, "Constitutional": 0.153},
    "mistral-7b": {"Varna": 0.457, "Constitutional": 0.255},
    "qwen2.5-7b": {"Varna": 0.466, "Constitutional": 0.274},
}


def true_pooled_rate(group_detail: dict) -> float:
    total_n = sum(g["n"] for g in group_detail.values())
    return sum(g["rate"] * g["n"] for g in group_detail.values()) / total_n


def random_partition_ami(group_detail: dict, pooled_rate: float, n_buckets: int, rng) -> float:
    """One random partition of the fine-grained groups into n_buckets,
    ensuring every bucket is non-empty (resamples the assignment if not)."""
    names = list(group_detail.keys())
    while True:
        assignment = rng.integers(0, n_buckets, size=len(names))
        if len(set(assignment)) == n_buckets:
            break  # every bucket has at least one group

    bucket_rates = {}
    for b in range(n_buckets):
        members = [names[i] for i in range(len(names)) if assignment[i] == b]
        total_n = sum(group_detail[m]["n"] for m in members)
        weighted_rate = sum(group_detail[m]["rate"] * group_detail[m]["n"] for m in members) / total_n
        bucket_rates[f"bucket_{b}"] = weighted_rate

    result = compute_ami(bucket_rates, pooled_rate)
    return result["ami"]


def run_for_model(model_key: str, group_detail: dict, rng) -> dict:
    pooled = true_pooled_rate(group_detail)
    random_amis = np.array([
        random_partition_ami(group_detail, pooled, N_BUCKETS, rng)
        for _ in range(N_RANDOM_PARTITIONS)
    ])

    actual = ACTUAL_PARTITION_AMI.get(model_key, {})
    percentiles = {}
    for label, ami_val in actual.items():
        pct = (random_amis < ami_val).mean() * 100
        percentiles[label] = pct

    return {
        "model": model_key,
        "n_groups": len(group_detail),
        "pooled_rate": pooled,
        "random_ami_mean": float(random_amis.mean()),
        "random_ami_std": float(random_amis.std()),
        "random_ami_min": float(random_amis.min()),
        "random_ami_max": float(random_amis.max()),
        "actual_partitions": actual,
        "actual_percentiles": percentiles,
        "random_amis": random_amis,  # kept for plotting, stripped before JSON dump
    }


def make_figure(all_results: list, outdir: Path):
    fig, axes = plt.subplots(1, 3, figsize=(9.5, 2.8), sharey=True)
    for ax, res in zip(axes, all_results):
        ax.hist(res["random_amis"], bins=30, color="#8fa8c9", edgecolor="white", alpha=0.85)
        for label, val in res["actual_partitions"].items():
            color = "#d62728" if label == "Varna" else "#2ca02c"
            ax.axvline(val, color=color, linewidth=1.8, linestyle="--",
                       label=f"{label} (p{res['actual_percentiles'][label]:.0f})")
        ax.set_title(res["model"], fontsize=8.5)
        ax.set_xlabel("AMI (random 4-group partition)", fontsize=7)
        ax.legend(fontsize=6.5)
    axes[0].set_ylabel("Count (of 1000 random partitions)")
    fig.suptitle("Random-partition ablation: where do the chosen partitions fall?", fontsize=9.5, y=1.05)
    fig.tight_layout()
    fig.savefig(outdir / "fig5_random_partition_ablation.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {outdir / 'fig5_random_partition_ablation.png'}")


def main():
    if len(sys.argv) != 3:
        print("Usage: python random_partition_ablation.py <l4_group_detail.json> <output_dir>")
        sys.exit(1)

    in_path = Path(sys.argv[1])
    outdir = Path(sys.argv[2])
    outdir.mkdir(parents=True, exist_ok=True)

    with open(in_path) as f:
        all_detail = json.load(f)

    rng = np.random.default_rng(SEED)
    all_results = []
    print(f"Running {N_RANDOM_PARTITIONS} random {N_BUCKETS}-group partitions per model...\n")
    for model_key, group_detail in all_detail.items():
        res = run_for_model(model_key, group_detail, rng)
        all_results.append(res)
        print(f"--- {model_key} ({res['n_groups']} fine-grained groups) ---")
        print(f"  Pooled rate: {res['pooled_rate']:.3f}")
        print(f"  Random-partition AMI: mean={res['random_ami_mean']:.3f}, "
              f"std={res['random_ami_std']:.3f}, range=[{res['random_ami_min']:.3f}, {res['random_ami_max']:.3f}]")
        for label, val in res["actual_partitions"].items():
            pct = res["actual_percentiles"][label]
            print(f"  Actual {label} partition: AMI={val:.3f} -> percentile {pct:.1f} "
                  f"of random distribution")
        print()

    make_figure(all_results, outdir)

    # Strip the raw arrays before JSON serialization
    summary = [{k: v for k, v in r.items() if k != "random_amis"} for r in all_results]
    with open(outdir / "random_partition_ablation_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved: {outdir / 'random_partition_ablation_summary.json'}")


if __name__ == "__main__":
    main()
