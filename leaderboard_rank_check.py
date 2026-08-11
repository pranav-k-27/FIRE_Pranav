"""
Leaderboard rank-flip check -- does the relative ranking of models change
depending on whether you rank by pooled rate vs. worst-case (fine-grained)
rate? Uses ONLY data already saved in ami_cells_v2.json -- no new Kaggle
run needed.

Key insight this checks for: pooled_rate is fixed for a given (model,
dataset) pair regardless of granularity level (pooling all items together
doesn't depend on how you'd later re-group them), so a naive "pooled
leaderboard" never changes across levels. The real question is whether
ranking models by their POOLED score gives a different order than ranking
them by their WORST-CASE (fine-grained) score -- i.e., does a model that
looks best under pooled reporting stop looking best once you account for
its worst-affected subgroup?

Usage:
    python leaderboard_rank_check.py ./kaggle_output/ami_cells_v2.json
"""

import json
import sys
from pathlib import Path

import pandas as pd


def main():
    if len(sys.argv) != 2:
        print("Usage: python leaderboard_rank_check.py <ami_cells_v2.json>")
        sys.exit(1)

    with open(sys.argv[1]) as f:
        cells = json.load(f)

    df = pd.DataFrame(cells)

    print("=" * 78)
    print("LEADERBOARD RANK-FLIP CHECK")
    print("=" * 78)

    results = []
    for dataset in sorted(df["dataset"].unique()):
        sub = df[df["dataset"] == dataset]
        # pooled_rate is identical across levels for a given model -- take
        # the finest available level's row per model for both pooled and
        # worst-case comparison (finest level = most granular evidence)
        finest_level = sub.groupby("model")["n_groups"].idxmax()
        finest = sub.loc[finest_level].set_index("model")

        pooled_rank = finest["pooled_rate"].rank(method="min")
        worst_rank = finest["worst_rate"].rank(method="min")

        print(f"\n--- {dataset} (finest level per model) ---")
        table = pd.DataFrame({
            "pooled_rate": finest["pooled_rate"].round(3),
            "pooled_rank": pooled_rank.astype(int),
            "worst_rate": finest["worst_rate"].round(3),
            "worst_rank": worst_rank.astype(int),
            "AMI": finest["ami"].round(3),
        }).sort_values("pooled_rank")
        print(table.to_string())

        flipped = (pooled_rank != worst_rank).any()
        n_flipped_pairs = 0
        models = list(finest.index)
        for i in range(len(models)):
            for j in range(i + 1, len(models)):
                m1, m2 = models[i], models[j]
                pooled_order = finest.loc[m1, "pooled_rate"] < finest.loc[m2, "pooled_rate"]
                worst_order = finest.loc[m1, "worst_rate"] < finest.loc[m2, "worst_rate"]
                if pooled_order != worst_order:
                    n_flipped_pairs += 1
                    print(f"  RANK FLIP: {m1} vs {m2} -- order reverses between "
                          f"pooled and worst-case ranking")

        results.append({
            "dataset": dataset,
            "any_rank_change": bool(flipped),
            "n_pairwise_flips": n_flipped_pairs,
            "n_pairs_total": len(models) * (len(models) - 1) // 2,
        })

    print("\n" + "=" * 78)
    print("SUMMARY")
    print("=" * 78)
    summary_df = pd.DataFrame(results)
    print(summary_df.to_string(index=False))

    out_path = Path(sys.argv[1]).parent / "leaderboard_rank_check_summary.csv"
    summary_df.to_csv(out_path, index=False)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
