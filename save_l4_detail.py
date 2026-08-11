"""
Small, targeted addition to save FULL per-group (rate, n) detail at the
finest IndiCASA caste level (L4, after fold-up) for all 3 models. This is
the one piece of data needed for the random-partition ablation that was
NOT saved by the original run_all.py (which only persisted summary
statistics -- worst/median/p90/pooled -- not the full per-group breakdown).

This is a SMALL, cheap addition: per the smoke-test timing projection,
IndiCASA caste across all levels and all 3 models took ~6 minutes total;
this script only scores L4 (the finest level) for all 3 models, so it
should take well under that.

Run this the same way as run_all.py (same Kaggle kernel pattern) -- see
kaggle_entry_l4detail.py / push_l4detail_to_kaggle.py.
"""

import json

from data_loaders import load_indicasa_caste, merge_small_groups
from scoring_models import MODELS, load_model, score_stereotype_preference

INDICASA_CASTE_CSV = "./data/IndiCASA/IndiCASA_dataset/csv_datasets/IndiCASA/IndiCASA_dataset - caste.csv"


def main():
    df = load_indicasa_caste(INDICASA_CASTE_CSV)
    df = merge_small_groups(df, "l4_group", min_n=5)

    all_detail = {}
    for model_key in MODELS:
        print(f"\n{'=' * 60}\nScoring {model_key} on IndiCASA L4 (finest level)\n{'=' * 60}")
        model, tokenizer = load_model(model_key)

        scored = score_stereotype_preference(model, tokenizer, df, group_col="l4_group_merged")

        group_detail = {}
        for group_name, group_df in scored.groupby("l4_group_merged"):
            group_detail[group_name] = {
                "rate": float(group_df["prefers_stereotype"].mean()),
                "n": int(len(group_df)),
            }
        all_detail[model_key] = group_detail

        print(f"Saved detail for {len(group_detail)} groups:")
        for g, d in sorted(group_detail.items(), key=lambda x: -x[1]["n"]):
            print(f"  {g:25s} n={d['n']:3d}  rate={d['rate']:.3f}")

        del model, tokenizer
        import torch, gc
        gc.collect()
        torch.cuda.empty_cache()

    with open("l4_group_detail.json", "w") as f:
        json.dump(all_detail, f, indent=2)
    print("\nSaved l4_group_detail.json")


if __name__ == "__main__":
    main()
