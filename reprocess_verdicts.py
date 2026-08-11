"""
Local reprocessing script -- fixes the RQ3 verdict-flip results using the
SELF-REFERENTIAL threshold (pooled_rate + 1*sigma_between) instead of the
externally-borrowed fixed threshold (0.2) that produced a degenerate
zero-flip result on the first full run.

Requires NO Kaggle, NO GPU, NO model inference -- every number needed
(pooled_rate, worst_rate, sigma_between) is already saved in
ami_cells.json from the completed full run. This is pure local
reprocessing of existing results.

Run this on your machine:
    python reprocess_verdicts.py ./kaggle_output/ami_cells.json

Produces ami_cells_v2.json (same directory as the input file) with
corrected/added flip fields for ALL cells across all three datasets
(previously only indicasa_caste had any flip data at all).
"""

import json
import sys
from pathlib import Path

from metrics import detect_flip_selfref


def main():
    if len(sys.argv) != 2:
        print("Usage: python reprocess_verdicts.py <path-to-ami_cells.json>")
        sys.exit(1)

    in_path = Path(sys.argv[1])
    with open(in_path) as f:
        cells = json.load(f)

    n_flips = 0
    for c in cells:
        flip_result = detect_flip_selfref(
            pooled_rate=c["pooled_rate"],
            worst_rate=c["worst_rate"],
            sigma_between=c["sigma_between"],
            higher_is_worse=True,  # true for all three datasets here:
            # IndiCASA/BhED-religion use stereotype-preference rate (higher=worse),
            # SANSKRITI's stored 'worst_rate' is already an ERROR rate (1-accuracy,
            # see run_all.py's run_sanskriti -- higher error = worse), so
            # higher_is_worse=True is correct for all three, not just two.
        )
        # Overwrite the old (degenerate, fixed-threshold) flip fields if
        # present, and add them fresh for cells that never had any
        # (bhed_religion, sanskriti).
        c.pop("pooled_verdict", None)
        c.pop("worst_verdict", None)
        c.pop("flipped", None)
        c.pop("threshold_used", None)
        c.update(flip_result)
        if flip_result["flipped"]:
            n_flips += 1

    out_path = in_path.parent / "ami_cells_v2.json"
    with open(out_path, "w") as f:
        json.dump(cells, f, indent=2)

    print(f"Reprocessed {len(cells)} cells.")
    print(f"Flips under self-referential threshold: {n_flips}/{len(cells)}")
    print(f"Saved: {out_path}")

    print("\nPer-dataset breakdown:")
    for dataset in sorted(set(c["dataset"] for c in cells)):
        sub = [c for c in cells if c["dataset"] == dataset]
        sub_flips = sum(1 for c in sub if c["flipped"])
        print(f"  {dataset:16s} {sub_flips}/{len(sub)} cells flipped")


if __name__ == "__main__":
    main()
