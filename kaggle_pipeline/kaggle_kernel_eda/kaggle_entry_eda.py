"""
Kaggle entry point for the EDA-ONLY pre-flight kernel.

Purpose: run this BEFORE the full run_all.py GPU pipeline. It costs no
GPU time and takes ~1-2 minutes. If anything here looks wrong, fix it
now -- it's a 30-second problem here vs. a wasted multi-hour GPU run if
caught only after full scoring starts.

Expects:
  - The 'YOUR-KAGGLE-USERNAME/fire-ami-code' dataset attached (for data_loaders.py)
  - enable_internet: true (to git-clone the two small repos and load
    SANSKRITI from HuggingFace)
  - enable_gpu: false is fine here -- nothing in this script touches a
    model or a GPU.

Writes /kaggle/working/eda_report.json, which `kaggle kernels output`
will retrieve, and prints the eyeball summary directly into the kernel
logs so you can read it without even downloading the file.
"""

import json
import subprocess
import sys
from pathlib import Path

# --- Make the code dataset importable ---------------------------------
CODE_DATASET_DIR = Path("/kaggle/input/fire-ami-code")
if CODE_DATASET_DIR.exists():
    sys.path.insert(0, str(CODE_DATASET_DIR))
else:
    print("[WARN] /kaggle/input/fire-ami-code not found -- assuming "
          "data_loaders.py is already on sys.path some other way")

from data_loaders import run_all_eda  # noqa: E402 -- must come after sys.path fix

WORKDIR = Path("/kaggle/working")
DATA_DIR = WORKDIR / "data"
DATA_DIR.mkdir(exist_ok=True)


def clone_repo(url: str, dest: Path):
    if dest.exists():
        print(f"[INFO] {dest} already exists, skipping clone")
        return
    print(f"[INFO] Cloning {url} -> {dest}")
    subprocess.run(["git", "clone", "--depth", "1", url, str(dest)], check=True)


def main():
    clone_repo("https://github.com/khyatikhandelwal/Indian-LLMs-Bias.git",
               DATA_DIR / "Indian-LLMs-Bias")
    clone_repo("https://github.com/cerai-iitm/IndiCASA.git",
               DATA_DIR / "IndiCASA")

    indicasa_caste_csv = str(
        DATA_DIR / "IndiCASA" / "IndiCASA_dataset" / "csv_datasets" / "IndiCASA"
        / "IndiCASA_dataset - caste.csv"
    )
    bhed_caste_csv = str(DATA_DIR / "Indian-LLMs-Bias" / "Data" / "Caste.csv")
    bhed_religion_csv = str(DATA_DIR / "Indian-LLMs-Bias" / "Data" / "India_Religious.csv")

    print("[INFO] Loading SANSKRITI from HuggingFace...")
    from datasets import load_dataset
    sanskriti_df = load_dataset("13ari/Sanskriti", split="train").to_pandas()
    print(f"[INFO] SANSKRITI loaded: {len(sanskriti_df)} rows")

    report = run_all_eda(
        indicasa_caste_path=indicasa_caste_csv,
        bhed_caste_path=bhed_caste_csv,
        bhed_religion_path=bhed_religion_csv,
        sanskriti_df=sanskriti_df,
    )

    out_path = WORKDIR / "eda_report.json"
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2, default=str)
    print(f"\n[INFO] Saved {out_path}")

    # Inline eyeball summary (same logic as day1_eda_smoketest.py Cell 3,
    # duplicated here so this script has no dependency beyond data_loaders.py)
    print("\n" + "=" * 70)
    print("DAY 1 EYEBALL SUMMARY -- things worth a second look")
    print("=" * 70)
    issues = []

    ic = report.get("indicasa_caste", {})
    if ic.get("duplicate_sentences", 0) > 0:
        issues.append(f"IndiCASA caste: {ic['duplicate_sentences']} exact duplicate sentences "
                       f"in raw file (note in Methods/Limitations)")

    bc = report.get("bhed_caste", {})
    if bc.get("multi_label_rows", 0) > 0:
        issues.append(f"Indian-BhED caste: {bc['multi_label_rows']} multi-label row(s) "
                       f"will be skipped by the scorer")

    br = report.get("bhed_religion", {})
    if br.get("dominant_group_pct", 0) > 55:
        issues.append(f"Indian-BhED religion: dominant group is {br['dominant_group_pct']}% "
                       f"of cleaned data -- expect wide CIs on minority groups")

    sk = report.get("sanskriti", {})
    if sk.get("unmapped_state_rows", 0) > 0:
        issues.append(f"SANSKRITI: {sk['unmapped_state_rows']} rows have an unmapped state -- "
                       f"FIX STATE_TO_REGION before scoring")
    enc = sk.get("possible_encoding_artifact_columns", {})
    if enc:
        issues.append(f"SANSKRITI: possible encoding artifacts in columns: {list(enc.keys())}")
    min_state_n = sk.get("sample_min_state_n")
    if min_state_n is not None and min_state_n < 30:
        issues.append(f"SANSKRITI: smallest post-sample state n = {min_state_n} -- "
                       f"consider raising min_n_per_state floor")

    if not issues:
        print("No red flags found. Proceed to the full run_all.py GPU pipeline.")
    else:
        print(f"{len(issues)} item(s) worth a second look:\n")
        for i, msg in enumerate(issues, 1):
            print(f"  {i}. {msg}")
        print("\nMost of these were already anticipated in the protocol's known "
              "limitations. If something here is NEW/surprising, stop and "
              "investigate before pushing the full GPU run.")


if __name__ == "__main__":
    main()
