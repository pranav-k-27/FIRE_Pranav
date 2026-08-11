"""
Push + run the SMOKE TEST kernel (YOUR-KAGGLE-USERNAME/fire-ami-smoketest).

This is Stage B-preview -- run this AFTER the EDA kernel (Stage A) looks
clean, and BEFORE the full run_all.py pipeline (Stage B proper). It costs
real GPU minutes (unlike the EDA kernel) but only a few, since it's
deliberately tiny -- 1 model, ~15-30 items per dataset.

IMPORTANT: run_smoketest.py must be added to CODE_FILES in
push_to_kaggle.py (or staged here directly) for this to work -- see the
stage_smoketest_file() addition below, which copies it in without
requiring you to edit push_to_kaggle.py's CODE_FILES list.

Usage:
    python push_smoketest_to_kaggle.py

After it finishes:
    python check_status.py YOUR-KAGGLE-USERNAME/fire-ami-smoketest
    python pull_output.py YOUR-KAGGLE-USERNAME/fire-ami-smoketest ./kaggle_output_smoketest

Read the log for "SMOKE TEST COMPLETE" and the projected full-run time
estimate before deciding to launch push_to_kaggle.py (the real, multi-
hour full run).
"""

import shutil
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

from kaggle.api.kaggle_api_extended import KaggleApi  # noqa: E402
from push_to_kaggle import stage_dataset_files, PROJECT_ROOT, DATASET_DIR  # noqa: E402

SMOKETEST_KERNEL_DIR = PROJECT_ROOT / "kaggle_kernel_smoketest"


def stage_smoketest_file():
    """Copy run_smoketest.py into the code dataset directory alongside
    the files push_to_kaggle.py's CODE_FILES already stages, without
    requiring an edit to that list."""
    src = PROJECT_ROOT / "run_smoketest.py"
    dst = DATASET_DIR / "run_smoketest.py"
    shutil.copy(src, dst)
    print(f"Staged {src.name} -> {dst}")


def main():
    api = KaggleApi()
    api.authenticate()
    print(f"Authenticated as: {api.config_values.get('username')}")

    stage_dataset_files()
    stage_smoketest_file()

    dataset_ref = "YOUR-KAGGLE-USERNAME/fire-ami-code"
    try:
        api.dataset_status(dataset_ref)
        exists = True
    except Exception:
        exists = False

    if exists:
        print(f"Updating existing dataset {dataset_ref} ...")
        api.dataset_create_version(str(DATASET_DIR), version_notes="Update from push_smoketest_to_kaggle.py")
    else:
        print(f"Creating new dataset {dataset_ref} ...")
        api.dataset_create_new(str(DATASET_DIR), public=False)

    print("Pushing smoke-test kernel YOUR-KAGGLE-USERNAME/fire-ami-smoketest (GPU, tiny job) ...")
    api.kernels_push(str(SMOKETEST_KERNEL_DIR))
    print("\nDone. The smoke-test kernel is now queued/running on Kaggle.")
    print("Check status:  python check_status.py YOUR-KAGGLE-USERNAME/fire-ami-smoketest")
    print("Pull output:   python pull_output.py YOUR-KAGGLE-USERNAME/fire-ami-smoketest ./kaggle_output_smoketest")


if __name__ == "__main__":
    main()
