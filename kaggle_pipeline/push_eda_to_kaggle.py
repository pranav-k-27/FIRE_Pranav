"""
Push + run the EDA-ONLY pre-flight kernel (YOUR-KAGGLE-USERNAME/fire-ami-eda).

This is Stage A of the two-stage flow -- run this FIRST, before
push_to_kaggle.py (Stage B, the full GPU model-scoring pipeline).

Reuses the same code-dataset staging logic from push_to_kaggle.py so
data_loaders.py etc. only need to be maintained in one place. Pushes a
SEPARATE kernel (fire-ami-eda, not fire-ami-pipeline) so this never
overwrites or interferes with the main pipeline kernel.

Usage:
    python push_eda_to_kaggle.py

After it finishes:
    kaggle kernels status YOUR-KAGGLE-USERNAME/fire-ami-eda
    kaggle kernels output YOUR-KAGGLE-USERNAME/fire-ami-eda -p ./kaggle_output_eda

Read kaggle_output_eda/eda_report.json and/or the kernel log (the
eyeball summary prints directly into the log) before proceeding to
push_to_kaggle.py.
"""

from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

from kaggle.api.kaggle_api_extended import KaggleApi  # noqa: E402
from push_to_kaggle import stage_dataset_files, PROJECT_ROOT  # noqa: E402 -- reuse existing logic

EDA_KERNEL_DIR = PROJECT_ROOT / "kaggle_kernel_eda"


def main():
    api = KaggleApi()
    api.authenticate()
    print(f"Authenticated as: {api.config_values.get('username')}")

    # Same code dataset used by both kernels -- staging it here keeps it
    # current even if you run this before push_to_kaggle.py today.
    stage_dataset_files()

    dataset_ref = "YOUR-KAGGLE-USERNAME/fire-ami-code"
    try:
        api.dataset_status(dataset_ref)
        exists = True
    except Exception:
        exists = False

    from push_to_kaggle import DATASET_DIR
    if exists:
        print(f"Updating existing dataset {dataset_ref} ...")
        api.dataset_create_version(str(DATASET_DIR), version_notes="Update from push_eda_to_kaggle.py")
    else:
        print(f"Creating new dataset {dataset_ref} ...")
        api.dataset_create_new(str(DATASET_DIR), public=False)

    print("Pushing EDA preflight kernel YOUR-KAGGLE-USERNAME/fire-ami-eda (no GPU) ...")
    api.kernels_push(str(EDA_KERNEL_DIR))
    print("\nDone. The EDA kernel is now queued/running on Kaggle.")
    print("Check status:  kaggle kernels status YOUR-KAGGLE-USERNAME/fire-ami-eda")
    print("Pull output:   kaggle kernels output YOUR-KAGGLE-USERNAME/fire-ami-eda -p ./kaggle_output_eda")
    print("\nOnce this looks clean, proceed to: python push_to_kaggle.py")


if __name__ == "__main__":
    main()
