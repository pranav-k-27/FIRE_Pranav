"""
Push + run the L4-DETAIL kernel (YOUR-KAGGLE-USERNAME/fire-ami-l4detail).

Usage:
    python push_l4detail_to_kaggle.py

After it finishes:
    python check_status.py YOUR-KAGGLE-USERNAME/fire-ami-l4detail
    python pull_output.py YOUR-KAGGLE-USERNAME/fire-ami-l4detail ./kaggle_output_l4detail

Then run the random-partition ablation locally on your machine:
    python random_partition_ablation.py ./kaggle_output_l4detail/l4_group_detail.json ./paper_figures
"""

import shutil
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

from kaggle.api.kaggle_api_extended import KaggleApi  # noqa: E402
from push_to_kaggle import stage_dataset_files, PROJECT_ROOT, DATASET_DIR  # noqa: E402

L4DETAIL_KERNEL_DIR = PROJECT_ROOT / "kaggle_kernel_l4detail"


def stage_l4detail_file():
    src = PROJECT_ROOT / "save_l4_detail.py"
    dst = DATASET_DIR / "save_l4_detail.py"
    shutil.copy(src, dst)
    print(f"Staged {src.name} -> {dst}")


def main():
    api = KaggleApi()
    api.authenticate()
    print(f"Authenticated as: {api.config_values.get('username')}")

    stage_dataset_files()
    stage_l4detail_file()

    dataset_ref = "YOUR-KAGGLE-USERNAME/fire-ami-code"
    try:
        api.dataset_status(dataset_ref)
        exists = True
    except Exception:
        exists = False

    if exists:
        print(f"Updating existing dataset {dataset_ref} ...")
        api.dataset_create_version(str(DATASET_DIR), version_notes="Update from push_l4detail_to_kaggle.py")
    else:
        print(f"Creating new dataset {dataset_ref} ...")
        api.dataset_create_new(str(DATASET_DIR), public=False)

    print("Pushing L4-detail kernel YOUR-KAGGLE-USERNAME/fire-ami-l4detail (GPU, small job) ...")
    api.kernels_push(str(L4DETAIL_KERNEL_DIR))
    print("\nDone. Check status:  python check_status.py YOUR-KAGGLE-USERNAME/fire-ami-l4detail")
    print("Pull output:          python pull_output.py YOUR-KAGGLE-USERNAME/fire-ami-l4detail ./kaggle_output_l4detail")


if __name__ == "__main__":
    main()
