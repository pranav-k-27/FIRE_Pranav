"""
Check kernel status via the same working auth pattern as push_eda_to_kaggle.py
/ push_to_kaggle.py (load .env into THIS process, not the shell).

The plain `kaggle kernels status ...` CLI command does NOT read your
project's .env file -- it's a separate process that only looks at its own
expected locations (~/.kaggle/access_token, or a shell-exported env var).
This script sidesteps that by authenticating the same way your push
scripts already do successfully.

Usage:
    python check_status.py YOUR-KAGGLE-USERNAME/fire-ami-eda
    python check_status.py YOUR-KAGGLE-USERNAME/fire-ami-pipeline
"""

import sys

from dotenv import load_dotenv

load_dotenv()

from kaggle.api.kaggle_api_extended import KaggleApi  # noqa: E402


def main():
    if len(sys.argv) != 2:
        print("Usage: python check_status.py <owner>/<kernel-slug>")
        print("e.g.:  python check_status.py YOUR-KAGGLE-USERNAME/fire-ami-eda")
        sys.exit(1)

    kernel_ref = sys.argv[1]

    api = KaggleApi()
    api.authenticate()
    print(f"Authenticated as: {api.config_values.get('username')}")

    status = api.kernels_status(kernel_ref)
    print(f"\nKernel: {kernel_ref}")
    print(f"Status: {status}")


if __name__ == "__main__":
    main()
