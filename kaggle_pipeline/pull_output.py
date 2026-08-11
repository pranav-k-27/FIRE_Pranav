"""
Pull kernel output (logs + eda_report.json / smoketest_results.json /
all_results.json / etc.) via the same working auth pattern as
push_eda_to_kaggle.py / push_to_kaggle.py.

Same reasoning as check_status.py -- the plain `kaggle kernels output ...`
CLI command doesn't read your project's .env, this script does.

Uses force=True by default: kernels_output() otherwise skips files that
already exist locally, which can silently leave you looking at a STALE
copy from a previous run after you've pushed code changes and re-run the
kernel. Forcing a fresh download every time costs a few extra seconds and
removes that entire class of confusion.

Usage:
    python pull_output.py YOUR-KAGGLE-USERNAME/fire-ami-eda ./kaggle_output_eda
    python pull_output.py YOUR-KAGGLE-USERNAME/fire-ami-pipeline ./kaggle_output
"""

import sys

from dotenv import load_dotenv

load_dotenv()

from kaggle.api.kaggle_api_extended import KaggleApi  # noqa: E402


def main():
    if len(sys.argv) != 3:
        print("Usage: python pull_output.py <owner>/<kernel-slug> <output-dir>")
        print("e.g.:  python pull_output.py YOUR-KAGGLE-USERNAME/fire-ami-eda ./kaggle_output_eda")
        sys.exit(1)

    kernel_ref = sys.argv[1]
    out_dir = sys.argv[2]

    api = KaggleApi()
    api.authenticate()
    print(f"Authenticated as: {api.config_values.get('username')}")

    files, log_url = api.kernels_output(kernel_ref, out_dir, force=True)
    print(f"\nDownloaded/refreshed {len(files)} file(s) to {out_dir}:")
    for f in files:
        print(f"  - {f}")
    if log_url:
        print(f"\nLog file: {log_url}")

    print(f"\nNext: open the files in {out_dir} and the log -- for the EDA "
          f"kernel look for 'DAY 1 EYEBALL SUMMARY', for the smoke-test "
          f"kernel look for 'SMOKE TEST COMPLETE', for the full pipeline "
          f"look at all_results.json / ami_cells.json.")


if __name__ == "__main__":
    main()

