"""
Entry point for the SMOKE TEST kernel (YOUR-KAGGLE-USERNAME/fire-ami-smoketest).

Mirrors kaggle_entry.py exactly (same pip installs, same data cloning,
same HF login) but calls run_smoketest.main() instead of run_all.main().
This is a SEPARATE kernel from fire-ami-pipeline -- running this never
touches or risks the real full-pipeline kernel.

GPU IS needed here (unlike the EDA-only kernel) since this tests real
model loading and scoring on Kaggle hardware.
"""

import os
import subprocess
import sys

subprocess.run(
    ["pip", "install", "-q", "transformers", "accelerate", "bitsandbytes",
     "datasets", "scipy", "kagglehub", "huggingface_hub", "ftfy"],
    check=True,
)

CODE_DIR = "/kaggle/input/fire-ami-code"
if CODE_DIR not in sys.path:
    sys.path.insert(0, CODE_DIR)

hf_token_path = os.path.join(CODE_DIR, "hf_token.txt")
if os.path.isfile(hf_token_path):
    from huggingface_hub import login
    with open(hf_token_path) as f:
        login(token=f.read().strip())
else:
    print("[WARN] No hf_token.txt found -- if the HF path for a gated "
          "model fails, the auto-fallback to kagglehub should still work "
          "for llama-3.2-3b (this smoke test's model).")

os.makedirs("./data", exist_ok=True)

if not os.path.isdir("./data/Indian-LLMs-Bias"):
    subprocess.run(
        ["git", "clone", "--depth", "1",
         "https://github.com/khyatikhandelwal/Indian-LLMs-Bias.git",
         "./data/Indian-LLMs-Bias"],
        check=True,
    )
if not os.path.isdir("./data/IndiCASA"):
    subprocess.run(
        ["git", "clone", "--depth", "1",
         "https://github.com/cerai-iitm/IndiCASA.git",
         "./data/IndiCASA"],
        check=True,
    )

import run_smoketest  # noqa: E402 -- must come after sys.path/data setup above

run_smoketest.main()
