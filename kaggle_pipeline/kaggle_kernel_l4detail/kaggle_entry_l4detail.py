"""
Entry point for the L4-DETAIL kernel (YOUR-KAGGLE-USERNAME/fire-ami-l4detail).
Mirrors kaggle_entry.py's setup exactly, but runs save_l4_detail.py
instead of run_all.py. Small, fast job -- only scores IndiCASA caste's
finest (L4) level, needed for the random-partition ablation.
"""

import os
import subprocess
import sys

subprocess.run(
    ["pip", "install", "-q", "transformers", "accelerate", "bitsandbytes",
     "datasets", "scipy", "kagglehub", "huggingface_hub"],
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

os.makedirs("./data", exist_ok=True)
if not os.path.isdir("./data/IndiCASA"):
    subprocess.run(
        ["git", "clone", "--depth", "1",
         "https://github.com/cerai-iitm/IndiCASA.git",
         "./data/IndiCASA"],
        check=True,
    )
# Indian-BhED not needed for this job -- only IndiCASA L4 detail is scored.

import save_l4_detail  # noqa: E402

save_l4_detail.main()
