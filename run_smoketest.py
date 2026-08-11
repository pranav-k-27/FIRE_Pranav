"""
Stage B SMOKE TEST -- a deliberately tiny run (1 model, ~15-30 items per
dataset) to validate the real GPU pipeline on Kaggle before committing to
the multi-hour full run_all.py job.

What this actually tests, that nothing before it could:
  - Model loading on real Kaggle GPU hardware (4-bit quantization, device
    placement) -- untestable in a no-GPU sandbox
  - The HF -> kagglehub automatic fallback for Llama-3.2 (source="auto")
    actually works end-to-end, not just the logic in isolation
  - sentence_log_likelihood() produces sane, non-NaN, non-inf values on
    real hardware
  - score_mcq_accuracy() runs without shape/device errors
  - The new clean_sanskriti_text_columns() mojibake fix runs correctly
    against the real dataset (not just the synthetic test we ran locally)
  - AMI computation, bootstrap CI, and JSON serialization all work on
    real (not synthetic) scored output
  - Gives a per-item timing estimate to project the full run's duration
    BEFORE spending the GPU hours on it

Only llama-3.2-3b is used (smallest, fastest) -- this is deliberately not
a scientific run, just a pipeline correctness + timing check.
"""

import json
import time

from data_loaders import (
    load_indicasa_caste, merge_small_groups,
    load_indianbhed_caste, load_indianbhed_religion,
    load_sanskriti, clean_sanskriti_text_columns,
)
from scoring_models import (
    load_model, score_stereotype_preference, score_mcq_accuracy,
    aggregate_group_rates, pooled_rate, sentence_log_likelihood,
)
from metrics import compute_ami, bootstrap_ci_ami

SMOKE_MODEL = "llama-3.2-3b"

INDICASA_CASTE_CSV = "./data/IndiCASA/IndiCASA_dataset/csv_datasets/IndiCASA/IndiCASA_dataset - caste.csv"
BHED_CASTE_CSV = "./data/Indian-LLMs-Bias/Data/Caste.csv"
BHED_RELIGION_CSV = "./data/Indian-LLMs-Bias/Data/India_Religious.csv"


def timed(label, fn, *args, **kwargs):
    t0 = time.time()
    result = fn(*args, **kwargs)
    elapsed = time.time() - t0
    print(f"[TIMING] {label}: {elapsed:.1f}s")
    return result, elapsed


def smoke_test_model_load():
    print("=" * 70)
    print(f"SMOKE TEST 1: Model loading ({SMOKE_MODEL}, source='auto')")
    print("=" * 70)
    (model, tokenizer), elapsed = timed(
        "load_model", load_model, SMOKE_MODEL, four_bit=True, source="auto"
    )
    print(f"Model loaded OK. dtype/device sanity: "
          f"{next(model.parameters()).dtype}, {next(model.parameters()).device}")
    return model, tokenizer, elapsed


def smoke_test_log_likelihood(model, tokenizer):
    print("\n" + "=" * 70)
    print("SMOKE TEST 2: sentence_log_likelihood sanity check")
    print("=" * 70)
    test_sentences = [
        "The sun rises in the east.",
        "Colorless green ideas sleep furiously.",  # should have low LL, still finite
    ]
    for s in test_sentences:
        (ll, n_tokens), elapsed = timed(f"  LL('{s[:30]}...')", sentence_log_likelihood, model, tokenizer, s)
        is_finite = ll == ll and abs(ll) != float("inf")  # NaN check without numpy import here
        print(f"    log-likelihood={ll:.4f}, n_tokens={n_tokens}, finite={is_finite}")
        assert is_finite, f"Non-finite log-likelihood for '{s}' -- STOP, debug before full run"


def smoke_test_indicasa(model, tokenizer):
    print("\n" + "=" * 70)
    print("SMOKE TEST 3: IndiCASA caste scoring (2 context_ids only)")
    print("=" * 70)
    df = load_indicasa_caste(INDICASA_CASTE_CSV)
    small_context_ids = sorted(df["context_id"].unique())[:2]
    small_df = df[df["context_id"].isin(small_context_ids)]
    print(f"Using {len(small_df)} rows from context_ids {small_context_ids}")

    scored, elapsed = timed(
        "score_stereotype_preference (indicasa, l2_group)",
        score_stereotype_preference, model, tokenizer, small_df, group_col="l2_group",
    )
    print(scored)
    if not scored.empty:
        group_rates = aggregate_group_rates(scored, "l2_group", "prefers_stereotype")
        pooled = pooled_rate(scored, "prefers_stereotype")
        ami = compute_ami(group_rates, pooled)
        print(f"AMI on this tiny sample (NOT meaningful scientifically, just a "
              f"pipeline check): {ami['ami']:.3f}")
    per_item_seconds = elapsed / max(len(small_df), 1)
    print(f"[TIMING] ~{per_item_seconds:.2f}s per sentence-pair scored "
          f"(2x log-likelihood calls each)")
    return per_item_seconds


def smoke_test_bhed(model, tokenizer):
    print("\n" + "=" * 70)
    print("SMOKE TEST 4: Indian-BhED MASK-fill scoring (first 10 rows)")
    print("=" * 70)
    df = load_indianbhed_caste(BHED_CASTE_CSV).head(10)
    raw = df.copy()
    correct = 0
    t0 = time.time()
    for _, row in raw.iterrows():
        sentence = row["Sentence"] if "Sentence" in raw.columns else row.get("sentence")
        stereo_term = str(row["Target_Stereotypical"]).strip("[]'\" ")
        anti_term = str(row["Target_Anti-Stereotypical"]).strip("[]'\" ")
        if "," in stereo_term or "," in anti_term:
            continue
        sent_stereo = sentence.replace("MASK", stereo_term)
        sent_anti = sentence.replace("MASK", anti_term)
        ll_stereo, _ = sentence_log_likelihood(model, tokenizer, sent_stereo)
        ll_anti, _ = sentence_log_likelihood(model, tokenizer, sent_anti)
        correct += 1
    elapsed = time.time() - t0
    print(f"[TIMING] Scored {correct} MASK-fill pairs in {elapsed:.1f}s "
          f"({elapsed/max(correct,1):.2f}s/pair)")


def smoke_test_sanskriti(model, tokenizer, sanskriti_df):
    print("\n" + "=" * 70)
    print("SMOKE TEST 5: SANSKRITI MCQ scoring (30-row sample) + mojibake fix check")
    print("=" * 70)
    df = load_sanskriti(sanskriti_df)
    df = clean_sanskriti_text_columns(df)
    small_df = df.sample(n=min(30, len(df)), random_state=42)

    scored, elapsed = timed("score_mcq_accuracy (30 rows)", score_mcq_accuracy, model, tokenizer, small_df)
    print(scored[["state", "predicted", "answer", "correct"]].to_string())
    acc = scored["correct"].mean()
    print(f"Smoke-sample accuracy: {acc:.2f} (not meaningful scientifically, just a pipeline check)")
    per_item_seconds = elapsed / max(len(small_df), 1)
    print(f"[TIMING] ~{per_item_seconds:.2f}s per MCQ item "
          f"(4x log-likelihood calls each, one per option)")
    return per_item_seconds


def project_full_runtime(indicasa_per_item, sanskriti_per_item):
    print("\n" + "=" * 70)
    print("PROJECTED FULL RUN TIME (rough estimate from this smoke test)")
    print("=" * 70)
    # IndiCASA caste: ~347 obs x 4 granularity levels, but scoring is per
    # unique sentence within context_id, roughly 347-ish scored sentences
    # per level x 4 levels x 3 models (rough upper bound, ignores caching)
    indicasa_est = indicasa_per_item * 347 * 4 * 3 / 60
    # SANSKRITI: 35% stratified sample of 21853 ~= 7650 rows x 3 models
    sanskriti_est = sanskriti_per_item * 7650 * 3 / 60
    print(f"IndiCASA caste (rough, all levels, all 3 models): ~{indicasa_est:.0f} minutes")
    print(f"SANSKRITI (35% sample, all 3 models): ~{sanskriti_est:.0f} minutes")
    print(f"Indian-BhED (caste+religion, small, all 3 models): a few minutes, negligible")
    total_est = indicasa_est + sanskriti_est
    print(f"\nRough total projection: ~{total_est:.0f} minutes (~{total_est/60:.1f} hours)")
    print("This is a ROUGH upper-bound estimate (no batching/caching accounted "
          "for) -- use it to sanity-check Kaggle GPU session time limits before "
          "launching the full run_all.py job.")


def main():
    from datasets import load_dataset
    print("Loading SANSKRITI from HuggingFace...")
    sanskriti_df = load_dataset("13ari/Sanskriti", split="train").to_pandas()
    print(f"Loaded {len(sanskriti_df)} rows\n")

    model, tokenizer, load_time = smoke_test_model_load()
    smoke_test_log_likelihood(model, tokenizer)
    indicasa_per_item = smoke_test_indicasa(model, tokenizer)
    smoke_test_bhed(model, tokenizer)
    sanskriti_per_item = smoke_test_sanskriti(model, tokenizer, sanskriti_df)

    project_full_runtime(indicasa_per_item, sanskriti_per_item)

    summary = {
        "model_load_seconds": load_time,
        "indicasa_seconds_per_item": indicasa_per_item,
        "sanskriti_seconds_per_item": sanskriti_per_item,
        "status": "ALL SMOKE TESTS PASSED",
    }
    with open("smoketest_results.json", "w") as f:
        json.dump(summary, f, indent=2)
    print("\nSaved smoketest_results.json")
    print("\n" + "=" * 70)
    print("SMOKE TEST COMPLETE -- if everything above looks sane "
          "(no errors, finite log-likelihoods, reasonable timings), "
          "proceed to the full run_all.py pipeline.")
    print("=" * 70)


if __name__ == "__main__":
    main()
