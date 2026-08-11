# Aggregation Matters: Evaluation Code and Analysis Pipeline

Code accompanying the paper *"Aggregation Matters: Evaluating the Effect of
Grouping Strategy on India-Centric LLM Bias Benchmarks"* (submitted to FIRE
2026, under double-blind review). This repository is anonymized for review;
author identity has been removed from all files and commit history.

## What this repository contains

| File | Purpose |
|---|---|
| `data_loaders.py` | Loads and granularity-maps Indian-BhED, IndiCASA, and SANSKRITI; includes the pre-specified small-group fold-up hierarchy for jati-level categories and the mojibake-cleaning utility for SANSKRITI text fields |
| `scoring_models.py` | Model loading (HuggingFace, with an automatic Kaggle Model Hub fallback for gated weights) and scoring functions (log-likelihood stereotype preference, log-likelihood MCQ accuracy) |
| `metrics.py` | Aggregation Masking Index (AMI) computation, bootstrap confidence intervals, the self-referential subgroup-deviation criterion, and the exploratory AMI-vs-variance covariate analysis |
| `run_all.py` | Orchestrates the full evaluation pipeline across all three benchmarks, all granularity levels, and all three models |
| `run_smoketest.py` | A small, fast pipeline validation run (1 model, ~15-30 items per dataset) used before committing to the full run |
| `save_l4_detail.py` | Saves the full per-community (rate, n) breakdown at IndiCASA caste's finest level, needed for the random-partition ablation |
| `random_partition_ablation.py` | The random-partition ablation (Section 6.2 of the paper): generates 1,000 random four-group partitions and locates the real-world Varna and Constitutional partitions within that distribution |
| `leaderboard_rank_check.py` | The leaderboard rank-sensitivity check (Section 6.6): compares model rankings under pooled vs. worst-case scoring |
| `reprocess_verdicts.py` | Recomputes the self-referential subgroup-deviation flag from already-saved results, with no GPU/re-inference required |
| `generate_paper_figures.py` | Generates all tables and figures used in the paper directly from result JSON files |
| `kaggle_pipeline/` | Scripts for running the pipeline on Kaggle (see note below) |

## Reproducing the results

1. Install dependencies: `pip install -r requirements.txt`
2. Obtain the three benchmark datasets:
   - Indian-BhED: `github.com/khyatikhandelwal/Indian-LLMs-Bias`
   - IndiCASA: `github.com/cerai-iitm/IndiCASA`
   - SANSKRITI: `huggingface.co/datasets/13ari/Sanskriti`
3. Run the smoke test first: `python run_smoketest.py` (validates the pipeline on one model before committing to the full run)
4. Run the full pipeline: `python run_all.py` (produces `all_results.json` and `ami_cells.json`)
5. Reprocess with the self-referential deviation criterion: `python reprocess_verdicts.py ami_cells.json`
6. Run the random-partition ablation (requires `save_l4_detail.py`'s output first): `python random_partition_ablation.py l4_group_detail.json ./figures`
7. Generate all paper figures and tables: `python generate_paper_figures.py ami_cells_v2.json ./figures`

## `kaggle_pipeline/`

These scripts run the evaluation on Kaggle's free GPU tier. They have been
genericized for this anonymized release — replace `YOUR-KAGGLE-USERNAME`
with your own Kaggle username, and the dataset/kernel identifiers with
your own, before use. They require a Kaggle account and API token; see
[Kaggle's API documentation](https://www.kaggle.com/docs/api) for setup.

Note: the top-level `push_to_kaggle.py` and its corresponding
`kaggle_entry.py` (which run the full pipeline as a single Kaggle kernel)
follow the same pattern as the EDA/smoke-test/L4-detail variants included
here, but are not included separately in this release; adapt
`kaggle_pipeline/push_eda_to_kaggle.py` as a template if you need the
equivalent full-pipeline version.

## Models used

Llama-3.2-3B-Instruct, Mistral-7B-Instruct-v0.3, Qwen2.5-7B-Instruct, all
loaded in 4-bit quantization. All three are publicly available via
HuggingFace; Llama-3.2 additionally has a Kaggle Model Hub fallback path
in `scoring_models.py` for accounts without confirmed gated-repo access.

## Data-quality corrections applied

Two issues were identified during exploratory analysis and corrected
before scoring (see `data_loaders.py`):

1. A naming mismatch between the region-mapping table and the literal
   state label used in the released SANSKRITI data (`Jammu_kashmir`, not
   `Jammu_and_Kashmir`), affecting all Jammu & Kashmir items.
2. Double-encoded (mojibake) text in a small fraction of SANSKRITI's
   question, option, and answer fields, corrected with `ftfy` prior to
   scoring.

## License

MIT License (see `LICENSE`).

## Citation

Citation details withheld during double-blind review; a `CITATION.cff`
will be added upon acceptance.
