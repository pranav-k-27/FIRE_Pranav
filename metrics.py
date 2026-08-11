"""
Metrics module for the AMI paper: Aggregation Masking Index, bootstrap CIs,
verdict-flip detection, and theorem-validation statistics.

No model-inference dependencies. Operates on already-scored group rates
(a dict {group_name: rate} or a dataframe of per-item scores + group labels).
"""

import numpy as np
from scipy import stats as sps


# ---------------------------------------------------------------------------
# Core AMI computation (Section 3 of protocol)
# ---------------------------------------------------------------------------

def compute_ami(group_rates: dict, pooled_rate: float) -> dict:
    """
    group_rates: {group_name: rate} at the finest available granularity
    pooled_rate: the item-weighted pooled rate (what the benchmark's own
                 default protocol would report)

    Returns AMI, AMI-median, AMI-p90, and the underlying worst/median/p90
    group rates plus between-group sigma, for full transparency in tables.
    """
    if not group_rates:
        raise ValueError("group_rates is empty")

    rates = np.array(list(group_rates.values()), dtype=float)
    worst_group = max(group_rates, key=group_rates.get)
    worst = group_rates[worst_group]
    median = float(np.median(rates))
    p90 = float(np.percentile(rates, 90))
    sigma_between = float(np.std(rates, ddof=0))

    def _ami(target):
        if target == 0:
            return 0.0
        return max(0.0, (target - pooled_rate) / target)

    return {
        "ami": _ami(worst),
        "ami_median": _ami(median),
        "ami_p90": _ami(p90),
        "worst_group": worst_group,
        "worst_rate": worst,
        "median_rate": median,
        "p90_rate": p90,
        "pooled_rate": pooled_rate,
        "sigma_between": sigma_between,
        "n_groups": len(rates),
    }


# ---------------------------------------------------------------------------
# Bootstrap confidence intervals
# ---------------------------------------------------------------------------

def bootstrap_ci_rate(binary_outcomes: np.ndarray, n_boot: int = 1000,
                       ci: float = 0.95, seed: int = 42) -> tuple:
    """CI for a simple rate (e.g. stereotype-preference rate, accuracy)
    computed from an array of 0/1 outcomes."""
    rng = np.random.default_rng(seed)
    n = len(binary_outcomes)
    if n == 0:
        return (np.nan, np.nan)
    boot_means = np.empty(n_boot)
    for i in range(n_boot):
        sample = rng.choice(binary_outcomes, size=n, replace=True)
        boot_means[i] = sample.mean()
    lo = np.percentile(boot_means, (1 - ci) / 2 * 100)
    hi = np.percentile(boot_means, (1 + ci) / 2 * 100)
    return (float(lo), float(hi))


def bootstrap_ci_ami(group_item_outcomes: dict, pooled_outcomes: np.ndarray,
                      n_boot: int = 1000, ci: float = 0.95, seed: int = 42) -> tuple:
    """CI for AMI itself, by resampling items within each group and the
    pooled set jointly, recomputing AMI each draw.

    group_item_outcomes: {group_name: np.array of 0/1 item outcomes}
    pooled_outcomes: np.array of 0/1 item outcomes for the pooled set
    """
    rng = np.random.default_rng(seed)
    boot_amis = np.empty(n_boot)
    for i in range(n_boot):
        boot_group_rates = {}
        for g, arr in group_item_outcomes.items():
            if len(arr) == 0:
                continue
            sample = rng.choice(arr, size=len(arr), replace=True)
            boot_group_rates[g] = sample.mean()
        boot_pooled = rng.choice(pooled_outcomes, size=len(pooled_outcomes), replace=True).mean()
        result = compute_ami(boot_group_rates, boot_pooled)
        boot_amis[i] = result["ami"]
    lo = np.percentile(boot_amis, (1 - ci) / 2 * 100)
    hi = np.percentile(boot_amis, (1 + ci) / 2 * 100)
    return (float(lo), float(hi))


# ---------------------------------------------------------------------------
# Verdict-flip detection (Section 6 of protocol -- thresholds passed in,
# never invented inside this module)
# ---------------------------------------------------------------------------

def verdict(rate: float, threshold: float, higher_is_worse: bool = True) -> str:
    if higher_is_worse:
        return "FLAGGED" if rate >= threshold else "OK"
    else:
        return "FLAGGED" if rate <= threshold else "OK"


def detect_flip(pooled_rate: float, worst_rate: float, threshold: float,
                 higher_is_worse: bool = True) -> dict:
    v_pooled = verdict(pooled_rate, threshold, higher_is_worse)
    v_worst = verdict(worst_rate, threshold, higher_is_worse)
    return {
        "pooled_verdict": v_pooled,
        "worst_verdict": v_worst,
        "flipped": v_pooled != v_worst,
        "threshold_used": threshold,
    }


def detect_flip_selfref(pooled_rate: float, worst_rate: float, sigma_between: float,
                         higher_is_worse: bool = True, n_sigma: float = 1.0) -> dict:
    """
    Self-referential verdict-flip detection: a group is FLAGGED if its rate
    exceeds (or, for higher_is_worse=False, falls below) the cell's OWN
    pooled rate by more than n_sigma standard deviations of its OWN
    between-group spread (sigma_between). The pooled rate is compared
    against itself, so it is FLAGGED only in the degenerate case
    sigma_between <= 0 (i.e. it is always the reference point, never
    flagged by construction) -- flips are driven entirely by whether the
    worst group's rate exceeds the pooled-rate-plus-spread bound.

    This replaces an externally borrowed fixed threshold (e.g. 0.2, taken
    from a different framework's own reporting convention) with a
    threshold computed from data already produced by this exact cell,
    addressing the "you invented the cutoff" concern directly: the cutoff
    is derived from the cell's own pooled rate and its own measured
    heterogeneity, not picked externally. It also mirrors the convention
    already used for SANSKRITI's EDA ('more than 1 SD below the model's
    own overall mean').

    IMPORTANT: this fully replaces the old fixed threshold approach (see
    detect_flip() above, kept only for reference/backward compatibility --
    do not use it for new results, it produced a degenerate zero-flip
    result on the first full run because 0.2 was far below the actual
    observed pooled rates of 0.30-0.70 across every dataset).
    """
    self_threshold = pooled_rate + n_sigma * sigma_between if higher_is_worse else pooled_rate - n_sigma * sigma_between
    v_pooled = verdict(pooled_rate, self_threshold, higher_is_worse)
    v_worst = verdict(worst_rate, self_threshold, higher_is_worse)
    return {
        "pooled_verdict": v_pooled,
        "worst_verdict": v_worst,
        "flipped": v_pooled != v_worst,
        "self_threshold_used": self_threshold,
        "n_sigma": n_sigma,
    }


# ---------------------------------------------------------------------------
# RQ4: exploratory covariate analysis -- does AMI tend to track between-
# group variance across cells?
#
# IMPORTANT (post-review correction): this is EXPLORATORY, not a validation
# of a proven bound. We do NOT have a derived, provable lower bound of the
# form "gap >= c * sigma_between" for an unconstrained constant c -- a
# reviewer correctly pointed out that a worst-group with small population
# weight can deviate arbitrarily without forcing a large pooled-vs-worst
# gap, so no such bound holds in general. (A real, citable, DERIVABLE
# result in this space is the Bhatia-Davis inequality, which bounds
# Var(X) <= (max - mean)(mean - min) for bounded X -- this requires the
# min group rate as an extra input, not sigma alone, and is left as
# optional future rigor, not claimed here.)
#
# What we DO report honestly: whether AMI empirically correlates with
# sigma_between across the cells we collected. A weak or null correlation
# does NOT falsify Theorem 1 (which only claims pooled <= worst, and is
# unconditionally true) -- it just means variance alone is not a strong
# predictor of gap SIZE in our data. Report accordingly; do not describe
# this function's output as "validating" anything.
# ---------------------------------------------------------------------------

def variance_covariate_analysis(ami_results: list) -> dict:
    """
    ami_results: list of dicts, each the output of compute_ami() for one
    (model, dataset, axis, granularity-level) cell.

    Returns Spearman and Pearson correlation between AMI and sigma_between,
    plus a simple OLS slope, reported as an EXPLORATORY covariate
    relationship (see module-level note above) -- not a test of a proven
    bound.
    """
    amis = np.array([r["ami"] for r in ami_results])
    sigmas = np.array([r["sigma_between"] for r in ami_results])

    if len(amis) < 3:
        return {"note": "insufficient cells for correlation (<3)"}

    spearman_r, spearman_p = sps.spearmanr(sigmas, amis)
    pearson_r, pearson_p = sps.pearsonr(sigmas, amis)
    slope, intercept, r_value, p_value, std_err = sps.linregress(sigmas, amis)

    return {
        "spearman_r": spearman_r, "spearman_p": spearman_p,
        "pearson_r": pearson_r, "pearson_p": pearson_p,
        "ols_slope": slope, "ols_intercept": intercept, "ols_r2": r_value ** 2,
        "n_cells": len(amis),
        "interpretation": "EXPLORATORY covariate correlation, not a validated bound",
    }


# ---------------------------------------------------------------------------
# RQ5: cross-benchmark rank agreement (thin/secondary)
# ---------------------------------------------------------------------------

def cross_benchmark_rank_agreement(scores_a: dict, scores_b: dict) -> dict:
    """scores_a, scores_b: {model_name: pooled_bias_rate} from two different
    benchmarks. Returns Spearman rank correlation of model rankings."""
    models = sorted(set(scores_a) & set(scores_b))
    if len(models) < 3:
        return {"note": f"only {len(models)} shared models, correlation unreliable"}
    a = [scores_a[m] for m in models]
    b = [scores_b[m] for m in models]
    r, p = sps.spearmanr(a, b)
    return {"spearman_r": r, "spearman_p": p, "models": models}
