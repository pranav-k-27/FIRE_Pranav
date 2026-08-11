"""
Data loaders and granularity-mapping tables for the AMI paper.
Verified against real files on 2026-08-07 (Indian-BhED, IndiCASA) and the
HuggingFace dataset card (SANSKRITI).

This module has NO model-inference dependencies -- it is pure pandas/regex
and can be run/tested anywhere (including this sandbox). Model scoring
(log-likelihood, MCQ accuracy) lives in scoring_models.py, which requires
a GPU + internet access to HuggingFace and should be run on Kaggle.
"""

import re
import pandas as pd


# ---------------------------------------------------------------------------
# 4.1 IndiCASA -- Caste granularity continuum
# ---------------------------------------------------------------------------

# Verified term -> (L2 varna, L3 constitutional, L4 fine label)
# Built from the word-boundary term extraction run on the real CSV on 2026-08-07.
CASTE_TERM_MAP = {
    "brahmin":        {"l1": "brahmin_pool", "l2": "Brahmin",    "l3": "General", "l4": "Brahmin"},
    "kshatriya":       {"l1": "brahmin_pool", "l2": "Kshatriya",  "l3": "General", "l4": "Kshatriya"},
    "vaishya":         {"l1": "brahmin_pool", "l2": "Vaishya",    "l3": "General", "l4": "Vaishya"},
    "bania":           {"l1": "brahmin_pool", "l2": "Vaishya",    "l3": "General", "l4": "Bania"},
    "baniya":          {"l1": "brahmin_pool", "l2": "Vaishya",    "l3": "General", "l4": "Bania"},
    "kayastha":        {"l1": "brahmin_pool", "l2": "Vaishya",    "l3": "General", "l4": "Kayastha"},  # scribe caste, closer to forward
    "rajput":          {"l1": "brahmin_pool", "l2": "Kshatriya",  "l3": "General", "l4": "Rajput"},
    "maratha":         {"l1": "brahmin_pool", "l2": "Kshatriya",  "l3": "General", "l4": "Maratha"},

    "shudra":          {"l1": "dalit_pool",   "l2": "Shudra",     "l3": "OBC",     "l4": "Shudra"},
    "yadav":           {"l1": "dalit_pool",   "l2": "Shudra",     "l3": "OBC",     "l4": "Yadav"},
    "kurmi":           {"l1": "dalit_pool",   "l2": "Shudra",     "l3": "OBC",     "l4": "Kurmi"},
    "jat":             {"l1": "dalit_pool",   "l2": "Shudra",     "l3": "OBC",     "l4": "Jat"},
    "obc":             {"l1": "dalit_pool",   "l2": "Shudra",     "l3": "OBC",     "l4": "OBC"},

    "dalit":           {"l1": "dalit_pool",   "l2": "Shudra",     "l3": "SC",      "l4": "Dalit"},
    "chamar":          {"l1": "dalit_pool",   "l2": "Shudra",     "l3": "SC",      "l4": "Chamar"},
    "harijan":         {"l1": "dalit_pool",   "l2": "Shudra",     "l3": "SC",      "l4": "Harijan"},
    "mahar":           {"l1": "dalit_pool",   "l2": "Shudra",     "l3": "SC",      "l4": "Mahar"},
    "jatav":           {"l1": "dalit_pool",   "l2": "Shudra",     "l3": "SC",      "l4": "Jatav"},
    "sc":              {"l1": "dalit_pool",   "l2": "Shudra",     "l3": "SC",      "l4": "SC(unspecified)"},

    "adivasi":         {"l1": "dalit_pool",   "l2": "Shudra",     "l3": "ST",      "l4": "Adivasi"},
    "scheduled tribe": {"l1": "dalit_pool",   "l2": "Shudra",     "l3": "ST",      "l4": "Scheduled Tribe"},
    "st":              {"l1": "dalit_pool",   "l2": "Shudra",     "l3": "ST",      "l4": "ST(unspecified)"},
    "nadar":           {"l1": "dalit_pool",   "l2": "Shudra",     "l3": "OBC",     "l4": "Nadar"},
    "reddy":           {"l1": "brahmin_pool", "l2": "Kshatriya",  "l3": "General", "l4": "Reddy"},
}

MIN_GROUP_N = 5  # per protocol Section 4.1 / contingency rule


def _find_caste_terms(sentence: str) -> list:
    """Return all caste terms (lowercased keys of CASTE_TERM_MAP) found via
    word-boundary regex match in a sentence. Longer multi-word terms are
    checked first to avoid 'st'/'sc' false-triggering inside other terms.
    """
    s = sentence.lower()
    found = []
    terms_by_length = sorted(CASTE_TERM_MAP.keys(), key=len, reverse=True)
    consumed = set()
    for term in terms_by_length:
        pattern = r"\b" + re.escape(term) + r"\b"
        for m in re.finditer(pattern, s):
            span = (m.start(), m.end())
            if not any(span[0] < c[1] and span[1] > c[0] for c in consumed):
                found.append(term)
                consumed.add(span)
    return found


def load_indicasa_caste(path: str) -> pd.DataFrame:
    """Load IndiCASA caste CSV and attach granularity labels (L1-L4) plus
    context_id for scenario-matched comparison. Rows with zero recognized
    caste terms are dropped (rare; mostly generic "these people" phrasing
    that belongs to the socioeconomic axis, not caste).
    """
    df = pd.read_csv(path)
    records = []
    for _, row in df.iterrows():
        terms = _find_caste_terms(row["sentence"])
        if not terms:
            continue
        # A sentence may contain >1 term (e.g. paired comparisons); explode
        # so each (sentence, term) pair is one observation, tagged with the
        # sentence's stereotype/anti-stereotype label and context_id.
        for term in terms:
            labels = CASTE_TERM_MAP[term]
            records.append({
                "context_id": row["context_id"],
                "sentence": row["sentence"],
                "type": row["type"],  # stereotype / anti_stereotype
                "term": term,
                "l1_group": labels["l1"],
                "l2_group": labels["l2"],
                "l3_group": labels["l3"],
                "l4_group": labels["l4"],
            })
    out = pd.DataFrame(records)
    return out


def filter_min_n(df: pd.DataFrame, group_col: str, min_n: int = MIN_GROUP_N) -> pd.DataFrame:
    """Drop groups with fewer than min_n rows at a given granularity level.

    DEPRECATED for L4 use -- kept only for quick exploratory counts (as in
    the Day-1 verification tests). For the actual experiment pipeline, use
    merge_small_groups() instead, which folds small groups into their
    parent taxonomy level rather than discarding data. Silently dropping
    observations risks the appearance of post-hoc group selection; merging
    upward according to a pre-specified hierarchy does not.
    """
    counts = df[group_col].value_counts()
    keep = counts[counts >= min_n].index
    return df[df[group_col].isin(keep)].copy()


# Pre-specified fold-up hierarchy for IndiCASA caste: if a group at the
# finest level (L4, jati/named-community) has fewer than MIN_GROUP_N
# observations, its rows are relabeled to their L3 (constitutional)
# category instead of being dropped. This hierarchy is fixed BEFORE seeing
# results (locked as of the v3->v4 code pass) -- do not adjust it after
# looking at which groups end up small, per the "lock everything now"
# discipline from the third review.
CASTE_FOLD_UP_PARENT = {"l4_group": "l3_group"}


def merge_small_groups(df: pd.DataFrame, level_col: str, min_n: int = MIN_GROUP_N,
                        fold_up_map: dict = None) -> pd.DataFrame:
    """
    Fold groups below min_n at `level_col` into their parent taxonomy
    level, rather than dropping the observations. Returns a copy of df
    with a new column f"{level_col}_merged" containing either the
    original group label (if it had enough n) or the parent-level label
    (if it was folded up).

    fold_up_map: {level_col: parent_col}. Defaults to CASTE_FOLD_UP_PARENT
    if level_col == 'l4_group'; raises if no mapping is known for other
    level_col values (forces an explicit decision rather than a silent
    wrong default).
    """
    fold_up_map = fold_up_map or CASTE_FOLD_UP_PARENT
    if level_col not in fold_up_map:
        raise ValueError(
            f"No pre-specified fold-up parent for '{level_col}'. "
            f"Add it to the fold_up_map explicitly -- do not guess."
        )
    parent_col = fold_up_map[level_col]

    df = df.copy()
    counts = df[level_col].value_counts()
    small_groups = set(counts[counts < min_n].index)

    merged_col = f"{level_col}_merged"
    df[merged_col] = df.apply(
        lambda row: row[parent_col] if row[level_col] in small_groups else row[level_col],
        axis=1,
    )

    n_folded = df[level_col].isin(small_groups).sum()
    if n_folded > 0:
        folded_labels = sorted(small_groups)
        print(f"[INFO] merge_small_groups: folded {n_folded} rows from "
              f"{len(small_groups)} small groups ({folded_labels}) "
              f"up into '{parent_col}' (min_n={min_n})")

    # Second-order check: a fold-up target bucket can itself still land
    # below min_n (e.g. two differently-worded L4 terms fold into the same
    # parent label via different paths, but one path's resulting bucket
    # stays small). We do NOT recursively re-fold -- that would add a
    # second undisclosed hierarchy decision. Instead, flag it explicitly
    # so it gets a wide-CI warning downstream, same treatment as the
    # small Indian-BhED religion groups.
    post_counts = df[merged_col].value_counts()
    still_small = post_counts[post_counts < min_n].index.tolist()
    if still_small:
        print(f"[WARNING] After one fold-up pass, groups still below "
              f"min_n={min_n} in '{merged_col}': {still_small} -- "
              f"report these with wide bootstrap CIs, do not treat as "
              f"equally reliable as larger groups")

    return df


# ---------------------------------------------------------------------------
# Granularity-level ordinal labels for Day 4 figures (Figure 1: Granularity
# Sensitivity Curve). LOCKED per third-review Issue 4: the x-axis must be
# an ordered taxonomy label, NOT a raw group count -- L2 (varna, 4 groups)
# and L3 (constitutional, ~4 groups) have similar counts but represent
# different, non-nested sociological partitions, and plotting by count
# alone would wrongly imply "more groups = more granular = better."
# ---------------------------------------------------------------------------

CASTE_LEVEL_ORDER = [
    ("l1_group", "Binary"),
    ("l2_group", "Varna"),
    ("l3_group", "Constitutional"),
    ("l4_group_merged", "Jati (fine-grained, small groups folded up)"),
]

SANSKRITI_LEVEL_ORDER = [
    ("l1_group", "Pooled (All-India)"),
    ("l2_group", "Macro-region"),
    ("l3_group", "State/UT"),
]


# ---------------------------------------------------------------------------
# 4.2 SANSKRITI -- Region granularity continuum
# ---------------------------------------------------------------------------

STATE_TO_REGION = {
    # North
    "Delhi": "North", "Punjab": "North", "Haryana": "North", "Himachal_Pradesh": "North",
    "Jammu_and_Kashmir": "North", "Jammu_kashmir": "North",  # dataset uses the
    # latter spelling (confirmed against real SANSKRITI data on 2026-08-07);
    # both kept as keys defensively in case of future dataset revisions.
    "Ladakh": "North", "Uttarakhand": "North", "Chandigarh": "North",
    # South
    "Andhra_Pradesh": "South", "Telangana": "South", "Karnataka": "South", "Kerala": "South",
    "Tamil_Nadu": "South", "Puducherry": "South", "Lakshadweep": "South",
    # East
    "West_Bengal": "East", "Odisha": "East", "Jharkhand": "East", "Bihar": "East",
    # Northeast
    "Assam": "Northeast", "Arunachal_Pradesh": "Northeast", "Manipur": "Northeast",
    "Meghalaya": "Northeast", "Mizoram": "Northeast", "Nagaland": "Northeast",
    "Sikkim": "Northeast", "Tripura": "Northeast",
    # West
    "Maharashtra": "West", "Gujarat": "West", "Rajasthan": "West", "Goa": "West",
    "Dadra_and_Nagar_Haveli_and_Daman_and_Diu": "West",
    # Central
    "Madhya_Pradesh": "Central", "Chhattisgarh": "Central", "Uttar_Pradesh": "Central",
    # Islands (kept separate as their own bucket if small-n; documented in Limitations)
    "Andaman_and_Nicobar": "Islands",
}


def load_sanskriti(df: pd.DataFrame) -> pd.DataFrame:
    """Attach L1 (pooled)/L2 (macro-region)/L3 (state) labels to a SANSKRITI
    dataframe already loaded via `datasets.load_dataset('13ari/Sanskriti')`
    and converted to pandas. Unmapped states are logged, not silently dropped.
    """
    df = df.copy()
    df["l1_group"] = "ALL_INDIA"
    df["l2_group"] = df["state"].map(STATE_TO_REGION).fillna("Unmapped")
    df["l3_group"] = df["state"]
    unmapped = df.loc[df["l2_group"] == "Unmapped", "state"].unique().tolist()
    if unmapped:
        print(f"[WARNING] {len(unmapped)} state values not in STATE_TO_REGION: {unmapped}")
    return df


def clean_mojibake(text) -> str:
    """Fix UTF-8/Latin-1 mojibake in SANSKRITI's text fields using ftfy
    (pip installable, add to kaggle_entry.py's pip install line).

    A hand-rolled single-pass encode('latin1').decode('utf-8') was tried
    first and found insufficient: the real data contains DOUBLE-mojibaked
    text (e.g. 'ChÃÂ¼moukedima district', confirmed from the actual
    Kaggle EDA run on 2026-08-07), which raises UnicodeDecodeError on a
    single round-trip and needs iterative/smarter repair. ftfy handles
    both single- and double-mojibake correctly and leaves already-clean
    text untouched -- verified against the real example above, which
    correctly recovers to 'Chümoukedima district'.
    """
    if not isinstance(text, str) or "Ã" not in text:
        return text
    import ftfy
    return ftfy.fix_text(text)


def clean_sanskriti_text_columns(df: pd.DataFrame,
                                   columns=("question", "option1", "option2", "option3",
                                            "option4", "answer")) -> pd.DataFrame:
    """Apply clean_mojibake() to the columns that matter for scoring
    correctness (deliberately excluding 'short explaination / source link',
    which is metadata, not something the scorer compares against). Prints
    a before/after affected-row count per column so the fix is auditable,
    not silent.
    """
    df = df.copy()
    for col in columns:
        if col not in df.columns:
            continue
        before = df[col].astype(str).str.contains("Ã", na=False).sum()
        df[col] = df[col].map(clean_mojibake)
        after = df[col].astype(str).str.contains("Ã", na=False).sum()
        if before > 0:
            print(f"[INFO] clean_sanskriti_text_columns: '{col}' "
                  f"{before} -> {after} rows still containing 'Ã' after cleaning")
    return df


def stratified_sample_sanskriti(df: pd.DataFrame, frac: float = 0.35, seed: int = 42,
                                  min_n_per_state: int = 50) -> pd.DataFrame:
    """Stratified sample by state, enforcing a floor on per-state n so the
    L3 (fine-grained) analysis stays valid after subsampling."""
    parts = []
    for state, g in df.groupby("state"):
        n = max(min_n_per_state, int(len(g) * frac))
        n = min(n, len(g))
        parts.append(g.sample(n=n, random_state=seed))
    return pd.concat(parts, ignore_index=True)


# ---------------------------------------------------------------------------
# 4.3 Indian-BhED -- motivating example + thin religion confirmation
# ---------------------------------------------------------------------------

def load_indianbhed_caste(path: str) -> pd.DataFrame:
    """Load as-is: only ever produces the L1 binary split (Dalit-pool vs
    Brahmin-pool). No finer level exists -- see protocol Section 4.3."""
    df = pd.read_csv(path)
    def pool(x):
        x = str(x).strip("[]'\" ").lower()
        return "dalit_pool" if "dalit" in x else ("brahmin_pool" if "brahmin" in x else "other")
    df["l1_group_stereo"] = df["Target_Stereotypical"].map(pool)
    df["l1_group_antistereo"] = df["Target_Anti-Stereotypical"].map(pool)
    return df


RELIGION_MAP = {
    "hindu": "Hindu", "hindus": "Hindu", "hinduism": "Hindu",
    "muslim": "Muslim", "muslims": "Muslim", "islam": "Muslim", "islamic": "Muslim",
    "christian": "Christian", "christianity": "Christian",
    "buddhist": "Buddhist", "buddhists": "Buddhist", "buddhism": "Buddhist",
    "sikh": "Sikh", "sikhs": "Sikh",
    "jain": "Jain", "jainism": "Jain",
}


def load_indianbhed_religion(path: str, min_n: int = 4) -> pd.DataFrame:
    """Load religion file and attach a cleaned single-label religion group
    where possible. Multi-label / messy entries (e.g. 'peacefule, dress')
    are dropped and counted, per the honesty note in protocol Section 4.3."""
    df = pd.read_csv(path)

    def clean(x):
        x = str(x).strip("[]'\" ").lower()
        # take first comma-separated token if multiple
        first = x.split(",")[0].strip().strip("'\" ")
        return RELIGION_MAP.get(first, None)

    df["religion_group"] = df["Target_Stereotypical"].map(clean)
    n_dropped = df["religion_group"].isna().sum()
    print(f"[INFO] Indian-BhED religion: dropped {n_dropped}/{len(df)} rows with unmapped/messy labels")
    df = df.dropna(subset=["religion_group"])
    counts = df["religion_group"].value_counts()
    print(f"[INFO] Religion group counts after cleaning:\n{counts}")
    small = counts[counts < min_n].index.tolist()
    if small:
        print(f"[WARNING] Groups below min_n={min_n}, report with wide CIs: {small}")
    return df


# ---------------------------------------------------------------------------
# EDA (Exploratory Data Analysis) -- run these BEFORE any model scoring.
# Purpose: understand group balance, sentence-length effects, missingness,
# and context/scenario structure well enough to sanity-check every number
# that comes out of the scoring pipeline later. Each function prints a
# human-readable report AND returns a dict of the same numbers, so it can
# be used both interactively (Kaggle notebook cell) and programmatically
# (e.g. dumped to JSON alongside the experiment results for the paper's
# reproducibility package).
# ---------------------------------------------------------------------------

def _sentence_len_stats(sentences: pd.Series) -> dict:
    lengths = sentences.astype(str).str.split().str.len()
    return {
        "mean_words": round(float(lengths.mean()), 2),
        "median_words": float(lengths.median()),
        "min_words": int(lengths.min()),
        "max_words": int(lengths.max()),
        "std_words": round(float(lengths.std()), 2),
    }


def eda_indicasa_caste(path: str = None, df: pd.DataFrame = None) -> dict:
    """EDA for IndiCASA caste data, BEFORE granularity mapping (raw file)
    and AFTER (mapped + folded), so you can see what the term-matching
    and fold-up steps actually did to the data.
    """
    assert path or df is not None, "provide either a path or a raw dataframe"
    raw = pd.read_csv(path) if path else df.copy()

    print("=" * 70)
    print("EDA: IndiCASA -- Caste (raw file, before term-matching)")
    print("=" * 70)
    print(f"Total rows: {len(raw)}")
    print(f"Distinct context_id scenarios: {raw['context_id'].nunique()}")
    print(f"type (stereotype/anti_stereotype) balance:\n{raw['type'].value_counts()}")
    print(f"Missing values per column:\n{raw.isna().sum()}")
    len_stats = _sentence_len_stats(raw["sentence"])
    print(f"Sentence length (words): {len_stats}")
    items_per_context = raw.groupby("context_id").size()
    print(f"Items per context_id: min={items_per_context.min()}, "
          f"max={items_per_context.max()}, mean={items_per_context.mean():.1f}")
    dup_sentences = raw["sentence"].duplicated().sum()
    print(f"Exact duplicate sentences: {dup_sentences}")

    # After term-matching + granularity mapping + fold-up (requires a path;
    # load_indicasa_caste re-reads and reprocesses the raw CSV rather than
    # taking an in-memory dataframe, since term-matching runs off the raw
    # 'sentence' column directly).
    if path:
        mapped = load_indicasa_caste(path)
        print()
        print("-" * 70)
        print("EDA: IndiCASA -- Caste (after term-matching + granularity mapping)")
        print("-" * 70)
        print(f"Observations after exploding (sentence, term) pairs: {len(mapped)}")
        n_sentences_no_term = raw.drop_duplicates("sentence").shape[0] - mapped.drop_duplicates("sentence").shape[0]
        print(f"Raw sentences with zero recognized caste term (excluded from mapped data): "
              f"~{n_sentences_no_term}")
        for level in ["l1_group", "l2_group", "l3_group", "l4_group"]:
            vc = mapped[level].value_counts()
            print(f"\n{level} distribution ({vc.shape[0]} groups):")
            print(vc)
        merged = merge_small_groups(mapped, "l4_group", min_n=5)
        print(f"\nl4_group_merged distribution after folding small groups up:")
        print(merged["l4_group_merged"].value_counts())
    else:
        print("[INFO] No path provided -- skipping post-mapping section "
              "(term-matching requires re-reading the raw sentence column)")

    return {
        "raw_n": len(raw),
        "n_context_ids": int(raw["context_id"].nunique()),
        "type_balance": raw["type"].value_counts().to_dict(),
        "missing_values": raw.isna().sum().to_dict(),
        "sentence_length": len_stats,
        "items_per_context_min_max_mean": (
            int(items_per_context.min()), int(items_per_context.max()), round(float(items_per_context.mean()), 2)
        ),
        "duplicate_sentences": int(dup_sentences),
    }


def eda_indianbhed_caste(path: str) -> dict:
    print("=" * 70)
    print("EDA: Indian-BhED -- Caste")
    print("=" * 70)
    raw = pd.read_csv(path)
    print(f"Total rows: {len(raw)}")
    print(f"Columns: {raw.columns.tolist()}")
    print(f"Missing values per column:\n{raw.isna().sum()}")
    print(f"\nTarget_Stereotypical value counts:\n{raw['Target_Stereotypical'].value_counts()}")
    print(f"\nTarget_Anti-Stereotypical value counts:\n{raw['Target_Anti-Stereotypical'].value_counts()}")
    multi_label = raw["Target_Stereotypical"].astype(str).str.contains(",").sum()
    print(f"\nMulti-label rows (contain comma, will be skipped by the pairwise scorer): {multi_label}")
    if "Sentence" in raw.columns:
        len_stats = _sentence_len_stats(raw["Sentence"])
        print(f"Sentence length (words): {len_stats}")
        mask_present = raw["Sentence"].astype(str).str.contains("MASK").sum()
        print(f"Rows containing 'MASK' placeholder: {mask_present}/{len(raw)}")
    else:
        len_stats = None
    return {
        "raw_n": len(raw),
        "missing_values": raw.isna().sum().to_dict(),
        "target_stereotypical_counts": raw["Target_Stereotypical"].value_counts().to_dict(),
        "multi_label_rows": int(multi_label),
        "sentence_length": len_stats,
    }


def eda_indianbhed_religion(path: str) -> dict:
    print("=" * 70)
    print("EDA: Indian-BhED -- Religion")
    print("=" * 70)
    raw = pd.read_csv(path)
    print(f"Total rows: {len(raw)}")
    print(f"Missing values per column:\n{raw.isna().sum()}")
    print(f"\nTarget_Stereotypical raw value counts (before cleaning):\n{raw['Target_Stereotypical'].value_counts()}")

    cleaned = load_indianbhed_religion(path)
    print(f"\nRows retained after cleaning to single-label religion terms: {len(cleaned)}/{len(raw)}")
    counts = cleaned["religion_group"].value_counts()
    pct_of_dominant = (counts.iloc[0] / counts.sum() * 100) if len(counts) else 0
    print(f"Most dominant group makes up {pct_of_dominant:.1f}% of the cleaned data "
          f"-- a strong class-imbalance signal to keep in mind when reading any per-group score")
    return {
        "raw_n": len(raw),
        "cleaned_n": len(cleaned),
        "religion_group_counts": counts.to_dict(),
        "dominant_group_pct": round(float(pct_of_dominant), 1),
    }


def eda_sanskriti(df: pd.DataFrame, frac_for_sample_preview: float = 0.35) -> dict:
    """
    df: the SANSKRITI dataframe already loaded via
    datasets.load_dataset('13ari/Sanskriti').to_pandas() on Kaggle.
    Cannot be run in an offline sandbox -- call this on Kaggle right after
    loading, before any model scoring, to sanity-check the data first.
    """
    print("=" * 70)
    print("EDA: SANSKRITI")
    print("=" * 70)
    print(f"Total rows: {len(df)}")
    print(f"Columns: {df.columns.tolist()}")
    print(f"Missing values per column:\n{df.isna().sum()}")

    print(f"\nDistinct states/UTs: {df['state'].nunique()}")
    state_counts = df["state"].value_counts()
    print(f"Items per state: min={state_counts.min()} ({state_counts.idxmin()}), "
          f"max={state_counts.max()} ({state_counts.idxmax()}), mean={state_counts.mean():.1f}")
    print(f"\nFull per-state item counts:\n{state_counts}")

    print(f"\nAttribute distribution ({df['attribute'].nunique()} attributes):")
    print(df["attribute"].value_counts())

    if "question_type" in df.columns:
        print(f"\nQuestion type distribution:")
        print(df["question_type"].value_counts())

    mapped = load_sanskriti(df)
    print(f"\nRegion (L2) distribution after state-to-region mapping:")
    print(mapped["l2_group"].value_counts())
    unmapped_n = (mapped["l2_group"] == "Unmapped").sum()
    if unmapped_n:
        print(f"[WARNING] {unmapped_n} rows have an unmapped state -- "
              f"check STATE_TO_REGION dict for missing entries before scoring")

    sampled = stratified_sample_sanskriti(df, frac=frac_for_sample_preview)
    sampled_state_counts = sampled["state"].value_counts()
    print(f"\nAfter stratified {frac_for_sample_preview:.0%} sample "
          f"(min_n_per_state=50 floor): total rows = {len(sampled)}, "
          f"smallest state n = {sampled_state_counts.min()} ({sampled_state_counts.idxmin()})")

    encoding_flags = df.astype(str).apply(
        lambda col: col.str.contains("Ã", na=False).sum()
    )
    flagged_cols = encoding_flags[encoding_flags > 0]
    if len(flagged_cols):
        print(f"\n[WARNING] Possible UTF-8 mis-decode artifacts ('Ã') found in columns:\n{flagged_cols}")

    return {
        "raw_n": len(df),
        "n_states": int(df["state"].nunique()),
        "state_item_counts": state_counts.to_dict(),
        "attribute_counts": df["attribute"].value_counts().to_dict(),
        "region_counts_after_mapping": mapped["l2_group"].value_counts().to_dict(),
        "unmapped_state_rows": int(unmapped_n),
        "sample_n_after_stratified_35pct": len(sampled),
        "sample_min_state_n": int(sampled_state_counts.min()),
        "possible_encoding_artifact_columns": flagged_cols.to_dict() if len(flagged_cols) else {},
    }


def run_all_eda(indicasa_caste_path: str, bhed_caste_path: str, bhed_religion_path: str,
                 sanskriti_df: pd.DataFrame = None) -> dict:
    """Convenience wrapper: runs every EDA function and returns one combined
    dict, suitable for json.dump()-ing alongside the experiment results so
    the paper's reproducibility package includes the exact data profile
    the experiments were run against."""
    report = {}
    report["indicasa_caste"] = eda_indicasa_caste(path=indicasa_caste_path)
    report["bhed_caste"] = eda_indianbhed_caste(bhed_caste_path)
    report["bhed_religion"] = eda_indianbhed_religion(bhed_religion_path)
    if sanskriti_df is not None:
        report["sanskriti"] = eda_sanskriti(sanskriti_df)
    else:
        print("[INFO] Skipping SANSKRITI EDA -- no dataframe provided "
              "(load it on Kaggle via datasets.load_dataset('13ari/Sanskriti') first)")
    return report
