#!/usr/bin/env python3
"""
Generate Experiment 1 CSV: validation of parent-child feature relationships.

Works from checkpoint or latest session JSON + reviews CSV. No Neo4j needed.

Output columns:
  parent_feature   - LLM label (Ollama) or most-specific-feature fallback
  child_feature    - most specific feature from the cluster
  sibling_features - other canonical features in the cluster
  example_reviews  - up to 3 reviews mentioning the child feature
  app_name         - source app
  n_siblings       - sibling count
  cluster_size     - features after deduplication
  tree_id          - unique cluster identifier

Stratification: equal distribution across cluster-size buckets; one row per cluster.

Usage:
  .venv/bin/python scripts/generate_experiment1.py [--n 300] [--out data/experiment1.csv]
"""
import argparse
import json
import logging
import random
import re
import sys
from collections import defaultdict
from pathlib import Path

import pandas as pd
import requests

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

REVIEWS_CSV     = PROJECT_ROOT / "data/input/endpoint_1_process_reviews/mobile_apps/mobilerec_reviews_pipeline.csv"
CHECKPOINT_FILE = PROJECT_ROOT / "evaluation_results/mobile_pipeline_checkpoint.json"
OLLAMA_BASE_URL = "http://localhost:11434"
OLLAMA_MODEL    = "llama3.2:3b"
RANDOM_SEED     = 42

# ---------------------------------------------------------------------------
# Quality thresholds
# ---------------------------------------------------------------------------
MAX_CLUSTER_SIZE  = 12   # garbage-dump guard
MIN_CLUSTER_SIZE  = 2    # need at least one sibling
MIN_FEATURE_LEN   = 3    # chars
MAX_NOISE_RATIO   = 0.5
MIN_REVIEW_HITS   = 1    # child must appear in at least 1 review

NOISE_FEATURES = {
    "app", "apps", "use", "used", "user", "users", "get", "got", "make",
    "good", "bad", "best", "new", "old", "free", "pay", "paid",
    "calls", "log", "bug", "fix", "tap", "click", "open",
    "work", "works", "need", "want", "like", "love", "hate", "try",
    "time", "day", "year", "week", "month", "way", "thing", "things",
    "lot", "bit", "one", "two", "three", "four", "five",
    "imo", "btw", "lol", "omg", "fyi",
}

# ---------------------------------------------------------------------------
# Feature cleaning & deduplication
# ---------------------------------------------------------------------------

def is_noise(feature: str) -> bool:
    f = feature.strip().lower()
    if len(f) < MIN_FEATURE_LEN:
        return True
    if f in NOISE_FEATURES:
        return True
    # 1-2 char tokens
    if " " not in f and "-" not in f and len(f) <= 2:
        return True
    return False


def _stem(s: str) -> str:
    """Very lightweight stemmer: strip common suffixes."""
    for suffix in ("ings", "ing", "tion", "tions", "ies", "es", "s"):
        if s.endswith(suffix) and len(s) - len(suffix) >= 3:
            return s[: -len(suffix)]
    return s


def deduplicate_features(features: list[str]) -> list[str]:
    """
    Remove morphological duplicates within a cluster.
    For each group of variants (share/sharing/sharing documents),
    keep only the most specific (longest) representative.
    """
    features = [f for f in features if not is_noise(f)]
    if not features:
        return []

    # Group by stem of first word
    groups: dict[str, list[str]] = defaultdict(list)
    for f in features:
        first_word = f.split()[0].lower()
        groups[_stem(first_word)].append(f)

    canonical = []
    for group in groups.values():
        # Within each stem group, keep the longest (most specific) feature
        # unless one is a substring of another → keep the longer
        group_sorted = sorted(group, key=len, reverse=True)
        kept = [group_sorted[0]]
        for candidate in group_sorted[1:]:
            # Keep if it's meaningfully different from all kept ones
            if not any(candidate.lower() in k.lower() or k.lower() in candidate.lower()
                       for k in kept):
                kept.append(candidate)
        canonical.extend(kept)

    return canonical


def is_quality_cluster(features: list[str]) -> tuple[bool, str]:
    if len(features) > MAX_CLUSTER_SIZE:
        return False, f"too large ({len(features)})"
    if len(features) < MIN_CLUSTER_SIZE:
        return False, f"too small ({len(features)})"
    noise_n = sum(1 for f in features if is_noise(f))
    if noise_n / len(features) > MAX_NOISE_RATIO:
        return False, f"too noisy ({noise_n}/{len(features)})"
    clean = deduplicate_features(features)
    if len(clean) < MIN_CLUSTER_SIZE:
        return False, f"only {len(clean)} after dedup"
    return True, "ok"


# ---------------------------------------------------------------------------
# Parent label generation
# ---------------------------------------------------------------------------

def _specificity(feature: str) -> int:
    """Score how specific a feature is — prefer multi-word, longer features."""
    return len(feature.split()) * 10 + len(feature)


def fallback_label(features: list[str]) -> str:
    """Best single-feature label when Ollama is unavailable."""
    return sorted(features, key=_specificity, reverse=True)[0]


_PROMPT_LEAK_WORDS = {"product", "analyst", "labeling", "human", "study", "category", "features"}


def _is_valid_label(label: str, features: list[str]) -> bool:
    if not label or len(label) > 50:
        return False
    words = label.split()
    if len(words) > 6:
        return False  # prompt leakage — too long
    if any(c in label for c in (",", "_", ":", ".")):
        return False
    if label in [f.lower() for f in features]:
        return False
    if sum(1 for w in words if w in _PROMPT_LEAK_WORDS) >= 2:
        return False
    return True


def generate_label_ollama(features: list[str]) -> str:
    # Few-shot completion format — small models complete patterns better than following instructions
    prompt = (
        "Features: camera zoom, zoom controls, manual focus\n"
        "Category: camera controls\n\n"
        "Features: push notification, alerts, message\n"
        "Category: notification settings\n\n"
        "Features: credit card, payment, checkout\n"
        "Category: payment methods\n\n"
        f"Features: {', '.join(features[:6])}\n"
        "Category:"
    )
    try:
        resp = requests.post(
            f"{OLLAMA_BASE_URL}/api/generate",
            json={"model": OLLAMA_MODEL, "prompt": prompt, "stream": False},
            timeout=30,
        )
        if resp.ok:
            label = resp.json().get("response", "").strip().strip('"').lower()
            label = label.splitlines()[0].strip()  # take only first line
            if _is_valid_label(label, features):
                return label
    except Exception as e:
        logger.debug(f"Ollama failed: {e}")
    return fallback_label(features)


# ---------------------------------------------------------------------------
# Review search
# ---------------------------------------------------------------------------

def find_reviews(reviews_df: pd.DataFrame, app_name: str, feature: str, n: int = 3) -> list[str]:
    app_reviews = reviews_df[reviews_df["app_name"] == app_name]["review"].dropna()
    pattern = re.compile(re.escape(feature), re.IGNORECASE)
    matches = app_reviews[app_reviews.str.contains(pattern, regex=True)].tolist()
    random.shuffle(matches)
    return matches[:n]


def best_child(features: list[str], reviews_df: pd.DataFrame, app_name: str) -> tuple[str, list[str]]:
    """
    Pick the child feature that is:
      1. Most specific (multi-word preferred over single-word)
      2. Has at least MIN_REVIEW_HITS review mentions
    Returns (child_feature, reviews) or (None, []) if none qualify.
    """
    candidates = sorted(features, key=_specificity, reverse=True)
    for feat in candidates:
        reviews = find_reviews(reviews_df, app_name, feat)
        if len(reviews) >= MIN_REVIEW_HITS:
            return feat, reviews
    return None, []


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_clusters() -> list[dict]:
    if CHECKPOINT_FILE.exists():
        with open(CHECKPOINT_FILE) as f:
            cp = json.load(f)
        clusters = []
        for app_name, v in cp.get("completed_apps", {}).items():
            if v.get("status") != "success":
                continue
            for cid, feats in v.get("clusters", {}).items():
                if isinstance(feats, list) and feats:
                    clusters.append({"app_name": app_name, "cluster_id": cid,
                                     "features": feats, "tree_id": f"{app_name}__cluster_{cid}"})
        if clusters:
            logger.info(f"Loaded {len(clusters)} raw clusters from checkpoint")
            return clusters

    results_dir = PROJECT_ROOT / "evaluation_results"
    sessions = sorted(results_dir.glob("test_session_*.json"),
                      key=lambda p: p.stat().st_mtime, reverse=True)
    for s in sessions:
        if s.stat().st_size > 100_000:
            with open(s) as f:
                data = json.load(f)
            configs = data.get("configurations", [])
            if not configs:
                continue
            best = max(configs, key=lambda c: len(c.get("best_selections", {})))
            clusters = []
            for app_name, sel in best.get("best_selections", {}).items():
                for cid, feats in sel.get("candidate", {}).get("clustering", {}).get("clusters", {}).items():
                    if isinstance(feats, list) and feats:
                        clusters.append({"app_name": app_name, "cluster_id": cid,
                                         "features": feats, "tree_id": f"{app_name}__cluster_{cid}"})
            if clusters:
                logger.info(f"Loaded {len(clusters)} raw clusters from {s.name}")
                return clusters
    return []


# ---------------------------------------------------------------------------
# Stratified sampling
# ---------------------------------------------------------------------------

def stratified_sample(rows_by_bucket: dict, n: int) -> list:
    buckets = sorted(rows_by_bucket.keys())
    quota, remainder = n // len(buckets), n % len(buckets)
    selected = []
    for i, b in enumerate(buckets):
        pool = rows_by_bucket[b][:]
        random.shuffle(pool)
        selected.extend(pool[: quota + (1 if i < remainder else 0)])
    if len(selected) < n:
        used = set(id(r) for r in selected)
        rest = [r for b in buckets for r in rows_by_bucket[b] if id(r) not in used]
        random.shuffle(rest)
        selected.extend(rest[: n - len(selected)])
    return selected[:n]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n",       type=int, default=300)
    parser.add_argument("--out",     type=str, default="data/experiment1.csv")
    args = parser.parse_args()

    random.seed(RANDOM_SEED)
    out_path = PROJECT_ROOT / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)

    raw = load_clusters()
    if not raw:
        logger.error("No clusters found. Run run_mobile_pipeline.py first.")
        sys.exit(1)

    # Quality filter + deduplication
    kept, rejected = [], []
    for c in raw:
        ok, reason = is_quality_cluster(c["features"])
        if not ok:
            rejected.append(reason)
            continue
        c["features"] = deduplicate_features(c["features"])
        if len(c["features"]) >= MIN_CLUSTER_SIZE:
            kept.append(c)
        else:
            rejected.append("dedup left <2 features")

    logger.info(f"Quality filter: {len(kept)} kept / {len(raw)} total  ({len(rejected)} rejected)")

    logger.info("Loading reviews CSV...")
    reviews_df = pd.read_csv(REVIEWS_CSV).dropna(subset=["review"])
    reviews_df["review"] = reviews_df["review"].astype(str)

    use_ollama = False
    try:
        r = requests.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=5)
        use_ollama = r.ok
        logger.info("Ollama reachable — using LLM labels.")
    except Exception:
        logger.warning("Ollama not reachable — using most-specific-feature fallback for labels.")

    rows, no_review_skipped = [], 0
    for idx, c in enumerate(kept, 1):
        child, reviews = best_child(c["features"], reviews_df, c["app_name"])
        if child is None:
            no_review_skipped += 1
            continue
        siblings = [f for f in c["features"] if f != child]
        label = generate_label_ollama(c["features"]) if use_ollama else fallback_label(c["features"])
        logger.info(f"[{idx}/{len(kept)}] {c['app_name']} → '{label}'")
        rows.append({
            "parent_feature":  label,
            "child_feature":   child,
            "sibling_features": json.dumps(siblings),
            "example_reviews": json.dumps(reviews),
            "app_name":        c["app_name"],
            "n_siblings":      len(siblings),
            "cluster_size":    len(c["features"]),
            "tree_id":         c["tree_id"],
        })

    logger.info(f"Rows after review filter: {len(rows)}  ({no_review_skipped} dropped — no review hits)")
    if len(rows) < args.n:
        logger.warning(f"Only {len(rows)} rows (target {args.n}) — will grow as pipeline completes.")

    by_size = defaultdict(list)
    for row in rows:
        by_size[row["cluster_size"]].append(row)
    logger.info(f"Cluster-size distribution: { {s: len(v) for s, v in sorted(by_size.items())} }")

    selected = stratified_sample(by_size, min(args.n, len(rows)))
    random.shuffle(selected)

    df = pd.DataFrame(selected, columns=[
        "parent_feature", "child_feature", "sibling_features",
        "example_reviews", "app_name", "n_siblings", "cluster_size", "tree_id",
    ])
    df.to_csv(out_path, index=False)
    logger.info(f"Saved: {out_path}  ({len(df)} rows)")

    logger.info("\n=== Sample rows ===")
    for _, row in df.sample(min(6, len(df)), random_state=0).iterrows():
        siblings = json.loads(row["sibling_features"])
        reviews  = json.loads(row["example_reviews"])
        logger.info(
            f"  [{row['app_name']}]\n"
            f"    parent:   {row['parent_feature']}\n"
            f"    child:    {row['child_feature']}\n"
            f"    siblings: {siblings}\n"
            f"    review:   {reviews[0][:80] if reviews else '(none)'}\n"
        )


if __name__ == "__main__":
    main()
