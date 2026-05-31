#!/usr/bin/env python3
"""
Generate Experiment 2 data: feature tree vs feature flat list.

Works entirely from the checkpoint or latest session JSON + the reviews CSV.
Does NOT require Neo4j.

For each selected app:
  tree_json - hierarchical view: clusters → features, each with embedded reviews
  list_json - flat view: all features of the app with embedded reviews

Quality filters applied:
  - Clusters with > MAX_CLUSTER_SIZE features are excluded (garbage dumps)
  - Noise features are removed
  - Apps with < MIN_APP_FEATURES total clean features are excluded

Selection: stratified between Q1 and Q3 of clean-feature-count per app.
Target: 60 apps.

Output:
  data/experiment2.json      (array, one entry per app)
  data/experiment2_flat.csv  (tabular, tree_json and list_json as strings)

Usage:
  .venv/bin/python scripts/generate_experiment2.py [--n 60] [--out data/experiment2.json]
"""
import argparse
import json
import logging
import random
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import requests

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

REVIEWS_CSV = PROJECT_ROOT / "data/input/endpoint_1_process_reviews/mobile_apps/mobilerec_reviews_pipeline.csv"
CHECKPOINT_FILE = PROJECT_ROOT / "evaluation_results/mobile_pipeline_checkpoint.json"
OLLAMA_BASE_URL = "http://localhost:11434"
OLLAMA_MODEL = "llama3.2:3b"
REVIEWS_PER_FEATURE = 3
RANDOM_SEED = 42

# ---------------------------------------------------------------------------
# Quality filters (same thresholds as experiment1)
# ---------------------------------------------------------------------------

MAX_CLUSTER_SIZE = 12
MIN_CLUSTER_SIZE = 2
MIN_FEATURE_LENGTH = 3
MIN_APP_FEATURES = 5   # app must have at least this many clean features total

NOISE_FEATURES = {
    "app", "apps", "use", "used", "user", "users", "get", "got", "make",
    "good", "bad", "best", "new", "old", "free", "pay", "paid",
    "call", "calls", "log", "bug", "fix", "tap", "click", "open",
    "work", "works", "need", "want", "like", "love", "hate", "try",
    "time", "day", "year", "week", "month", "way", "thing", "things",
    "lot", "bit", "one", "two", "three", "four", "five",
    "imo", "btw", "lol", "omg", "fyi",
}
MAX_NOISE_RATIO = 0.5


def is_noise_feature(feature: str) -> bool:
    f = feature.strip().lower()
    if len(f) < MIN_FEATURE_LENGTH:
        return True
    if f in NOISE_FEATURES:
        return True
    if " " not in f and "-" not in f and len(f) <= 2:
        return True
    return False


def _stem(s: str) -> str:
    """Very lightweight stemmer: strip common suffixes."""
    for suffix in ("ings", "ing", "tion", "tions", "ies", "es", "s"):
        if s.endswith(suffix) and len(s) - len(suffix) >= 3:
            return s[: -len(suffix)]
    return s


def _specificity(feature: str) -> int:
    """Score how specific a feature is — prefer multi-word, longer features."""
    return len(feature.split()) * 10 + len(feature)


def deduplicate_features(features: list[str]) -> list[str]:
    """
    Remove morphological duplicates within a cluster.
    For each group of variants (share/sharing/sharing documents),
    keep only the most specific (longest) representative.
    """
    features = [f for f in features if not is_noise_feature(f)]
    if not features:
        return []

    groups: dict[str, list[str]] = defaultdict(list)
    for f in features:
        first_word = f.split()[0].lower()
        groups[_stem(first_word)].append(f)

    canonical = []
    for group in groups.values():
        group_sorted = sorted(group, key=len, reverse=True)
        kept = [group_sorted[0]]
        for candidate in group_sorted[1:]:
            if not any(candidate.lower() in k.lower() or k.lower() in candidate.lower()
                       for k in kept):
                kept.append(candidate)
        canonical.extend(kept)

    return canonical


def clean_features(features: list[str]) -> list[str]:
    return deduplicate_features(features)


def fallback_label(features: list[str]) -> str:
    """Best single-feature label when Ollama is unavailable."""
    return sorted(features, key=_specificity, reverse=True)[0]


def is_quality_cluster(features: list[str]) -> bool:
    if not (MIN_CLUSTER_SIZE <= len(features) <= MAX_CLUSTER_SIZE):
        return False
    noise_count = sum(1 for f in features if is_noise_feature(f))
    if noise_count / len(features) > MAX_NOISE_RATIO:
        return False
    return len(deduplicate_features(features)) >= MIN_CLUSTER_SIZE


# ---------------------------------------------------------------------------
# Cluster label generation
# ---------------------------------------------------------------------------

_PROMPT_LEAK_WORDS = {"product", "analyst", "labeling", "human", "study", "category", "features"}


def _is_valid_label(label: str, features: list[str]) -> bool:
    if not label or len(label) > 50:
        return False
    words = label.split()
    if len(words) > 6:
        return False
    if any(c in label for c in (",", "_", ":", ".")):
        return False
    if label in [f.lower() for f in features]:
        return False
    if sum(1 for w in words if w in _PROMPT_LEAK_WORDS) >= 2:
        return False
    return True


def generate_label_ollama(features: list[str]) -> str:
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
            label = label.splitlines()[0].strip()
            if _is_valid_label(label, features):
                return label
    except Exception:
        pass
    return fallback_label(features)


# ---------------------------------------------------------------------------
# Review search
# ---------------------------------------------------------------------------

def find_reviews(reviews_df: pd.DataFrame, app_name: str, feature: str, n: int = REVIEWS_PER_FEATURE) -> list[str]:
    app_reviews = reviews_df[reviews_df["app_name"] == app_name]["review"].dropna()
    pattern = re.compile(re.escape(feature), re.IGNORECASE)
    matches = app_reviews[app_reviews.str.contains(pattern, regex=True)].tolist()
    random.shuffle(matches)
    return matches[:n]


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_app_clusters() -> dict[str, list[dict]]:
    """Return {app_name: [{cluster_id, features}]} from checkpoint or session."""
    if CHECKPOINT_FILE.exists():
        with open(CHECKPOINT_FILE) as f:
            cp = json.load(f)
        completed = cp.get("completed_apps", {})
        apps = {}
        for app_name, v in completed.items():
            if v.get("status") != "success":
                continue
            clusters = [
                {"cluster_id": cid, "features": feats}
                for cid, feats in v.get("clusters", {}).items()
                if isinstance(feats, list) and feats
            ]
            if clusters:
                apps[app_name] = clusters
        if apps:
            logger.info(f"Loaded {len(apps)} apps from checkpoint")
            return apps

    # Fallback: session file
    results_dir = PROJECT_ROOT / "evaluation_results"
    sessions = sorted(results_dir.glob("test_session_*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
    for s in sessions:
        if s.stat().st_size > 100_000:
            with open(s) as f:
                data = json.load(f)
            configs = data.get("configurations", [])
            if not configs:
                continue
            best = max(configs, key=lambda c: len(c.get("best_selections", {})))
            apps = {}
            for app_name, sel in best.get("best_selections", {}).items():
                raw = sel.get("candidate", {}).get("clustering", {}).get("clusters", {})
                clusters = [{"cluster_id": cid, "features": feats} for cid, feats in raw.items() if isinstance(feats, list) and feats]
                if clusters:
                    apps[app_name] = clusters
            if apps:
                logger.info(f"Loaded {len(apps)} apps from session {s.name}")
                return apps

    return {}


# ---------------------------------------------------------------------------
# Tree / list builders
# ---------------------------------------------------------------------------

def build_tree_json(app_name: str, clusters: list[dict], reviews_df: pd.DataFrame, use_ollama: bool) -> dict:
    tree = {"app": app_name, "clusters": []}
    for c in clusters:
        features = clean_features(c["features"])
        if not features:
            continue
        # Sort by specificity so the most descriptive features come first
        features = sorted(features, key=_specificity, reverse=True)
        label = generate_label_ollama(features) if use_ollama else fallback_label(features)
        logger.info(f"  cluster label: '{label}'  ← {features[:3]}")
        tree["clusters"].append({
            "label": label,
            "features": [
                {"name": f, "reviews": find_reviews(reviews_df, app_name, f)}
                for f in features
            ],
        })
    return tree


def build_list_json(app_name: str, clusters: list[dict], reviews_df: pd.DataFrame) -> dict:
    seen: set[str] = set()
    features = []
    for c in clusters:
        # Sort by specificity within each cluster before adding to flat list
        for f in sorted(clean_features(c["features"]), key=_specificity, reverse=True):
            if f not in seen:
                seen.add(f)
                features.append({"name": f, "reviews": find_reviews(reviews_df, app_name, f)})
    return {"app": app_name, "features": features}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=60)
    parser.add_argument("--out", type=str, default="data/experiment2.json")
    args = parser.parse_args()

    random.seed(RANDOM_SEED)
    out_path = PROJECT_ROOT / args.out
    out_flat = out_path.parent / (out_path.stem + "_flat.csv")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    raw_apps = load_app_clusters()
    if not raw_apps:
        logger.error("No data found. Run run_mobile_pipeline.py first.")
        sys.exit(1)

    # Apply quality filters per app
    apps: dict[str, list[dict]] = {}
    for app_name, clusters in raw_apps.items():
        good_clusters = [c for c in clusters if is_quality_cluster(c["features"])]
        total_clean = sum(len(clean_features(c["features"])) for c in good_clusters)
        if total_clean >= MIN_APP_FEATURES and good_clusters:
            apps[app_name] = good_clusters

    logger.info(f"Apps after quality filter: {len(apps)} / {len(raw_apps)}")

    logger.info("Loading reviews CSV...")
    reviews_df = pd.read_csv(REVIEWS_CSV).dropna(subset=["review"])
    reviews_df["review"] = reviews_df["review"].astype(str)

    use_ollama = False
    try:
        r = requests.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=5)
        use_ollama = r.ok
        logger.info("Ollama reachable — generating LLM cluster labels.")
    except Exception:
        logger.warning("Ollama not reachable — using most-specific-feature fallback for labels.")

    # Compute clean feature count per app for stratification
    app_sizes = {
        app: sum(len(clean_features(c["features"])) for c in clusters)
        for app, clusters in apps.items()
    }
    sizes = np.array(list(app_sizes.values()))
    q1, q3 = np.percentile(sizes, 25), np.percentile(sizes, 75)
    logger.info(
        f"Clean features per app — min:{sizes.min()} Q1:{q1:.0f} "
        f"median:{np.median(sizes):.0f} Q3:{q3:.0f} max:{sizes.max()}"
    )

    eligible = {a: c for a, c in apps.items() if q1 <= app_sizes[a] <= q3}
    logger.info(f"Apps in [Q1={q1:.0f}, Q3={q3:.0f}]: {len(eligible)} / {len(apps)}")

    if len(eligible) < args.n:
        logger.warning(f"Only {len(eligible)} eligible apps — using all {len(apps)} apps.")
        eligible = apps

    sampled = random.sample(list(eligible.keys()), min(args.n, len(eligible)))
    logger.info(f"Building representations for {len(sampled)} apps...")

    records, flat_rows = [], []
    for idx, app_name in enumerate(sampled, 1):
        clusters = eligible[app_name]
        logger.info(f"[{idx}/{len(sampled)}] {app_name} ({len(clusters)} clusters)...")
        tree = build_tree_json(app_name, clusters, reviews_df, use_ollama)
        flat = build_list_json(app_name, clusters, reviews_df)
        n_features = app_sizes[app_name]
        n_clusters = len(clusters)

        records.append({
            "app_name": app_name,
            "n_clusters": n_clusters,
            "n_features": n_features,
            "tree_json": tree,
            "list_json": flat,
        })
        flat_rows.append({
            "app_name": app_name,
            "n_clusters": n_clusters,
            "n_features": n_features,
            "tree_json": json.dumps(tree, ensure_ascii=False),
            "list_json": json.dumps(flat, ensure_ascii=False),
        })

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(records, f, indent=2, ensure_ascii=False)
    logger.info(f"Experiment 2 JSON: {out_path}  ({len(records)} apps)")

    pd.DataFrame(flat_rows).to_csv(out_flat, index=False)
    logger.info(f"Experiment 2 flat CSV: {out_flat}")

    # Sanity check: print one sample tree
    if records:
        sample = records[0]
        logger.info(f"\n=== Sample: {sample['app_name']} ({sample['n_clusters']} clusters, {sample['n_features']} features) ===")
        for cl in sample["tree_json"]["clusters"][:3]:
            feats = [f["name"] for f in cl["features"]]
            logger.info(f"  [{cl['label']}]: {feats}")


if __name__ == "__main__":
    main()
