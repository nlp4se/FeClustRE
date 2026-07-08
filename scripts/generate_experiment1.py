#!/usr/bin/env python3
"""
Generate Experiment 1 CSV: validation of parent-child feature relationships.

Requirements (advisor):
  - n = 300 rows
  - Columns: parent_feature, child_feature, sibling_features, example_reviews (n=3)
  - Stratification by tree depth (level of child node in dendrogram)
  - One row per tree (no two rows from the same cluster/tree)

Works from checkpoint + reviews CSV. No Neo4j needed.

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
from scipy.cluster.hierarchy import linkage, to_tree
from scipy.spatial.distance import pdist
from sentence_transformers import SentenceTransformer

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config import config

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

_cfg = config["default"]
REVIEWS_CSV = PROJECT_ROOT / "data/input/endpoint_1_process_reviews/mobile_apps/mobilerec_reviews_pipeline_large.csv"
CHECKPOINT_FILE = PROJECT_ROOT / "evaluation_results/mobile_pipeline_checkpoint.json"
RANDOM_SEED = 42
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

MIN_CLUSTER_SIZE = 3
MAX_CLUSTER_SIZE = 10
MIN_FEATURE_LEN = 3
MIN_REVIEW_HITS = 1
N_REVIEWS = 3

NOISE_FEATURES = {
    "app", "apps", "use", "used", "user", "users", "get", "got", "make",
    "good", "bad", "best", "new", "old", "free", "pay", "paid",
    "call", "calls", "log", "bug", "fix", "tap", "click", "open",
    "work", "works", "need", "want", "like", "love", "hate", "try",
    "time", "day", "year", "week", "month", "way", "thing", "things",
    "lot", "bit", "one", "two", "three", "four", "five",
    "imo", "btw", "lol", "omg", "fyi",
}

_embed_model: SentenceTransformer | None = None


def _get_embed_model() -> SentenceTransformer:
    global _embed_model
    if _embed_model is None:
        logger.info(f"Loading embedding model {EMBEDDING_MODEL}...")
        _embed_model = SentenceTransformer(EMBEDDING_MODEL)
    return _embed_model


def _is_noise(f: str) -> bool:
    f = f.strip().lower()
    return len(f) < MIN_FEATURE_LEN or f in NOISE_FEATURES


def _stem(s: str) -> str:
    for suffix in ("ings", "ing", "tion", "tions", "ies", "es", "s"):
        if s.endswith(suffix) and len(s) - len(suffix) >= 3:
            return s[: -len(suffix)]
    return s


def _specificity(feature: str) -> int:
    return len(feature.split()) * 10 + len(feature)


def deduplicate_features(features: list[str]) -> list[str]:
    features = [f for f in features if not _is_noise(f)]
    if not features:
        return []
    groups: dict[str, list[str]] = defaultdict(list)
    for f in features:
        groups[_stem(f.split()[0].lower())].append(f)
    canonical = []
    for group in groups.values():
        group.sort(key=len, reverse=True)
        kept = [group[0]]
        for c in group[1:]:
            if not any(c.lower() in k.lower() or k.lower() in c.lower() for k in kept):
                kept.append(c)
        canonical.extend(kept)
    return canonical


def _parent_same_as_child(parent: str, child: str) -> bool:
    p, c = parent.strip().lower(), child.strip().lower()
    return p == c or p in c or c in p


def build_dendrogram(features: list[str]) -> dict:
    if len(features) == 1:
        return {"label": features[0], "is_leaf": True, "children": []}
    embeddings = _get_embed_model().encode(features)
    Z = linkage(pdist(embeddings, metric="cosine"), method="average")
    root = to_tree(Z)

    def recurse(node) -> dict:
        if node.is_leaf():
            return {"label": features[node.id], "is_leaf": True, "children": []}
        return {"label": None, "is_leaf": False,
                "children": [recurse(node.left), recurse(node.right)]}

    return recurse(root)


def _leaf_depth(tree: dict, feature_name: str, depth: int = 0) -> int:
    if tree.get("is_leaf"):
        return depth if tree["label"] == feature_name else -1
    for child in tree.get("children", []):
        d = _leaf_depth(child, feature_name, depth + 1)
        if d >= 0:
            return d
    return -1


def _find_reviews(reviews_df: pd.DataFrame, app_name: str, feature: str,
                  n: int = N_REVIEWS) -> list[str]:
    app_reviews = reviews_df[reviews_df["app_name"] == app_name]["review"].dropna()
    pattern = re.compile(re.escape(feature), re.IGNORECASE)
    matches = app_reviews[app_reviews.str.contains(pattern, regex=True)].tolist()
    random.shuffle(matches)
    return matches[:n]


def _best_child(features: list[str], reviews_df: pd.DataFrame,
                app_name: str) -> tuple[str | None, list[str]]:
    candidates = sorted(features, key=_specificity, reverse=True)
    for feat in candidates:
        reviews = _find_reviews(reviews_df, app_name, feat)
        if len(reviews) >= MIN_REVIEW_HITS:
            return feat, reviews
    return None, []


def _load_clusters() -> tuple[list[dict], dict]:
    with open(CHECKPOINT_FILE) as f:
        cp = json.load(f)
    clusters = []
    for app_name, v in cp.get("completed_apps", {}).items():
        if v.get("status") != "success":
            continue
        labels = v.get("labels", {})
        for cid, feats in v.get("clusters", {}).items():
            if isinstance(feats, list) and feats:
                clusters.append({
                    "app_name": app_name, "cluster_id": cid,
                    "features": feats, "tree_id": f"{app_name}__cluster_{cid}",
                    "label": labels.get(str(cid)),
                })
    return clusters, cp.get("provenance", {})


def _stratified_sample(rows_by_bucket: dict[int, list], n: int) -> list:
    buckets = sorted(rows_by_bucket.keys())
    if not buckets:
        return []
    quota, remainder = n // len(buckets), n % len(buckets)
    selected = []
    for i, b in enumerate(buckets):
        pool = rows_by_bucket[b][:]
        random.shuffle(pool)
        selected.extend(pool[: quota + (1 if i < remainder else 0)])
    if len(selected) < n:
        used = {id(r) for r in selected}
        rest = [r for b in buckets for r in rows_by_bucket[b] if id(r) not in used]
        random.shuffle(rest)
        selected.extend(rest[: n - len(selected)])
    return selected[:n]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=300)
    parser.add_argument("--out", type=str, default="data/experiment1.csv")
    args = parser.parse_args()

    random.seed(RANDOM_SEED)
    out_path = PROJECT_ROOT / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)

    raw, provenance = _load_clusters()
    labeled_count = sum(1 for c in raw if c["label"])
    logger.info(f"Provenance: {provenance}")
    logger.info(f"Loaded {len(raw)} raw clusters ({labeled_count} with pipeline labels)")

    kept = []
    for c in raw:
        c["features"] = deduplicate_features(c["features"])
        if MIN_CLUSTER_SIZE <= len(c["features"]) <= MAX_CLUSTER_SIZE:
            kept.append(c)
    logger.info(f"After quality filter: {len(kept)} clusters (size [{MIN_CLUSTER_SIZE}, {MAX_CLUSTER_SIZE}])")

    logger.info("Loading reviews CSV...")
    reviews_df = pd.read_csv(REVIEWS_CSV).dropna(subset=["review"])
    reviews_df["review"] = reviews_df["review"].astype(str)

    logger.info("Building dendrograms and selecting parent-child pairs...")
    rows = []
    skipped_no_review, skipped_same, skipped_no_label = 0, 0, 0

    for idx, c in enumerate(kept, 1):
        child, reviews = _best_child(c["features"], reviews_df, c["app_name"])
        if child is None:
            skipped_no_review += 1
            continue

        label = c.get("label")
        if not label:
            label = sorted(c["features"], key=_specificity, reverse=True)[0]
            if label == child:
                candidates = sorted(c["features"], key=_specificity, reverse=True)
                label = candidates[1] if len(candidates) > 1 else candidates[0]
        if not label:
            skipped_no_label += 1
            continue
        if _parent_same_as_child(label, child):
            skipped_same += 1
            continue

        siblings = [f for f in c["features"] if f != child]
        tree = build_dendrogram(c["features"])
        child_lvl = _leaf_depth(tree, child)
        if child_lvl < 0:
            child_lvl = 1

        rows.append({
            "parent_feature":   label,
            "child_feature":    child,
            "sibling_features": json.dumps(siblings),
            "example_reviews":  json.dumps(reviews),
            "app_name":         c["app_name"],
            "n_siblings":       len(siblings),
            "cluster_size":     len(c["features"]),
            "child_depth":      child_lvl,
            "tree_id":          c["tree_id"],
            "model_type":       provenance.get("model_type", "unknown"),
            "embedding_type":   provenance.get("embedding_type", "unknown"),
            "strategy":         provenance.get("selection_strategy", "unknown"),
            "sample_size":      provenance.get("sample_size", "unknown"),
        })
        if idx % 100 == 0:
            logger.info(f"  processed {idx}/{len(kept)} clusters...")

    logger.info(
        f"Valid rows: {len(rows)} "
        f"(skipped: {skipped_no_review} no reviews, {skipped_same} parent=child, "
        f"{skipped_no_label} no label)"
    )
    if len(rows) < args.n:
        logger.warning(f"Only {len(rows)} rows available (target {args.n}).")

    by_depth = defaultdict(list)
    for row in rows:
        by_depth[row["child_depth"]].append(row)
    logger.info(f"Depth distribution: { {d: len(v) for d, v in sorted(by_depth.items())} }")

    selected = _stratified_sample(by_depth, min(args.n, len(rows)))
    random.shuffle(selected)

    df = pd.DataFrame(selected, columns=[
        "parent_feature", "child_feature", "sibling_features",
        "example_reviews", "app_name", "n_siblings", "cluster_size",
        "child_depth", "tree_id",
        "model_type", "embedding_type", "strategy", "sample_size",
    ])
    df.to_csv(out_path, index=False)
    logger.info(f"Saved: {out_path}  ({len(df)} rows)")

    logger.info(f"\n=== Summary ===")
    logger.info(f"  Total rows: {len(df)}")
    logger.info(f"  Apps covered: {df['app_name'].nunique()}")
    logger.info(f"  Depth range: {df['child_depth'].min()} - {df['child_depth'].max()}")
    depth_dist = df["child_depth"].value_counts().sort_index()
    logger.info(f"  Depth stratification: {dict(depth_dist)}")


if __name__ == "__main__":
    main()
