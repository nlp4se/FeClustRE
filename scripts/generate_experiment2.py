#!/usr/bin/env python3
"""
Generate Experiment 2 data: feature tree vs feature flat list.

Requirements (advisor):
  - n = 60 rows
  - Each row: tree_json (hierarchical with reviews) + list_json (flat with reviews)
  - Stratification between Q1 and Q3 values according to tree size

Works from checkpoint + reviews CSV. No Neo4j needed.

Output:
  data/experiment2.json      (array, one entry per cluster)
  data/experiment2_flat.csv   (tabular, tree_json and list_json as strings)

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
APPS_CSV = PROJECT_ROOT / "data/input/endpoint_1_process_reviews/mobile_apps/mobilerec_apps.csv"
CHECKPOINT_FILE = PROJECT_ROOT / "evaluation_results/mobile_pipeline_checkpoint.json"
OLLAMA_BASE_URL = _cfg.OLLAMA_BASE_URL
OLLAMA_MODEL = _cfg.OLLAMA_MODEL
REVIEWS_PER_FEATURE = 3
RANDOM_SEED = 42
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

MIN_FEATURE_LENGTH = 3
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
    return len(f) < MIN_FEATURE_LENGTH or f in NOISE_FEATURES


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


def fallback_label(features: list[str]) -> str:
    return sorted(features, key=_specificity, reverse=True)[0]


def _leaf_features_in_subtree(node, features: list[str]) -> list[str]:
    if node.is_leaf():
        return [features[node.id]]
    return _leaf_features_in_subtree(node.left, features) + _leaf_features_in_subtree(node.right, features)


def build_dendrogram(features: list[str]) -> dict:
    if len(features) == 1:
        return {"label": features[0], "is_leaf": True, "children": []}
    embeddings = _get_embed_model().encode(features)
    Z = linkage(pdist(embeddings, metric="cosine"), method="average")
    root, _ = to_tree(Z, rd=True)

    def recurse(node) -> dict:
        if node.is_leaf():
            return {"label": features[node.id], "is_leaf": True, "children": []}
        subtree_feats = _leaf_features_in_subtree(node, features)
        return {
            "label": fallback_label(subtree_feats),
            "is_leaf": False,
            "children": [recurse(node.left), recurse(node.right)],
        }

    return recurse(root)


def tree_depth(node: dict) -> int:
    children = node.get("children") or []
    if not children:
        return 1
    return 1 + max(tree_depth(child) for child in children)


def find_reviews(reviews_df: pd.DataFrame, app_name: str, feature: str,
                 n: int = REVIEWS_PER_FEATURE) -> list[str]:
    app_reviews = reviews_df[reviews_df["app_name"] == app_name]["review"].dropna()
    pattern = re.compile(re.escape(feature), re.IGNORECASE)
    matches = app_reviews[app_reviews.str.contains(pattern, regex=True)].tolist()
    random.shuffle(matches)
    return matches[:n]


def attach_reviews(node: dict, app_name: str, reviews_df: pd.DataFrame) -> None:
    if node.get("is_leaf"):
        node["name"] = node["label"]
        node["reviews"] = find_reviews(reviews_df, app_name, node["label"])
        return
    for child in node.get("children", []):
        attach_reviews(child, app_name, reviews_df)


def generate_label_ollama(features: list[str], max_attempts: int = 5) -> str:
    import requests
    feature_lines = "\n".join(f"- {f}" for f in features)
    prompt = (
        "You name feature categories for mobile app review clusters.\n\n"
        f"Features in this cluster:\n{feature_lines}\n\n"
        "Write one category name (2-4 words). Reply with ONLY the name."
    )
    for _ in range(max_attempts):
        try:
            resp = requests.post(
                f"{OLLAMA_BASE_URL}/api/chat",
                json={"model": OLLAMA_MODEL,
                      "messages": [{"role": "user", "content": prompt}],
                      "stream": False},
                timeout=60,
            )
            if resp.ok:
                raw = resp.json().get("message", {}).get("content", "").strip()
                label = re.sub(r"[^a-zA-Z0-9\s\-]", "", raw.splitlines()[0]).strip().lower()
                if label and 1 <= len(label.split()) <= 6:
                    return label
        except Exception:
            pass
    return fallback_label(features)


def _load_app_metadata() -> dict[str, dict]:
    if not APPS_CSV.exists():
        return {}
    df = pd.read_csv(APPS_CSV)
    return {
        row["app_name"]: {
            "app_package": row.get("app_package", ""),
            "app_category": row.get("app_category", "unknown") or "unknown",
        }
        for _, row in df.iterrows()
    }


def _load_app_clusters() -> tuple[dict[str, list[dict]], dict]:
    with open(CHECKPOINT_FILE) as f:
        cp = json.load(f)
    apps = {}
    for app_name, v in cp.get("completed_apps", {}).items():
        if v.get("status") != "success":
            continue
        app_labels = v.get("labels", {})
        clusters = [
            {"cluster_id": cid, "features": feats, "label": app_labels.get(str(cid))}
            for cid, feats in v.get("clusters", {}).items()
            if isinstance(feats, list) and feats
        ]
        if clusters:
            apps[app_name] = clusters
    return apps, cp.get("provenance", {})


def build_cluster_tree_json(app_name, cluster_id, features, reviews_df, label):
    features = sorted(features, key=_specificity, reverse=True)
    subtree = build_dendrogram(features)
    attach_reviews(subtree, app_name, reviews_df)
    depth = 1 + tree_depth(subtree)
    return {
        "app": app_name, "cluster_id": cluster_id, "label": label, "depth": depth,
        "tree": {"label": label, "is_leaf": False,
                 "children": subtree["children"] if not subtree.get("is_leaf") else [subtree]},
    }


def build_cluster_list_json(app_name, cluster_id, features, reviews_df):
    features = sorted(features, key=_specificity, reverse=True)
    return {
        "app": app_name, "cluster_id": cluster_id,
        "features": [{"name": f, "reviews": find_reviews(reviews_df, app_name, f)} for f in features],
    }


def _stratified_sample(rows_by_bucket, n):
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
    parser.add_argument("--n", type=int, default=60)
    parser.add_argument("--out", type=str, default="data/experiment2.json")
    args = parser.parse_args()

    random.seed(RANDOM_SEED)
    out_path = PROJECT_ROOT / args.out
    out_flat = out_path.parent / (out_path.stem + "_flat.csv")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    app_meta = _load_app_metadata()
    raw_apps, provenance = _load_app_clusters()
    logger.info(f"Provenance: {provenance}")
    if not raw_apps:
        logger.error("No data found. Run run_mobile_pipeline.py first.")
        sys.exit(1)

    all_clusters = []
    for app_name, app_clusters in raw_apps.items():
        meta = app_meta.get(app_name, {})
        for c in app_clusters:
            features = deduplicate_features(list(c["features"]))
            if len(features) < 2:
                continue
            all_clusters.append({
                "app_name": app_name,
                "app_package": meta.get("app_package", ""),
                "app_category": meta.get("app_category", "unknown"),
                "cluster_id": str(c["cluster_id"]),
                "features": features,
                "n_features": len(features),
                "tree_id": f"{app_name}__cluster_{c['cluster_id']}",
                "label": c.get("label"),
            })

    logger.info(f"Non-singleton clusters after cleaning: {len(all_clusters)}")

    sizes = np.array([c["n_features"] for c in all_clusters])
    q1 = int(np.percentile(sizes, 25))
    q3 = int(np.percentile(sizes, 75))
    logger.info(f"Tree size stats — min:{sizes.min()} Q1:{q1} median:{int(np.median(sizes))} Q3:{q3} max:{sizes.max()}")

    eligible = [c for c in all_clusters if q1 <= c["n_features"] <= q3]
    logger.info(f"Clusters in [Q1={q1}, Q3={q3}]: {len(eligible)}")

    if len(eligible) < args.n:
        logger.warning(f"Only {len(eligible)} clusters in IQR (target {args.n}). Using all available.")

    has_pipeline_labels = any(c.get("label") for c in eligible)
    use_ollama = False
    if not has_pipeline_labels:
        try:
            import requests
            r = requests.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=5)
            use_ollama = r.ok
        except Exception:
            pass
    else:
        logger.info("Using pipeline labels from checkpoint.")

    by_size = defaultdict(list)
    for c in eligible:
        by_size[c["n_features"]].append(c)
    logger.info(f"Size distribution in IQR: { {s: len(v) for s, v in sorted(by_size.items())} }")

    target_n = min(args.n, len(eligible))
    sampled = _stratified_sample(by_size, target_n)
    random.shuffle(sampled)
    logger.info(f"Selected {len(sampled)} clusters (stratified by tree size in [Q1, Q3])")

    logger.info("Loading reviews CSV...")
    reviews_df = pd.read_csv(REVIEWS_CSV).dropna(subset=["review"])
    reviews_df["review"] = reviews_df["review"].astype(str)

    records, flat_rows = [], []
    for idx, c in enumerate(sampled, 1):
        app_name = c["app_name"]
        cluster_id = c["cluster_id"]

        if c.get("label"):
            label = c["label"]
        elif use_ollama:
            label = generate_label_ollama(c["features"])
        else:
            label = fallback_label(c["features"])

        logger.info(f"[{idx}/{len(sampled)}] {app_name} cluster {cluster_id} ({c['n_features']} features) -> '{label}'")

        tree = build_cluster_tree_json(app_name, cluster_id, c["features"], reviews_df, label)
        flat = build_cluster_list_json(app_name, cluster_id, c["features"], reviews_df)

        record = {
            "tree_id": c["tree_id"], "app_name": app_name,
            "app_package": c["app_package"], "app_category": c["app_category"],
            "cluster_id": cluster_id, "label": label,
            "n_features": c["n_features"], "tree_depth": tree.get("depth", 0),
            "model_type": provenance.get("model_type", "unknown"),
            "embedding_type": provenance.get("embedding_type", "unknown"),
            "strategy": provenance.get("selection_strategy", "unknown"),
            "sample_size": provenance.get("sample_size", "unknown"),
            "tree_json": tree, "list_json": flat,
        }
        records.append(record)
        flat_rows.append({
            **{k: v for k, v in record.items() if k not in ("tree_json", "list_json")},
            "tree_json": json.dumps(tree, ensure_ascii=False),
            "list_json": json.dumps(flat, ensure_ascii=False),
        })

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(records, f, indent=2, ensure_ascii=False)
    logger.info(f"Experiment 2 JSON: {out_path}  ({len(records)} clusters)")

    pd.DataFrame(flat_rows).to_csv(out_flat, index=False)
    logger.info(f"Experiment 2 flat CSV: {out_flat}")

    if records:
        sizes_sel = [r["n_features"] for r in records]
        apps_sel = {r["app_name"] for r in records}
        logger.info(f"\n=== Summary ===")
        logger.info(f"  Total clusters: {len(records)}")
        logger.info(f"  Apps covered: {len(apps_sel)}")
        logger.info(f"  Tree size range: {min(sizes_sel)} - {max(sizes_sel)}")


if __name__ == "__main__":
    main()
