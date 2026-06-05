#!/usr/bin/env python3
"""
Generate Experiment 2 data: feature tree vs feature flat list.

Works entirely from the checkpoint or latest session JSON + the reviews CSV.
Does NOT require Neo4j.

Each row is one cluster (cluster-level entity):
  tree_json - hierarchical dendrogram: cluster label → internal groups → features (3+ levels)
  list_json - flat view: same cluster features with embedded reviews (no label)

Quality filters applied:
  - Clusters outside [MIN_CLUSTER_SIZE, MAX_CLUSTER_SIZE] are excluded
  - Tree depth must be in [MIN_TREE_DEPTH, MAX_TREE_DEPTH] (default 3–8)
  - All features from each kept cluster are used for labeling and saved in tree/list JSON

Selection (cluster-level):
  - Only clusters with n_features in [MIN_CLUSTER_SIZE, MAX_CLUSTER_SIZE]
  - Stratified across app categories (mobilerec_apps.csv)
  - Within each category, stratified by cluster size
  - At most MAX_CLUSTERS_PER_APP clusters per app
Target: 60 clusters (default).

Output:
  data/experiment2.json      (array, one entry per cluster)
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
from scipy.cluster.hierarchy import linkage, to_tree
from scipy.spatial.distance import pdist
from sentence_transformers import SentenceTransformer

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config import config

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

_cfg = config["default"]
REVIEWS_CSV = PROJECT_ROOT / "data/input/endpoint_1_process_reviews/mobile_apps/mobilerec_reviews_pipeline.csv"
APPS_CSV = PROJECT_ROOT / "data/input/endpoint_1_process_reviews/mobile_apps/mobilerec_apps.csv"
CHECKPOINT_FILE = PROJECT_ROOT / "evaluation_results/mobile_pipeline_checkpoint.json"
OLLAMA_BASE_URL = _cfg.OLLAMA_BASE_URL
OLLAMA_MODEL = _cfg.OLLAMA_MODEL
REVIEWS_PER_FEATURE = 3
RANDOM_SEED = 42

# ---------------------------------------------------------------------------
# Quality filters (same thresholds as experiment1)
# ---------------------------------------------------------------------------

MAX_CLUSTER_SIZE = 30
MIN_CLUSTER_SIZE = 10
MIN_TREE_DEPTH = 2
MAX_TREE_DEPTH = 5
MAX_CLUSTERS_PER_APP = 2
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
MIN_FEATURE_LENGTH = 3

_embed_model: SentenceTransformer | None = None
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
    return MIN_CLUSTER_SIZE <= len(features) <= MAX_CLUSTER_SIZE


def get_embed_model() -> SentenceTransformer:
    global _embed_model
    if _embed_model is None:
        logger.info(f"Loading embedding model {EMBEDDING_MODEL} for intra-cluster trees...")
        _embed_model = SentenceTransformer(EMBEDDING_MODEL)
    return _embed_model


def _leaf_features_in_subtree(node, features: list[str]) -> list[str]:
    if node.is_leaf():
        return [features[node.id]]
    return _leaf_features_in_subtree(node.left, features) + _leaf_features_in_subtree(node.right, features)


def build_dendrogram(features: list[str]) -> dict:
    """Binary feature dendrogram (same linkage approach as taxonomy_service)."""
    if len(features) == 1:
        return {"label": features[0], "is_leaf": True, "children": []}

    embeddings = get_embed_model().encode(features)
    linkage_matrix = linkage(pdist(embeddings, metric="cosine"), method="average")
    root, _ = to_tree(linkage_matrix, rd=True)

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


def cluster_tree_depth(features: list[str]) -> int:
    """Depth of dendrogram plus one for the LLM cluster label at the root."""
    if len(features) < 2:
        return 1
    return 1 + tree_depth(build_dendrogram(features))


def attach_reviews(node: dict, app_name: str, reviews_df: pd.DataFrame) -> None:
    if node.get("is_leaf"):
        node["name"] = node["label"]
        node["reviews"] = find_reviews(reviews_df, app_name, node["label"])
        return
    for child in node.get("children", []):
        attach_reviews(child, app_name, reviews_df)


# ---------------------------------------------------------------------------
# Cluster label generation (aligned with generate_experiment1.py)
# ---------------------------------------------------------------------------

_PROMPT_LEAK_WORDS = {"product", "analyst", "labeling", "human", "study", "category", "features"}


def _normalize_feature(text: str) -> str:
    return re.sub(r"\s+", " ", text.lower().strip())


def _label_equals_feature(label: str, features: list[str]) -> bool:
    norm = _normalize_feature(label)
    return any(norm == _normalize_feature(f) for f in features)


def _parse_label_response(raw: str) -> str:
    label = raw.strip().strip('"').strip("'").lower()
    label = label.splitlines()[0].strip()
    for prefix in ("category name:", "category:", "name:", "label:"):
        if label.startswith(prefix):
            label = label[len(prefix):].strip()
    label = re.sub(r"[^a-zA-Z0-9\s\-]", "", label).strip()
    return label


def _is_valid_label(label: str, features: list[str]) -> bool:
    if not label or len(label) > 50:
        return False
    words = label.split()
    if len(words) > 6 or len(words) < 1:
        return False
    if any(c in label for c in (",", "_", ":", ".")):
        return False
    if _label_equals_feature(label, features):
        return False
    if sum(1 for w in words if w in _PROMPT_LEAK_WORDS) >= 2:
        return False
    return True


def _build_zero_shot_prompt(features: list[str]) -> str:
    feature_lines = "\n".join(f"- {f}" for f in features)
    return (
        "You name feature categories for mobile app review clusters.\n\n"
        f"Features in this cluster:\n{feature_lines}\n\n"
        "Write one category name that:\n"
        "1. Is as close to this domain as possible, but broader than any single feature.\n"
        "2. Covers all features in the cluster.\n"
        "3. Is NOT identical to any feature listed above.\n"
        "4. Is a short phrase (2-4 words).\n\n"
        "Reply with ONLY the category name. No explanation, punctuation, or other text."
    )


def generate_label_ollama(features: list[str], max_attempts: int = 10) -> str:
    prompt = _build_zero_shot_prompt(features)
    for attempt in range(1, max_attempts + 1):
        try:
            resp = requests.post(
                f"{OLLAMA_BASE_URL}/api/chat",
                json={
                    "model": OLLAMA_MODEL,
                    "messages": [{"role": "user", "content": prompt}],
                    "stream": False,
                },
                timeout=60,
            )
            if not resp.ok:
                logger.debug(f"Ollama HTTP {resp.status_code} (attempt {attempt}/{max_attempts})")
                continue
            raw = resp.json().get("message", {}).get("content", "")
            label = _parse_label_response(raw)
            if not label:
                logger.warning(f"Empty label (attempt {attempt}/{max_attempts})")
                continue
            if _label_equals_feature(label, features):
                logger.warning(
                    f"Label '{label}' equals a feature — retry {attempt}/{max_attempts}"
                )
                continue
            if _is_valid_label(label, features):
                return label
            logger.warning(f"Label '{label}' rejected by validation (attempt {attempt}/{max_attempts})")
        except Exception as e:
            logger.debug(f"Ollama failed (attempt {attempt}/{max_attempts}): {e}")
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

def load_app_metadata() -> dict[str, dict]:
    if not APPS_CSV.exists():
        logger.warning(f"Apps CSV not found at {APPS_CSV}")
        return {}
    df = pd.read_csv(APPS_CSV)
    return {
        row["app_name"]: {
            "app_package": row.get("app_package", ""),
            "app_category": row.get("app_category", "unknown") or "unknown",
        }
        for _, row in df.iterrows()
    }


def load_provenance() -> dict:
    if CHECKPOINT_FILE.exists():
        with open(CHECKPOINT_FILE) as f:
            cp = json.load(f)
        return cp.get("provenance", {})
    return {}


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
# Cluster collection / sampling
# ---------------------------------------------------------------------------

def collect_quality_clusters(
    raw_apps: dict[str, list[dict]],
    app_meta: dict[str, dict],
) -> list[dict]:
    """Flatten clusters within size bounds and dendrogram depth in [MIN_TREE_DEPTH, MAX_TREE_DEPTH]."""
    clusters = []
    rejected_size, rejected_shallow, rejected_deep = 0, 0, 0
    for app_name, app_clusters in raw_apps.items():
        meta = app_meta.get(app_name, {})
        for c in app_clusters:
            features = list(c["features"])
            n_features = len(features)
            if not is_quality_cluster(features):
                rejected_size += 1
                continue
            depth = cluster_tree_depth(features)
            if depth < MIN_TREE_DEPTH:
                rejected_shallow += 1
                continue
            if depth > MAX_TREE_DEPTH:
                rejected_deep += 1
                continue
            clusters.append({
                "app_name": app_name,
                "app_package": meta.get("app_package", ""),
                "app_category": meta.get("app_category", "unknown"),
                "cluster_id": str(c["cluster_id"]),
                "features": features,
                "n_features": n_features,
                "tree_depth": depth,
                "tree_id": f"{app_name}__cluster_{c['cluster_id']}",
            })
    logger.info(
        f"Rejected {rejected_size} clusters outside size "
        f"[{MIN_CLUSTER_SIZE}, {MAX_CLUSTER_SIZE}], "
        f"{rejected_shallow} with depth < {MIN_TREE_DEPTH}, "
        f"{rejected_deep} with depth > {MAX_TREE_DEPTH}"
    )
    return clusters


def stratified_sample(rows_by_bucket: dict, n: int) -> list:
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


def _pick_with_app_cap(
    pool: list[dict],
    n: int,
    per_app_counts: dict[str, int],
    max_per_app: int,
) -> list[dict]:
    """Pick up to n clusters from pool, respecting per-app cap."""
    random.shuffle(pool)
    picked = []
    for c in pool:
        if len(picked) >= n:
            break
        app = c["app_name"]
        if per_app_counts.get(app, 0) >= max_per_app:
            continue
        picked.append(c)
        per_app_counts[app] = per_app_counts.get(app, 0) + 1
    return picked


def stratified_sample_clusters(
    clusters: list[dict],
    n: int,
    max_per_app: int = MAX_CLUSTERS_PER_APP,
) -> list[dict]:
    """
    Stratify across app categories, then cluster size within each category.
    Enforces MIN/MAX cluster size and a per-app cap for diversity.
    """
    eligible = [
        c for c in clusters
        if MIN_CLUSTER_SIZE <= c["n_features"] <= MAX_CLUSTER_SIZE
    ]
    if not eligible:
        return []

    by_category: dict[str, list[dict]] = defaultdict(list)
    for c in eligible:
        by_category[c["app_category"]].append(c)

    categories = sorted(by_category.keys())
    quota, remainder = n // len(categories), n % len(categories)

    selected: list[dict] = []
    per_app_counts: dict[str, int] = defaultdict(int)
    used_ids: set[int] = set()

    for i, category in enumerate(categories):
        cat_quota = quota + (1 if i < remainder else 0)
        if cat_quota <= 0:
            continue

        by_size: dict[int, list[dict]] = defaultdict(list)
        for c in by_category[category]:
            by_size[c["n_features"]].append(c)

        cat_candidates = stratified_sample(by_size, cat_quota)
        cat_picked = _pick_with_app_cap(
            cat_candidates, cat_quota, per_app_counts, max_per_app
        )

        if len(cat_picked) < cat_quota:
            picked_ids = {id(c) for c in cat_picked}
            spill = [
                c for c in by_category[category]
                if id(c) not in used_ids and id(c) not in picked_ids
            ]
            cat_picked.extend(
                _pick_with_app_cap(
                    spill,
                    cat_quota - len(cat_picked),
                    per_app_counts,
                    max_per_app,
                )
            )

        for c in cat_picked:
            used_ids.add(id(c))
        selected.extend(cat_picked)

    if len(selected) < n:
        spill = [c for c in eligible if id(c) not in used_ids]
        random.shuffle(spill)
        extra = _pick_with_app_cap(spill, n - len(selected), per_app_counts, max_per_app)
        for c in extra:
            used_ids.add(id(c))
        selected.extend(extra)

    return selected[:n]


# ---------------------------------------------------------------------------
# Tree / list builders (single cluster)
# ---------------------------------------------------------------------------

def build_cluster_tree_json(
    app_name: str,
    cluster_id: str,
    features: list[str],
    reviews_df: pd.DataFrame,
    use_ollama: bool,
) -> tuple[dict, str]:
    features = sorted(features, key=_specificity, reverse=True)
    label = generate_label_ollama(features) if use_ollama else fallback_label(features)
    subtree = build_dendrogram(features)
    attach_reviews(subtree, app_name, reviews_df)
    depth = 1 + tree_depth(subtree)
    return {
        "app": app_name,
        "cluster_id": cluster_id,
        "label": label,
        "depth": depth,
        "tree": {
            "label": label,
            "is_leaf": False,
            "children": subtree["children"] if not subtree.get("is_leaf") else [subtree],
        },
    }, label


def build_cluster_list_json(
    app_name: str,
    cluster_id: str,
    features: list[str],
    reviews_df: pd.DataFrame,
) -> dict:
    features = sorted(features, key=_specificity, reverse=True)
    return {
        "app": app_name,
        "cluster_id": cluster_id,
        "features": [
            {"name": f, "reviews": find_reviews(reviews_df, app_name, f)}
            for f in features
        ],
    }


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

    provenance = load_provenance()
    if provenance:
        logger.info(f"Provenance: {provenance}")
    else:
        logger.warning("No provenance found in checkpoint — model/embedding/strategy unknown.")

    app_meta = load_app_metadata()
    raw_apps = load_app_clusters()
    if not raw_apps:
        logger.error("No data found. Run run_mobile_pipeline.py first.")
        sys.exit(1)

    all_clusters = collect_quality_clusters(raw_apps, app_meta)
    logger.info(
        f"Quality clusters: {len(all_clusters)} from {len(raw_apps)} apps "
        f"(size [{MIN_CLUSTER_SIZE}, {MAX_CLUSTER_SIZE}], "
        f"depth [{MIN_TREE_DEPTH}, {MAX_TREE_DEPTH}])"
    )
    if not all_clusters:
        logger.error("No quality clusters found after filtering.")
        sys.exit(1)

    by_category = defaultdict(list)
    for c in all_clusters:
        by_category[c["app_category"]].append(c)
    logger.info(
        f"Categories: {len(by_category)} — "
        f"{ {cat: len(v) for cat, v in sorted(by_category.items())} }"
    )

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

    sizes = np.array([c["n_features"] for c in all_clusters])
    logger.info(
        f"Cluster sizes — min:{sizes.min()} Q1:{np.percentile(sizes, 25):.0f} "
        f"median:{np.median(sizes):.0f} Q3:{np.percentile(sizes, 75):.0f} max:{sizes.max()}"
    )

    by_size = defaultdict(list)
    for c in all_clusters:
        by_size[c["n_features"]].append(c)
    logger.info(f"Cluster-size distribution: { {s: len(v) for s, v in sorted(by_size.items())} }")

    target_n = min(args.n, len(all_clusters))
    sampled = stratified_sample_clusters(all_clusters, target_n)
    random.shuffle(sampled)
    logger.info(
        f"Selected {len(sampled)} clusters "
        f"(stratified by category + size, max {MAX_CLUSTERS_PER_APP}/app)..."
    )

    bad_size = [
        c for c in sampled
        if not (MIN_CLUSTER_SIZE <= c["n_features"] <= MAX_CLUSTER_SIZE)
    ]
    if bad_size:
        logger.error(
            f"{len(bad_size)} sampled clusters outside "
            f"[{MIN_CLUSTER_SIZE}, {MAX_CLUSTER_SIZE}] — aborting."
        )
        sys.exit(1)

    bad_depth = [
        c for c in sampled
        if not (MIN_TREE_DEPTH <= c["tree_depth"] <= MAX_TREE_DEPTH)
    ]
    if bad_depth:
        logger.error(
            f"{len(bad_depth)} sampled clusters outside "
            f"depth [{MIN_TREE_DEPTH}, {MAX_TREE_DEPTH}] — aborting."
        )
        sys.exit(1)

    sel_by_cat = defaultdict(int)
    sel_by_size = defaultdict(int)
    sel_by_app = defaultdict(int)
    for c in sampled:
        sel_by_cat[c["app_category"]] += 1
        sel_by_size[c["n_features"]] += 1
        sel_by_app[c["app_name"]] += 1
    logger.info(f"Selected by category: {dict(sorted(sel_by_cat.items()))}")
    logger.info(f"Selected by size: {dict(sorted(sel_by_size.items()))}")
    logger.info(f"Apps represented: {len(sel_by_app)} (max {max(sel_by_app.values())}/app)")

    records, flat_rows = [], []
    for idx, c in enumerate(sampled, 1):
        app_name = c["app_name"]
        cluster_id = c["cluster_id"]
        logger.info(
            f"[{idx}/{len(sampled)}] {app_name} cluster {cluster_id} "
            f"({c['n_features']} features)..."
        )
        tree, label = build_cluster_tree_json(
            app_name, cluster_id, c["features"], reviews_df, use_ollama
        )
        logger.info(f"  cluster label: '{label}'")
        flat = build_cluster_list_json(app_name, cluster_id, c["features"], reviews_df)

        record = {
            "tree_id":        c["tree_id"],
            "app_name":       app_name,
            "app_package":    c["app_package"],
            "app_category":   c["app_category"],
            "cluster_id":     cluster_id,
            "label":          label,
            "n_features":     c["n_features"],
            "tree_depth":     tree.get("depth", c.get("tree_depth")),
            "model_type":     provenance.get("model_type", "unknown"),
            "embedding_type": provenance.get("embedding_type", "unknown"),
            "strategy":       provenance.get("selection_strategy", "unknown"),
            "sample_size":    provenance.get("sample_size", "unknown"),
            "tree_json":      tree,
            "list_json":      flat,
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
        sample = records[0]
        logger.info(
            f"\n=== Sample: {sample['app_name']} / {sample['label']} "
            f"({sample['n_features']} features, depth {sample.get('tree_depth')}) ==="
        )


if __name__ == "__main__":
    main()
