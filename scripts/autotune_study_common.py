"""Shared scoring and metrics for the autotune study (run + visualize)."""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import pandas as pd

# Production auto_tune uses steps=8; study uses 12 for smoother sweep curves (same algorithm).
THRESHOLD_RANGE = (0.1, 0.9)
THRESHOLD_STEPS = 12
TOP_K_AUTOTUNE = 3
SELECTION_STRATEGY = "balanced"
RANDOM_SEED = 42
SAMPLE_SIZES = [50, 100, 200, 300, 500, 1000, 0]  # 0 = all reviews per app
MODEL_TYPES = ["t-frex", "transfeatex", "hybrid"]
EMBEDDING_TYPE = "allmini"

STUDY_DIR = Path(__file__).resolve().parent.parent / "evaluation_results" / "autotune_study"
SWEEP_CSV = STUDY_DIR / "sweep_records.csv"
SELECTION_CSV = STUDY_DIR / "selection_summary.csv"
CONFIG_JSON = STUDY_DIR / "config.json"
CHECKPOINT_JSON = STUDY_DIR / "checkpoint.json"
FIGURES_DIR = STUDY_DIR / "figures"
FEATURE_CACHE_DIR = STUDY_DIR / "feature_cache"
RAW_FEATURE_SOURCES = ("t-frex", "transfeatex")
LEGACY_MODEL_TYPE = "t-frex"


def migrate_study_csv(path: Path) -> bool:
    """Normalize legacy CSV rows that predate the model_type column. Returns True if rewritten."""
    if not path.exists():
        return False

    with path.open(newline="") as f:
        reader = csv.reader(f)
        try:
            header = next(reader)
        except StopIteration:
            return False

        if "model_type" in header:
            return False

        rows = list(reader)

    migrated = []
    legacy_width = len(header)
    for row in rows:
        if len(row) == legacy_width:
            migrated.append([LEGACY_MODEL_TYPE, *row])
        elif len(row) == legacy_width + 1:
            migrated.append(row)
        else:
            raise ValueError(
                f"{path.name}: expected {legacy_width} or {legacy_width + 1} fields, got {len(row)}"
            )

    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["model_type", *header])
        writer.writerows(migrated)

    return True


def load_study_csv(path: Path) -> pd.DataFrame:
    migrate_study_csv(path)
    return pd.read_csv(path)


def cluster_stats(clusters: dict) -> dict:
    if not clusters:
        return {
            "n_clusters": 0,
            "avg_cluster_size": 0.0,
            "n_singletons": 0,
            "singleton_ratio": 0.0,
        }
    sizes = [len(f) for f in clusters.values()]
    n_singletons = sum(1 for s in sizes if s == 1)
    n_clusters = len(clusters)
    return {
        "n_clusters": n_clusters,
        "avg_cluster_size": round(sum(sizes) / n_clusters, 3),
        "n_singletons": n_singletons,
        "singleton_ratio": round(n_singletons / n_clusters, 4),
    }


def autotune_score(silhouette: float, singleton_ratio: float) -> float:
    return float(silhouette) * (1.0 - float(singleton_ratio))


def balanced_components(
    silhouette: float,
    davies_bouldin: float,
    n_clusters: int,
    avg_cluster_size: float,
) -> dict:
    db_inv = 1.0 / (1.0 + float(davies_bouldin))
    cluster_penalty = 1.0 / (1.0 + abs(n_clusters - 5))
    size_penalty = 1.0 / (1.0 + abs(avg_cluster_size - 10))
    total = (
        silhouette * 0.4
        + db_inv * 0.3
        + cluster_penalty * 0.15
        + size_penalty * 0.15
    )
    return {
        "balanced_score": round(total, 6),
        "sil_component": round(silhouette * 0.4, 6),
        "db_component": round(db_inv * 0.3, 6),
        "cluster_penalty_component": round(cluster_penalty * 0.15, 6),
        "size_penalty_component": round(size_penalty * 0.15, 6),
    }


def rank_autotune_results(results: list[dict]) -> list[dict]:
    """Same ranking as clustering_service.auto_tune_clustering."""
    ranked = sorted(
        results,
        key=lambda r: autotune_score(
            r["metrics"].get("silhouette_score", -1),
            r.get("singleton_ratio", 0),
        ),
        reverse=True,
    )
    return ranked


def pick_balanced_from_top3(top3: list[dict]) -> dict:
    best = None
    best_score = -1.0
    for r in top3:
        bc = balanced_components(
            r["metrics"]["silhouette_score"],
            r["metrics"]["davies_bouldin_score"],
            r["n_clusters"],
            r["avg_cluster_size"],
        )
        if bc["balanced_score"] > best_score:
            best_score = bc["balanced_score"]
            best = {**r, **bc}
    return best


def result_at_threshold(all_results: list[dict], target: float) -> dict | None:
    if not all_results:
        return None
    return min(all_results, key=lambda r: abs(r["threshold"] - target))


def max_silhouette_result(all_results: list[dict]) -> dict | None:
    if not all_results:
        return None
    return max(all_results, key=lambda r: r["metrics"].get("silhouette_score", -1))


def median_threshold_result(all_results: list[dict]) -> dict | None:
    if not all_results:
        return None
    thresholds = [r["threshold"] for r in all_results]
    med = float(np.median(thresholds))
    return result_at_threshold(all_results, med)


def random_threshold_results(all_results: list[dict], n_draws: int = 5, seed: int = RANDOM_SEED) -> list[dict]:
    if not all_results:
        return []
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(all_results), size=n_draws)
    return [all_results[i] for i in idx]


def checkpoint_key(model_type: str, app_name: str, sample_size: int) -> str:
    return f"{model_type}__{app_name}__{sample_size}"


def legacy_checkpoint_key(app_name: str, sample_size: int) -> str:
    """Pre–multi-model checkpoint entries (t-frex only)."""
    return f"{app_name}__{sample_size}"


def is_checkpoint_done(completed: dict, model_type: str, app_name: str, sample_size: int) -> bool:
    if checkpoint_key(model_type, app_name, sample_size) in completed:
        return True
    if model_type == "t-frex" and legacy_checkpoint_key(app_name, sample_size) in completed:
        return True
    return False


def flatten_result_row(
    model_type: str,
    app_name: str,
    sample_size: int,
    n_reviews: int,
    n_unique_features: int,
    r: dict,
    *,
    autotune_rank: int | None = None,
    selected: bool = False,
    baseline: str | None = None,
) -> dict:
    row = {
        "model_type": model_type,
        "app_name": app_name,
        "sample_size": sample_size,
        "n_reviews": n_reviews,
        "n_unique_features": n_unique_features,
        "height_threshold": r["threshold"],
        "silhouette_score": r["metrics"]["silhouette_score"],
        "davies_bouldin_score": r["metrics"]["davies_bouldin_score"],
        "n_clusters": r["n_clusters"],
        "avg_cluster_size": r["avg_cluster_size"],
        "n_singletons": r["n_singletons"],
        "singleton_ratio": r["singleton_ratio"],
        "autotune_score": round(
            autotune_score(r["metrics"]["silhouette_score"], r["singleton_ratio"]), 6
        ),
        "autotune_rank": autotune_rank,
        "selected": selected,
        "baseline": baseline or "",
    }
    if "balanced_score" in r:
        row.update({
            "balanced_score": r["balanced_score"],
            "sil_component": r.get("sil_component"),
            "db_component": r.get("db_component"),
            "cluster_penalty_component": r.get("cluster_penalty_component"),
            "size_penalty_component": r.get("size_penalty_component"),
        })
    return row


def write_config(model_types: list[str], path: Path = CONFIG_JSON) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    cfg = {
        "threshold_range": list(THRESHOLD_RANGE),
        "threshold_steps": THRESHOLD_STEPS,
        "production_threshold_steps": 8,
        "top_k_autotune": TOP_K_AUTOTUNE,
        "selection_strategy": SELECTION_STRATEGY,
        "random_seed": RANDOM_SEED,
        "sample_sizes_per_app": SAMPLE_SIZES,
        "model_types": model_types,
        "embedding_type": EMBEDDING_TYPE,
        "balanced_weights": {
            "silhouette": 0.4,
            "davies_bouldin_inv": 0.3,
            "cluster_count_penalty": 0.15,
            "avg_size_penalty": 0.15,
        },
        "balanced_targets": {"n_clusters": 5, "avg_cluster_size": 10},
    }
    path.write_text(json.dumps(cfg, indent=2))
