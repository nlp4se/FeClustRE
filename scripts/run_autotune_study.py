#!/usr/bin/env python3
"""
Collect autotuning sweep data for the journal extension.

For each app × review sample size × model type (t-frex, transfeatex, hybrid):
  - Raw features cached per review uid (incremental across sample sizes; hybrid reuses both sources)
  - Post-process + cluster per model type (shared post-processing pipeline)
  - Sweep height_threshold, record all metrics
  - Rank by autotune score (silhouette × (1 − singleton_ratio))
  - Select final config with balanced strategy on top-3
  - Record baselines (median / max-silhouette / fixed-0.5 / random)

No Flask, Neo4j, or Ollama.

Outputs (evaluation_results/autotune_study/):
  sweep_records.csv      — one row per (model_type, app, sample_size, threshold)
  selection_summary.csv  — one row per (model_type, app, sample_size) + baselines
  config.json, checkpoint.json

Usage:
  .venv/bin/python scripts/run_autotune_study.py
  .venv/bin/python scripts/run_autotune_study.py --resume
  .venv/bin/python scripts/run_autotune_study.py --apps 5   # smoke test
  .venv/bin/python scripts/run_autotune_study.py --models t-frex hybrid

Requires TRANSFEATEX_URL for transfeatex and hybrid (see .env).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.autotune_study_common import (
    CHECKPOINT_JSON,
    CONFIG_JSON,
    EMBEDDING_TYPE,
    FEATURE_CACHE_DIR,
    MODEL_TYPES,
    RANDOM_SEED,
    RAW_FEATURE_SOURCES,
    SAMPLE_SIZES,
    SELECTION_CSV,
    STUDY_DIR,
    SWEEP_CSV,
    THRESHOLD_RANGE,
    THRESHOLD_STEPS,
    TOP_K_AUTOTUNE,
    autotune_score,
    balanced_components,
    checkpoint_key,
    cluster_stats,
    flatten_result_row,
    is_checkpoint_done,
    migrate_study_csv,
    max_silhouette_result,
    median_threshold_result,
    pick_balanced_from_top3,
    random_threshold_results,
    rank_autotune_results,
    result_at_threshold,
    write_config,
)
from utils.health_checks import check_transfeatex
from services.clustering_service import HierarchicalClusterer
from services.feature_extraction_service import FeatureExtractor
from services.preprocessing_service import ReviewPreprocessor

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

CSV_FILE = PROJECT_ROOT / "data/input/endpoint_1_process_reviews/mobile_apps/mobilerec_reviews_pipeline_large.csv"


def resolve_model_types(requested: list[str]) -> list[str]:
    """Return model types that can run; transfeatex/hybrid need a healthy TransFeatEx endpoint."""
    normalized = []
    for m in requested:
        m = FeatureExtractor._normalize_model_type(m)
        if m not in MODEL_TYPES:
            logger.warning(f"Unknown model type '{m}' — skipped")
            continue
        normalized.append(m)

    if not normalized:
        normalized = ["t-frex"]

    tfex = check_transfeatex()
    needs_tfex = {"transfeatex", "hybrid"}
    if needs_tfex.intersection(normalized) and tfex.get("status") != "healthy":
        msg = tfex.get("message") or tfex.get("error", "TransFeatEx unavailable")
        for m in list(normalized):
            if m in needs_tfex:
                logger.error(f"Cannot run '{m}': {msg}")
                normalized.remove(m)
        if not normalized:
            raise SystemExit(
                "No runnable model types. Set TRANSFEATEX_URL and start TransFeatEx for hybrid/transfeatex."
            )

    return normalized


def load_checkpoint() -> dict:
    if CHECKPOINT_JSON.exists():
        return json.loads(CHECKPOINT_JSON.read_text())
    return {"completed": {}, "started_at": datetime.now().isoformat()}


def save_checkpoint(cp: dict) -> None:
    STUDY_DIR.mkdir(parents=True, exist_ok=True)
    cp["last_updated"] = datetime.now().isoformat()
    CHECKPOINT_JSON.write_text(json.dumps(cp, indent=2))


def sample_reviews(df: pd.DataFrame, app_name: str, sample_size: int) -> pd.DataFrame:
    app_df = df[df["app_name"] == app_name]
    if sample_size <= 0 or sample_size >= len(app_df):
        return app_df
    return app_df.sample(n=sample_size, random_state=RANDOM_SEED)


def prepare_sample(
    df: pd.DataFrame, app_name: str, sample_size: int, preprocessor: ReviewPreprocessor
) -> pd.DataFrame:
    """Return sampled reviews with stable uid and preprocessed text (order preserved)."""
    app_df = sample_reviews(df, app_name, sample_size).copy()
    if "uid" not in app_df.columns:
        app_df["uid"] = app_df.index.astype(str)
    app_df["processed_text"] = [
        preprocessor.preprocess_text(t) for t in app_df["review"].astype(str)
    ]
    return app_df


def _cache_file_for_app(app_name: str) -> Path:
    digest = hashlib.sha1(app_name.encode("utf-8")).hexdigest()
    return FEATURE_CACHE_DIR / f"{digest}.json"


class RawFeatureCache:
    """Per-app cache of raw (pre-postprocess) features keyed by review uid."""

    def __init__(self, app_name: str):
        self.app_name = app_name
        self.path = _cache_file_for_app(app_name)
        self._data: dict[str, dict[str, list[str]]] = {s: {} for s in RAW_FEATURE_SOURCES}
        if self.path.exists():
            loaded = json.loads(self.path.read_text())
            for source in RAW_FEATURE_SOURCES:
                self._data[source] = {
                    uid: list(features)
                    for uid, features in loaded.get(source, {}).items()
                }

    def _save(self) -> None:
        FEATURE_CACHE_DIR.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(self._data))

    def ensure(self, source: str, extractor: FeatureExtractor, sample_df: pd.DataFrame) -> None:
        uids = sample_df["uid"].astype(str).tolist()
        texts = sample_df["processed_text"].tolist()
        missing = [(uid, text) for uid, text in zip(uids, texts) if uid not in self._data[source]]
        n_hit = len(uids) - len(missing)
        if not missing:
            logger.debug(f"[{source}] {self.app_name}: cache hit {n_hit}/{len(uids)} reviews")
            return

        logger.info(
            f"[{source}] {self.app_name}: cache hit {n_hit}/{len(uids)}, "
            f"extracting {len(missing)} new reviews"
        )
        missing_uids, missing_texts = zip(*missing)
        raw_lists = extractor.extract_features_raw(list(missing_texts))
        for uid, feats in zip(missing_uids, raw_lists):
            self._data[source][uid] = list(feats)
        self._save()

    def raw_features_for_model(self, model_type: str, uids: list[str]) -> list[list[str]]:
        if model_type == "t-frex":
            return [list(self._data["t-frex"].get(uid, [])) for uid in uids]
        if model_type == "transfeatex":
            return [list(self._data["transfeatex"].get(uid, [])) for uid in uids]
        if model_type == "hybrid":
            return [
                list(set(self._data["t-frex"].get(uid, [])) | set(self._data["transfeatex"].get(uid, [])))
                for uid in uids
            ]
        raise ValueError(f"Unknown model_type: {model_type}")


def postprocess_features(extractor: FeatureExtractor, raw_features: list[list[str]]) -> list[list[str]]:
    if extractor.enable_postprocessing and extractor.postprocessor:
        return extractor.postprocessor.process_features_list(raw_features)
    return raw_features


def enrich_result(threshold: float, metrics: dict, clusters: dict) -> dict:
    stats = cluster_stats(clusters)
    return {
        "threshold": float(threshold),
        "metrics": metrics,
        "clusters": clusters,
        **stats,
    }


def run_sweep(clusterer: HierarchicalClusterer, features: list[str], embeddings: np.ndarray) -> list[dict]:
    results = []
    for threshold in np.linspace(THRESHOLD_RANGE[0], THRESHOLD_RANGE[1], THRESHOLD_STEPS):
        clusterer.height_threshold = float(threshold)
        out = clusterer.perform_clustering(features, embeddings)
        if out["n_clusters"] < 2:
            continue

        labels = []
        for feature in features:
            for cid, feats in out["clusters"].items():
                if feature in feats:
                    labels.append(int(cid))
                    break

        metrics = clusterer.evaluate_clustering(features, embeddings, labels)
        if "error" in metrics:
            continue

        results.append(enrich_result(float(threshold), metrics, out["clusters"]))
    return results


def process_app_sample(
    model_type: str,
    app_name: str,
    sample_size: int,
    n_reviews: int,
    features_per_review: list[list[str]],
    extractor: FeatureExtractor,
    clusterer: HierarchicalClusterer,
) -> tuple[list[dict], dict | None, str | None]:
    all_features = [f for flist in features_per_review for f in flist]
    unique_features = sorted(set(all_features))
    n_unique = len(unique_features)

    if n_unique < 4:
        return [], None, f"only {n_unique} unique features (need ≥4)"

    embeddings = extractor.get_embeddings(unique_features)
    all_results = run_sweep(clusterer, unique_features, embeddings)
    if not all_results:
        return [], None, "no valid threshold in sweep"

    ranked = rank_autotune_results(all_results)
    top3 = ranked[:TOP_K_AUTOTUNE]

    for i, r in enumerate(top3):
        r.update(balanced_components(
            r["metrics"]["silhouette_score"],
            r["metrics"]["davies_bouldin_score"],
            r["n_clusters"],
            r["avg_cluster_size"],
        ))

    selected = pick_balanced_from_top3(top3)

    sweep_rows = []
    top3_thresholds = {r["threshold"] for r in top3}
    for rank, r in enumerate(ranked, start=1):
        row = flatten_result_row(
            model_type, app_name, sample_size, n_reviews, n_unique, r,
            autotune_rank=rank,
            selected=(r["threshold"] == selected["threshold"]),
        )
        if r["threshold"] in top3_thresholds:
            t3 = next(x for x in top3 if x["threshold"] == r["threshold"])
            row.update({
                "balanced_score": t3["balanced_score"],
                "sil_component": t3["sil_component"],
                "db_component": t3["db_component"],
                "cluster_penalty_component": t3["cluster_penalty_component"],
                "size_penalty_component": t3["size_penalty_component"],
            })
        sweep_rows.append(row)

    baselines = {
        "median_threshold": median_threshold_result(all_results),
        "max_silhouette": max_silhouette_result(all_results),
        "fixed_0.5": result_at_threshold(all_results, 0.5),
        "random_mean": None,
    }
    random_draws = random_threshold_results(all_results)
    if random_draws:
        baselines["random_mean"] = {
            "threshold": float(np.mean([r["threshold"] for r in random_draws])),
            "metrics": {
                "silhouette_score": float(np.mean([r["metrics"]["silhouette_score"] for r in random_draws])),
                "davies_bouldin_score": float(np.mean([r["metrics"]["davies_bouldin_score"] for r in random_draws])),
            },
            "n_clusters": round(float(np.mean([r["n_clusters"] for r in random_draws])), 2),
            "avg_cluster_size": round(float(np.mean([r["avg_cluster_size"] for r in random_draws])), 2),
            "n_singletons": round(float(np.mean([r["n_singletons"] for r in random_draws])), 2),
            "singleton_ratio": round(float(np.mean([r["singleton_ratio"] for r in random_draws])), 4),
        }

    summary = {
        "model_type": model_type,
        "app_name": app_name,
        "sample_size": sample_size,
        "n_reviews": n_reviews,
        "n_unique_features": n_unique,
        "n_sweep_points": len(all_results),
        "selected_threshold": selected["threshold"],
        "selected_silhouette": selected["metrics"]["silhouette_score"],
        "selected_davies_bouldin": selected["metrics"]["davies_bouldin_score"],
        "selected_n_clusters": selected["n_clusters"],
        "selected_avg_cluster_size": selected["avg_cluster_size"],
        "selected_singleton_ratio": selected["singleton_ratio"],
        "selected_autotune_score": autotune_score(
            selected["metrics"]["silhouette_score"], selected["singleton_ratio"]
        ),
        "selected_balanced_score": selected["balanced_score"],
        "selected_autotune_rank": next(
            i for i, r in enumerate(ranked, 1) if r["threshold"] == selected["threshold"]
        ),
    }

    for name, br in baselines.items():
        if br is None:
            summary[f"baseline_{name}_silhouette"] = None
            summary[f"baseline_{name}_balanced_score"] = None
            continue
        bc = balanced_components(
            br["metrics"]["silhouette_score"],
            br["metrics"]["davies_bouldin_score"],
            br["n_clusters"],
            br["avg_cluster_size"],
        )
        summary[f"baseline_{name}_threshold"] = br["threshold"]
        summary[f"baseline_{name}_silhouette"] = br["metrics"]["silhouette_score"]
        summary[f"baseline_{name}_n_clusters"] = br["n_clusters"]
        summary[f"baseline_{name}_singleton_ratio"] = br["singleton_ratio"]
        summary[f"baseline_{name}_balanced_score"] = bc["balanced_score"]
        summary["selected_beats_baseline_" + name] = (
            selected["balanced_score"] > bc["balanced_score"]
        )

    return sweep_rows, summary, None


def append_csv(path: Path, rows: list[dict], columns: list[str] | None = None) -> None:
    if not rows:
        return
    if path.exists():
        migrate_study_csv(path)

    df = pd.DataFrame(rows)
    if columns:
        df = df.reindex(columns=columns, fill_value=None)

    if path.exists():
        existing = pd.read_csv(path)
        all_cols = list(dict.fromkeys([*existing.columns, *df.columns]))
        df = df.reindex(columns=all_cols)
        df.to_csv(path, mode="a", index=False, header=False)
    else:
        df.to_csv(path, index=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--apps", type=int, default=0, help="Limit to first N apps (0 = all)")
    parser.add_argument(
        "--models",
        nargs="+",
        default=MODEL_TYPES,
        help=f"Feature extractors to evaluate (default: {' '.join(MODEL_TYPES)})",
    )
    args = parser.parse_args()

    model_types = resolve_model_types(args.models)
    STUDY_DIR.mkdir(parents=True, exist_ok=True)
    for csv_path in (SWEEP_CSV, SELECTION_CSV):
        if migrate_study_csv(csv_path):
            logger.info(f"Migrated legacy rows in {csv_path.name} (added model_type=t-frex)")
    write_config(model_types)

    logger.info(f"Loading {CSV_FILE.name}...")
    reviews_df = pd.read_csv(CSV_FILE).dropna(subset=["review"])
    reviews_df["review"] = reviews_df["review"].astype(str)

    apps = sorted(reviews_df["app_name"].unique())
    if args.apps > 0:
        apps = apps[: args.apps]
    logger.info(
        f"Models: {model_types}, apps: {len(apps)}, "
        f"sample sizes per app: {SAMPLE_SIZES}"
    )

    cp = load_checkpoint() if args.resume else {"completed": {}, "started_at": datetime.now().isoformat()}
    completed = cp["completed"]

    preprocessor = ReviewPreprocessor()
    clusterer = HierarchicalClusterer()
    extractors: dict[str, FeatureExtractor] = {}

    def get_extractor(model_type: str) -> FeatureExtractor:
        if model_type not in extractors:
            logger.info(f"Loading FeatureExtractor ({model_type}, {EMBEDDING_TYPE})...")
            extractors[model_type] = FeatureExtractor(
                model_type=model_type, embedding_model=EMBEDDING_TYPE
            )
        return extractors[model_type]

    needs_tfrex = {"t-frex", "hybrid"}.intersection(model_types)
    needs_transfeatex = {"transfeatex", "hybrid"}.intersection(model_types)
    if needs_tfrex:
        get_extractor("t-frex")
    if needs_transfeatex:
        get_extractor("transfeatex")

    total = len(model_types) * len(apps) * len(SAMPLE_SIZES)
    done_before = sum(
        1
        for mt in model_types
        for app in apps
        for ss in SAMPLE_SIZES
        if is_checkpoint_done(completed, mt, app, ss)
    )
    logger.info(f"Checkpoint: {done_before} / {total} (model, app, sample_size) triples already done")

    embedding_extractor = get_extractor("t-frex")

    for app_name in apps:
        cache = RawFeatureCache(app_name)
        tfrex_extractor = extractors.get("t-frex")
        transfeatex_extractor = extractors.get("transfeatex")

        for sample_size in SAMPLE_SIZES:
            pending = [
                mt for mt in model_types
                if not is_checkpoint_done(completed, mt, app_name, sample_size)
            ]
            if not pending:
                continue

            sample_df = prepare_sample(reviews_df, app_name, sample_size, preprocessor)
            uids = sample_df["uid"].astype(str).tolist()
            n_reviews = len(sample_df)

            if needs_tfrex and tfrex_extractor is not None:
                cache.ensure("t-frex", tfrex_extractor, sample_df)
            if needs_transfeatex and transfeatex_extractor is not None:
                cache.ensure("transfeatex", transfeatex_extractor, sample_df)

            for model_type in pending:
                key = checkpoint_key(model_type, app_name, sample_size)
                t0 = time.time()
                try:
                    raw = cache.raw_features_for_model(model_type, uids)
                    features_per_review = postprocess_features(embedding_extractor, raw)
                    sweep_rows, summary, err = process_app_sample(
                        model_type, app_name, sample_size, n_reviews,
                        features_per_review, embedding_extractor, clusterer,
                    )
                    elapsed = time.time() - t0

                    if err:
                        completed[key] = {"status": "skipped", "reason": err, "elapsed": elapsed}
                        logger.warning(f"SKIP [{model_type}] {app_name} n={sample_size}: {err}")
                    else:
                        append_csv(SWEEP_CSV, sweep_rows)
                        append_csv(SELECTION_CSV, [summary])
                        completed[key] = {"status": "success", "elapsed": elapsed}
                        logger.info(
                            f"OK [{model_type}] {app_name} sample={sample_size or 'all'} "
                            f"({summary['n_reviews']} reviews, {summary['n_unique_features']} features, "
                            f"thr={summary['selected_threshold']:.3f}, "
                            f"bal={summary['selected_balanced_score']:.4f}) "
                            f"[{elapsed:.1f}s]"
                        )
                except Exception as e:
                    completed[key] = {"status": "failed", "error": str(e)}
                    logger.exception(f"FAIL [{model_type}] {app_name} n={sample_size}: {e}")

                cp["completed"] = completed
                save_checkpoint(cp)

    n_ok = sum(
        1
        for mt in model_types
        for app in apps
        for ss in SAMPLE_SIZES
        if completed.get(checkpoint_key(mt, app, ss), {}).get("status") == "success"
        or (mt == "t-frex" and completed.get(f"{app}__{ss}", {}).get("status") == "success")
    )
    logger.info(f"Done. {n_ok}/{total} successful. Data in {STUDY_DIR}")


if __name__ == "__main__":
    main()
