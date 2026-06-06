#!/usr/bin/env python3
"""
Run the FeClustRE pipeline on the full mobile apps dataset, one app at a time.

Checkpoints each app result to disk immediately after processing so a crash
can be resumed from the last completed app.

Checkpoint file: evaluation_results/mobile_pipeline_checkpoint.json
Final session:   evaluation_results/test_session_mobile_<timestamp>.json

Prerequisites: Flask app running on localhost:3000
  .venv/bin/python app.py

Usage:
  .venv/bin/python scripts/run_mobile_pipeline.py [--sample 300] [--resume]
"""
import argparse
import json
import logging
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from urllib.parse import quote

import pandas as pd
import requests

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

#CSV_FILE = PROJECT_ROOT / "data/input/endpoint_1_process_reviews/mobile_apps/mobilerec_reviews_pipeline.csv"
CSV_FILE = PROJECT_ROOT / "data/input/endpoint_1_process_reviews/mobile_apps/mobilerec_reviews_pipeline_large.csv"
BASE_URL = "http://localhost:3000"
CHECKPOINT_FILE = PROJECT_ROOT / "evaluation_results/mobile_pipeline_checkpoint.json"
MODEL_TYPE = "t-frex"
EMBEDDING_TYPE = "allmini"
SELECTION_STRATEGY = "balanced"


def load_checkpoint() -> dict:
    if CHECKPOINT_FILE.exists():
        with open(CHECKPOINT_FILE) as f:
            return json.load(f)
    return {"completed_apps": {}, "started_at": datetime.now().isoformat()}


def save_checkpoint(checkpoint: dict):
    CHECKPOINT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(CHECKPOINT_FILE, "w") as f:
        json.dump(checkpoint, f, indent=2, default=str)


def process_app(app_name: str, app_df: pd.DataFrame) -> dict | None:
    """Send one app's reviews to the Flask API and return the result."""
    csv_bytes = app_df.to_csv(index=False).encode("utf-8")
    try:
        resp = requests.post(
            f"{BASE_URL}/process_reviews/upload",
            files={"file": (f"{app_name}.csv", csv_bytes, "text/csv")},
            params={"model_type": MODEL_TYPE, "embedding_type": EMBEDDING_TYPE},
            timeout=None,
        )
        if resp.ok:
            data = resp.json()
            results = data.get("results", {})
            return results.get(app_name)
        else:
            logger.error(f"API error for '{app_name}': {resp.status_code} {resp.text[:200]}")
            return None
    except Exception as e:
        logger.error(f"Request failed for '{app_name}': {e}")
        return None


def auto_select_best(app_result: dict, strategy: str = "balanced") -> dict | None:
    """Pick the best clustering candidate from an app result."""
    candidates = app_result.get("clustering_results", {}).get("candidates", [])
    if not candidates:
        return None

    scored = []
    for i, c in enumerate(candidates):
        m = c.get("summary", {}).get("metrics", {})
        sil = m.get("silhouette_score", 0)
        db = 1 / (1 + m.get("davies_bouldin_score", 1))
        n = c.get("summary", {}).get("n_clusters", 1)
        avg = c.get("summary", {}).get("avg_cluster_size", 1)

        if strategy == "balanced":
            score = sil * 0.4 + db * 0.3 + (1 / (1 + abs(n - 5))) * 0.15 + (1 / (1 + abs(avg - 10))) * 0.15
        else:
            score = sil

        scored.append((i, score, c))

    best_i, best_score, best_c = max(scored, key=lambda x: x[1])
    return {"candidate_index": best_i, "score": best_score, "candidate": best_c}


def save_to_neo4j(app_name: str, selection: dict) -> dict | None:
    """Call save_selected_clustering — optional, skipped if Neo4j is down."""
    clustering = selection["candidate"].get("clustering", {})
    try:
        resp = requests.post(
            f"{BASE_URL}/save_selected_clustering/{quote(app_name, safe='')}",
            json={"clustering": clustering, "provenance": {
                "model_type": MODEL_TYPE,
                "embedding_type": EMBEDDING_TYPE,
                "selection_strategy": SELECTION_STRATEGY,
                "selection_score": round(float(selection["score"]), 4),
            }},
            timeout=30,
        )
        if resp.ok:
            return resp.json()
        logger.warning(f"Neo4j save skipped for '{app_name}': {resp.text[:100]}")
    except Exception as e:
        logger.warning(f"Neo4j save skipped for '{app_name}': {e}")
    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample", type=int, default=300, help="Reviews per app (default: 300). Use 0 for all reviews.")
    parser.add_argument("--resume", action="store_true", help="Resume from checkpoint")
    parser.add_argument("--workers", type=int, default=1,
                        help="Parallel app workers (default: 1). Set >1 only if Flask runs with threaded=True)")
    args = parser.parse_args()

    # Check app is running
    try:
        requests.get(f"{BASE_URL}/ping", timeout=5)
    except Exception:
        logger.error(f"Flask app not reachable at {BASE_URL}. Start it with: .venv/bin/python app.py")
        sys.exit(1)

    logger.info(f"Loading CSV: {CSV_FILE.name}")
    df = pd.read_csv(CSV_FILE).dropna(subset=["review"])
    df["review"] = df["review"].astype(str)

    # Sample per app (0 = all reviews)
    if args.sample > 0:
        df = df.groupby("app_name").apply(
            lambda x: x.sample(min(len(x), args.sample), random_state=42)
        ).reset_index(drop=True)
    logger.info(f"Sampled {len(df)} rows across {df['app_name'].nunique()} apps ({args.sample or 'all'}/app max)")

    checkpoint = load_checkpoint() if args.resume else {
        "completed_apps": {},
        "started_at": datetime.now().isoformat(),
        "provenance": {
            "model_type": MODEL_TYPE,
            "embedding_type": EMBEDDING_TYPE,
            "selection_strategy": SELECTION_STRATEGY,
            "sample_size": args.sample,
        },
    }
    completed = checkpoint["completed_apps"]

    if completed:
        logger.info(f"Resuming — {len(completed)} apps already done, skipping them.")

    apps = sorted(df["app_name"].unique())
    remaining = [a for a in apps if a not in completed]
    logger.info(f"Apps to process: {len(remaining)} / {len(apps)}")

    checkpoint_lock = threading.Lock()
    done_count = [0]

    def process_one(app_name: str) -> tuple[str, dict]:
        app_df = df[df["app_name"] == app_name].copy()
        t0 = time.time()
        result = process_app(app_name, app_df)
        elapsed = time.time() - t0
        if result is None:
            return app_name, {"status": "failed", "elapsed": elapsed}
        selection = auto_select_best(result, SELECTION_STRATEGY)
        neo4j_result = save_to_neo4j(app_name, selection) if selection else None
        clusters = selection["candidate"]["clustering"].get("clusters", {}) if selection else {}
        return app_name, {
            "status": "success",
            "elapsed": elapsed,
            "unique_features": result.get("unique_features", 0),
            "clusters": clusters,
            "neo4j_saved": neo4j_result is not None,
        }

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(process_one, app_name): app_name for app_name in remaining}
        for future in as_completed(futures):
            app_name, entry = future.result()
            with checkpoint_lock:
                done_count[0] += 1
                completed[app_name] = entry
                checkpoint["completed_apps"] = completed
                checkpoint["last_updated"] = datetime.now().isoformat()
                save_checkpoint(checkpoint)
            if entry["status"] == "success":
                logger.info(
                    f"[{done_count[0]}/{len(remaining)}] {app_name}: "
                    f"{entry['elapsed']:.1f}s — {entry['unique_features']} features, "
                    f"{len(entry['clusters'])} clusters"
                )
            else:
                logger.warning(f"[{done_count[0]}/{len(remaining)}] {app_name}: skipped (no result)")

    # Summary
    success = [a for a, v in completed.items() if v.get("status") == "success"]
    neo4j_saved = [a for a, v in completed.items() if v.get("neo4j_saved")]
    logger.info(f"\nDone. {len(success)}/{len(apps)} apps successful, {len(neo4j_saved)} saved to Neo4j.")
    logger.info(f"Checkpoint: {CHECKPOINT_FILE}")

    # Write final session JSON compatible with generate_experiment scripts
    final = {
        "session_id": f"mobile_pipeline_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        "start_time": checkpoint.get("started_at"),
        "end_time": datetime.now().isoformat(),
        "configurations": [{
            "csv_file": str(CSV_FILE),
            "model_type": MODEL_TYPE,
            "embedding_type": EMBEDDING_TYPE,
            "sample_size": args.sample,
            "selection_strategy": SELECTION_STRATEGY,
            "best_selections": {
                app_name: {
                    "score": 0,
                    "candidate": {"clustering": {"clusters": v["clusters"]}}
                }
                for app_name, v in completed.items()
                if v.get("status") == "success" and v.get("clusters")
            }
        }]
    }
    out = PROJECT_ROOT / f"evaluation_results/test_session_mobile_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(out, "w") as f:
        json.dump(final, f, indent=2, default=str)
    logger.info(f"Session file: {out}")


if __name__ == "__main__":
    main()
