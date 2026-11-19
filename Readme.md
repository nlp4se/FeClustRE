# FeClustRE

FeClustRE turns flat lists of app features mined from user reviews into **hierarchical, semantically labelled taxonomies**, one per app. This repository holds the framework and tool for the paper *App Review Mining through Hybrid Feature Extraction and Hierarchical Semantic Taxonomies*.

The pipeline has three stages:
1. **Hybrid feature extraction:** the union of a syntactic method (TransFeatEx) and an LLM-based method (T-FREX), with shared pre- and post-processing.
2. **Hierarchical clustering:** `all-MiniLM-L6-v2` embeddings, average-linkage agglomerative clustering on cosine distance, with the cut threshold τ\* auto-tuned per app.
3. **Semantic tagging:** each cluster is labelled with `llama3.2:3b`, and the taxonomies are stored in Neo4j.

What this repository covers in the paper's evaluation:

| RQ | What | Where |
|----|------|-------|
| RQ₁ | Extraction correctness on two annotated benchmarks | Separate replication package (see the paper's Data Availability Statement) |
| RQ₂ | Auto-tuned clustering on 100 apps and 7 review sample sizes | [RQ₂ — autotune study](#rq₂--hierarchical-clustering-autotune-study) |
| RQ₃ | Taxonomies for the human evaluation experiments | [Experiments](#experiments) |

---

## Table of Contents

- [Pipeline](#pipeline)
- [Setup](#setup)
- [Usage](#usage)
- [Experiments](#experiments)
  - [RQ₂ — Hierarchical clustering (autotune study)](#rq₂--hierarchical-clustering-autotune-study)
- [Dataset](#dataset)
- [Troubleshooting](#troubleshooting)

---

## Pipeline

```
CSV reviews → pre-process → TransFeatEx ∪ T-FREX → post-process → embed (all-MiniLM-L6-v2)
  → agglomerative clustering (auto-tuned τ*) → LLM tagging (Ollama) → Neo4j taxonomy
```

| Stage | Component |
|-------|-----------|
| Feature extraction | Hybrid: TransFeatEx (syntactic) ∪ T-FREX (LLM-based NER); each can also run alone |
| Embedding | all-MiniLM-L6-v2 |
| Clustering | Agglomerative (average linkage, cosine distance), auto-tuned threshold τ* |
| Labeling | Ollama llama3.2:3b |
| Storage | Neo4j 5.15 |

---

## Setup

### Docker (recommended)

```bash
git clone https://github.com/nlp4se/FeClustRE.git && cd FeClustRE
docker compose up --build
```

| Service | URL |
|---------|-----|
| API | http://localhost:3000 |
| Neo4j | http://localhost:7474 |
| Ollama | http://localhost:11434 |

### Local

```bash
python3.10 -m venv .venv && .venv/bin/pip install -r requirements.txt

# Start Neo4j + Ollama
docker run -d --name neo4j -e NEO4J_AUTH=neo4j/12345678 -p 7474:7474 -p 7687:7687 neo4j:5.15
ollama serve & ollama pull llama3.2:3b

# Start API
.venv/bin/python app.py
```

---

## Usage

### Run the pipeline

```bash
.venv/bin/python scripts/run_mobile_pipeline.py          # fresh run
.venv/bin/python scripts/run_mobile_pipeline.py --resume  # resume after interruption
```

### Query results (Neo4j)

```cypher
MATCH (app:App)-[:HAS_MINI_TAXONOMY]->(root:MiniTaxonomyNode)
OPTIONAL MATCH (root)-[:HAS_CHILD*]->(leaf)
WHERE NOT (leaf)-[:HAS_CHILD]->()
RETURN app.name, root.llm_tag, collect(DISTINCT leaf.feature)
```

---

## Experiments

### Generate

```bash
# Experiment 1 — parent/child validation (n=300, stratified by tree depth)
.venv/bin/python scripts/generate_experiment1.py

# Experiment 2 — hierarchical taxonomy vs flat feature list
.venv/bin/python scripts/generate_experiment2.py
```

### Visualize

```bash
.venv/bin/streamlit run scripts/visualize_experiments.py
```

### Reset

```bash
# Regenerate experiments only
rm -f data/experiment1.csv data/experiment2.json data/experiment2_flat.csv

# Full reset (pipeline + experiments)
rm -f evaluation_results/mobile_pipeline_checkpoint.json
```

### RQ₂ — Hierarchical clustering (autotune study)

Offline evaluation of hierarchical clustering and threshold auto-tuning on the 100-app MobileRec dataset (see [Dataset](#dataset)). Clustering only — no Neo4j, Ollama, or LLM tagging.

**Step-by-step replication**

1. **Environment.** Python 3.10, plus the plotting libraries from `requirements-dev.txt`:
   ```bash
   python3.10 -m venv .venv
   .venv/bin/pip install -r requirements.txt -r requirements-dev.txt
   ```
   A CUDA GPU is used automatically for T-FREX and the embeddings if available (CPU works, slower).
2. **TransFeatEx service** (needed for `transfeatex` and `hybrid`). It is a separate repository, [NLP_pipeline](https://github.com/nlp4se/NLP_pipeline) (Python 3.9, spaCy 3.2):
   ```bash
   git clone https://github.com/nlp4se/NLP_pipeline.git && cd NLP_pipeline
   python3.9 -m venv .venv && .venv/bin/pip install -r requirements.txt
   .venv/bin/python -m spacy download en_core_web_trf
   nohup .venv/bin/python wsgi.py > transfeatex.log 2>&1 &   # serves on port 3004 (CPU)
   ```
   To run it on GPU, start it single-threaded (concurrent GPU requests crash spaCy/thinc):
   ```bash
   .venv/bin/pip install "cupy-cuda12x<13"
   nohup .venv/bin/python -c "import spacy, runpy; spacy.require_gpu(); \
     runpy.run_path('wsgi.py', run_name='tfx')['create_app']().run(host='0.0.0.0', port=3004, threaded=False)" \
     > transfeatex.log 2>&1 &
   ```
3. **Configure** `.env` (copy from `.env.example`): `TRANSFEATEX_URL=http://localhost:3004`. Check it is reachable:
   ```bash
   .venv/bin/python -c "from dotenv import load_dotenv; load_dotenv(); from utils.health_checks import check_transfeatex; print(check_transfeatex())"
   ```
   ⚠️ If the endpoint is unreachable, the study **does not stop**: unavailable extractors are dropped with only an error log, and failed TransFeatEx batches (HTTP 500 / timeout) silently yield no features. Check the health check above and grep the run log for `ERROR` before trusting results.
4. **Smoke test** (a few minutes): `.venv/bin/python scripts/run_autotune_study.py --apps 5`. `selection_summary.csv` must contain rows for all 3 extractors and all 7 sample sizes. Then reset (see **Reset**) before the full run.
5. **Full run** (100 apps × 7 $N$ × 3 extractors = 2,100 runs; several hours, CPU-bound clustering). Run it detached and resume if interrupted:
   ```bash
   nohup .venv/bin/python scripts/run_autotune_study.py > autotune_run.log 2>&1 &
   .venv/bin/python scripts/run_autotune_study.py --resume   # after an interruption
   ```
6. **Figures**: `.venv/bin/python scripts/visualize_autotune_study.py` (reuses the CSVs; no recomputation). The paper figure is also written to `figures/4_framework/6_sample-size-stability.{png,pdf}`.
7. **Paper table** (hybrid, mean over apps per $N$; `0` = All):
   ```bash
   .venv/bin/python -c "import pandas as pd; s=pd.read_csv('evaluation_results/autotune_study/selection_summary.csv'); \
     print(s[s.model_type=='hybrid'].groupby('sample_size')[['selected_silhouette','selected_davies_bouldin','selected_balanced_score','selected_n_clusters','selected_threshold']].agg(['mean','std']).round(3))"
   ```
   Small differences in the 3rd decimal between runs are expected (TransFeatEx/T-FREX numerical nondeterminism, e.g. CPU vs GPU).

The results of the reference run (`selection_summary.csv`, `sweep_records.csv`, `config.json`) are versioned in `evaluation_results/autotune_study/`, so steps 6–7 can be reproduced without rerunning the study.

**Design** (see paper §Hierarchical Clustering):

| Factor | Setting |
|--------|---------|
| Apps | 100 (all apps in dataset) |
| Review subsample per app | $N \in \{50, 100, 200, 300, 500, 1000, \text{All}\}$ (`0` = all reviews in code) |
| Feature extractors | `hybrid`, `transfeatex` (paper: syntactic-only), `t-frex` (paper: LLM-only) |
| Embedding | `all-MiniLM-L6-v2` (`EMBEDDING_TYPE=allmini`) |
| Linkage thresholds $\tau$ | 12 values, uniform on $[0.1, 0.9]$; partitions with $<2$ clusters discarded |
| Auto-tuner | Rank by $\hat{s}=s\times(1-\rho)$; keep top 3; select $\tau^*$ by balanced score (silhouette + Davies–Bouldin + structural penalties) |
| Baselines | Random $\tau$, median $\tau$, max-silhouette $\tau$, fixed $\tau{=}0.5$ |

Constants and scoring live in `scripts/autotune_study_common.py`. Raw features are cached per review `uid` so larger $N$ reuses extraction from smaller subsamples; hybrid unions cached `t-frex` + `transfeatex` features.

**Run**

```bash
# Requires TRANSFEATEX_URL for transfeatex / hybrid (see .env)
.venv/bin/python scripts/run_autotune_study.py
.venv/bin/python scripts/run_autotune_study.py --resume          # after interruption
.venv/bin/python scripts/run_autotune_study.py --apps 5            # smoke test
.venv/bin/python scripts/run_autotune_study.py --models hybrid t-frex
.venv/bin/python scripts/run_autotune_study.py --csv data/input/endpoint_1_process_reviews/mobile_apps/mobilerec_reviews_pipeline.csv   # other reviews file

.venv/bin/python scripts/visualize_autotune_study.py               # main figures
.venv/bin/python scripts/visualize_autotune_study.py --exemplar   # optional per-app diagnostics
```

**Outputs** (`evaluation_results/autotune_study/`):

| File | Role |
|------|------|
| `sweep_records.csv` | One row per (extractor, app, $N$, $\tau$) |
| `selection_summary.csv` | One row per (extractor, app, $N$): selected $\tau^*$, metrics, baselines |
| `config.json` | Recorded study parameters |
| `figures/sample_size_stability.*` | Sample-size stability (all extractors); copied to `figures/4_framework/6_sample-size-stability.*` (paper figure) |
| `figures/baseline_comparison.*` | Auto-tuner vs fixed/random $\tau$ baselines |

**Reset**

```bash
rm -f evaluation_results/autotune_study/checkpoint.json
rm -f evaluation_results/autotune_study/sweep_records.csv evaluation_results/autotune_study/selection_summary.csv
# Optional: rm -rf evaluation_results/autotune_study/feature_cache
```

---

## Dataset

### Input format

| Column | Required | Description |
|--------|----------|-------------|
| `app_name` | yes | App display name |
| `review` | yes | Raw review text |
| `score` | no | Star rating (1-5) |

### Provided datasets

- **Mobile Apps** — 100 apps from MobileRec (10 most popular categories × 10 most reviewed apps), in `data/input/endpoint_1_process_reviews/mobile_apps/`:
  - `mobilerec_reviews_pipeline_large.csv` — default input of the pipeline and the RQ₂ study.
  - `mobilerec_reviews_pipeline.csv` — 30-day window (2022-03-11 to 2022-04-10); select it in the RQ₂ study with `--csv`.

---

## Troubleshooting

| Problem | Fix |
|---------|-----|
| Ollama model not found | `ollama pull llama3.2:3b` |
| Neo4j auth failure | Check `NEO4J_PASSWORD` matches (default: `12345678`) |
| Slow first run | T-FREX downloads ~1-2 GB from HuggingFace on first use |
| TransfeatEx not configured | Expected when `TRANSFEATEX_URL` is unset; T-FREX works without it |
