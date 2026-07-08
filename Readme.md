# FeClustRE

[![arXiv](https://img.shields.io/badge/arXiv-2510.18799-b31b1b.svg)](https://doi.org/10.48550/arXiv.2510.18799)
[![Python 3.10](https://img.shields.io/badge/python-3.10-blue.svg)](https://www.python.org/downloads/release/python-3100/)

Hierarchical clustering and semantic tagging of app features from user reviews.

> Tiessler & Motger. **FeClustRE: Hierarchical Clustering and Semantic Tagging of App Features from User Reviews.** arXiv:2510.18799, 2025.

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
CSV reviews → T-FREX NER → post-process → embed (all-MiniLM-L6-v2)
  → agglomerative clustering → LLM tagging (Ollama) → Neo4j taxonomy
```

| Stage | Component |
|-------|-----------|
| Feature extraction | T-FREX / TransfeatEx NER |
| Embedding | all-MiniLM-L6-v2 |
| Clustering | Agglomerative (balanced / silhouette / conservative) |
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

# Experiment 2 — tree vs flat list (n=60, stratified by Q1-Q3 tree size)
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

Offline evaluation of hierarchical clustering and threshold auto-tuning on the 100-app MobileRec dataset (`data/input/endpoint_1_process_reviews/mobile_apps/mobilerec_reviews_pipeline_large.csv`). Clustering only — no Neo4j, Ollama, or LLM tagging.

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

.venv/bin/python scripts/visualize_autotune_study.py               # main figures
.venv/bin/python scripts/visualize_autotune_study.py --exemplar   # optional per-app diagnostics
```

**Outputs** (`evaluation_results/autotune_study/`):

| File | Role |
|------|------|
| `sweep_records.csv` | One row per (extractor, app, $N$, $\tau$) |
| `selection_summary.csv` | One row per (extractor, app, $N$): selected $\tau^*$, metrics, baselines |
| `config.json` | Recorded study parameters |
| `figures/sample_size_stability.*` | Sample-size stability (all extractors) |
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

- **AI Assistants** — 6 apps (Claude, DeepSeek, Gemini, Le Chat, Copilot, Perplexity)
- **Mobile Apps** — 100 apps, 117K reviews (MobileRec 2022)

---

## Troubleshooting

| Problem | Fix |
|---------|-----|
| Ollama model not found | `ollama pull llama3.2:3b` |
| Neo4j auth failure | Check `NEO4J_PASSWORD` matches (default: `12345678`) |
| Slow first run | T-FREX downloads ~1-2 GB from HuggingFace on first use |
| TransfeatEx not configured | Expected when `TRANSFEATEX_URL` is unset; T-FREX works without it |
