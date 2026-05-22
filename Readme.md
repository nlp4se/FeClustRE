# FeClustRE: Feature Clustering and Semantic Tagging of Mobile App Review Feature Taxonomies

[![CI](https://github.com/nlp4se/FeClustRE/actions/workflows/ci.yml/badge.svg)](https://github.com/nlp4se/FeClustRE/actions/workflows/ci.yml)
[![Coverage](https://codecov.io/gh/nlp4se/FeClustRE/branch/master/graph/badge.svg)](https://codecov.io/gh/nlp4se/FeClustRE)
[![arXiv](https://img.shields.io/badge/arXiv-2510.18799-b31b1b.svg)](https://doi.org/10.48550/arXiv.2510.18799)
[![Python 3.10](https://img.shields.io/badge/python-3.10-blue.svg)](https://www.python.org/downloads/release/python-3100/)

## Overview

**FeClustRE** is a system for extracting **app features** from reviews, grouping them into **hierarchical taxonomies**, and assigning **semantic labels** to each cluster using **LLMs**. 
The pipeline stores taxonomies in **Neo4j** and provides utilities to assess the quality and coherence of these structures.

This repository serves as the full **replication package** for our paper, including code, metrics, and automated evaluation framework.

## Paper

> Max Tiessler, Quim Motger. **FeClustRE: Hierarchical Clustering and Semantic Tagging of App Features from User Reviews.** arXiv:2510.18799 [cs.SE], October 2025.
> https://doi.org/10.48550/arXiv.2510.18799

---

## Quick Start - Replication Package

### 1. Clone

```bash
git clone https://github.com/nlp4se/FeClustRE.git
cd feclustre
```

### 2. Start the Reproduction Stack

```bash
docker compose up --build
```

This starts:

- FeClustRE API on `http://localhost:3000`
- Neo4j Browser on `http://localhost:7474`
- Neo4j Bolt on `bolt://localhost:7687`
- Ollama on `http://localhost:11434`
- A one-shot Ollama model pull for `qwen:1.8b`

To use another Ollama model:

```bash
OLLAMA_MODEL=your-model docker compose up --build
```

### 3. Local Python Environment

Requires **Python 3.10**. ML packages (torch, transformers, sentence-transformers) are pinned to
versions that are tested on Python 3.10. Python 3.11 may work but is not validated.

```bash
python3.10 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -r requirements.txt
```

For notebook and visualisation work install the dev dependencies instead:

```bash
.venv/bin/python -m pip install -r requirements-dev.txt
```

### 4. Start Services and API (non-Docker)

If you are running without Docker Compose, start services in this order:

**1. Neo4j**

```bash
# Option A: Docker only Neo4j
docker run -d --name neo4j \
  -e NEO4J_AUTH=neo4j/12345678 \
  -p 7474:7474 -p 7687:7687 \
  neo4j:5.15

# Option B: existing Neo4j installation — just start it
```

**2. Ollama**

```bash
ollama serve &
ollama pull qwen:1.8b
```

**3. Environment**

Copy `.env.example` to `.env` and adjust values if needed:

```bash
cp .env.example .env
```

**4. Flask API**

```bash
.venv/bin/python app.py
```

The API starts on `http://localhost:3000`.

Verify the process is alive:

```bash
curl http://localhost:3000/ping
```

Check dependency status:

```bash
curl http://localhost:3000/health
```

### 5. Run Systematic Experiments

All experiments from the paper can be replicated using the systematic testing framework. The script uploads the CSV files to the running API, iterates over every parameter combination, evaluates clustering quality, and writes results to `evaluation_results/`.

The API must be running (step 2 or step 4) before executing any experiment.

#### Dataset selection

Use `--dataset` to choose which dataset the experiment runs on:

```bash
# AI assistants dataset (default — used in the paper)
python3 test/test_config.py <mode> --dataset ai_assistants

# MobileRec dataset (100 apps, 10 categories)
python3 test/test_config.py <mode> --dataset mobile_apps
```

The `mobile_apps` option requires `mobilerec_reviews_pipeline.csv` to be present. If it is missing, generate it by running the full `MobileRec.ipynb` notebook first.

You can also point directly at any pipeline-compatible CSV file:

```bash
python3 test/test_config.py <mode> --csv-files path/to/your.csv
```

#### Running experiments

```bash
# Mock test (quick validation, first 2 apps/files, small samples)
python3 test/test_config.py mock

# Full comprehensive test (all apps, multiple configurations)
python3 test/test_config.py full

# Semantic experiment (hybrid model, maximum data, balanced strategy)
python3 test/test_config.py semantic

# Examples with dataset selection
python3 test/test_config.py mock --dataset mobile_apps
python3 test/test_config.py full --dataset mobile_apps
python3 test/test_config.py semantic --dataset mobile_apps
```

Results, visualisations, and the summary report are written to `evaluation_results/` in the project root. The session ID printed at the end identifies the output subfolder.

#### Experiment configurations

**Mock Test:**
- Apps: first 2 files of the selected dataset
- Models: transfeatex, t-frex, hybrid
- Embeddings: allmini, sentence-t5
- Sample sizes: 10, 20, 50
- Strategies: balanced, silhouette, conservative

**Full Test:**
- Apps: all files of the selected dataset (6 AI assistants or 100 mobile apps)
- Models: t-frex, transfeatex, hybrid
- Embeddings: allmini, sentence-t5
- Sample sizes: 1000, 2000, 5000, 50000
- Strategies: balanced, silhouette, conservative

**Semantic Experiment:**
- Apps: all files of the selected dataset
- Models: hybrid
- Embeddings: allmini
- Sample sizes: no limit (all available reviews)
- Strategies: balanced


---

## Troubleshooting

### Ollama model not found

`/health` reports `"error": "Model 'qwen:1.8b' not found in Ollama"`.

```bash
ollama pull qwen:1.8b
```

The first pull downloads ~1 GB and takes a few minutes. The app will start fine
while the pull is in progress — only label generation will fail until the model
is available.

### Neo4j authentication failure

`/health` reports `"error": "Cannot connect to Neo4j"`.

Check that the values in `.env` match your Neo4j instance:

```
NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=12345678
```

If you changed the password when starting Neo4j, update `NEO4J_PASSWORD` to match.

### TransfeatEx unavailable

`/health` reports `"status": "not_configured"` for `transfeatex` when
`TRANSFEATEX_URL` is not set. This is expected and does not affect T-FREX mode
— the overall health status remains `healthy`. Set `TRANSFEATEX_URL` in `.env`
to the service address to enable it.
Experiments that specify `model: transfeatex` or `model: hybrid` will fail fast
with a clear error if the URL is not configured.

### First model download is slow

T-FREX (`quim-motger/t-frex-bert-base-uncased`) and the sentence-transformer
embedding model are downloaded from Hugging Face on first use. This can take
several minutes. The download is cached in `~/.cache/huggingface/` and is only
needed once.

---

## Dataset

The dataset is available in `/data/input/endpoint_1_process_reviews/` directory.

### Input CSV format

All CSV files uploaded to `/process_reviews/upload` must follow this schema:

| Column | Required | Type | Description |
|---|---|---|---|
| `app_name` | **yes** | string | Display name of the app. Used to group reviews. Rows without this are silently dropped. |
| `review` | **yes** | string | Raw review text. Empty or missing rows are silently dropped. |
| `app_package` | no | string | Package/bundle identifier (e.g. `com.discord`). Stored in Neo4j. Defaults to `unknown`. |
| `app_categoryId` | no | string | App category label. Stored in Neo4j. Defaults to `unknown`. |
| `score` | no | int 1–5 | Star rating. Defaults to `0` if missing or unparseable. |
| `reviewId` | no | string | Optional review identifier. Defaults to empty string. |

Any extra columns in the CSV are ignored.

### Provided datasets

#### AI Assistants (`/data/input/endpoint_1_process_reviews/ai_assistants/`)

Six Google Play review exports, one file per app, already in the required format:

- `Claude_by_Anthropic.csv`
- `DeepSeek_-_AI_Assistant.csv`
- `Google_Gemini.csv`
- `Le_Chat_by_Mistral_AI.csv`
- `Microsoft_Copilot.csv`
- `Perplexity_-_Ask_Anything.csv`

#### Mobile Apps (`/data/input/endpoint_1_process_reviews/mobile_apps/`)

A curated subset of the [MobileRec](https://huggingface.co/datasets/recmeapp/mobilerec) dataset: 100 apps across 10 Google Play categories, 117,820 reviews in a 30-day window (2022-03-11 → 2022-04-10).

Run `MobileRec.ipynb` to generate `mobilerec_reviews_pipeline.csv` before running experiments with this dataset. The notebook downloads the source data from Hugging Face (~4 GB) and produces a pipeline-compatible CSV.

---

## Querying Results

After running experiments, taxonomies are stored in Neo4j and can be queried directly.

### Query All Taxonomies

```cypher
MATCH (app:App)-[:HAS_MINI_TAXONOMY]->(root:MiniTaxonomyNode)
WHERE NOT ()-[:HAS_CHILD]->(root)
OPTIONAL MATCH (root)-[:HAS_CHILD*]->(leaf)
WHERE NOT (leaf)-[:HAS_CHILD]->()
RETURN app.name as app_name, 
       root.llm_tag as taxonomy_label,
       count(DISTINCT leaf) as leaf_count,
       collect(DISTINCT leaf.feature) as features
```
