# FeClustRE: Feature Clustering and Semantic Tagging of Mobile App Review Feature Taxonomies

[![CI](https://github.com/nlp4se/FeClustRE/actions/workflows/ci.yml/badge.svg)](https://github.com/nlp4se/FeClustRE/actions/workflows/ci.yml)
[![Coverage](https://raw.githubusercontent.com/nlp4se/FeClustRE/extension-magazine/badges/coverage.svg)](https://github.com/nlp4se/FeClustRE/actions/workflows/ci.yml)
[![arXiv](https://img.shields.io/badge/arXiv-2510.18799-b31b1b.svg)](https://doi.org/10.48550/arXiv.2510.18799)
[![Python 3.10](https://img.shields.io/badge/python-3.10-blue.svg)](https://www.python.org/downloads/release/python-3100/)

> Max Tiessler, Quim Motger. **FeClustRE: Hierarchical Clustering and Semantic Tagging of App Features from User Reviews.** arXiv:2510.18799 [cs.SE], October 2025.
> https://doi.org/10.48550/arXiv.2510.18799

FeClustRE extracts app features from reviews, groups them into hierarchical taxonomies, and assigns semantic labels using an LLM. Taxonomies are stored in Neo4j.

---

## Quickstart (Docker)

```bash
git clone https://github.com/nlp4se/FeClustRE.git && cd FeClustRE
docker compose up --build
```

| Service | URL |
|---------|-----|
| FeClustRE API | http://localhost:3000 |
| Neo4j Browser | http://localhost:7474 |
| Ollama | http://localhost:11434 |

The stack pulls `llama3.2:3b` automatically on first start. To use a different model:

```bash
OLLAMA_MODEL=your-model docker compose up --build
```

Verify the API is up:

```bash
curl http://localhost:3000/ping
curl http://localhost:3000/health
```

---

## Local Setup (no Docker)

Requires **Python 3.10**.

```bash
python3.10 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -r requirements.txt
cp .env.example .env   # edit NEO4J_* and OLLAMA_* if needed
```

Start dependencies:

```bash
# Neo4j
docker run -d --name neo4j -e NEO4J_AUTH=neo4j/12345678 \
  -p 7474:7474 -p 7687:7687 neo4j:5.15

# Ollama
ollama serve &
ollama pull llama3.2:3b
```

Start the API:

```bash
.venv/bin/python app.py   # http://localhost:3000
```

---

## Run Experiments

The API must be running before executing any experiment.

```bash
# Quick validation (2 apps, small samples)
python3 test/test_config.py mock

# Full paper replication (all apps, all configs)
python3 test/test_config.py full

# Semantic experiment (hybrid model, all reviews, balanced strategy)
python3 test/test_config.py semantic
```

Use `--dataset` to switch between datasets:

```bash
python3 test/test_config.py full --dataset ai_assistants   # default
python3 test/test_config.py full --dataset mobile_apps     # requires mobilerec_reviews_pipeline.csv
```

To generate `mobilerec_reviews_pipeline.csv`, run `MobileRec.ipynb` first (downloads ~4 GB from Hugging Face).

Results are written to `evaluation_results/`.

---

## Human Study — Experiment Data

These steps produce the datasets for the magazine paper's user study.

### 1. Start infrastructure

```bash
docker compose up -d neo4j ollama
docker exec $(docker ps -q --filter name=ollama) ollama pull llama3.2:3b
```

### 2. Start the API

```bash
.venv/bin/python app.py
```

### 3. Run the mobile-apps pipeline

Safe to interrupt and resume.

```bash
.venv/bin/python scripts/run_mobile_pipeline.py          # fresh run (~30–60 min)
.venv/bin/python scripts/run_mobile_pipeline.py --resume # resume after interruption
```

Check progress:

```bash
python3 -c "
import json; cp = json.load(open('evaluation_results/mobile_pipeline_checkpoint.json'))
done = sum(1 for v in cp['completed_apps'].values() if v.get('status') == 'success')
print(f'{done}/100 apps done')
"
```

### 4. Generate experiment datasets

```bash
# Experiment 1 — parent/child feature validation (n=300)
.venv/bin/python scripts/generate_experiment1.py --n 300 --out data/experiment1.csv

# Experiment 2 — tree vs flat list (n=60 apps)
.venv/bin/python scripts/generate_experiment2.py --n 60 --out data/experiment2.json
```

### 5. Visualise

```bash
.venv/bin/streamlit run scripts/visualize_experiments.py  # http://localhost:8501
```

### Reset

```bash
# Regenerate experiment files only (keep pipeline clusters)
rm -f data/experiment1.csv data/experiment2.json data/experiment2_flat.csv
.venv/bin/python scripts/generate_experiment1.py --n 300 --out data/experiment1.csv
.venv/bin/python scripts/generate_experiment2.py --n 60  --out data/experiment2.json

# Full reset (re-run pipeline + experiments)
rm -f evaluation_results/mobile_pipeline_checkpoint.json \
      evaluation_results/test_session_mobile_*.json \
      data/experiment1.csv data/experiment2.json data/experiment2_flat.csv
.venv/bin/python scripts/run_mobile_pipeline.py
.venv/bin/python scripts/generate_experiment1.py --n 300 --out data/experiment1.csv
.venv/bin/python scripts/generate_experiment2.py --n 60  --out data/experiment2.json

# Full reset including Neo4j graph
docker compose stop neo4j && docker volume rm feclustre_neo4j_data
docker compose up -d neo4j
# then re-run pipeline + generate (commands above)
```

---

## Dataset

### Input CSV format

| Column | Required | Type | Description |
|--------|----------|------|-------------|
| `app_name` | yes | string | App display name — used to group reviews |
| `review` | yes | string | Raw review text |
| `app_package` | no | string | Package identifier (e.g. `com.discord`) |
| `app_categoryId` | no | string | App category label |
| `score` | no | int 1–5 | Star rating |
| `reviewId` | no | string | Review identifier |

Extra columns are ignored.

### Provided datasets

**AI Assistants** (`data/input/endpoint_1_process_reviews/ai_assistants/`)
Six Google Play exports: Claude, DeepSeek, Gemini, Le Chat, Copilot, Perplexity.

**Mobile Apps** (`data/input/endpoint_1_process_reviews/mobile_apps/`)
100 apps, 10 categories, 117 820 reviews (MobileRec, 2022-03-11 → 2022-04-10).
Generate `mobilerec_reviews_pipeline.csv` by running `MobileRec.ipynb` first.

---

## Querying Results

```cypher
MATCH (app:App)-[:HAS_MINI_TAXONOMY]->(root:MiniTaxonomyNode)
WHERE NOT ()-[:HAS_CHILD]->(root)
OPTIONAL MATCH (root)-[:HAS_CHILD*]->(leaf)
WHERE NOT (leaf)-[:HAS_CHILD]->()
RETURN app.name AS app_name,
       root.llm_tag AS taxonomy_label,
       count(DISTINCT leaf) AS leaf_count,
       collect(DISTINCT leaf.feature) AS features
```

---

## Troubleshooting

**Ollama model not found** — `/health` reports `"Model 'llama3.2:3b' not found"`:
```bash
ollama pull llama3.2:3b
```

**Neo4j auth failure** — check `.env` values match your instance:
```
NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=12345678
```

**TransfeatEx not configured** — `/health` shows `"status": "not_configured"` for `transfeatex`. This is expected when `TRANSFEATEX_URL` is unset; T-FREX mode works without it. Set `TRANSFEATEX_URL` in `.env` to enable hybrid/transfeatex experiments.

**Slow first run** — T-FREX and the sentence-transformer model download from Hugging Face on first use (~1–2 GB, cached in `~/.cache/huggingface/`).
