# FeClustRE Infrastructure

This document explains how the project is intended to run and which external
systems it depends on. It reflects the current codebase state, including known
gaps found during the audit.

## Runtime Components

FeClustRE is a Flask-based Python API that processes mobile app reviews into
feature taxonomies.

The runtime has four main parts:

1. Flask API server
   - Entry point: `app.py`
   - Default port: `3000`
   - Main endpoints:
     - `GET /ping`
     - `GET /health`
     - `POST /process_reviews/upload`
     - `POST /save_selected_clustering/<app_name>`
     - `GET /mini_taxonomies/<app_name>`
     - `GET /llm_taxonomy_metrics`

2. Neo4j database
   - Stores apps, reviews, extracted features, clusters, and taxonomy nodes.
   - Docker Compose service: `neo4j` in `docker-compose.yml`
   - Exposed ports:
     - `7474` for Neo4j Browser
     - `7687` for Bolt

3. Feature extraction and embeddings
   - T-FREX model: `quim-motger/t-frex-bert-base-uncased`
   - Embedding models:
     - `all-MiniLM-L6-v2`
     - `sentence-transformers/sentence-t5-base`
   - TransfeatEx is also referenced, but the current implementation points to a
     private network address and is not portable as-is.

4. Ollama
   - Used for cluster label generation.
   - Current default model: `qwen:1.8b`
   - Current default base URL: `http://localhost:11434`

## Data Flow

The intended pipeline is:

1. A CSV is uploaded to `POST /process_reviews/upload`.
2. The CSV is parsed into apps and reviews.
3. Reviews are cleaned and tokenized.
4. Features are extracted from review text.
5. Reviews and features are stored in Neo4j.
6. Unique features are embedded.
7. Hierarchical clustering generates candidate clusters.
8. The API returns candidate clustering options.
9. A selected clustering is posted to `POST /save_selected_clustering/<app_name>`.
10. Ollama generates semantic labels for each selected cluster.
11. Mini taxonomies are stored in Neo4j.
12. Taxonomy metrics can be queried from Neo4j.

## Configuration

The current configuration lives in `config.py`.

Important environment variables:

```bash
FLASK_DEBUG=false
NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=12345678
NEO4J_DATABASE=neo4j
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=qwen:1.8b
TRANSFEATEX_URL=http://example:3004
LOG_LEVEL=INFO
```

Important issue: the current `config.py` default for `NEO4J_URI` is
`bolt://10.4.63.10:7687`, while the README and Docker Compose flow imply local
Neo4j at `bolt://localhost:7687`. The local value should become the default for
replication.

## Local Infrastructure

The repository includes `docker-compose.yml` for Neo4j only.

Current Compose behavior:

```bash
docker compose up -d
```

This starts Neo4j with:

```text
username: neo4j
password: 12345678
database: neo4j
```

Ollama is not managed by Docker Compose in the current repo. It must be started
separately:

```bash
ollama run qwen:1.8b
```

## Repository Data

The repository tracks replication data under:

```text
data/input/
data/results/
```

These are treated as part of the replication package, not local generated
output. Do not broadly ignore or delete these directories without first deciding
whether the project is still meant to be a full replication package.

Generated local output should go to ignored directories such as:

```text
test_cache/
evaluation_results/
logs/
```

## Known Infrastructure Problems

These are current blockers or portability issues:

1. `requirements.txt` is UTF-16 encoded. Standard Python tooling usually expects
   UTF-8 text.
2. The README uses `python`, but the local environment may only provide
   `python3`.
3. Flask is required by `app.py`, but it was not installed in the audited local
   environment.
4. Neo4j defaults point to a private IP instead of the local Docker service.
5. TransfeatEx is hardcoded to `http://10.4.63.10:3004/extract-features`.
6. Heavy services are initialized at module import time in `app.py`.
7. App startup can trigger model loading and database connection setup before a
   simple `/ping` request is possible.
8. The README and test scripts reference `ChatGPT.csv`, which is not present in
   the current dataset.

## Target Runtime Shape

The maintainable target architecture should be:

```text
Client/test runner
      |
      v
Flask app factory
      |
      +--> Lazy FeatureExtractor
      +--> Lazy Neo4jConnection
      +--> Lazy TaxonomyBuilder
      +--> Health checks that report missing services without crashing startup
      |
      v
Neo4j + Ollama + optional TransfeatEx
```

The important design principle is simple: starting the API should prove the web
server can start. It should not require every expensive model and external
service to be healthy before the process can boot.
