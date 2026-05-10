# FeClustRE: Feature Clustering and Semantic Tagging of Mobile App Review Feature Taxonomies

## Overview

**FeClustRE** is a system for extracting **app features** from reviews, grouping them into **hierarchical taxonomies**, and assigning **semantic labels** to each cluster using **LLMs**. 
The pipeline stores taxonomies in **Neo4j** and provides utilities to assess the quality and coherence of these structures.

This repository serves as the full **replication package** for our paper, including code, metrics, and automated evaluation framework.

---

## Quick Start - Replication Package

### 1. Clone

```bash
git clone https://github.com/your-org/feclustre.git
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

All experiments from the paper can be replicated using the systematic testing framework:

```bash
# Mock test (quick validation with 2 apps, small samples)
python3 test/test_config.py mock

# Full comprehensive test (all apps, multiple configurations)
python3 test/test_config.py full

# Semantic experiment (hybrid model, maximum data, balanced strategy)
python3 test/test_config.py semantic
```

#### Experiment Configurations

**Mock Test:**
- Apps: Claude by Anthropic, Perplexity
- Models: transfeatex, t-frex, hybrid
- Embeddings: allmini, sentence-t5
- Sample sizes: 10, 20, 50
- Strategies: balanced, silhouette, conservative

**Full Test:**
- Apps: All 6 included AI assistants (Claude, DeepSeek, Gemini, Le Chat, Copilot, Perplexity)
- Models: t-frex, transfeatex, hybrid
- Embeddings: allmini, sentence-t5
- Sample sizes: 1000, 2000, 5000, 50000
- Strategies: balanced, silhouette, conservative

**Semantic Experiment:**
- Apps: All 6 included AI assistants
- Models: hybrid
- Embeddings: allmini
- Sample sizes: No limit (all available reviews)
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

`/health` reports `"status": "unhealthy"` for `transfeatex`. This is expected
unless you have access to a running TransfeatEx service. Set `TRANSFEATEX_URL`
in `.env` to the service address, or leave it empty to skip TransfeatEx.
Experiments that specify `model: transfeatex` or `model: hybrid` will fail fast
with a clear error if the URL is not configured.

### First model download is slow

T-FREX (`quim-motger/t-frex-bert-base-uncased`) and the sentence-transformer
embedding model are downloaded from Hugging Face on first use. This can take
several minutes. The download is cached in `~/.cache/huggingface/` and is only
needed once.

---

## Dataset

The dataset is available in `/data/input/endpoint_1_process_reviews/` directory with two categories:

### AI Assistants (`/data/input/endpoint_1_process_reviews/ai_assistants/`)
- Claude_by_Anthropic.csv
- DeepSeek_-_AI_Assistant.csv
- Google_Gemini.csv
- Le_Chat_by_Mistral_AI.csv
- Microsoft_Copilot.csv
- Perplexity_-_Ask_Anything.csv


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
