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

```bash
python3 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -r requirements.txt
```

### 4. Run Systematic Experiments

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
