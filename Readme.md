# FeClustRE: Feature Clustering and Semantic Tagging of Mobile App Review Feature Taxonomies

## Overview

**FeClustRE** is a system for extracting **app features** from reviews, grouping them into **hierarchical taxonomies**, and assigning **semantic labels** to each cluster using **LLMs**. 
The pipeline stores taxonomies in **Neo4j** and provides utilities to assess the quality and coherence of these structures.

This repository serves as the full **replication package** for our paper, including code, metrics, and automated evaluation framework.

---

## Quick Start - Replication Package

### 1. Install Dependencies

```bash
git clone https://github.com/your-org/feclustre.git
cd feclustre
pip install -r requirements.txt
```

### 2. Start Neo4j

```bash
docker run -d --name neo4j \
  -p7474:7474 -p7687:7687 \
  -e NEO4J_AUTH=neo4j/12345678 \
  neo4j:5.15
```

### 3. Start Ollama with Qwen

```bash
ollama run qwen:1.8b
```

Ensure your `config.py` includes:

```python
OLLAMA_BASE_URL = "http://localhost:11434"
OLLAMA_MODEL = "qwen:1.8b"
```

### 4. Start the Backend API Server

```bash
python app.py
```

The API server will be available at `http://localhost:3000`

### 5. Run Systematic Experiments

All experiments from the paper can be replicated using the systematic testing framework:

```bash
# Mock test (quick validation with 2 apps, small samples)
python test_config.py mock

# Full comprehensive test (all apps, multiple configurations)
python test_config.py full

# Semantic experiment (hybrid model, maximum data, balanced strategy)
python test_config.py semantic
```

#### Experiment Configurations

**Mock Test:**
- Apps: ChatGPT, Claude by Anthropic
- Models: transfeatex, t-frex, hybrid
- Embeddings: allmini, sentence-t5
- Sample sizes: 10, 20, 50
- Strategies: balanced, silhouette, conservative

**Full Test:**
- Apps: All 7 AI assistants (ChatGPT, Claude, DeepSeek, Gemini, Le Chat, Copilot, Perplexity)
- Models: t-frex, transfeatex, hybrid
- Embeddings: allmini, sentence-t5
- Sample sizes: 1000, 2000, 5000, 50000
- Strategies: balanced, silhouette, conservative

**Semantic Experiment:**
- Apps: All 7 AI assistants
- Models: hybrid
- Embeddings: allmini
- Sample sizes: No limit (all available reviews)
- Strategies: balanced

---

## Dataset

The dataset is available in `/data/input/endpoint_1_process_reviews/` directory with two categories:

### AI Assistants (`/data/input/endpoint_1_process_reviews/ai_assistants/`)
- ChatGPT.csv
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
