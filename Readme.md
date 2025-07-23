# FeClustRE: Feature Clustering and Semantic Tagging of Mobile App Review Feature Taxonomies

## Overview

**FeClustRE** is a system for extracting **app features** from reviews, grouping them into **hierarchical taxonomies**, and assigning **semantic labels** to each cluster using **LLMs**. 
The pipeline stores taxonomies in **Neo4j** and provides utilities to assess the quality and coherence of these structures.

This repository serves as the full **replication package** for our paper, including code, metrics, and queries.

---

## System Architecture

1. **Feature Extraction:** Extracts candidate feature phrases from app reviews using pattern matching or pretrained extractors.
2. **Clustering:** Groups features into clusters via hierarchical clustering and cuts the dendrogram at a chosen threshold.
3. **Taxonomy Construction:** Each cluster is transformed into a tree structure and stored in Neo4j. No LLM is involved in this step.
4. **LLM Tagging:** The root of each taxonomy is assigned a human-readable label by an LLM based on its child features.
5. **Analysis Tools:** Provides similarity analysis, singleton detection, and low-quality label diagnostics.

---

## Setup Instructions

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

### 4. Run the API Server

```bash
python app.py
```

---

## Key API Endpoints

| Endpoint                                    | Description                                          |
| ------------------------------------------- | ---------------------------------------------------- |
| `POST /process_reviews/upload`              | Upload and parse review CSV                          |
| `GET /cluster_features/<app_name>`          | Generate clusters of features                        |
| `POST /save_selected_clustering/<app_name>` | Save clusters, build taxonomies, tag roots using LLM |
| `GET /llm_taxonomy_metrics`                 | Analyze tag similarity and quality                   |
| `GET /get_app_clustering/<app_name>`        | Retrieve clusters and labels                         |

---

## Neo4j Taxonomy Schema

Each taxonomy is structured as a tree:

* `MiniTaxonomyNode` (root): `llm_tag`, `feature`, `is_leaf=False`
* `MiniTaxonomyNode` (leaf): `feature`, `is_leaf=True`
* `HAS_CHILD`: hierarchical relation

### Query to View All Mini Taxonomies

```cypher
MATCH (root:MiniTaxonomyNode)
WHERE NOT ()-[:HAS_CHILD]->(root)
OPTIONAL MATCH (root)-[:HAS_CHILD*]->(descendant)
RETURN root, descendant
```

Use Neo4j Browser to visualize.

---

## LLM Prompt for Tagging

Used for tagging (not constructing) cluster roots:

```text
You are an assistant that assigns a single, concise category name to a list of app feature keywords.
Respond with only the category name. Do not explain or include extra text.

Examples:
Features: send message, chat group, reply dm
Category: Messaging

Features: freeze screen, crash often, app bug
Category: App Stability

Features: [cluster features go here]
Category:
```

---

## Taxonomy Metrics

Available at `/llm_taxonomy_metrics`:

* **Singleton Clusters:** Trees with a root and exactly one leaf
* **Tag Similarity Clusters:** Groups of semantically similar LLM tags
* **High-Similarity Pairs:** Pairs of distinct labels with cosine similarity > 0.9
* **Low-Quality Tags:** Detected via regex (e.g. "Unknown", "Cluster 12")
* **Tag Frequency Stats**

Sample output:

```json
{
  "singleton_clusters": {
    "count": 70,
    "examples": ["mini_taxonomy_root_5611...", "..."]
  },
  "high_similarity_pairs": [
    {
      "tag_a": "Chatting",
      "tag_b": "Messaging",
      "similarity": 0.93
    }
  ],
  "tag_similarity_clusters": [...],
  "low_quality_tags": ["Unknown"],
  "tag_statistics": {
    "total_tags": 70,
    "distinct_tags": 69,
    "most_common_tags": [["Messaging", 4], ...]
  }
}
```


