from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import AgglomerativeClustering
from collections import defaultdict
import numpy as np
import re
import psutil
import torch
from flask import Flask, request, jsonify
import csv
import logging
import sys
from datetime import datetime
from collections import Counter

from services.neo4j_service import Neo4jConnection
from services.preprocessing_service import ReviewPreprocessor
from services.feature_extraction_service import FeatureExtractor
from services.clustering_service import HierarchicalClusterer
from services.taxonomy_service import TaxonomyBuilder
from config import config
from utils.health_checks import (
    check_transfeatex, check_tfrex_model, check_embedding_model,
    check_nltk_data, check_ollama, check_neo4j
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.config.from_object(config['default'])

# Initialize services
neo4j_conn = Neo4jConnection()
preprocessor = ReviewPreprocessor()
feature_extractor = FeatureExtractor()
clusterer = HierarchicalClusterer()
taxonomy_builder = TaxonomyBuilder(neo4j_conn, feature_extractor)


@app.route('/ping')
def ping():
    return 'pong'


@app.route('/health')
def health_check():
    health_status = {
        "timestamp": datetime.now().isoformat(),
        "services": {
            "neo4j": check_neo4j(neo4j_conn),
            "nltk": check_nltk_data(),
            "ollama": check_ollama(app.config),
        },
        "models": {
            "tfrex": check_tfrex_model(feature_extractor),
            "embeddings": check_embedding_model(feature_extractor),
            "transfeatex": check_transfeatex()
        },
        "system": {
            "python_version": sys.version,
            "cpu_usage": psutil.cpu_percent(),
            "memory_usage": psutil.virtual_memory().percent,
            "disk_usage": psutil.disk_usage('/').percent,
            "torch_version": torch.__version__
        }
    }

    if torch.cuda.is_available():
        health_status["system"]["cuda_device"] = torch.cuda.get_device_name(0)
        health_status["system"]["cuda_memory"] = {
            "allocated": torch.cuda.memory_allocated(0),
            "reserved": torch.cuda.memory_reserved(0)
        }

    all_statuses = list(health_status["services"].values()) + list(health_status["models"].values())
    health_status["status"] = "healthy" if all(s["status"] == "healthy" for s in all_statuses) else "unhealthy"
    status_code = 200 if health_status["status"] == "healthy" else 503

    return jsonify(health_status), status_code


@app.route('/llm_taxonomy_metrics', methods=['GET'])
def llm_taxonomy_metrics():
    try:
        # Step 1: Fetch comprehensive taxonomy data
        with neo4j_conn.driver.session(database=neo4j_conn.database) as session:
            root_results = session.run("""
                MATCH (root:MiniTaxonomyNode)
                WHERE NOT (()-[:HAS_CHILD]->(root)) AND root.llm_tag IS NOT NULL
                RETURN root.id AS root_id, 
                       root.llm_tag AS tag, 
                       root.session_id AS session_id
            """)
            root_data = [(r["root_id"], r["tag"].strip(), r["session_id"]) for r in root_results if r["tag"]]

            structure_results = session.run("""
                MATCH (root:MiniTaxonomyNode)
                WHERE NOT (()-[:HAS_CHILD]->(root)) AND root.llm_tag IS NOT NULL
                OPTIONAL MATCH path = (root)-[:HAS_CHILD*]->(leaf)
                WHERE NOT (leaf)-[:HAS_CHILD]->()
                WITH root, 
                     max(length(path)) as max_depth,
                     count(DISTINCT leaf) as leaf_count,
                     collect(DISTINCT leaf.feature) as leaf_features
                RETURN root.id AS root_id,
                       root.llm_tag AS tag,
                       COALESCE(max_depth, 0) AS depth,
                       COALESCE(leaf_count, 0) AS leaves,
                       leaf_features
            """)
            structure_data = {
                r["root_id"]: {
                    "tag": r["tag"],
                    "depth": r["depth"],
                    "leaves": r["leaves"],
                    "leaf_features": r["leaf_features"]
                }
                for r in structure_results
            }

        if not root_data or len(root_data) < 2:
            return jsonify({"error": "Not enough labeled mini taxonomies found"}), 400

        root_ids, tags, session_ids = zip(*root_data)

        # Step 2: Basic statistics and quality analysis
        tag_counter = Counter(tags)
        session_counter = Counter(session_ids)

        duplicate_tags = {tag: count for tag, count in tag_counter.items() if count > 1}

        low_quality_patterns = [
            r"unknown",
            r"internal",
            r"cluster \d+",
            r"category \d+",
            r"group \d+"
        ]

        low_quality_tags = []
        for tag in set(tags):
            for pattern in low_quality_patterns:
                if re.fullmatch(pattern, tag.lower()):
                    low_quality_tags.append({
                        "tag": tag,
                        "pattern": pattern,
                        "count": tag_counter[tag]
                    })
                    break

        structure_metrics = {
            "depth_distribution": {},
            "leaf_count_distribution": {},
            "singleton_taxonomies": [],
            "large_taxonomies": [],
            "empty_taxonomies": []
        }

        depths = []
        leaf_counts = []

        for root_id in root_ids:
            if root_id in structure_data:
                depth = structure_data[root_id]["depth"]
                leaves = structure_data[root_id]["leaves"]
                tag = structure_data[root_id]["tag"]

                depths.append(depth)
                leaf_counts.append(leaves)

                if leaves <= 1:
                    structure_metrics["singleton_taxonomies"].append({
                        "root_id": root_id,
                        "tag": tag,
                        "leaves": leaves
                    })
                elif leaves >= 10:
                    structure_metrics["large_taxonomies"].append({
                        "root_id": root_id,
                        "tag": tag,
                        "leaves": leaves,
                        "depth": depth
                    })
                elif leaves == 0:
                    structure_metrics["empty_taxonomies"].append({
                        "root_id": root_id,
                        "tag": tag
                    })

        depth_counter = Counter(depths)
        leaf_counter = Counter(leaf_counts)

        structure_metrics["depth_distribution"] = dict(depth_counter.most_common())
        structure_metrics["leaf_count_distribution"] = dict(leaf_counter.most_common())
        structure_metrics["avg_depth"] = round(np.mean(depths), 2) if depths else 0
        structure_metrics["avg_leaves"] = round(np.mean(leaf_counts), 2) if leaf_counts else 0

        distinct_tags = list(set(tags))
        if len(distinct_tags) >= 2:
            embeddings = feature_extractor.get_embeddings(distinct_tags)
            tag_to_index = {tag: i for i, tag in enumerate(distinct_tags)}
            sim_matrix = cosine_similarity(embeddings)

            # Cluster similar tags with multiple thresholds
            similarity_analysis = {}
            thresholds = [0.7, 0.8, 0.9]

            for threshold in thresholds:
                clustering = AgglomerativeClustering(
                    n_clusters=None,
                    distance_threshold=1 - threshold,
                    affinity='precomputed',
                    linkage='average'
                )
                clusters = clustering.fit(1 - sim_matrix)

                clustered_tags = defaultdict(list)
                for idx, cluster_id in enumerate(clusters.labels_):
                    clustered_tags[cluster_id].append(distinct_tags[idx])

                meaningful_groups = [
                    {
                        "tags": tag_list,
                        "count": len(tag_list),
                        "avg_similarity": round(float(np.mean([
                            sim_matrix[tag_to_index[tag1]][tag_to_index[tag2]]
                            for i, tag1 in enumerate(tag_list)
                            for j, tag2 in enumerate(tag_list)
                            if i < j
                        ])), 4) if len(tag_list) > 1 else 1.0
                    }
                    for tag_list in clustered_tags.values()
                    if len(tag_list) > 1
                ]
                meaningful_groups.sort(key=lambda x: x["count"], reverse=True)

                similarity_analysis[f"threshold_{int(threshold * 100)}"] = {
                    "groups": meaningful_groups[:10],
                    "total_groups": len(meaningful_groups),
                    "tags_in_groups": sum(g["count"] for g in meaningful_groups),
                    "singleton_tags": len(distinct_tags) - sum(g["count"] for g in meaningful_groups)
                }

            similar_pairs = []
            merge_threshold = 0.85
            for i in range(len(distinct_tags)):
                for j in range(i + 1, len(distinct_tags)):
                    sim = float(sim_matrix[i][j])
                    if sim >= merge_threshold:
                        tag_a, tag_b = distinct_tags[i], distinct_tags[j]
                        similar_pairs.append({
                            "tag_a": tag_a,
                            "tag_b": tag_b,
                            "similarity": round(sim, 4),
                            "count_a": tag_counter[tag_a],
                            "count_b": tag_counter[tag_b],
                            "merge_candidate": sim >= 0.9
                        })
            similar_pairs.sort(key=lambda x: x["similarity"], reverse=True)

        else:
            similarity_analysis = {}
            similar_pairs = []

        content_analysis = analyze_taxonomy_content_quality(structure_data)

        session_analysis = {}
        if len(session_counter) > 1:
            for session_id, count in session_counter.most_common():
                session_tags = [tag for _, tag, sid in root_data if sid == session_id]
                session_analysis[session_id] = {
                    "taxonomy_count": count,
                    "unique_tags": len(set(session_tags)),
                    "duplicate_tags": len(session_tags) - len(set(session_tags)),
                    "top_tags": Counter(session_tags).most_common(5)
                }

        return jsonify({
            "overview": {
                "total_taxonomies": len(tags),
                "distinct_tags": len(distinct_tags),
                "duplicate_tags": len(duplicate_tags),
                "low_quality_count": len(low_quality_tags),
                "sessions": len(session_counter)
            },
            "tag_statistics": {
                "most_common_tags": tag_counter.most_common(15),
                "duplicate_tags": duplicate_tags,
                "low_quality_tags": low_quality_tags
            },
            "structure_analysis": structure_metrics,
            "similarity_analysis": similarity_analysis,
            "merge_candidates": similar_pairs[:20],
            "content_quality": content_analysis,
            "session_breakdown": dict(list(session_analysis.items())[:5]),
        })

    except Exception as e:
        logger.error(f"Failed to compute LLM taxonomy metrics: {e}", exc_info=True)
        return jsonify({"error": "Internal server error"}), 500


def analyze_taxonomy_content_quality(structure_data):
    content_metrics = {
        "feature_overlap": [],
        "semantic_coherence": [],
        "naming_patterns": {}
    }

    # Analyze feature overlap between taxonomies
    all_features = []
    taxonomy_features = {}

    for root_id, data in structure_data.items():
        features = data.get("leaf_features", [])
        if features:
            taxonomy_features[data["tag"]] = set(features)
            all_features.extend(features)

    # Find overlapping features
    feature_counter = Counter(all_features)
    overlapping_features = {feat: count for feat, count in feature_counter.items() if count > 1}

    if overlapping_features:
        content_metrics["feature_overlap"] = [
            {
                "feature": feat,
                "appears_in_taxonomies": count,
                "taxonomies": [tag for tag, features in taxonomy_features.items() if feat in features]
            }
            for feat, count in Counter(overlapping_features).most_common(10)
        ]

    # Analyze naming patterns
    tag_words = []
    for data in structure_data.values():
        tag_words.extend(data["tag"].lower().split())

    word_counter = Counter(tag_words)
    content_metrics["naming_patterns"] = {
        "most_common_words": word_counter.most_common(10),
        "unique_words": len(set(tag_words)),
        "total_words": len(tag_words)
    }

    return content_metrics


def generate_taxonomy_recommendations(duplicate_tags, low_quality_tags, similar_pairs,
                                      structure_metrics, content_analysis):
    recommendations = []

    # Duplicate tag recommendations
    if duplicate_tags:
        recommendations.append({
            "type": "duplicates",
            "priority": "high",
            "message": f"Found {len(duplicate_tags)} duplicate tags that should be merged or renamed",
            "action": "Review duplicate tags and either merge taxonomies or improve label generation",
            "examples": list(duplicate_tags.keys())[:5]
        })

    # Low quality tag recommendations
    if low_quality_tags:
        recommendations.append({
            "type": "quality",
            "priority": "medium",
            "message": f"Found {len(low_quality_tags)} low-quality generic tags",
            "action": "Improve label generation to create more specific, distinctive labels",
            "examples": [tag["tag"] for tag in low_quality_tags[:5]]
        })

    # Similarity-based merge recommendations
    high_similarity_pairs = [p for p in similar_pairs if p["similarity"] >= 0.9]
    if high_similarity_pairs:
        recommendations.append({
            "type": "merging",
            "priority": "medium",
            "message": f"Found {len(high_similarity_pairs)} pairs with >90% similarity that could be merged",
            "action": "Consider merging highly similar taxonomies",
            "examples": [f"'{p['tag_a']}' ↔ '{p['tag_b']}' ({p['similarity']})"
                         for p in high_similarity_pairs[:3]]
        })

    # Structure recommendations
    singleton_count = len(structure_metrics.get("singleton_taxonomies", []))
    if singleton_count > len(structure_metrics.get("large_taxonomies", [])) * 3:
        recommendations.append({
            "type": "structure",
            "priority": "low",
            "message": f"High ratio of singleton taxonomies ({singleton_count}) suggests over-segmentation",
            "action": "Consider adjusting clustering parameters to create larger, more meaningful groups"
        })

    # Content overlap recommendations
    overlap_features = content_analysis.get("feature_overlap", [])
    if len(overlap_features) > 10:
        recommendations.append({
            "type": "content",
            "priority": "medium",
            "message": f"High feature overlap ({len(overlap_features)} features) between taxonomies",
            "action": "Review clustering logic - features appearing in multiple taxonomies may indicate poor separation"
        })

    return recommendations


@app.route('/process_reviews/upload', methods=['POST'])
def process_reviews_upload():
    try:
        if 'file' not in request.files:
            return jsonify({"error": "No file uploaded"}), 400

        file = request.files['file']
        if file.filename == '':
            return jsonify({"error": "No file selected"}), 400

        if not file.filename.endswith('.csv'):
            return jsonify({"error": "File must be a CSV"}), 400

        csv_content = file.read().decode('utf-8')
        extractor = get_feature_extractor()
        return _process_csv_data(csv_content, extractor)

    except Exception as e:
        logger.error(f"Error processing uploaded file: {str(e)}")
        return jsonify({"error": str(e)}), 500


@app.route('/save_selected_clustering/<app_name>', methods=['POST'])
def save_selected_clustering(app_name):
    try:
        data = request.get_json()

        if "clustering" not in data:
            return jsonify({"error": "Missing 'clustering' in request body"}), 400

        clustering_result = convert_numpy_types(data["clustering"])
        clusters = clustering_result.get("clusters", {})
        logger.info(f"Generating semantic labels for {len(clusters)} clusters in '{app_name}'...")

        labels = taxonomy_builder.store_llm_taxonomy(app_name, clusters, method="llm-clustering")

        merge_results = taxonomy_builder.merge_mini_taxonomies(app_name)

        logger.info(f"Clustering result saved for '{app_name}'.")
        return jsonify({
            "status": "success",
            "message": f"Clustering saved for app '{app_name}'",
            "n_clusters": clustering_result.get("n_clusters"),
            "metrics": clustering_result.get("metrics"),
            "merge_results": merge_results
        })

    except Exception as e:
        logger.error(f"Failed to save clustering for '{app_name}': {e}", exc_info=True)
        return jsonify({"error": "Internal server error"}), 500


def _parse_csv_data(csv_data):
    csv_reader = csv.DictReader(csv_data.splitlines())
    reviews_data = list(csv_reader)

    if not reviews_data:
        raise ValueError("No reviews found in CSV")

    apps = {}
    for row in reviews_data:
        app_name = row['app_name']
        if app_name not in apps:
            apps[app_name] = {
                'package': row['app_package'],
                'category': row['app_categoryId'],
                'reviews': []
            }
        apps[app_name]['reviews'].append(row)

    return apps


def _process_app_reviews(app_name, reviews, extractor):
    logger.info(f"Processing app: {app_name}")

    processed_reviews = []
    all_processed_texts = []

    for review in reviews:
        original_text = review['review']
        processed_text = preprocessor.preprocess_text(original_text)
        processed_reviews.append({
            'review_id': review['reviewId'],
            'processed_text': processed_text,
            'original_text': original_text,
            'score': int(review['score'])
        })
        all_processed_texts.append(processed_text)

    features_per_review = extractor.extract_features(all_processed_texts)
    return processed_reviews, features_per_review


def _store_app_data(app_name, app_data, processed_reviews, features_per_review):
    # Create app node
    neo4j_conn.create_app_node(app_name, app_data['package'], app_data['category'])

    # Store reviews with features
    for i, review_data in enumerate(processed_reviews):
        review_features = features_per_review[i] if i < len(features_per_review) else []
        neo4j_conn.create_review_with_features(
            app_name,
            review_data['review_id'],
            review_data['processed_text'],
            review_data['original_text'],
            review_data['score'],
            review_features
        )


def _extract_and_aggregate_features(features_per_review):
    all_features = []
    for features in features_per_review:
        all_features.extend(features)

    unique_features = list(set(all_features))
    return all_features, unique_features


def compute_jaccard_similarity(candidate_a, candidate_b):
    def to_sets(clustering):
        return [set(cluster) for cluster in clustering.get("clusters", {}).values()]

    clusters_a = to_sets(candidate_a["clustering"])
    clusters_b = to_sets(candidate_b["clustering"])

    if not clusters_a or not clusters_b:
        return 0.0

    similarities = []
    for ca in clusters_a:
        best = 0.0
        for cb in clusters_b:
            intersection = len(ca & cb)
            union = len(ca | cb)
            score = intersection / union if union else 0.0
            best = max(best, score)
        similarities.append(best)

    return round(np.mean(similarities), 4) if similarities else 0.0


def _perform_clustering_analysis(app_name, unique_features, taxonomy_tree=None, extractor=None):
    if len(unique_features) < 4:
        return {
            "auto_tuning_completed": False,
            "message": f"Need at least 4 features for clustering. Found {len(unique_features)}."
        }

    logger.info("Performing hierarchical clustering with active learning...")
    feature_embeddings = extractor.get_embeddings(unique_features)
    tuning_result = clusterer.auto_tune_clustering(unique_features, feature_embeddings)
    best_options = tuning_result['best_options']

    clustering_candidates = []
    for i, option in enumerate(best_options):
        clustering_data = option['clustering']
        clustering_data['metrics'] = option['metrics']
        clustering_data['height_threshold'] = option['threshold']
        clustering_data['sibling_threshold'] = clusterer.sibling_threshold
        clustering_data = convert_numpy_types(clustering_data)

        if taxonomy_tree:
            clustering_data["hierarchy"] = {}
            for cluster_id, cluster_features in clustering_data["clusters"].items():
                clustering_data["hierarchy"][str(cluster_id)] = taxonomy_builder._extract_subtree_structures(
                    cluster_features, taxonomy_tree
                )

        n_clusters = len(clustering_data.get("clusters", {}))
        avg_cluster_size = round(
            sum(len(v) for v in clustering_data["clusters"].values()) / n_clusters if n_clusters > 0 else 0, 2
        )
        top_features = [c[0] for c in list(clustering_data["clusters"].values())[:3] if c]

        summary = {
            "index": i,
            "threshold": option["threshold"],
            "n_clusters": n_clusters,
            "avg_cluster_size": avg_cluster_size,
            "top_features": top_features,
            "metrics": option["metrics"]
        }

        clustering_candidates.append({
            "summary": summary,
            "clustering": clustering_data
        })

    # Add similarity to best candidate
    best_candidate = clustering_candidates[0]
    for candidate in clustering_candidates:
        similarity = compute_jaccard_similarity(best_candidate, candidate)
        candidate["summary"]["similarity_to_best"] = round(similarity, 4)

    return {
        "auto_tuning_completed": True,
        "candidates": clustering_candidates,
        "message": "Top clustering candidates generated. Use summary to choose one and save via /save_selected_clustering."
    }


def _build_taxonomy(app_name, unique_features, method="bert", feature_extractor=None):
    if len(unique_features) >= 4:
        feature_embeddings = feature_extractor.get_embeddings(unique_features)
        return taxonomy_builder.build_and_store_taxonomy(app_name, unique_features, feature_embeddings, method=method)
    return None


def _create_app_result(processed_reviews, all_features, unique_features, clustering_results, taxonomy_result):
    return {
        'processed_reviews': len(processed_reviews),
        'total_features': len(all_features),
        'unique_features': len(unique_features),
        'clustering_results': clustering_results,
        'top_features': dict(sorted(Counter(all_features).items(), key=lambda x: x[1], reverse=True)[:10]),
        'taxonomy': taxonomy_result
    }


def _process_csv_data(csv_data, extractor=None):
    try:
        if extractor is None:
            extractor = get_feature_extractor()

        logger.info(f"Using model_type='{extractor.model_type}' to process apps")

        apps = _parse_csv_data(csv_data)
        results = {}

        for app_name, app_data in apps.items():
            processed_reviews, features_per_review = _process_app_reviews(app_name, app_data['reviews'], extractor)
            _store_app_data(app_name, app_data, processed_reviews, features_per_review)

            all_features, unique_features = _extract_and_aggregate_features(features_per_review)
            logger.info(f"Found {len(unique_features)} unique features")

            taxonomy_result = _build_taxonomy(app_name, unique_features, method=extractor.model_type,
                                              feature_extractor=extractor)
            taxonomy_tree = taxonomy_result.get("taxonomy_tree", {}) if taxonomy_result else {}

            clustering_results = _perform_clustering_analysis(app_name, unique_features, taxonomy_tree, extractor)
            if clustering_results.get("candidates"):
                for candidate in clustering_results["candidates"]:
                    candidate["clustering"]["taxonomy_tree"] = taxonomy_tree

            clustering_results = convert_numpy_types(clustering_results)
            taxonomy_result = convert_numpy_types(taxonomy_result)

            results[app_name] = _create_app_result(
                processed_reviews, all_features, unique_features,
                clustering_results, taxonomy_result
            )

        return jsonify({
            "status": "success",
            "message": "Complete pipeline executed: CSV -> preprocessing -> Neo4j -> clustering -> taxonomy",
            "results": results
        })

    except Exception as e:
        logger.error(f"Error in complete pipeline: {str(e)}")
        return jsonify({"error": str(e)}), 500


def convert_numpy_types(obj):
    if isinstance(obj, dict):
        return {str(key) if isinstance(key, np.integer) else key: convert_numpy_types(value) for key, value in
                obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    else:
        return obj


def get_feature_extractor():
    model_type = request.args.get("model_type", "tfrex").lower()
    return FeatureExtractor(model_type=model_type)


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=3000)
