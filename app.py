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
        # Step 1: Fetch LLM tags from Neo4j
        with neo4j_conn.driver.session(database=neo4j_conn.database) as session:
            results = session.run("""
                MATCH (n:MiniTaxonomyNode)
                WHERE NOT (()-[:HAS_CHILD]->(n)) AND n.llm_tag IS NOT NULL
                RETURN n.id AS id, n.llm_tag AS tag
            """)
            label_data = [(record["id"], record["tag"].strip()) for record in results if record["tag"]]

        if not label_data or len(label_data) < 2:
            return jsonify({"error": "Not enough labeled mini taxonomies found"}), 400

        label_ids, tags = zip(*label_data)
        tag_counter = Counter(tags)

        # Step 2: Detect singleton clusters (root with only one child)
        with neo4j_conn.driver.session(database=neo4j_conn.database) as session:
            singleton_result = session.run("""
                MATCH (root:MiniTaxonomyNode)-[:HAS_CHILD]->(child)
                WITH root, count(child) AS child_count
                WHERE child_count = 1
                RETURN root.id AS root_id
            """)
            singleton_roots = [r["root_id"] for r in singleton_result]

        # Step 3: Embed distinct tags
        distinct_tags = list(set(tags))
        embeddings = feature_extractor.get_embeddings(distinct_tags)
        tag_to_index = {tag: i for i, tag in enumerate(distinct_tags)}

        # Step 4: Cluster similar tags
        sim_matrix = cosine_similarity(embeddings)
        clustering = AgglomerativeClustering(
            n_clusters=None,
            distance_threshold=0.25,
            affinity='precomputed',
            linkage='average'
        )
        clusters = clustering.fit(1 - sim_matrix)

        clustered_tags = defaultdict(list)
        for idx, cluster_id in enumerate(clusters.labels_):
            clustered_tags[cluster_id].append(distinct_tags[idx])

        cluster_infos = []
        for cluster_id, tag_list in clustered_tags.items():
            cluster_embeddings = [embeddings[tag_to_index[tag]] for tag in tag_list]
            avg_sim = np.mean(cosine_similarity(cluster_embeddings))
            cluster_infos.append({
                "tag_group": tag_list,
                "avg_similarity": round(float(avg_sim), 4),
                "count": len(tag_list)
            })
        cluster_infos.sort(key=lambda x: x["count"], reverse=True)

        # Step 5: High similarity tag pairs (e.g., for merging)
        similar_pairs = []
        threshold = 0.9
        for i in range(len(distinct_tags)):
            for j in range(i + 1, len(distinct_tags)):
                sim = float(sim_matrix[i][j])
                if sim >= threshold:
                    similar_pairs.append({
                        "tag_a": distinct_tags[i],
                        "tag_b": distinct_tags[j],
                        "similarity": round(sim, 4)
                    })
        similar_pairs.sort(key=lambda x: x["similarity"], reverse=True)

        # Step 6: Detect low-quality tags
        low_quality_tags = [
            tag for tag in distinct_tags
            if re.fullmatch(r"(unknown|internal|cluster \d+)", tag.lower())
        ]

        return jsonify({
            "tag_statistics": {
                "total_tags": len(tags),
                "distinct_tags": len(distinct_tags),
                "most_common_tags": tag_counter.most_common(10),
                "singleton_clusters": {
                    "count": len(singleton_roots),
                    "examples": singleton_roots[:10]
                }
            },
            "tag_similarity_clusters": cluster_infos[:10],  # Top 10 groups
            "high_similarity_pairs": similar_pairs[:25],     # Top 25 highly similar
            "low_quality_tags": low_quality_tags
        })

    except Exception as e:
        logger.error(f"Failed to compute LLM taxonomy metrics: {e}", exc_info=True)
        return jsonify({"error": "Internal server error"}), 500

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


@app.route('/get_app_clustering/<app_name>')
def get_app_clustering(app_name):
    try:
        clustering = neo4j_conn.get_app_clustering(app_name)
        if not clustering:
            return jsonify({"error": "No clustering results found for this app"}), 404
        return jsonify({"app_name": app_name, "clustering": clustering})
    except Exception as e:
        logger.error(f"Error getting clustering data: {str(e)}")
        return jsonify({"error": str(e)}), 500


@app.route('/get_app_data/<app_name>')
def get_app_data(app_name):
    try:
        reviews = neo4j_conn.get_app_reviews(app_name)
        features = neo4j_conn.get_app_features(app_name)
        return jsonify({"app_name": app_name, "reviews": reviews, "features": features})
    except Exception as e:
        logger.error(f"Error getting app data: {str(e)}")
        return jsonify({"error": str(e)}), 500


@app.route('/cluster_features/<app_name>')
def cluster_features(app_name):
    try:
        height_threshold = float(request.args.get('height_threshold', 0.5))
        sibling_threshold = float(request.args.get('sibling_threshold', 0.3))
        features_data = neo4j_conn.get_app_features(app_name)

        if not features_data:
            return jsonify({"error": "No features found for this app"}), 404

        all_features = [f['feature_name'] for f in features_data if 'feature_name' in f]
        unique_features = list(set(all_features))

        if len(unique_features) < 2:
            return jsonify({"error": "Not enough features for clustering"}), 400

        embeddings = feature_extractor.get_embeddings(unique_features)
        clusterer.height_threshold = height_threshold
        clusterer.sibling_threshold = sibling_threshold
        clustering_result = clusterer.perform_clustering(unique_features, embeddings)

        return jsonify({
            "app_name": app_name,
            "clustering_result": clustering_result,
            "parameters": {
                "height_threshold": height_threshold,
                "sibling_threshold": sibling_threshold
            }
        })
    except Exception as e:
        logger.error(f"Error clustering features: {str(e)}")
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

        taxonomy_builder.store_llm_taxonomy(app_name, clusters, method="llm-clustering")

        logger.info(f"Clustering result saved for '{app_name}'.")
        return jsonify({
            "status": "success",
            "message": f"Clustering saved for app '{app_name}'",
            "n_clusters": clustering_result.get("n_clusters"),
            "metrics": clustering_result.get("metrics")
        })

    except Exception as e:
        logger.error(f"Failed to save clustering for '{app_name}': {e}", exc_info=True)
        return jsonify({"error": "Internal server error"}), 500



@app.route('/get_app_subclusters/<app_name>')
def get_app_subclusters(app_name):
    try:
        with neo4j_conn.driver.session(database=neo4j_conn.database) as session:
            result = session.run("""
                MATCH (a:App {name: $app_name})-[:HAS_SUBCLUSTER]->(sc:Subcluster)
                RETURN sc.cluster_id AS cluster_id, sc.features AS features, sc.session_id AS session_id
                ORDER BY sc.cluster_id
            """, app_name=app_name)
            return jsonify([record.data() for record in result])
    except Exception as e:
        logger.error(f"Error fetching subclusters: {str(e)}")
        return jsonify({"error": str(e)}), 500


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
