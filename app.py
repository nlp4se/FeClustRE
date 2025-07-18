import nltk
import psutil
import torch
from flask import Flask, request, jsonify
import csv
import json
import logging
import sys
from datetime import datetime
from collections import Counter
import requests

from services.neo4j_service import Neo4jConnection
from services.preprocessing_service import ReviewPreprocessor
from services.feature_extraction_service import FeatureExtractor
from services.clustering_service import HierarchicalClusterer
from services.taxonomy_service import TaxonomyBuilder

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Initialize services
neo4j_conn = Neo4jConnection()
preprocessor = ReviewPreprocessor()
feature_extractor = FeatureExtractor()
clusterer = HierarchicalClusterer()
taxonomy_builder = TaxonomyBuilder(neo4j_conn)


@app.route('/ping')
def ping():
    return 'pong'


@app.route('/health')
def health_check():
    def check_neo4j():
        try:
            with neo4j_conn.driver.session(database=neo4j_conn.database) as session:
                if session.run("RETURN 1 as test").single()["test"] == 1:
                    return {"status": "healthy", "database": neo4j_conn.database, "uri": neo4j_conn.uri}
        except:
            pass
        return {"status": "unhealthy", "error": "Cannot connect to Neo4j", "database": neo4j_conn.database,
                "uri": neo4j_conn.uri}

    def check_tfrex_model():
        try:
            if feature_extractor.model and feature_extractor.tokenizer:
                test = feature_extractor.extract_features(["test app notification"])
                is_valid = test is not None and hasattr(test, '__len__') and len(test) > 0
                return {"status": "healthy", "model_name": feature_extractor.model_name, "test_result": is_valid}
            else:
                raise ValueError("Model or tokenizer not loaded")
        except Exception as e:
            return {"status": "unhealthy", "error": str(e),
                    "model_name": getattr(feature_extractor, "model_name", "unknown")}

    def check_embedding_model():
        try:
            if feature_extractor.embedding_model:
                test_embeddings = feature_extractor.get_embeddings(["test text"])
                is_valid = test_embeddings is not None and hasattr(test_embeddings, 'shape') and test_embeddings.shape[0] > 0
                return {"status": "healthy", "model_name": "all-MiniLM-L6-v2", "test_result": is_valid}
            else:
                raise ValueError("Embedding model not loaded")
        except Exception as e:
            return {"status": "unhealthy", "error": str(e), "model_name": "all-MiniLM-L6-v2"}

    def check_nltk_data():
        try:
            required = ['punkt', 'stopwords', 'wordnet']
            for item in required:
                try:
                    nltk.data.find(f'corpora/{item}' if item != 'punkt' else f'tokenizers/{item}')
                except LookupError:
                    nltk.download(item)
            return {"status": "healthy", "data_available": required}
        except Exception as e:
            return {"status": "unhealthy", "error": str(e)}

    def check_ollama():
        try:
            response = requests.get("http://localhost:11434/api/tags", timeout=3)
            if response.status_code != 200:
                return {"status": "unhealthy", "error": f"Status code: {response.status_code}"}

            models = response.json().get("models", [])
            qwen_installed = any("qwen" in model.get("name", "").lower() for model in models)

            if not qwen_installed:
                return {"status": "unhealthy", "error": "Qwen model not found in Ollama"}

            chat_resp = requests.post(
                "http://localhost:11434/api/chat",
                json={"model": "qwen:7b", "messages": [{"role": "user", "content": "Say hello"}]},
                timeout=5
            )

            if chat_resp.status_code == 200:
                return {"status": "healthy", "model": "qwen:7b"}
            else:
                return {"status": "unhealthy", "error": f"Chat failed with status {chat_resp.status_code}"}
        except Exception as e:
            return {"status": "unhealthy", "error": str(e), "model": "qwen:7b"}

    health_status = {
        "timestamp": datetime.now().isoformat(),
        "services": {
            "neo4j": check_neo4j(),
            "nltk": check_nltk_data(),
            "ollama": check_ollama()
        },
        "models": {
            "tfrex": check_tfrex_model(),
            "embeddings": check_embedding_model()
        },
        "system": {
            "python_version": sys.version,
            "cpu_usage": psutil.cpu_percent(),
            "memory_usage": psutil.virtual_memory().percent,
            "disk_usage": psutil.disk_usage('/').percent,
            "cuda_available": torch.cuda.is_available(),
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
        return _process_csv_data(csv_content)

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

        # Accept directly under 'clustering' key instead of 'clustering_result'
        if "clustering" not in data:
            return jsonify({"error": "Missing 'clustering' in request body"}), 400

        clustering_result = convert_numpy_types(data["clustering"])

        # Save to Neo4j
        neo4j_conn.create_clustering_result(app_name, clustering_result)

        return jsonify({
            "status": "success",
            "message": f"Clustering saved for app '{app_name}'",
            "n_clusters": clustering_result.get("n_clusters"),
            "metrics": clustering_result.get("metrics")
        })

    except Exception as e:
        logger.error(f"Error saving clustering result: {str(e)}")
        return jsonify({"error": str(e)}), 500


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


def _process_app_reviews(app_name, reviews):
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

    features_per_review = feature_extractor.extract_features(all_processed_texts)

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


def _perform_clustering_analysis(app_name, unique_features):
    if len(unique_features) < 4:
        return {
            "auto_tuning_completed": False,
            "message": f"Need at least 4 features for clustering. Found {len(unique_features)}."
        }

    logger.info("Performing hierarchical clustering with active learning...")
    feature_embeddings = feature_extractor.get_embeddings(unique_features)
    tuning_result = clusterer.auto_tune_clustering(unique_features, feature_embeddings)
    best_options = tuning_result['best_options']

    clustering_candidates = []
    for i, option in enumerate(best_options):
        clustering_data = option['clustering']
        clustering_data['metrics'] = option['metrics']
        clustering_data['height_threshold'] = option['threshold']
        clustering_data['sibling_threshold'] = clusterer.sibling_threshold
        clustering_data = convert_numpy_types(clustering_data)

        clusters = clustering_data.get("clusters", {})
        n_clusters = len(clusters)
        avg_cluster_size = round(
            sum(len(v) for v in clusters.values()) / n_clusters if n_clusters > 0 else 0, 2
        )
        top_features = []
        for cluster in list(clusters.values())[:3]:
            if cluster:
                top_features.append(cluster[0])

        metrics = option["metrics"]

        summary = {
            "index": i,
            "threshold": option["threshold"],
            "n_clusters": n_clusters,
            "avg_cluster_size": avg_cluster_size,
            "top_features": top_features,
            "metrics": metrics,

        }

        clustering_candidates.append({
            "summary": summary,
            "clustering": clustering_data
        })

    return {
        "auto_tuning_completed": True,
        "candidates": clustering_candidates,
        "message": "Top clustering candidates generated. Use summary to choose one and save via /save_selected_clustering."
    }


def _build_taxonomy(app_name, unique_features, method="bert"):
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


def _process_csv_data(csv_data):
    try:
        # Parse CSV and group by apps
        apps = _parse_csv_data(csv_data)
        results = {}

        # Process each app
        for app_name, app_data in apps.items():
            # Process reviews and extract features
            processed_reviews, features_per_review = _process_app_reviews(app_name, app_data['reviews'])

            # Store in Neo4j
            _store_app_data(app_name, app_data, processed_reviews, features_per_review)

            # Aggregate features
            all_features, unique_features = _extract_and_aggregate_features(features_per_review)
            logger.info(f"Found {len(unique_features)} unique features")

            # Perform clustering analysis
            clustering_results = _perform_clustering_analysis(app_name, unique_features)

            # Build taxonomy
            taxonomy_result = _build_taxonomy(app_name, unique_features)

            # Create result summary
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
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


def convert_numpy_types(obj):
    import numpy as np
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


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=3000)
