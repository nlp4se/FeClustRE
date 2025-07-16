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

from services.neo4j_service import Neo4jConnection
from services.preprocessing_service import ReviewPreprocessor
from services.feature_extraction_service import FeatureExtractor
from services.clustering_service import HierarchicalClusterer

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Initialize services
neo4j_conn = Neo4jConnection()
preprocessor = ReviewPreprocessor()
feature_extractor = FeatureExtractor()
clusterer = HierarchicalClusterer()


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
        return {"status": "unhealthy", "error": "Cannot connect to Neo4j", "database": neo4j_conn.database, "uri": neo4j_conn.uri}

    def check_tfrex_model():
        try:
            if feature_extractor.model and feature_extractor.tokenizer:
                test = feature_extractor.extract_features(["test app notification"])
                is_valid = test is not None and hasattr(test, '__len__') and len(test) > 0
                return {
                    "status": "healthy",
                    "model_name": feature_extractor.model_name,
                    "test_result": is_valid
                }
            else:
                raise ValueError("Model or tokenizer not loaded")
        except Exception as e:
            return {
                "status": "unhealthy",
                "error": str(e),
                "model_name": getattr(feature_extractor, "model_name", "unknown")
            }

    def check_embedding_model():
        try:
            if feature_extractor.embedding_model:
                test_embeddings = feature_extractor.get_embeddings(["test text"])

                is_valid = (
                        test_embeddings is not None and
                        hasattr(test_embeddings, 'shape') and
                        test_embeddings.shape[0] > 0
                )

                return {
                    "status": "healthy",
                    "model_name": "all-MiniLM-L6-v2",
                    "test_result": is_valid
                }
            else:
                raise ValueError("Embedding model not loaded")
        except Exception as e:
            return {
                "status": "unhealthy",
                "error": str(e),
                "model_name": "all-MiniLM-L6-v2"
            }

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

    health_status = {
        "timestamp": datetime.now().isoformat(),
        "services": {
            "neo4j": check_neo4j(),
            "nltk": check_nltk_data()
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

    # Determine overall health
    all_statuses = list(health_status["services"].values()) + list(health_status["models"].values())
    health_status["status"] = "healthy" if all(s["status"] == "healthy" for s in all_statuses) else "unhealthy"
    status_code = 200 if health_status["status"] == "healthy" else 503

    return jsonify(health_status), status_code


@app.route('/process_reviews', methods=['POST'])
def process_reviews():
    try:
        data = request.get_json()
        csv_data = data.get('csv_data', '')
        if not csv_data:
            return jsonify({"error": "No CSV data provided"}), 400
        return _process_csv_data(csv_data)

    except Exception as e:
        logger.error(f"Error processing reviews: {str(e)}")
        return jsonify({"error": str(e)}), 500


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


@app.route('/get_app_data/<app_name>')
def get_app_data(app_name):
    try:
        reviews = neo4j_conn.get_app_reviews(app_name)
        features = neo4j_conn.get_app_features(app_name)

        return jsonify({
            "app_name": app_name,
            "reviews": reviews,
            "features": features
        })

    except Exception as e:
        logger.error(f"Error getting app data: {str(e)}")
        return jsonify({"error": str(e)}), 500


@app.route('/cluster_features/<app_name>')
def cluster_features(app_name):
    try:
        height_threshold = float(request.args.get('height_threshold', 0.5))
        sibling_threshold = float(request.args.get('sibling_threshold', 0.3))

        # Get app features
        features_data = neo4j_conn.get_app_features(app_name)

        if not features_data:
            return jsonify({"error": "No features found for this app"}), 404

        # Extract all features
        all_features = []
        for feature_data in features_data:
            features = json.loads(feature_data['features']) if feature_data['features'] else []
            all_features.extend(features)

        # Remove duplicates
        unique_features = list(set(all_features))

        if len(unique_features) < 2:
            return jsonify({"error": "Not enough features for clustering"}), 400

        # Get embeddings
        embeddings = feature_extractor.get_embeddings(unique_features)

        # Update clustering thresholds
        clusterer.height_threshold = height_threshold
        clusterer.sibling_threshold = sibling_threshold

        # Perform clustering
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


def _process_csv_data(csv_data):
    try:
        csv_reader = csv.DictReader(csv_data.splitlines())
        reviews_data = list(csv_reader)

        if not reviews_data:
            return jsonify({"error": "No reviews found in CSV"}), 400

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

        results = {}

        for app_name, app_data in apps.items():
            logger.info(f"Processing app: {app_name}")

            # Create app node
            neo4j_conn.create_app_node(app_name, app_data['package'], app_data['category'])

            # Process reviews
            processed_reviews = []
            all_processed_texts = []

            for review in app_data['reviews']:
                original_text = review['review']
                processed_text = preprocessor.preprocess_text(original_text)

                processed_reviews.append({
                    'review_id': review['reviewId'],
                    'processed_text': processed_text,
                    'original_text': original_text,
                    'score': int(review['score'])
                })

                all_processed_texts.append(processed_text)

            # Extract features using T-FREX
            logger.info(f"Extracting features for {len(all_processed_texts)} reviews")
            features_per_review = feature_extractor.extract_features(all_processed_texts)

            # Get embeddings for clustering
            embeddings = feature_extractor.get_embeddings(all_processed_texts)

            # Collect all features for clustering
            all_features = []
            for features in features_per_review:
                all_features.extend(features)

            # Remove duplicates for clustering
            unique_features = list(set(all_features))

            # Perform clustering if we have enough features
            clustering_result = {}
            if len(unique_features) >= 2:
                feature_embeddings = feature_extractor.get_embeddings(unique_features)
                clustering_result = clusterer.perform_clustering(unique_features, feature_embeddings)

            # Calculate word statistics
            word_stats = dict(Counter(all_features))

            # Save to Neo4j
            for i, review_data in enumerate(processed_reviews):
                review_features = features_per_review[i] if i < len(features_per_review) else []
                neo4j_conn.create_review_node(
                    app_name,
                    review_data['review_id'],
                    review_data['processed_text'],
                    review_data['original_text'],
                    review_data['score'],
                    review_features
                )

            # Save feature statistics
            neo4j_conn.create_feature_statistics(app_name, word_stats)

            results[app_name] = {
                'processed_reviews': len(processed_reviews),
                'total_features': len(all_features),
                'unique_features': len(unique_features),
                'clustering_result': clustering_result,
                'top_features': dict(sorted(word_stats.items(), key=lambda x: x[1], reverse=True)[:10])
            }

        return jsonify({
            "status": "success",
            "message": "Reviews processed successfully",
            "results": results
        })

    except Exception as e:
        logger.error(f"Error processing CSV data: {str(e)}")
        return jsonify({"error": str(e)}), 500



if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=3000)