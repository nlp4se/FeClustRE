import logging
import json
import csv
from io import StringIO
from typing import List, Dict, Any, Optional


def setup_logging(log_level: str = 'INFO', log_format: str = None):
    if log_format is None:
        log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'

    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format=log_format,
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('app.log')
        ]
    )


def parse_csv_data(csv_content: str) -> List[Dict[str, Any]]:
    try:
        csv_file = StringIO(csv_content)
        reader = csv.DictReader(csv_file)
        return list(reader)
    except Exception as e:
        logging.error(f"Error parsing CSV: {str(e)}")
        return []


def group_reviews_by_app(reviews_data: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    apps = {}

    for row in reviews_data:
        app_name = row.get('app_name', '')
        if not app_name:
            continue

        if app_name not in apps:
            apps[app_name] = {
                'package': row.get('app_package', ''),
                'category': row.get('app_categoryId', ''),
                'reviews': []
            }
        apps[app_name]['reviews'].append(row)

    return apps


def validate_review_data(review: Dict[str, Any]) -> bool:
    required_fields = ['reviewId', 'review', 'score']
    return all(field in review and review[field] is not None for field in required_fields)


def clean_feature_text(text: str) -> str:
    if not text or not isinstance(text, str):
        return ""

    # Remove extra whitespace
    text = ' '.join(text.split())

    # Remove special characters at the beginning and end
    text = text.strip('.,!?;:')

    return text


def merge_similar_features(features: List[str], similarity_threshold: float = 0.8) -> List[str]:
    from difflib import SequenceMatcher

    if not features:
        return []

    unique_features = []

    for feature in features:
        feature_clean = clean_feature_text(feature)
        if not feature_clean:
            continue

        # Check if similar feature already exists
        is_similar = False
        for existing_feature in unique_features:
            similarity = SequenceMatcher(None, feature_clean.lower(), existing_feature.lower()).ratio()
            if similarity >= similarity_threshold:
                is_similar = True
                break

        if not is_similar:
            unique_features.append(feature_clean)

    return unique_features


def calculate_feature_importance(features: List[str], word_stats: Dict[str, int]) -> Dict[str, float]:
    if not features or not word_stats:
        return {}

    importance_scores = {}
    max_count = max(word_stats.values()) if word_stats else 1

    for feature in features:
        count = word_stats.get(feature, 0)
        # Normalize to 0-1 range
        importance_scores[feature] = count / max_count

    return importance_scores


def filter_features_by_importance(features: List[str], importance_scores: Dict[str, float],
                                  min_importance: float = 0.1) -> List[str]:
    return [
        feature for feature in features
        if importance_scores.get(feature, 0) >= min_importance
    ]


def create_feature_summary(features_per_review: List[List[str]]) -> Dict[str, Any]:
    all_features = []
    for features in features_per_review:
        all_features.extend(features)

    if not all_features:
        return {
            'total_features': 0,
            'unique_features': 0,
            'avg_features_per_review': 0,
            'top_features': []
        }

    from collections import Counter

    feature_counts = Counter(all_features)
    unique_features = len(feature_counts)
    avg_features = len(all_features) / len(features_per_review) if features_per_review else 0

    return {
        'total_features': len(all_features),
        'unique_features': unique_features,
        'avg_features_per_review': round(avg_features, 2),
        'top_features': feature_counts.most_common(10)
    }


def validate_clustering_parameters(height_threshold: float, sibling_threshold: float) -> bool:
    return (0.0 <= height_threshold <= 1.0 and
            0.0 <= sibling_threshold <= 1.0 and
            height_threshold >= sibling_threshold)


def format_clustering_results(clustering_result: Dict[str, Any]) -> Dict[str, Any]:
    if not clustering_result:
        return {}

    formatted = {
        'summary': {
            'total_clusters': clustering_result.get('n_clusters', 0),
            'total_features': sum(len(features) for features in clustering_result.get('clusters', {}).values())
        },
        'clusters': []
    }

    clusters = clustering_result.get('clusters', {})
    hierarchy = clustering_result.get('hierarchy', {})

    for cluster_id, features in clusters.items():
        cluster_info = {
            'id': cluster_id,
            'size': len(features),
            'features': features,
            'hierarchy': hierarchy.get(cluster_id, {})
        }
        formatted['clusters'].append(cluster_info)

    formatted['clusters'].sort(key=lambda x: x['size'], reverse=True)

    return formatted


def save_results_to_json(results: Dict[str, Any], filename: str) -> bool:
    try:
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        return True
    except Exception as e:
        logging.error(f"Error saving results to {filename}: {str(e)}")
        return False


def load_results_from_json(filename: str) -> Optional[Dict[str, Any]]:
    try:
        with open(filename, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        logging.error(f"Error loading results from {filename}: {str(e)}")
        return None



