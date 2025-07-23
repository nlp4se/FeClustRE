import logging
import numpy as np
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics.pairwise import cosine_similarity
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import pdist, squareform

logger = logging.getLogger(__name__)


class HierarchicalClusterer:
    def __init__(self, height_threshold=0.5, sibling_threshold=0.3):
        self.height_threshold = height_threshold
        self.sibling_threshold = sibling_threshold

    def perform_clustering(self, features, embeddings):
        if not features or len(features) < 2:
            return {"clusters": [], "hierarchy": {}, "n_clusters": 0}

        try:
            if not isinstance(embeddings, np.ndarray):
                embeddings = np.array(embeddings)

            clustering = AgglomerativeClustering(
                n_clusters=None,
                distance_threshold=self.height_threshold,
                linkage='average',
                metric='cosine'
            )

            cluster_labels = clustering.fit_predict(embeddings)

            clusters = {}
            for i, label in enumerate(cluster_labels):
                if label not in clusters:
                    clusters[label] = []
                clusters[label].append(features[i])

            hierarchy = self._create_hierarchy(clusters, embeddings, cluster_labels, features)

            cluster_stats = self._calculate_cluster_stats(clusters, embeddings, cluster_labels)

            return {
                "clusters": clusters,
                "hierarchy": hierarchy,
                "n_clusters": len(clusters),
                "cluster_stats": cluster_stats
            }

        except Exception as e:
            logger.error(f"Error in hierarchical clustering: {str(e)}")
            return {"clusters": [], "hierarchy": {}, "n_clusters": 0}

    def _create_hierarchy(self, clusters, embeddings, cluster_labels, features):
        hierarchy = {}

        try:
            for cluster_id, cluster_features in clusters.items():
                cluster_indices = [i for i, label in enumerate(cluster_labels) if label == cluster_id]

                if len(cluster_indices) > 1:
                    cluster_embeddings = embeddings[cluster_indices]
                    similarity_matrix = cosine_similarity(cluster_embeddings)
                    avg_similarities = []
                    for i in range(len(cluster_embeddings)):
                        similarities = similarity_matrix[i]
                        avg_sim = np.mean(similarities[similarities != 1.0])  # Exclude self-similarity
                        avg_similarities.append(avg_sim if not np.isnan(avg_sim) else 0.0)

                    # Classify as parnt or child based on average similarity
                    parent_features = []
                    child_features = []

                    for i, feature in enumerate(cluster_features):
                        if avg_similarities[i] > self.sibling_threshold:
                            parent_features.append(feature)
                        else:
                            child_features.append(feature)

                    # If no clear parent-child distinction, treat all as siblings
                    if not parent_features:
                        parent_features = cluster_features
                        child_features = []

                    hierarchy[cluster_id] = {
                        "parent_features": parent_features,
                        "child_features": child_features,
                        "avg_similarity": float(np.mean(avg_similarities)) if avg_similarities else 0.0,
                        "cluster_coherence": float(self._calculate_cluster_coherence(cluster_embeddings))
                    }
                else:
                    hierarchy[cluster_id] = {
                        "parent_features": cluster_features,
                        "child_features": [],
                        "avg_similarity": 1.0,
                        "cluster_coherence": 1.0
                    }

            return hierarchy

        except Exception as e:
            logger.error(f"Error creating hierarchy: {str(e)}")
            return {}

    def _calculate_cluster_coherence(self, cluster_embeddings):
        if len(cluster_embeddings) < 2:
            return 1.0

        try:
            # Calculate pairwise similarities
            similarity_matrix = cosine_similarity(cluster_embeddings)

            # Get upper triangle (excluding diagonal)
            upper_triangle = similarity_matrix[np.triu_indices_from(similarity_matrix, k=1)]

            # Return average similarity as Python float
            return float(np.mean(upper_triangle))

        except Exception as e:
            logger.error(f"Error calculating cluster coherence: {str(e)}")
            return 0.0

    def _calculate_cluster_stats(self, clusters, embeddings, cluster_labels):
        stats = {}

        try:
            for cluster_id, cluster_features in clusters.items():
                cluster_indices = [i for i, label in enumerate(cluster_labels) if label == cluster_id]
                cluster_embeddings = embeddings[cluster_indices]

                stats[cluster_id] = {
                    "size": len(cluster_features),
                    "features": cluster_features,
                    "coherence": float(self._calculate_cluster_coherence(cluster_embeddings)),
                    "centroid": cluster_embeddings.mean(axis=0).tolist() if len(cluster_embeddings) > 0 else []
                }

            return stats

        except Exception as e:
            logger.error(f"Error calculating cluster stats: {str(e)}")
            return {}

    def get_dendrogram_data(self, embeddings):
        try:
            if len(embeddings) < 2:
                return None

            # Calculate distance matrix
            distances = pdist(embeddings, metric='cosine')

            # Perform linkage
            linkage_matrix = linkage(distances, method='average')

            return {
                "linkage_matrix": linkage_matrix.tolist(),
                "distance_threshold": self.height_threshold
            }

        except Exception as e:
            logger.error(f"Error generating dendrogram data: {str(e)}")
            return None

    def update_thresholds(self, height_threshold=None, sibling_threshold=None):
        if height_threshold is not None:
            self.height_threshold = height_threshold
        if sibling_threshold is not None:
            self.sibling_threshold = sibling_threshold

        logger.info(f"Updated thresholds - Height: {self.height_threshold}, Sibling: {self.sibling_threshold}")

    def get_optimal_clusters(self, embeddings, max_clusters=10):
        try:
            from sklearn.metrics import silhouette_score

            if len(embeddings) < 4:
                return 1

            best_score = -1
            best_clusters = 1

            for n_clusters in range(2, min(max_clusters + 1, len(embeddings))):
                clustering = AgglomerativeClustering(
                    n_clusters=n_clusters,
                    linkage='average',
                    metric='cosine'
                )

                cluster_labels = clustering.fit_predict(embeddings)
                score = silhouette_score(embeddings, cluster_labels, metric='cosine')

                if score > best_score:
                    best_score = score
                    best_clusters = n_clusters

            return best_clusters

        except Exception as e:
            logger.error(f"Error finding optimal clusters: {str(e)}")
            return 1

    def evaluate_clustering(self, features, embeddings, cluster_labels):
        from sklearn.metrics import silhouette_score, davies_bouldin_score

        if len(set(cluster_labels)) < 2:
            return {"error": "Need at least 2 clusters for evaluation"}

        try:
            silhouette = silhouette_score(embeddings, cluster_labels, metric='cosine')
            davies_bouldin = davies_bouldin_score(embeddings, cluster_labels)

            return {
                "silhouette_score": float(silhouette),
                "davies_bouldin_score": float(davies_bouldin),
                "n_clusters": len(set(cluster_labels)),
            }
        except Exception as e:
            return {"error": str(e)}

    def auto_tune_clustering(self, features, embeddings, threshold_range=(0.1, 0.9), steps=8):
        results = []
        thresholds = np.linspace(threshold_range[0], threshold_range[1], steps)

        for threshold in thresholds:
            self.height_threshold = threshold
            clustering_result = self.perform_clustering(features, embeddings)

            if clustering_result['n_clusters'] < 2:
                continue

            # Get cluster labels
            cluster_labels = []
            for feature in features:
                for cluster_id, cluster_features in clustering_result['clusters'].items():
                    if feature in cluster_features:
                        cluster_labels.append(int(cluster_id))
                        break

            metrics = self.evaluate_clustering(features, embeddings, cluster_labels)

            results.append({
                "threshold": float(threshold),
                "metrics": metrics,
                "clustering": clustering_result
            })

        # Sort by silhouette score (best first)
        results.sort(key=lambda x: x['metrics'].get('silhouette_score', -1), reverse=True)

        return {
            "best_options": results[:3],
            "all_results": results
        }