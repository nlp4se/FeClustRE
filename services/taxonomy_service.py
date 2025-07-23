import uuid
from datetime import datetime
import numpy as np
from scipy.cluster.hierarchy import linkage, to_tree, fcluster
from sklearn.metrics.pairwise import cosine_distances, cosine_similarity
import logging
import re
import requests
import json

logger = logging.getLogger(__name__)


class TaxonomyBuilder:
    def __init__(self, neo4j_conn, feature_extractor, height_threshold=0.5):
        self.neo4j_conn = neo4j_conn
        self.feature_extractor = feature_extractor
        self.height_threshold = height_threshold

    def build_and_store_taxonomy(self, app_name, features, embeddings, method="bert"):
        if len(features) < 2:
            logger.warning("Not enough features to build taxonomy.")
            return {"error": "Not enough features for taxonomy"}

        embeddings = np.array(embeddings)
        distance_matrix = cosine_distances(embeddings)
        linkage_matrix = linkage(distance_matrix, method='average')

        tree_root, _ = to_tree(linkage_matrix, rd=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        session_id = f"{app_name}_taxonomy_{method}_{timestamp}"

        with self.neo4j_conn.driver.session(database=self.neo4j_conn.database) as session:
            node_counter = [0]
            root_id = f"taxonomy_node_{node_counter[0]}"
            session.write_transaction(
                self._create_taxonomy_tree_with_app_link,
                tree_root, features, session_id, app_name, method, node_counter, root_id=root_id
            )

        best_cut, top_results = self._tune_subcluster_cut(linkage_matrix, features, embeddings)

        for result in top_results:
            result["subclusters"] = {
                str(k): v for k, v in result["subclusters"].items()
            }

        return {
            "taxonomy_session_id": session_id,
            "top_subcluster_options": top_results[:3],
            "message": "Top subcluster cut options generated. Use /save_subclusters/<app_name> to persist one."
        }

    def save_selected_subclusters(self, app_name, session_id, selected_subclusters):
        with self.neo4j_conn.driver.session(database=self.neo4j_conn.database) as session:
            session.write_transaction(self._store_subclusters, app_name, session_id, selected_subclusters)

    def _create_taxonomy_tree_with_app_link(self, tx, node, features, session_id, app_name, method, node_counter,
                                            parent_id=None, root_id=None):
        is_leaf = node.is_leaf()
        feature = features[node.id] if is_leaf else None

        node_id = f"taxonomy_node_{node_counter[0]}"
        node_counter[0] += 1

        tx.run("""
            CREATE (n:TaxonomyNode {
                id: $node_id,
                is_leaf: $is_leaf,
                feature: $feature,
                session_id: $session_id
            })
        """, node_id=node_id, is_leaf=is_leaf, feature=feature, session_id=session_id)

        if parent_id:
            tx.run("""
                MATCH (p:TaxonomyNode {id: $parent_id}), (c:TaxonomyNode {id: $child_id})
                CREATE (p)-[:HAS_CHILD]->(c)
            """, parent_id=parent_id, child_id=node_id)

        if node_id == root_id:
            tx.run("""
                MATCH (a:App {name: $app_name})
                MATCH (root:TaxonomyNode {id: $root_id})
                MERGE (a)-[:HAS_TAXONOMY {method: $method}]->(root)
            """, app_name=app_name, root_id=root_id, method=method)

        if node.left:
            self._create_taxonomy_tree_with_app_link(tx, node.left, features, session_id, app_name, method,
                                                     node_counter, node_id, root_id)
        if node.right:
            self._create_taxonomy_tree_with_app_link(tx, node.right, features, session_id, app_name, method,
                                                     node_counter, node_id, root_id)

    def _cut_tree(self, linkage_matrix, features, height_threshold):
        cluster_assignments = fcluster(linkage_matrix, t=height_threshold, criterion='distance')
        subclusters = {}
        for feature, cluster_id in zip(features, cluster_assignments):
            if cluster_id not in subclusters:
                subclusters[cluster_id] = []
            subclusters[cluster_id].append(feature)
        return subclusters

    def _store_subclusters(self, tx, app_name, session_id, subclusters):
        for cluster_id, feature_list in subclusters.items():
            cluster_uuid = str(uuid.uuid4())
            tx.run("""
                MATCH (a:App {name: $app_name})
                CREATE (sc:Subcluster {
                    id: $cluster_uuid,
                    cluster_id: $cluster_id,
                    session_id: $session_id
                })
            """, app_name=app_name, cluster_uuid=cluster_uuid, cluster_id=cluster_id, session_id=session_id)

            for feature in feature_list:
                tx.run("""
                    MATCH (sc:Subcluster {id: $cluster_uuid})
                    MATCH (f:Feature {name: $feature})
                    MERGE (sc)-[:HAS_FEATURE]->(f)
                """, cluster_uuid=cluster_uuid, feature=feature)

    def _tune_subcluster_cut(self, linkage_matrix, features, embeddings):
        thresholds = np.linspace(0.2, 1.0, 10)
        embeddings_dict = {f: e for f, e in zip(features, embeddings)}
        results = []

        for t in thresholds:
            subclusters = self._cut_tree(linkage_matrix, features, t)
            if len(subclusters) == 0:
                continue

            avg_size = np.mean([len(v) for v in subclusters.values()])
            coverage = len(set([f for v in subclusters.values() for f in v])) / len(features)
            coherence = self._compute_avg_coherence(subclusters, embeddings_dict)
            n_singletons = sum(1 for v in subclusters.values() if len(v) == 1)
            singleton_ratio = n_singletons / len(subclusters)
            score = (len(subclusters) * avg_size * coherence * coverage) * (1 - singleton_ratio)

            results.append({
                "height_threshold": float(t),
                "n_subclusters": len(subclusters),
                "avg_size": round(avg_size, 2),
                "avg_coherence": round(coherence, 3),
                "coverage": round(coverage, 3),
                "singleton_ratio": round(singleton_ratio, 3),
                "score": round(score, 3),
                "subclusters": subclusters
            })

        results.sort(key=lambda x: x["score"], reverse=True)
        return results[0]["height_threshold"], results[:3]

    def _compute_avg_coherence(self, subclusters, embeddings_dict):
        scores = []
        for feats in subclusters.values():
            emb = np.array([embeddings_dict[f] for f in feats if f in embeddings_dict])
            if len(emb) > 1:
                sim = cosine_similarity(emb)
                sim_vals = sim[np.triu_indices_from(sim, k=1)]
                scores.append(np.mean(sim_vals))
        return np.mean(scores) if scores else 0.0

    def _extract_subtree_structures(self, features, taxonomy_tree):
        parent_features = set()
        child_features = set()

        for parent, children in taxonomy_tree.items():
            if parent in features:
                parent_features.add(parent)
                for child in children:
                    if child in features:
                        child_features.add(child)

        if not parent_features and not child_features:
            sorted_feats = sorted(features, key=len)
            return {
                "parent_features": [sorted_feats[0]] if sorted_feats else [],
                "child_features": sorted_feats[1:] if len(sorted_feats) > 1 else []
            }

        return {
            "parent_features": list(parent_features),
            "child_features": list(child_features)
        }

    def store_llm_taxonomy(self, app_name, clusters, method="llm-clustering"):
        from datetime import datetime
        from scipy.cluster.hierarchy import linkage, to_tree
        from scipy.spatial.distance import pdist
        import uuid

        session_id = f"{app_name}_llm_clusters_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        node_counter = [0]
        labels = {}

        def _create_node(tx, node_id, feature, is_leaf, session_id, llm_tag=None):
            tx.run("""
                MERGE (n:MiniTaxonomyNode {id: $id})
                SET n.is_leaf = $is_leaf,
                    n.feature = $feature,
                    n.session_id = $session_id,
                    n.llm_tag = $llm_tag
            """, id=node_id, is_leaf=is_leaf, feature=feature, session_id=session_id, llm_tag=llm_tag)

        def _create_link(tx, parent_id, child_id):
            tx.run("""
                MATCH (p:MiniTaxonomyNode {id: $parent_id}),
                      (c:MiniTaxonomyNode {id: $child_id})
                MERGE (p)-[:HAS_CHILD]->(c)
            """, parent_id=parent_id, child_id=child_id)

        def _build_tree(tx, subtree, parent_id):
            node_id = f"mini_taxonomy_node_{node_counter[0]}"
            node_counter[0] += 1
            feature = subtree.get("feature", "internal")
            is_leaf = subtree.get("is_leaf", False)
            _create_node(tx, node_id, feature, is_leaf, session_id)
            _create_link(tx, parent_id, node_id)
            for child in subtree.get("children", []):
                _build_tree(tx, child, node_id)

        def _extract_tree(features):
            if len(features) == 1:
                return {"feature": features[0], "is_leaf": True, "children": []}
            embeddings = self.feature_extractor.get_embeddings(features)
            dists = pdist(embeddings, metric="cosine")
            tree = to_tree(linkage(dists, method="average"))

            def build_subtree(node):
                if node.is_leaf():
                    return {"feature": features[node.id], "is_leaf": True, "children": []}
                return {
                    "feature": "internal",
                    "is_leaf": False,
                    "children": [build_subtree(node.left), build_subtree(node.right)]
                }

            return build_subtree(tree)

        for cluster_id, features in clusters.items():
            label = generate_cluster_label(features)
            labels[cluster_id] = label
            try:
                subtree = _extract_tree(features)
                root_id = f"mini_taxonomy_root_{uuid.uuid4().hex[:8]}"
                with self.neo4j_conn.driver.session(database=self.neo4j_conn.database) as session:
                    session.write_transaction(_create_node, root_id, label, False, session_id, label)
                    session.write_transaction(
                        lambda tx: tx.run("""
                            MATCH (a:App {name: $app_name}), (r:MiniTaxonomyNode {id: $root_id})
                            MERGE (a)-[:HAS_MINI_TAXONOMY {method: $method}]->(r)
                        """, app_name=app_name, root_id=root_id, method=method)
                    )
                    session.write_transaction(_build_tree, subtree, root_id)
            except Exception as e:
                logger.error(f"Failed to store LLM taxonomy for cluster {cluster_id}: {e}", exc_info=True)

        return labels


def generate_cluster_label(features, mode="few-shot", base_url=None, model=None):
    from config import config
    cfg = config["default"]()

    base_url = base_url or cfg.OLLAMA_BASE_URL
    model = model or cfg.OLLAMA_MODEL

    try:
        logger.info(f"Requesting label from Qwen for cluster with {len(features)} features.")

        if mode == "no-shot":
            prompt = (
                "You are an assistant that assigns a single, concise category label to a list of app feature keywords.\n"
                "Given the list of features below, respond with only the best possible semantic category name that "
                "groups them together. Do not explain. Just return the name.\n\n"
                f"Features: {', '.join(features)}\n\n"
                "Category:"
            )
        elif mode == "few-shot":
            prompt = (
                "You are an assistant that assigns a single, concise category name to a list of app feature keywords.\n"
                "Your response must be only the category name. No explanations. No punctuation. No labels like 'Usage' or 'Explanation'.\n"
                "Return only the category name — a short phrase (1 to 4 words).\n\n"
                "Examples:\n"
                "Features: send message, chat group, reply dm\nCategory: Messaging\n\n"
                "Features: log in, password reset, cannot login\nCategory: Account Access\n\n"
                "Features: freeze screen, crash often, app bug\nCategory: App Stability\n\n"
                "Features: battery drain, overheat, fast battery use\nCategory: Battery Usage\n\n"
                "Features: share file, upload media, download image\nCategory: File Sharing\n\n"
                f"Features: {', '.join(features)}\nCategory:"
            )
        else:
            logger.warning(f"Unsupported prompt mode: {mode}")
            return "Unknown"

        response = requests.post(
            f"{base_url}/api/chat",
            json={
                "model": model,
                "messages": [{"role": "user", "content": prompt}]
            },
            stream=True,
            timeout=30
        )

        if response.status_code != 200:
            logger.warning(f"Ollama returned status {response.status_code} for model '{model}'")
            return "Unknown"

        full_content = ""
        for line in response.iter_lines(decode_unicode=True):
            if not line.strip():
                continue
            try:
                msg = json.loads(line)
                chunk = msg.get("message", {}).get("content", "")
                full_content += chunk
            except Exception as e:
                logger.warning(f"Skipping malformed line from Ollama: {e}")

        label = full_content.strip().split("\n")[0]
        label = re.sub(r"[^a-zA-Z0-9\s\-]", "", label).strip()

        if "explanation" in label.lower() or "usage" in label.lower() or not label:
            logger.warning("LLM error. Using fallback.")
            return "Unknown"

        logger.info(f"Generated label: '{label}' for cluster with features: {features[:3]}...")
        return label

    except Exception as e:
        logger.error(f"Error generating label with Qwen: {str(e)}")
        return "Unknown"
