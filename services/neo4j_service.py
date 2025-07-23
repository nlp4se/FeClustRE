import json
import uuid

from neo4j import GraphDatabase
from config import Config
from services.taxonomy_service import TaxonomyBuilder


class Neo4jConnection:
    def __init__(self, uri=None, user=None, password=None, database=None):
        self.uri = uri or Config.NEO4J_URI
        self.user = user or Config.NEO4J_USER
        self.password = password or Config.NEO4J_PASSWORD
        self.database = database or Config.NEO4J_DATABASE

        self.driver = GraphDatabase.driver(self.uri, auth=(self.user, self.password))

    def close(self):
        self.driver.close()

    def create_clustering_result(self, app_name, clustering_result, taxonomy_tree=None):
        self._create_clustering_result(app_name, clustering_result, taxonomy_tree)

    def create_app_node(self, app_name, app_package, category):
        with self.driver.session(database=self.database) as session:
            session.write_transaction(self._create_app, app_name, app_package, category)

    def create_review_with_features(self, app_name, review_id, processed_text, original_text, score, features=None):
        with self.driver.session(database=self.database) as session:
            session.write_transaction(self._create_review_with_features, app_name, review_id,
                                      processed_text, original_text, score, features)

    def create_feature_statistics(self, app_name, word_stats):
        with self.driver.session(database=self.database) as session:
            session.write_transaction(self._create_feature_stats, app_name, word_stats)

    def get_app_reviews(self, app_name):
        with self.driver.session(database=self.database) as session:
            result = session.read_transaction(self._get_reviews, app_name)
            return result

    def get_app_features(self, app_name):
        with self.driver.session(database=self.database) as session:
            result = session.read_transaction(self._get_features, app_name)
            return result

    def get_app_clustering(self, app_name):
        with self.driver.session(database=self.database) as session:
            result = session.read_transaction(self._get_clustering, app_name)
            return result

    def get_feature_statistics(self, app_name):
        with self.driver.session(database=self.database) as session:
            result = session.read_transaction(self._get_feature_stats, app_name)
            return result

    @staticmethod
    def _create_app(tx, app_name, app_package, category):
        query = """
        MERGE (a:App {name: $app_name, package: $app_package, category: $category})
        RETURN a
        """
        tx.run(query, app_name=app_name, app_package=app_package, category=category)

    @staticmethod
    def _create_review_with_features(tx, app_name, review_id, processed_text, original_text, score, features):
        query = """
        MATCH (a:App {name: $app_name})
        MERGE (r:Review {
            id: $review_id,
            processed_text: $processed_text,
            original_text: $original_text,
            score: $score
        })
        MERGE (a)-[:HAS_REVIEW]->(r)
        RETURN r
        """
        tx.run(query, app_name=app_name, review_id=review_id,
               processed_text=processed_text, original_text=original_text, score=score)

        if features:
            for feature in features:
                feature_query = """
                MATCH (r:Review {id: $review_id})
                MERGE (f:Feature {name: $feature})
                MERGE (r)-[:HAS_FEATURE]->(f)
                RETURN f
                """
                tx.run(feature_query, review_id=review_id, feature=feature)


    def get_clustering_by_session(self, session_id):
        with self.driver.session(database=self.database) as session:
            result = session.read_transaction(self._get_clustering_by_session, session_id)
            return result

    @staticmethod
    def _get_clustering_by_session(tx, session_id):
        query = """
        MATCH (cs:ClusteringSession {id: $session_id})
        OPTIONAL MATCH (cs)-[:HAS_CLUSTER]->(c:Cluster)
        RETURN cs, collect(c) as clusters
        """
        result = tx.run(query, session_id=session_id)
        return result.single().data() if result.single() else None

    @staticmethod
    def _create_feature_stats(tx, app_name, word_stats):
        query = """
        MATCH (a:App {name: $app_name})
        CREATE (fs:FeatureStatistics {
            word_counts: $word_stats,
            created_at: datetime()
        })
        CREATE (a)-[:HAS_FEATURE_STATS]->(fs)
        RETURN fs
        """
        tx.run(query, app_name=app_name, word_stats=json.dumps(word_stats))

    @staticmethod
    def _get_reviews(tx, app_name):
        query = """
        MATCH (a:App {name: $app_name})-[:HAS_REVIEW]->(r:Review)
        OPTIONAL MATCH (r)-[:HAS_FEATURE]->(f:Feature)
        RETURN r.id as review_id, r.processed_text as processed_text, 
               r.original_text as original_text, r.score as score,
               collect(f.name) as features
        """
        result = tx.run(query, app_name=app_name)
        return [record.data() for record in result]

    @staticmethod
    def _get_features(tx, app_name):
        query = """
        MATCH (a:App {name: $app_name})-[:HAS_REVIEW]->(r:Review)-[:HAS_FEATURE]->(f:Feature)
        RETURN DISTINCT f.name as feature_name, count(r) as review_count
        ORDER BY review_count DESC
        """
        result = tx.run(query, app_name=app_name)
        return [record.data() for record in result]

    @staticmethod
    def _get_clustering(tx, app_name):
        query = """
        MATCH (a:App {name: $app_name})-[:HAS_CLUSTER]->(c:Cluster)
        RETURN collect({
            id: c.id,
            size: c.size,
            features: c.features,
            avg_similarity: c.avg_similarity,
            cluster_coherence: c.cluster_coherence,
            parent_features: c.parent_features,
            child_features: c.child_features
        }) AS clusters
        """
        result = tx.run(query, app_name=app_name)
        record = result.single()
        if record:
            clusters = record["clusters"]
            return {
                "n_clusters": len(clusters),
                "clusters": clusters
            }
        return None

    @staticmethod
    def _get_feature_stats(tx, app_name):
        query = """
        MATCH (a:App {name: $app_name})-[:HAS_FEATURE_STATS]->(fs:FeatureStatistics)
        RETURN fs.word_counts as word_counts, fs.created_at as created_at
        ORDER BY fs.created_at DESC
        LIMIT 1
        """
        result = tx.run(query, app_name=app_name)
        record = result.single()
        return record.data() if record else None

    def _create_clustering_result(self, app_name, clustering_result, taxonomy_tree=None):
        session_id = str(uuid.uuid4())
        print(f"Creating clustering result for app '{app_name}' with session ID {session_id}")

        with self.driver.session(database=self.database) as session:
            clusters = clustering_result.get("clusters", {})
            hierarchy = clustering_result.get("hierarchy", {})

            for cluster_id, feature_list in clusters.items():
                metrics = clustering_result.get("metrics", {}).get(str(cluster_id), {})
                avg_similarity = metrics.get("avg_similarity", 0.0)
                cluster_coherence = metrics.get("coherence", 0.0)

                if str(cluster_id) not in hierarchy and taxonomy_tree:
                    hierarchy[str(cluster_id)] = TaxonomyBuilder(self).extract_subtree_structure(feature_list,
                                                                                                 taxonomy_tree)

                hierarchy_info = hierarchy.get(str(cluster_id), {})
                parent_features = hierarchy_info.get("parent_features", [])
                child_features = hierarchy_info.get("child_features", [])
                semantic_label = hierarchy_info.get("semantic_label", "Unknown")

                session.write_transaction(
                    self._create_cluster_and_features,
                    session_id,
                    cluster_id,
                    feature_list,
                    avg_similarity,
                    cluster_coherence,
                    parent_features,
                    child_features,
                    semantic_label
                )

    @staticmethod
    def _create_cluster_and_features(tx, session_id, cluster_id, feature_list, avg_similarity, cluster_coherence,
                                     parent_features, child_features, semantic_label):
        tx.run("""
            MERGE (c:Cluster {session_id: $session_id, cluster_id: $cluster_id})
            SET c.avg_similarity = $avg_similarity,
                c.cluster_coherence = $cluster_coherence,
                c.semantic_label = $semantic_label
        """, session_id=session_id, cluster_id=str(cluster_id),
               avg_similarity=avg_similarity,
               cluster_coherence=cluster_coherence,
               semantic_label=semantic_label)

        feature_label = "ClusterFeature"

        if parent_features and child_features:
            for parent_feature in parent_features:
                tx.run(f"""
                    MERGE (f:{feature_label} {{name: $feature}})
                    WITH f
                    MATCH (c:Cluster {{session_id: $session_id, cluster_id: $cluster_id}})
                    MERGE (c)-[:HAS_CHILD]->(f)
                """, session_id=session_id, cluster_id=str(cluster_id), feature=parent_feature)

            for parent in parent_features:
                for child in child_features:
                    tx.run(f"""
                        MATCH (pf:{feature_label} {{name: $parent}}), (cf:{feature_label} {{name: $child}})
                        MERGE (pf)-[:HAS_CHILD]->(cf)
                    """, parent=parent, child=child)
        else:
            for feature in feature_list:
                tx.run(f"""
                    MERGE (f:{feature_label} {{name: $feature}})
                    WITH f
                    MATCH (c:Cluster {{session_id: $session_id, cluster_id: $cluster_id}})
                    MERGE (c)-[:HAS_CHILD]->(f)
                """, session_id=session_id, cluster_id=str(cluster_id), feature=feature)

        # Ensure all features exist
        for feature in feature_list:
            tx.run(f"MERGE (:{feature_label} {{name: $name}})", name=feature)

