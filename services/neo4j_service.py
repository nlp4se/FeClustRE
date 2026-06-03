import json
import logging

from neo4j import GraphDatabase
from config import Config

logger = logging.getLogger(__name__)


class Neo4jConnection:
    _INDEXES_CREATED = False

    def __init__(self, uri=None, user=None, password=None, database=None):
        self.uri = uri or Config.NEO4J_URI
        self.user = user or Config.NEO4J_USER
        self.password = password or Config.NEO4J_PASSWORD
        self.database = database or Config.NEO4J_DATABASE

        self.driver = GraphDatabase.driver(
            self.uri,
            auth=(self.user, self.password),
            connection_timeout=3,
            max_transaction_retry_time=0,
        )
        self._ensure_indexes()

    def _ensure_indexes(self):
        if Neo4jConnection._INDEXES_CREATED:
            return
        try:
            with self.driver.session(database=self.database) as session:
                for stmt in [
                    "CREATE INDEX IF NOT EXISTS FOR (r:Review) ON (r.id)",
                    "CREATE INDEX IF NOT EXISTS FOR (a:App) ON (a.name)",
                    "CREATE INDEX IF NOT EXISTS FOR (f:Feature) ON (f.name, f.app_name)",
                    "CREATE INDEX IF NOT EXISTS FOR (t:TaxonomyNode) ON (t.id)",
                    "CREATE INDEX IF NOT EXISTS FOR (m:MiniTaxonomyNode) ON (m.id)",
                    "CREATE INDEX IF NOT EXISTS FOR (m:MiniTaxonomyNode) ON (m.app_name)",
                ]:
                    session.run(stmt)
            Neo4jConnection._INDEXES_CREATED = True
            logger.info("Neo4j indexes ensured")
        except Exception as e:
            logger.warning(f"Could not create indexes: {e}")

    def close(self):
        self.driver.close()

    def create_app_node(self, app_name, app_package, category):
        with self.driver.session(database=self.database) as session:
            session.write_transaction(self._create_app, app_name, app_package, category)

    def create_review_with_features(self, app_name, review_id, processed_text, original_text, score, features=None, model_type='unknown'):
        with self.driver.session(database=self.database) as session:
            session.write_transaction(self._create_review_with_features, app_name, review_id,
                                      processed_text, original_text, score, features, model_type)

    def create_reviews_with_features_batch(self, app_name, processed_reviews, features_per_review, model_type='unknown', batch_size=100):
        """Write reviews + features in batches using UNWIND for speed.

        Removes existing reviews for the app first to avoid slow MERGE
        relationship checks against thousands of stale edges.
        """
        self._delete_app_reviews_batched(app_name)

        for start in range(0, len(processed_reviews), batch_size):
            batch = processed_reviews[start:start + batch_size]
            batch_features = features_per_review[start:start + batch_size]

            rows = []
            for i, review_data in enumerate(batch):
                feats = batch_features[i] if i < len(batch_features) else []
                rows.append({
                    "review_id": review_data["review_id"],
                    "processed_text": review_data["processed_text"],
                    "original_text": review_data["original_text"],
                    "score": review_data["score"],
                    "features": feats,
                })

            with self.driver.session(database=self.database) as session:
                session.write_transaction(
                    self._create_reviews_batch, app_name, rows, model_type
                )

    def _delete_app_reviews_batched(self, app_name, batch_size=200):
        """Delete existing reviews and orphaned features in batches to avoid OOM."""
        total = 0
        while True:
            with self.driver.session(database=self.database) as session:
                deleted = session.write_transaction(
                    lambda tx: tx.run("""
                    MATCH (a:App {name: $app_name})-[:HAS_REVIEW]->(r:Review)
                    WITH r LIMIT $limit
                    DETACH DELETE r
                    RETURN count(*) AS deleted
                    """, app_name=app_name, limit=batch_size).single()["deleted"]
                )
                total += deleted
                if deleted == 0:
                    break
        if total > 0:
            logger.info(f"Deleted {total} stale reviews for '{app_name}'")
            with self.driver.session(database=self.database) as session:
                session.write_transaction(
                    lambda tx: tx.run("""
                    MATCH (f:Feature {app_name: $app_name})
                    WHERE NOT (f)<-[:HAS_FEATURE]-()
                    DELETE f
                    """, app_name=app_name)
                )

    @staticmethod
    def _create_reviews_batch(tx, app_name, rows, model_type):
        tx.run("""
        UNWIND $rows AS row
        MATCH (a:App {name: $app_name})
        MERGE (r:Review {id: row.review_id})
        SET r.processed_text = row.processed_text,
            r.original_text = row.original_text,
            r.score = row.score
        MERGE (a)-[:HAS_REVIEW]->(r)
        WITH r, row
        UNWIND row.features AS feat
        MERGE (f:Feature {name: feat, app_name: $app_name})
        SET f.model_type = $model_type
        MERGE (r)-[rel:HAS_FEATURE]->(f)
        SET rel.model_type = $model_type
        """, app_name=app_name, rows=rows, model_type=model_type)

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

    def get_feature_statistics(self, app_name):
        with self.driver.session(database=self.database) as session:
            result = session.read_transaction(self._get_feature_stats, app_name)
            return result

    @staticmethod
    def _create_app(tx, app_name, app_package, category):
        query = """
        MERGE (a:App {name: $app_name})
        SET a.package = $app_package, a.category = $category
        RETURN a
        """
        tx.run(query, app_name=app_name, app_package=app_package, category=category)

    @staticmethod
    def _create_review_with_features(tx, app_name, review_id, processed_text, original_text, score, features, model_type='unknown'):
        tx.run("""
        MATCH (a:App {name: $app_name})
        MERGE (r:Review {id: $review_id})
        SET r.processed_text = $processed_text,
            r.original_text = $original_text,
            r.score = $score
        MERGE (a)-[:HAS_REVIEW]->(r)
        """, app_name=app_name, review_id=review_id,
               processed_text=processed_text, original_text=original_text, score=score)

        if features:
            tx.run("""
            UNWIND $features AS feat
            MATCH (r:Review {id: $review_id})
            MERGE (f:Feature {name: feat, app_name: $app_name})
            SET f.model_type = $model_type
            MERGE (r)-[rel:HAS_FEATURE]->(f)
            SET rel.model_type = $model_type
            """, review_id=review_id, features=features,
                   app_name=app_name, model_type=model_type)


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
        record = result.single()
        return record.data() if record else None

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

