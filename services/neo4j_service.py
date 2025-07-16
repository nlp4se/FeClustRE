import json
from neo4j import GraphDatabase
from config import Config


class Neo4jConnection:
    def __init__(self, uri=None, user=None, password=None, database=None):
        self.uri = uri or Config.NEO4J_URI
        self.user = user or Config.NEO4J_USER
        self.password = password or Config.NEO4J_PASSWORD
        self.database = database or Config.NEO4J_DATABASE

        self.driver = GraphDatabase.driver(self.uri, auth=(self.user, self.password))

    def close(self):
        self.driver.close()

    def create_app_node(self, app_name, app_package, category):
        with self.driver.session(database=self.database) as session:
            session.write_transaction(self._create_app, app_name, app_package, category)

    def create_review_node(self,
                           app_name,
                           review_id,
                           processed_text,
                           original_text,
                           score,
                           features=None):
        with self.driver.session(database=self.database) as session:
            session.write_transaction(self._create_review,
                                      app_name,
                                      review_id,
                                      processed_text,
                                      original_text, score,
                                      features)

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
        MERGE (a:App {name: $app_name, package: $app_package, category: $category})
        RETURN a
        """
        tx.run(query, app_name=app_name, app_package=app_package, category=category)

    @staticmethod
    def _create_review(tx, app_name, review_id, processed_text, original_text, score, features):
        query = """
        MATCH (a:App {name: $app_name})
        CREATE (r:Review {
            id: $review_id,
            processed_text: $processed_text,
            original_text: $original_text,
            score: $score,
            features: $features
        })
        CREATE (a)-[:HAS_REVIEW]->(r)
        RETURN r
        """
        tx.run(query, app_name=app_name, review_id=review_id,
               processed_text=processed_text, original_text=original_text,
               score=score, features=json.dumps(features) if features else None)

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
        RETURN r.id as review_id, r.processed_text as processed_text, 
               r.original_text as original_text, r.score as score, r.features as features
        """
        result = tx.run(query, app_name=app_name)
        return [record.data() for record in result]

    @staticmethod
    def _get_features(tx, app_name):
        query = """
        MATCH (a:App {name: $app_name})-[:HAS_REVIEW]->(r:Review)
        WHERE r.features IS NOT NULL
        RETURN r.features as features
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
