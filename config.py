import os


class Config:
    # Flask Configuration
    DEBUG = os.environ.get('FLASK_DEBUG', 'False').lower() == 'true'

    # Neo4j Configuration
    NEO4J_URI = os.environ.get('NEO4J_URI', 'bolt://10.4.63.10:7687')
    NEO4J_USER = os.environ.get('NEO4J_USER', 'neo4j')
    NEO4J_PASSWORD = os.environ.get('NEO4J_PASSWORD', '12345678')
    NEO4J_DATABASE = os.environ.get('NEO4J_DATABASE', 'neo4j')

    # Models Configuration
    TFREX_MODEL = 'quim-motger/t-frex-bert-base-uncased'
    EMBEDDING_MODELS = {
        'allmini': 'all-MiniLM-L6-v2',
        'sentence-t5': 'sentence-transformers/sentence-t5-base'
    }
    DEFAULT_EMBEDDING_MODEL = 'allmini'
    OLLAMA_MODEL = os.environ.get('OLLAMA_MODEL', 'qwen:1.8b')
    OLLAMA_BASE_URL = os.environ.get('OLLAMA_BASE_URL', 'http://localhost:11434')

    # Feature Extraction Configuration
    FEATURE_CONFIDENCE_THRESHOLD = 0.5
    MAX_FEATURES_PER_REVIEW = 50

    # Clustering Configuration
    DEFAULT_HEIGHT_THRESHOLD = 0.5
    DEFAULT_SIBLING_THRESHOLD = 0.3
    MAX_CLUSTERS = 10

    # Processing Configuration
    BATCH_SIZE = 32
    MAX_TEXT_LENGTH = 512

    # Logging Configuration
    LOG_LEVEL = os.environ.get('LOG_LEVEL', 'INFO')
    LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'


class DevelopmentConfig(Config):
    DEBUG = True
    LOG_LEVEL = 'DEBUG'

config = {
    'default': DevelopmentConfig
}
