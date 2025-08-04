import logging

import requests
import torch
import os
import numpy as np
from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)

class FeatureExtractor:
    def __init__(self, model_type='tfrex'):
        self.model_type = model_type.lower()
        self.tokenizer = None
        self.model = None
        self.ner_pipeline = None
        self.embedding_model = None
        self.re_extractor = None
        self.model_name = None
        self._initialize_models()

    def _initialize_models(self):
        try:
            if self.model_type == 'tfrex':
                self.model_name = "quim-motger/t-frex-bert-base-uncased"
                logger.info("Loading T-FREX model...")
                self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
                self.model = AutoModelForTokenClassification.from_pretrained(self.model_name)
                self.ner_pipeline = pipeline(
                    "ner",
                    model=self.model,
                    tokenizer=self.tokenizer,
                    aggregation_strategy="simple",
                    device=0 if torch.cuda.is_available() else -1
                )

            elif self.model_type == 'transfeatex':
                self.model_name = "TransfeatEx (API)"
                use_vpn = os.environ.get('TRANSFEATEX_USE_VPN', 'true').lower() == 'true'
                if use_vpn:
                    self.transfeatex_endpoint = 'http://10.4.63.10:3004/extract-features'
                    logger.info("Configured TransfeatEx VPN endpoint.")
                else:
                    self.transfeatex_endpoint = os.environ.get('TRANSFEATEX_URL',
                                                               'http://gessi-chatbots.essi.upc.edu:3004') + '/extract-features-aux'
                    logger.info("Configured TransfeatEx original endpoint.")


            else:
                raise ValueError(f"Unsupported model type: {self.model_type}")

            # Load embedding model in both cases
            logger.info("Loading embedding model...")
            self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
            logger.info("Models loaded successfully")

        except Exception as e:
            logger.error(f"Error initializing models: {str(e)}")
            raise

    def extract_features(self, texts):
        if not texts:
            return []
        elif self.model_type == 'transfeatex':
            return self._extract_features_transfeatex(texts)
        else:
            return self._extract_features_tfrex(texts)

    def _extract_features_transfeatex(self, texts):
        features_per_text = []
        for text in texts:
            try:
                response = requests.post(self.transfeatex_endpoint, json={"text": text})
                if response.status_code == 200:
                    features = response.json()
                    features_per_text.append(features if isinstance(features, list) else [])
                else:
                    logger.warning(f"TransfeatEx returned status {response.status_code}")
                    features_per_text.append([])
            except Exception as e:
                logger.error(f"Error contacting TransfeatEx: {str(e)}")
                features_per_text.append([])
        return features_per_text

    def _extract_features_tfrex(self, texts):
        features_per_text = []

        for text in texts:
            if not text or not isinstance(text, str):
                features_per_text.append([])
                continue

            try:
                entities = self.ner_pipeline(text)

                features = []
                for entity in entities:
                    if entity['score'] > 0.5:
                        feature_text = entity['word'].strip()
                        if feature_text and len(feature_text) > 2:
                            features.append(feature_text)

                # Remove duplicates while preserving order
                unique_features = []
                seen = set()
                for feature in features:
                    if feature.lower() not in seen:
                        unique_features.append(feature)
                        seen.add(feature.lower())

                features_per_text.append(unique_features)

            except Exception as e:
                logger.error(f"Error extracting features from text: {str(e)}")
                features_per_text.append([])

        return features_per_text

    def extract_features_with_details(self, text):
        if self.model_type == 'rebert':
            # No detailed RE-BERT output is currently supported
            return []
        if not text or not isinstance(text, str):
            return []

        try:
            entities = self.ner_pipeline(text)

            detailed_features = []
            for entity in entities:
                detailed_features.append({
                    'text': entity['word'].strip(),
                    'label': entity['entity_group'],
                    'score': entity['score'],
                    'start': entity['start'],
                    'end': entity['end']
                })

            return detailed_features

        except Exception as e:
            logger.error(f"Error extracting detailed features: {str(e)}")
            return []

    def get_embeddings(self, texts):
        if not texts:
            return []

        try:
            valid_texts = [text for text in texts if text and isinstance(text, str)]

            if not valid_texts:
                return []

            embeddings = self.embedding_model.encode(valid_texts)
            return embeddings

        except Exception as e:
            logger.error(f"Error getting embeddings: {str(e)}")
            return []

    def get_model_info(self):
        return {
            'model_type': self.model_type,
            'model_name': self.model_name,
            'embedding_model': 'all-MiniLM-L6-v2',
            'device': 'cuda' if torch.cuda.is_available() else 'cpu',
            'model_loaded': (self.model is not None or self.re_extractor is not None),
            'tokenizer_loaded': self.tokenizer is not None
        }
