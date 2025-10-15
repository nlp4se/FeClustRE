import logging

import requests
import torch
import os
import numpy as np
from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline
from sentence_transformers import SentenceTransformer
from feature_post_processor import FeaturePostProcessor

from config import Config

logger = logging.getLogger(__name__)


class FeatureExtractor:
    def __init__(self, model_type='t-frex', embedding_model='allmini', enable_postprocessing=True):
        self.model_type = model_type.lower()
        self.embedding_model_key = embedding_model
        self.tokenizer = None
        self.model = None
        self.ner_pipeline = None
        self.embedding_model = None
        self.re_extractor = None
        self.model_name = None
        self.enable_postprocessing = enable_postprocessing

        logger.info(f"Initializing FeatureExtractor with model_type='{model_type}', "
                    f"embedding_model='{embedding_model}', postprocessing={'enabled' if enable_postprocessing else 'disabled'}")

        # Initialize post-processor
        if self.enable_postprocessing:
            self.postprocessor = FeaturePostProcessor(
                min_length=3,
                max_length=50,
                similarity_threshold=0.85
            )
            logger.info("Feature post-processor initialized and enabled")
        else:
            self.postprocessor = None
            logger.info("Feature post-processor disabled")

        self._initialize_models()
        self.batch_size = 100
        logger.info(f"FeatureExtractor initialization complete, batch_size={self.batch_size}")

    def _initialize_models(self):
        try:
            if self.model_type == 't-frex':
                self.model_name = "quim-motger/t-frex-bert-base-uncased"
                logger.info(f"Loading T-FREX model: {self.model_name}")
                self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
                logger.debug("T-FREX tokenizer loaded successfully")

                self.model = AutoModelForTokenClassification.from_pretrained(self.model_name)
                logger.debug("T-FREX model loaded successfully")

                device = 0 if torch.cuda.is_available() else -1
                logger.info(f"Creating NER pipeline on device: {'GPU' if device == 0 else 'CPU'}")

                self.ner_pipeline = pipeline(
                    "ner",
                    model=self.model,
                    tokenizer=self.tokenizer,
                    aggregation_strategy="simple",
                    device=device
                )
                logger.info("T-FREX NER pipeline created successfully")

            elif self.model_type == 'transfeatex':
                self.model_name = "TransfeatEx (API)"
                use_vpn = True
                # TODO dont hardcode use config.py
                if use_vpn:
                    self.transfeatex_endpoint = 'http://10.4.63.10:3004/extract-features'
                    logger.info(f"Configured TransfeatEx VPN endpoint: {self.transfeatex_endpoint}")
                else:
                    self.transfeatex_endpoint = os.environ.get('TRANSFEATEX_URL',
                                                               'http://gessi-chatbots.essi.upc.edu:3004') + '/extract-features'
                    logger.info(f"Configured TransfeatEx original endpoint: {self.transfeatex_endpoint}")

            elif self.model_type == 'hybrid':
                self.model_name = "hybrid"
                logger.info("Loading hybrid model (T-FREX + TransfeatEx)")

                self.tokenizer = AutoTokenizer.from_pretrained("quim-motger/t-frex-bert-base-uncased")
                self.model = AutoModelForTokenClassification.from_pretrained("quim-motger/t-frex-bert-base-uncased")
                self.ner_pipeline = pipeline("ner", model=self.model, tokenizer=self.tokenizer)
                self.transfeatex_endpoint = 'http://10.4.63.10:3004/extract-features'
                logger.info("Hybrid model configuration complete")

            else:
                raise ValueError(f"Unsupported model type: {self.model_type}")

            logger.info("Loading embedding model...")
            embedding_name = Config.EMBEDDING_MODELS.get(
                self.embedding_model_key,
                Config.EMBEDDING_MODELS[Config.DEFAULT_EMBEDDING_MODEL]
            )
            logger.info(f"Loading embedding model: {embedding_name}")
            self.embedding_model = SentenceTransformer(embedding_name)
            logger.info("All models loaded successfully")

        except Exception as e:
            logger.error(f"Error initializing models: {str(e)}", exc_info=True)
            raise

    def extract_features(self, texts):
        if not texts:
            logger.warning("No texts provided for feature extraction")
            return []

        logger.info(f"Starting feature extraction for {len(texts)} texts using {self.model_type}")

        # Step 1: Raw feature extraction
        if self.model_type == 'hybrid':
            raw_features = self._extract_features_hybrid(texts)
        elif self.model_type == 'transfeatex':
            raw_features = self._extract_features_transfeatex(texts)
        else:
            raw_features = self._extract_features_tfrex(texts)

        raw_total = sum(len(features) for features in raw_features)
        raw_unique = len(set(f for features in raw_features for f in features))
        logger.info(f"Raw extraction complete: {raw_total} total features, {raw_unique} unique")

        # Step 2: Post-processing (if enabled)
        if self.enable_postprocessing and self.postprocessor:
            logger.info("Starting feature post-processing...")
            processed_features = self.postprocessor.process_features_list(raw_features)

            stats = self.postprocessor.get_stats(raw_features, processed_features)
            logger.info(f"Feature extraction with post-processing complete: "
                        f"{stats['original_unique']} -> {stats['processed_unique']} unique features "
                        f"({stats['reduction_ratio']}% reduction)")

            return processed_features

        logger.info("Feature extraction complete (no post-processing)")
        return raw_features

    def extract_features_raw(self, texts):
        """Extract features without post-processing (for debugging)"""
        logger.debug(f"Extracting raw features for {len(texts)} texts (no post-processing)")

        if self.model_type == 'hybrid':
            return self._extract_features_hybrid(texts)
        elif self.model_type == 'transfeatex':
            return self._extract_features_transfeatex(texts)
        else:
            return self._extract_features_tfrex(texts)

    def _extract_features_hybrid(self, texts):
        logger.info("Running hybrid extraction (T-FREX + TransfeatEx)")

        logger.debug("Extracting with T-FREX...")
        tfrex_features = self._extract_features_tfrex(texts)
        tfrex_total = sum(len(features) for features in tfrex_features)

        logger.debug("Extracting with TransfeatEx...")
        transfeatex_features = self._extract_features_transfeatex(texts)
        transfeatex_total = sum(len(features) for features in transfeatex_features)

        logger.info(f"Hybrid extraction: T-FREX={tfrex_total}, TransfeatEx={transfeatex_total} features")

        combined_features = []
        for i in range(len(texts)):
            t_feat = set(tfrex_features[i] if i < len(tfrex_features) else [])
            tf_feat = set(transfeatex_features[i] if i < len(transfeatex_features) else [])
            combined = list(t_feat.union(tf_feat))
            combined_features.append(combined)

        combined_total = sum(len(features) for features in combined_features)
        logger.info(f"Hybrid combination complete: {combined_total} total features")
        return combined_features

    def _extract_features_transfeatex(self, texts):
        logger.info(f"Starting TransfeatEx extraction for {len(texts)} texts")
        all_features = []

        def chunked(iterable, size):
            for i in range(0, len(iterable), size):
                yield iterable[i:i + size]

        total_batches = (len(texts) + self.batch_size - 1) // self.batch_size
        logger.debug(f"Processing in {total_batches} batches of {self.batch_size}")

        for batch_idx, batch in enumerate(chunked(texts, self.batch_size)):
            logger.debug(f"Processing batch {batch_idx + 1}/{total_batches} ({len(batch)} texts)")

            payload = {
                "text": [
                    {"id": f"input_{batch_idx * self.batch_size + i}", "text": text}
                    for i, text in enumerate(batch)
                ]
            }

            try:
                response = requests.post(self.transfeatex_endpoint, json=payload, timeout=30)

                if response.status_code == 200:
                    response_data = response.json()

                    if not isinstance(response_data, list):
                        logger.warning(f"Batch {batch_idx + 1}: TransfeatEx response is not a list: {response_data}")
                        continue

                    if len(response_data) != len(batch):
                        logger.warning(f"Batch {batch_idx + 1}: Response length mismatch "
                                       f"(got {len(response_data)}, expected {len(batch)})")
                        continue

                    batch_features = 0
                    for item in response_data:
                        features = item.get("features", [])
                        if isinstance(features, list):
                            all_features.extend(features)
                            batch_features += len(features)
                        else:
                            logger.warning(f"Batch {batch_idx + 1}: Malformed features in response item: {item}")

                    logger.debug(f"Batch {batch_idx + 1}: Extracted {batch_features} features")

                else:
                    logger.error(f"Batch {batch_idx + 1}: TransfeatEx returned status {response.status_code}")

            except requests.exceptions.Timeout:
                logger.error(f"Batch {batch_idx + 1}: TransfeatEx request timeout")
            except Exception as e:
                logger.error(f"Batch {batch_idx + 1}: Error contacting TransfeatEx: {str(e)}")

        logger.info(f"TransfeatEx extraction complete: {len(all_features)} total features")
        return all_features

    def _extract_features_tfrex(self, texts):
        logger.info(f"Starting T-FREX extraction for {len(texts)} texts")
        features_per_text = []
        total_features = 0
        processed_texts = 0

        for i, text in enumerate(texts):
            if not text or not isinstance(text, str):
                features_per_text.append([])
                logger.debug(f"Text {i}: Skipped (empty or invalid)")
                continue

            try:
                entities = self.ner_pipeline(text)
                logger.debug(f"Text {i}: NER pipeline found {len(entities)} entities")

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
                total_features += len(unique_features)
                processed_texts += 1

                if len(unique_features) != len(features):
                    logger.debug(f"Text {i}: Deduplicated {len(features)} -> {len(unique_features)} features")

            except Exception as e:
                logger.error(f"Text {i}: Error extracting features: {str(e)}")
                features_per_text.append([])

        logger.info(f"T-FREX extraction complete: {total_features} features from {processed_texts}/{len(texts)} texts")
        return features_per_text

    def extract_features_with_details(self, text):
        if self.model_type == 'rebert':
            logger.warning("Detailed extraction not supported for RE-BERT")
            return []

        if not text or not isinstance(text, str):
            logger.warning("Invalid text provided for detailed extraction")
            return []

        logger.debug(f"Extracting detailed features from text: '{text[:50]}...'")

        try:
            entities = self.ner_pipeline(text)
            logger.debug(f"Found {len(entities)} entities with details")

            detailed_features = []
            for entity in entities:
                detailed_features.append({
                    'text': entity['word'].strip(),
                    'label': entity['entity_group'],
                    'score': entity['score'],
                    'start': entity['start'],
                    'end': entity['end']
                })

            logger.debug(f"Detailed extraction complete: {len(detailed_features)} features")
            return detailed_features

        except Exception as e:
            logger.error(f"Error extracting detailed features: {str(e)}")
            return []

    def get_embeddings(self, texts):
        if not texts:
            logger.warning("No texts provided for embedding generation")
            return []

        logger.info(f"Generating embeddings for {len(texts)} texts")

        try:
            valid_texts = [text for text in texts if text and isinstance(text, str)]

            if not valid_texts:
                logger.warning("No valid texts found for embedding generation")
                return []

            if len(valid_texts) != len(texts):
                logger.debug(f"Filtered {len(texts) - len(valid_texts)} invalid texts")

            logger.debug("Encoding texts with sentence transformer...")
            embeddings = self.embedding_model.encode(valid_texts)

            logger.info(f"Embedding generation complete: {embeddings.shape} matrix")
            return embeddings

        except Exception as e:
            logger.error(f"Error generating embeddings: {str(e)}")
            return []

    def configure_postprocessing(self, **kwargs):
        """Configure post-processing parameters"""
        if self.postprocessor:
            logger.info(f"Configuring post-processor with parameters: {kwargs}")
            for key, value in kwargs.items():
                if hasattr(self.postprocessor, key):
                    old_value = getattr(self.postprocessor, key)
                    setattr(self.postprocessor, key, value)
                    logger.info(f"Updated post-processor {key}: {old_value} -> {value}")
                else:
                    logger.warning(f"Unknown post-processor parameter: {key}")
        else:
            logger.warning("Cannot configure post-processor: not enabled")

    def get_postprocessing_stats(self, texts):
        """Get detailed post-processing statistics"""
        if not self.postprocessor:
            logger.warning("Post-processor not enabled for stats")
            return {"error": "Post-processor not enabled"}

        logger.info("Generating post-processing statistics...")
        raw_features = self.extract_features_raw(texts)
        processed_features = self.postprocessor.process_features_list(raw_features)

        stats = self.postprocessor.get_stats(raw_features, processed_features)
        logger.info(f"Post-processing stats generated: {stats}")
        return stats

    def get_model_info(self):
        info = {
            'model_type': self.model_type,
            'model_name': self.model_name,
            'embedding_model': self.embedding_model_key,
            'device': 'cuda' if torch.cuda.is_available() else 'cpu',
            'model_loaded': (self.model is not None or self.re_extractor is not None),
            'tokenizer_loaded': self.tokenizer is not None,
            'postprocessing_enabled': self.enable_postprocessing,
            'postprocessor_config': {
                'min_length': self.postprocessor.min_length if self.postprocessor else None,
                'max_length': self.postprocessor.max_length if self.postprocessor else None,
                'similarity_threshold': self.postprocessor.similarity_threshold if self.postprocessor else None
            } if self.postprocessor else None
        }

        logger.debug(f"Model info requested: {info}")
        return info