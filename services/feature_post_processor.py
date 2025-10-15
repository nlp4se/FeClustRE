import re
import string
from collections import Counter
from difflib import SequenceMatcher
import nltk
from nltk.corpus import stopwords
import logging

logger = logging.getLogger(__name__)


class FeaturePostProcessor:
    def __init__(self,
                 min_length=3,
                 max_length=50,
                 similarity_threshold=0.85):

        self.min_length = min_length
        self.max_length = max_length
        self.similarity_threshold = similarity_threshold

        try:
            self.stopwords = set(stopwords.words('english'))
            logger.info(f"Loaded {len(self.stopwords)} English stopwords")
        except LookupError:
            logger.warning("NLTK stopwords not found, downloading...")
            nltk.download('stopwords')
            self.stopwords = set(stopwords.words('english'))
            logger.info(f"Downloaded and loaded {len(self.stopwords)} English stopwords")

        logger.info(f"FeaturePostProcessor initialized: min_length={min_length}, "
                    f"max_length={max_length}, similarity_threshold={similarity_threshold}")

    def clean_feature(self, feature):
        if not feature or not isinstance(feature, str):
            return None

        original_feature = feature
        feature = feature.strip().lower()
        feature = ' '.join(feature.split())
        feature = feature.strip(string.punctuation.replace('-', '').replace('_', ''))

        if len(feature) < self.min_length or len(feature) > self.max_length:
            logger.debug(f"Filtered feature by length: '{original_feature}' -> rejected (length: {len(feature)})")
            return None

        if ' ' not in feature and feature in self.stopwords:
            logger.debug(f"Filtered stopword: '{original_feature}' -> rejected (stopword)")
            return None

        if re.match(r'^[0-9\s\-_.]+$', feature):
            logger.debug(f"Filtered numeric feature: '{original_feature}' -> rejected (numeric)")
            return None

        if not re.search(r'[a-zA-Z]', feature):
            logger.debug(f"Filtered non-alphabetic feature: '{original_feature}' -> rejected (no letters)")
            return None

        if feature != original_feature.strip().lower():
            logger.debug(f"Cleaned feature: '{original_feature}' -> '{feature}'")

        return feature

    def merge_similar_features(self, features):
        if not features:
            logger.debug("No features to merge")
            return []

        logger.debug(f"Starting feature merging with {len(features)} total features")

        feature_counts = Counter(features)
        unique_features = list(feature_counts.keys())
        logger.debug(f"Found {len(unique_features)} unique features before merging")

        groups = []
        used = set()
        merge_count = 0

        for feature in unique_features:
            if feature in used:
                continue

            group = [feature]
            used.add(feature)

            for other_feature in unique_features:
                if other_feature in used:
                    continue

                if self._are_similar(feature, other_feature):
                    group.append(other_feature)
                    used.add(other_feature)
                    merge_count += 1
                    logger.debug(f"Merged similar features: '{feature}' <-> '{other_feature}'")

            groups.append(group)

        logger.info(f"Feature merging complete: {merge_count} merges performed, "
                    f"{len(unique_features)} -> {len(groups)} unique features")

        result = []
        for group in groups:
            group_with_counts = [(f, feature_counts[f]) for f in group]
            group_with_counts.sort(key=lambda x: (-x[1], len(x[0])))

            representative = group_with_counts[0][0]
            total_count = sum(feature_counts[f] for f in group)

            if len(group) > 1:
                logger.debug(f"Group representative: '{representative}' represents {group} "
                             f"(total occurrences: {total_count})")

            result.extend([representative] * total_count)

        return result

    def _are_similar(self, feature1, feature2):
        if feature1 == feature2:
            return True

        if (feature1 + 's' == feature2) or (feature2 + 's' == feature1):
            logger.debug(f"Plural match: '{feature1}' <-> '{feature2}'")
            return True

        if (feature1 + 'es' == feature2) or (feature2 + 'es' == feature1):
            logger.debug(f"Plural -es match: '{feature1}' <-> '{feature2}'")
            return True

        suffixes = ['ing', 'ed', 'er', 'ly']
        for suffix in suffixes:
            if feature1.endswith(suffix) and feature1[:-len(suffix)] == feature2:
                logger.debug(f"Suffix match ({suffix}): '{feature1}' <-> '{feature2}'")
                return True
            if feature2.endswith(suffix) and feature2[:-len(suffix)] == feature1:
                logger.debug(f"Suffix match ({suffix}): '{feature1}' <-> '{feature2}'")
                return True

        similarity = SequenceMatcher(None, feature1, feature2).ratio()
        if similarity >= self.similarity_threshold:
            logger.debug(f"High similarity ({similarity:.3f}): '{feature1}' <-> '{feature2}'")
            return True

        return False

    def process_features_list(self, features_per_review):
        if not features_per_review:
            logger.warning("No features to process")
            return []

        logger.info(f"Starting feature post-processing for {len(features_per_review)} reviews")

        original_total = sum(len(features) for features in features_per_review)
        logger.debug(f"Total features before processing: {original_total}")

        cleaned_per_review = []
        total_cleaned = 0

        for i, review_features in enumerate(features_per_review):
            cleaned = []
            for feature in review_features:
                clean_feature = self.clean_feature(feature)
                if clean_feature:
                    cleaned.append(clean_feature)

            cleaned_per_review.append(cleaned)
            total_cleaned += len(cleaned)

            if len(cleaned) != len(review_features):
                logger.debug(f"Review {i}: {len(review_features)} -> {len(cleaned)} features after cleaning")

        logger.info(f"Cleaning complete: {original_total} -> {total_cleaned} features "
                    f"({original_total - total_cleaned} removed)")

        all_features = []
        for features in cleaned_per_review:
            all_features.extend(features)

        logger.debug(f"Flattened {total_cleaned} features for global merging")
        merged_features = self.merge_similar_features(all_features)

        feature_mapping = {}
        merged_counts = Counter(merged_features)
        original_counts = Counter(all_features)

        mapping_count = 0
        for original_feature in original_counts:
            for merged_feature in merged_counts:
                if (original_feature == merged_feature or
                        self._are_similar(original_feature, merged_feature)):
                    if original_feature != merged_feature:
                        mapping_count += 1
                        logger.debug(f"Feature mapping: '{original_feature}' -> '{merged_feature}'")
                    feature_mapping[original_feature] = merged_feature
                    break

        logger.info(f"Created {mapping_count} feature mappings")

        processed_per_review = []
        total_deduped = 0

        for i, review_features in enumerate(cleaned_per_review):
            processed = []
            for feature in review_features:
                mapped_feature = feature_mapping.get(feature, feature)
                if mapped_feature not in processed:
                    processed.append(mapped_feature)

            processed_per_review.append(processed)
            total_deduped += len(processed)

            if len(processed) != len(review_features):
                logger.debug(f"Review {i}: {len(review_features)} -> {len(processed)} features after deduplication")

        logger.info(f"Post-processing complete: {original_total} -> {total_deduped} final features")
        return processed_per_review

    def get_stats(self, original_features, processed_features):
        orig_flat = [f for review in original_features for f in review]
        proc_flat = [f for review in processed_features for f in review]

        orig_unique = len(set(orig_flat))
        proc_unique = len(set(proc_flat))

        stats = {
            'original_total': len(orig_flat),
            'original_unique': orig_unique,
            'processed_total': len(proc_flat),
            'processed_unique': proc_unique,
            'reduction_ratio': round((orig_unique - proc_unique) / orig_unique * 100, 1) if orig_unique > 0 else 0,
            'avg_per_review_before': round(len(orig_flat) / len(original_features), 1) if original_features else 0,
            'avg_per_review_after': round(len(proc_flat) / len(processed_features), 1) if processed_features else 0
        }

        logger.info(f"Processing statistics: {stats['original_unique']} -> {stats['processed_unique']} unique features "
                    f"({stats['reduction_ratio']}% reduction)")

        return stats