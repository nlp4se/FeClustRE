import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from services.feature_extraction_service import FeatureExtractor
from services.preprocessing_service import ReviewPreprocessor
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_tfrex_extraction():
    sample_reviews = [
        "This is very very usefull app please try it",
        "Buggy (eg. notifications just don't work for me), there's no way to quit when you are done, and it uses a "
        "lot of the battery",
        "it's ok. discord is a narc, but for irrelevant banter is fine. annoying with how everything is nickeled and "
        "dimed out of you though",
        "it's gotten so many bugs overtime and discord doesn't do anything to fix it. I can't access the shop or "
        "change pfps without being kicked out of app.",
        "Overall it's good, but it would be better if there was an \"delete all\" button for deleting messages.",
        "Why we have to verify i hate it other games dont have it",
        "kinda works, why push notifications work sometimes, then randomly stop working."
    ]

    try:
        logger.info("Initializing services...")
        preprocessor = ReviewPreprocessor()
        feature_extractor = FeatureExtractor()

        model_info = feature_extractor.get_model_info()
        logger.info(f"Model info: {model_info}")

        logger.info("Preprocessing reviews...")
        processed_reviews = []
        for review in sample_reviews:
            processed = preprocessor.clean_text(review)
            processed_reviews.append(processed)
            logger.info(f"Original: {review}")
            logger.info(f"Processed: {processed}")
            print("-" * 50)

        # Extract features
        logger.info("Extracting features using T-FREX...")
        features_per_review = feature_extractor.extract_features(processed_reviews)

        # Display results
        logger.info("\n" + "=" * 80)
        logger.info("FEATURE EXTRACTION RESULTS")
        logger.info("=" * 80)

        all_features = []
        for i, (original, processed, features) in enumerate(
                zip(sample_reviews, processed_reviews, features_per_review)):
            logger.info(f"\nReview {i + 1}:")
            logger.info(f"Original: {original}")
            logger.info(f"Processed: {processed}")
            logger.info(f"Features: {features}")
            all_features.extend(features)

            # Get detailed features for first few reviews
            if i < 3:
                detailed_features = feature_extractor.extract_features_with_details(processed)
                logger.info(f"Detailed features: {detailed_features}")

            print("-" * 50)

        # Feature statistics
        from collections import Counter
        feature_counts = Counter(all_features)

        logger.info(f"\nFeature Statistics:")
        logger.info(f"Total features extracted: {len(all_features)}")
        logger.info(f"Unique features: {len(feature_counts)}")
        logger.info(f"Average features per review: {len(all_features) / len(sample_reviews):.2f}")

        logger.info(f"\nTop 10 most common features:")
        for feature, count in feature_counts.most_common(10):
            logger.info(f"  {feature}: {count}")

        # Test embeddings
        logger.info("\nTesting embeddings...")
        unique_features = list(set(all_features))
        if unique_features:
            embeddings = feature_extractor.get_embeddings(unique_features)
            logger.info(f"Generated embeddings for {len(unique_features)} unique features")
            logger.info(f"Embedding shape: {embeddings.shape if hasattr(embeddings, 'shape') else 'N/A'}")

        logger.info("\nTest completed successfully!")
        return True

    except Exception as e:
        logger.error(f"Error during testing: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = test_tfrex_extraction()
    if success:
        print("\nT-FREX test passed")
    else:
        print("\nT-FREX test failed")
        sys.exit(1)
