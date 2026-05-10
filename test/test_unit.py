"""Unit tests for CSV parsing, preprocessing, post-processing, and clustering."""
import sys
import os
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ---------------------------------------------------------------------------
# CSV parsing
# ---------------------------------------------------------------------------

class TestParseCsvData(unittest.TestCase):
    def setUp(self):
        # Import lazily so the module doesn't trigger Flask/service init at
        # collection time when those packages may be absent.
        import importlib
        self.app_module = importlib.import_module("app")
        self._parse = self.app_module._parse_csv_data

    def test_valid_csv_returns_apps(self):
        csv = "app_name,app_package,app_categoryId,reviewId,review,score\n" \
              "MyApp,com.my,Tools,r1,Great app,5\n" \
              "MyApp,com.my,Tools,r2,Needs work,3\n"
        result = self._parse(csv)
        self.assertIn("MyApp", result)
        self.assertEqual(len(result["MyApp"]["reviews"]), 2)

    def test_empty_csv_raises(self):
        with self.assertRaises(ValueError):
            self._parse("app_name,review\n")

    def test_rows_with_empty_review_are_filtered(self):
        csv = "app_name,review\nMyApp,Good app\nMyApp,\nMyApp,   \n"
        result = self._parse(csv)
        self.assertEqual(len(result["MyApp"]["reviews"]), 1)

    def test_rows_with_missing_app_name_are_filtered(self):
        csv = "app_name,review\nMyApp,Good app\n,No name\n"
        result = self._parse(csv)
        self.assertEqual(len(result["MyApp"]["reviews"]), 1)

    def test_multiple_apps_are_separated(self):
        csv = "app_name,review\nApp1,Hello\nApp2,World\n"
        result = self._parse(csv)
        self.assertIn("App1", result)
        self.assertIn("App2", result)

    def test_all_rows_filtered_returns_empty_apps(self):
        csv = "app_name,review\n,\n,\n"
        result = self._parse(csv)
        self.assertEqual(result, {})


# ---------------------------------------------------------------------------
# Preprocessing
# ---------------------------------------------------------------------------

class TestReviewPreprocessor(unittest.TestCase):
    def setUp(self):
        from services.preprocessing_service import ReviewPreprocessor
        self.pp = ReviewPreprocessor()

    def test_none_input_returns_empty_string(self):
        self.assertEqual(self.pp.clean_text(None), "")

    def test_non_string_input_returns_empty_string(self):
        self.assertEqual(self.pp.clean_text(123), "")
        self.assertEqual(self.pp.clean_text([]), "")

    def test_empty_string_returns_empty(self):
        self.assertEqual(self.pp.clean_text(""), "")

    def test_urls_are_removed(self):
        result = self.pp.clean_text("Visit http://example.com for more info")
        self.assertNotIn("http", result)
        self.assertIn("for more info", result)

    def test_emojis_are_stripped(self):
        result = self.pp.clean_text("Great app 😀🎉")
        self.assertNotIn("😀", result)
        self.assertNotIn("🎉", result)

    def test_lowercase_conversion(self):
        result = self.pp.clean_text("This Is Mixed Case")
        self.assertEqual(result, result.lower())

    def test_preprocess_empty_string_returns_empty(self):
        self.assertEqual(self.pp.preprocess_text(""), "")

    def test_preprocess_whitespace_only_returns_empty(self):
        self.assertEqual(self.pp.preprocess_text("   "), "")

    def test_preprocess_normal_text_returns_tokens(self):
        result = self.pp.preprocess_text("The app crashes a lot")
        self.assertTrue(len(result) > 0)


# ---------------------------------------------------------------------------
# Feature post-processing
# ---------------------------------------------------------------------------

class TestFeaturePostProcessor(unittest.TestCase):
    def setUp(self):
        from services.feature_post_processor import FeaturePostProcessor
        self.fpp = FeaturePostProcessor()

    def test_none_input_is_rejected(self):
        self.assertIsNone(self.fpp.clean_feature(None))

    def test_empty_string_is_rejected(self):
        self.assertIsNone(self.fpp.clean_feature(""))

    def test_non_string_is_rejected(self):
        self.assertIsNone(self.fpp.clean_feature(42))

    def test_too_short_feature_is_rejected(self):
        # min_length default is 3
        self.assertIsNone(self.fpp.clean_feature("ab"))

    def test_numeric_only_is_rejected(self):
        self.assertIsNone(self.fpp.clean_feature("1234"))

    def test_stopword_only_is_rejected(self):
        # single-word stopwords like "the", "is", "in" should be rejected
        self.assertIsNone(self.fpp.clean_feature("the"))

    def test_valid_feature_is_returned(self):
        result = self.fpp.clean_feature("video call")
        self.assertEqual(result, "video call")

    def test_extra_whitespace_is_normalised(self):
        result = self.fpp.clean_feature("video   call")
        self.assertEqual(result, "video call")

    def test_clean_feature_is_deterministic(self):
        feature = "Push Notifications"
        self.assertEqual(self.fpp.clean_feature(feature), self.fpp.clean_feature(feature))

    def test_merge_similar_features_returns_list(self):
        features = ["video call", "video call", "push notification"]
        result = self.fpp.merge_similar_features(features)
        # identical strings may be re-expanded by count; result is still a list
        self.assertIsInstance(result, list)
        self.assertGreater(len(result), 0)

    def test_merge_similar_features_empty_input(self):
        self.assertEqual(self.fpp.merge_similar_features([]), [])


# ---------------------------------------------------------------------------
# Clustering edge cases
# ---------------------------------------------------------------------------

class TestHierarchicalClusterer(unittest.TestCase):
    def setUp(self):
        from services.clustering_service import HierarchicalClusterer
        self.clusterer = HierarchicalClusterer()

    def test_empty_features_returns_empty(self):
        result = self.clusterer.perform_clustering([], [])
        self.assertEqual(result["n_clusters"], 0)
        self.assertEqual(result["clusters"], [])

    def test_single_feature_returns_empty(self):
        import numpy as np
        result = self.clusterer.perform_clustering(
            ["login"],
            np.array([[0.1, 0.2, 0.3]])
        )
        self.assertEqual(result["n_clusters"], 0)

    def test_two_features_clusters_successfully(self):
        import numpy as np
        embeddings = np.array([[1.0, 0.0], [0.9, 0.1]])
        result = self.clusterer.perform_clustering(["login", "sign in"], embeddings)
        self.assertGreaterEqual(result["n_clusters"], 1)

    def test_get_optimal_handles_too_few_features(self):
        import numpy as np
        embeddings = np.array([[1.0, 0.0]])
        result = self.clusterer.get_optimal_clusters(embeddings, max_clusters=10)
        # returns int (optimal n_clusters); falls back to 1 when not enough data
        self.assertIsInstance(result, int)
        self.assertEqual(result, 1)


if __name__ == "__main__":
    unittest.main()
