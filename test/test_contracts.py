"""Contract tests: output shape and clear errors from external services."""
import sys
import os
import unittest
from unittest.mock import MagicMock, patch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _make_extractor_with_mock_pipeline(mock_ner_output):
    """Instantiate FeatureExtractor for t-frex with a mocked NER pipeline."""
    from services.feature_extraction_service import FeatureExtractor

    with patch("services.feature_extraction_service.AutoTokenizer"), \
         patch("services.feature_extraction_service.AutoModelForTokenClassification"), \
         patch("services.feature_extraction_service.pipeline"), \
         patch("services.feature_extraction_service.SentenceTransformer"):
        extractor = FeatureExtractor(model_type="t-frex", enable_postprocessing=False)

    extractor.ner_pipeline = MagicMock(return_value=mock_ner_output)
    return extractor


class TestExtractorOutputContract(unittest.TestCase):
    """Every extractor mode must return list[list[str]] from extract_features."""

    def _assert_contract(self, result, n_texts):
        self.assertIsInstance(result, list, "extract_features must return a list")
        self.assertEqual(len(result), n_texts, "one inner list per input text")
        for inner in result:
            self.assertIsInstance(inner, list, "each element must be a list")
            for item in inner:
                self.assertIsInstance(item, str, "features must be strings")

    # --- t-frex mode ---------------------------------------------------------

    def test_tfrex_returns_list_of_lists_with_features(self):
        mock_entities = [
            {"word": "video call", "score": 0.9},
            {"word": "notifications", "score": 0.8},
        ]
        extractor = _make_extractor_with_mock_pipeline(mock_entities)
        result = extractor.extract_features(["Great video call feature"])
        self._assert_contract(result, 1)
        self.assertIn("video call", result[0])

    def test_tfrex_empty_input_returns_empty_list(self):
        extractor = _make_extractor_with_mock_pipeline([])
        result = extractor.extract_features([])
        self.assertEqual(result, [])

    def test_tfrex_low_confidence_entities_are_excluded(self):
        mock_entities = [{"word": "login", "score": 0.3}]
        extractor = _make_extractor_with_mock_pipeline(mock_entities)
        result = extractor.extract_features(["login screen"])
        self._assert_contract(result, 1)
        self.assertEqual(result[0], [], "entities below 0.5 threshold must be dropped")

    def test_tfrex_empty_text_produces_empty_inner_list(self):
        extractor = _make_extractor_with_mock_pipeline([])
        result = extractor.extract_features(["", None])
        self._assert_contract(result, 2)
        for inner in result:
            self.assertEqual(inner, [])

    def test_tfrex_multiple_texts_produces_one_list_per_text(self):
        extractor = _make_extractor_with_mock_pipeline([{"word": "search", "score": 0.95}])
        result = extractor.extract_features(["text one", "text two", "text three"])
        self._assert_contract(result, 3)

    # --- transfeatex mode (missing URL should fail clearly) ------------------

    def test_transfeatex_missing_url_raises_value_error(self):
        from services.feature_extraction_service import FeatureExtractor

        with patch("services.feature_extraction_service.SentenceTransformer"), \
             patch("config.Config") as mock_cfg:
            mock_cfg.TRANSFEATEX_URL = None
            mock_cfg.EMBEDDING_MODELS = {"allmini": "all-MiniLM-L6-v2"}
            mock_cfg.DEFAULT_EMBEDDING_MODEL = "allmini"

            with self.assertRaises((ValueError, Exception)):
                FeatureExtractor(model_type="transfeatex", enable_postprocessing=False)


class TestModelAliasContract(unittest.TestCase):
    """Alias normalisation must be stable regardless of input casing/separator."""

    def _normalise(self, alias):
        from services.feature_extraction_service import FeatureExtractor
        return FeatureExtractor._normalize_model_type(alias)

    def test_tfrex_variants_all_resolve_to_canonical(self):
        for alias in ("tfrex", "t_frex", "t-frex", "T-FREX", "T_FREX"):
            self.assertEqual(self._normalise(alias), "t-frex", f"failed for {alias!r}")

    def test_transfeatex_variants_all_resolve_to_canonical(self):
        for alias in ("transfeatex", "transfeat-ex", "transfeat_ex"):
            self.assertEqual(self._normalise(alias), "transfeatex", f"failed for {alias!r}")

    def test_none_defaults_to_tfrex(self):
        self.assertEqual(self._normalise(None), "t-frex")

    def test_unknown_alias_is_passed_through(self):
        self.assertEqual(self._normalise("hybrid"), "hybrid")


class TestHealthCheckContract(unittest.TestCase):
    """Health checks must return a dict with a 'status' key and never raise."""

    def test_transfeatex_missing_url_reports_not_configured(self):
        from utils.health_checks import check_transfeatex
        from unittest.mock import patch

        with patch("utils.health_checks.Config") as mock_cfg:
            mock_cfg.TRANSFEATEX_URL = None
            result = check_transfeatex()

        self.assertIsInstance(result, dict)
        self.assertIn("status", result)
        self.assertEqual(result["status"], "not_configured")

    def test_transfeatex_connection_refused_reports_unhealthy(self):
        from utils.health_checks import check_transfeatex

        with patch("utils.health_checks.Config") as mock_cfg:
            mock_cfg.TRANSFEATEX_URL = "http://localhost:19999"
            result = check_transfeatex()

        self.assertIsInstance(result, dict)
        self.assertEqual(result["status"], "unhealthy")

    def test_nltk_data_check_returns_status_dict(self):
        from utils.health_checks import check_nltk_data
        result = check_nltk_data()
        self.assertIsInstance(result, dict)
        self.assertIn("status", result)


if __name__ == "__main__":
    unittest.main()
