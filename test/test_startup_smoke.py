import importlib
import importlib.util
import unittest


@unittest.skipIf(importlib.util.find_spec("flask") is None, "Flask is not installed")
class AppStartupSmokeTests(unittest.TestCase):
    def setUp(self):
        self.app_module = importlib.import_module("app")
        self.app_module._services.clear()

    def tearDown(self):
        self.app_module._services.clear()

    def test_create_app_does_not_initialize_services(self):
        flask_app = self.app_module.create_app()

        self.assertEqual(flask_app.config["NEO4J_URI"], "bolt://localhost:7687")
        self.assertEqual(self.app_module._services, {})

    def test_ping_does_not_initialize_services(self):
        client = self.app_module.app.test_client()

        response = client.get("/ping")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.data, b"pong")
        self.assertEqual(self.app_module._services, {})


class FeatureExtractorContractTests(unittest.TestCase):
    def test_model_type_aliases_are_normalized(self):
        from services.feature_extraction_service import FeatureExtractor

        cases = {
            "tfrex": "t-frex",
            "t_frex": "t-frex",
            "t-frex": "t-frex",
            "transfeat-ex": "transfeatex",
            "transfeat_ex": "transfeatex",
            "transfeatex": "transfeatex",
            "hybrid": "hybrid",
        }

        for raw, expected in cases.items():
            with self.subTest(raw=raw):
                self.assertEqual(FeatureExtractor._normalize_model_type(raw), expected)


if __name__ == "__main__":
    unittest.main()
