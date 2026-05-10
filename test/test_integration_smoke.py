"""Integration smoke tests: /health structure and CSV pipeline path."""
import importlib
import importlib.util
import json
import unittest
from unittest.mock import MagicMock, patch

_flask_available = importlib.util.find_spec("flask") is not None


@unittest.skipIf(not _flask_available, "Flask is not installed")
class HealthEndpointSmokeTests(unittest.TestCase):
    """
    /health must return structured JSON with a top-level 'status' field and
    per-service entries even when all dependencies are down.
    """

    def setUp(self):
        self.app_module = importlib.import_module("app")
        self.app_module._services.clear()
        self.client = self.app_module.app.test_client()

    def tearDown(self):
        self.app_module._services.clear()

    def _mock_unhealthy_services(self):
        unhealthy = {"status": "unhealthy", "error": "unavailable"}
        patcher_neo4j = patch("app.get_neo4j_connection", side_effect=Exception("no neo4j"))
        patcher_extractor = patch("app.get_default_feature_extractor", side_effect=Exception("no model"))
        patcher_nltk = patch("app.check_nltk_data", return_value=unhealthy)
        patcher_ollama = patch("app.check_ollama", return_value=unhealthy)
        patcher_transfeatex = patch("app.check_transfeatex", return_value=unhealthy)
        return [patcher_neo4j, patcher_extractor, patcher_nltk, patcher_ollama, patcher_transfeatex]

    def test_health_returns_json(self):
        patchers = self._mock_unhealthy_services()
        for p in patchers:
            p.start()
        try:
            response = self.client.get("/health")
            self.assertEqual(response.content_type, "application/json")
        finally:
            for p in patchers:
                p.stop()

    def test_health_has_top_level_status_field(self):
        patchers = self._mock_unhealthy_services()
        for p in patchers:
            p.start()
        try:
            response = self.client.get("/health")
            body = json.loads(response.data)
            self.assertIn("status", body)
            self.assertIn(body["status"], ("healthy", "unhealthy"))
        finally:
            for p in patchers:
                p.stop()

    def test_health_returns_503_when_unhealthy(self):
        patchers = self._mock_unhealthy_services()
        for p in patchers:
            p.start()
        try:
            response = self.client.get("/health")
            self.assertEqual(response.status_code, 503)
        finally:
            for p in patchers:
                p.stop()

    def test_health_has_services_section(self):
        patchers = self._mock_unhealthy_services()
        for p in patchers:
            p.start()
        try:
            response = self.client.get("/health")
            body = json.loads(response.data)
            self.assertIn("services", body)
            services = body["services"]
            self.assertIn("nltk", services)
            self.assertIn("ollama", services)
        finally:
            for p in patchers:
                p.stop()

    def test_health_has_models_section(self):
        patchers = self._mock_unhealthy_services()
        for p in patchers:
            p.start()
        try:
            response = self.client.get("/health")
            body = json.loads(response.data)
            self.assertIn("models", body)
            self.assertIn("transfeatex", body["models"])
        finally:
            for p in patchers:
                p.stop()


@unittest.skipIf(not _flask_available, "Flask is not installed")
class CsvUploadSmokeTests(unittest.TestCase):
    """
    A tiny CSV uploaded to /process_reviews/upload should either return a
    valid clustering response or a clear error — it must not crash unhandled.
    """

    TINY_CSV = (
        "app_name,app_package,app_categoryId,reviewId,review,score\n"
        "TestApp,com.test,Tools,r1,Great video call feature,5\n"
        "TestApp,com.test,Tools,r2,Push notifications are broken,2\n"
        "TestApp,com.test,Tools,r3,Login is too slow,3\n"
    )

    def setUp(self):
        self.app_module = importlib.import_module("app")
        self.app_module._services.clear()
        self.client = self.app_module.app.test_client()

    def tearDown(self):
        self.app_module._services.clear()

    def test_upload_with_no_file_returns_400(self):
        response = self.client.post("/process_reviews/upload")
        self.assertEqual(response.status_code, 400)
        body = json.loads(response.data)
        self.assertIn("error", body)

    def test_upload_non_csv_returns_400(self):
        data = {"file": (b"not a csv", "data.txt", "text/plain")}
        response = self.client.post(
            "/process_reviews/upload",
            data=data,
            content_type="multipart/form-data"
        )
        self.assertEqual(response.status_code, 400)

    def test_upload_empty_csv_returns_error(self):
        data = {"file": (b"app_name,review\n", "data.csv", "text/csv")}
        response = self.client.post(
            "/process_reviews/upload",
            data=data,
            content_type="multipart/form-data"
        )
        # Expect 4xx or 5xx with an error body — not an unhandled crash
        self.assertGreaterEqual(response.status_code, 400)
        body = json.loads(response.data)
        self.assertIn("error", body)

    def test_upload_tiny_csv_with_mocked_extractor_returns_response(self):
        mock_features = [["video call"], ["notifications"], ["login"]]

        mock_extractor = MagicMock()
        mock_extractor.extract_features.return_value = mock_features
        mock_extractor.get_embeddings.return_value = __import__("numpy").array([
            [1.0, 0.0, 0.0],
            [0.9, 0.1, 0.0],
            [0.0, 0.0, 1.0],
        ])

        mock_neo4j = MagicMock()
        mock_session = MagicMock()
        mock_neo4j.driver.session.return_value.__enter__ = MagicMock(return_value=mock_session)
        mock_neo4j.driver.session.return_value.__exit__ = MagicMock(return_value=False)
        mock_session.run.return_value = MagicMock()
        mock_neo4j.database = "neo4j"

        with patch("app.get_default_feature_extractor", return_value=mock_extractor), \
             patch("app.get_neo4j_connection", return_value=mock_neo4j):
            csv_bytes = self.TINY_CSV.encode("utf-8")
            data = {"file": (csv_bytes, "reviews.csv", "text/csv")}
            response = self.client.post(
                "/process_reviews/upload",
                data=data,
                content_type="multipart/form-data"
            )

        # Regardless of internal outcome, the endpoint must return JSON
        # and not a 500 unhandled exception
        self.assertNotEqual(response.status_code, 500)
        body = json.loads(response.data)
        self.assertIsInstance(body, dict)


if __name__ == "__main__":
    unittest.main()
