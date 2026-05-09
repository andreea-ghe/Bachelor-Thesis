"""
Tests for the FastAPI inference server.

Covers:
  1. Health and model listing endpoints
  2. Input validation on the /predict endpoint
  3. Successful inference with mocked model
"""

import sys
from unittest.mock import MagicMock, patch

from fastapi.testclient import TestClient

from .server import app

client = TestClient(app)


# ---------------------------------------------------------------------------
# 1. Health and model listing
# ---------------------------------------------------------------------------

class TestHealthEndpoint:
    def _mock_torch(self):
        """Create a minimal torch mock for environments without GPU."""
        mock_torch = MagicMock()
        mock_torch.cuda.is_available.return_value = False
        mock_torch.cuda.get_device_name.return_value = None
        return mock_torch

    def test_health_returns_200(self):
        with patch.dict(sys.modules, {"torch": self._mock_torch()}):
            response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"
        assert "cuda_available" in data

    def test_health_includes_gpu_info(self):
        with patch.dict(sys.modules, {"torch": self._mock_torch()}):
            response = client.get("/health")
        data = response.json()
        assert "gpu_name" in data


class TestModelsEndpoint:
    def test_models_returns_200(self):
        response = client.get("/models")
        assert response.status_code == 200

    def test_models_returns_all_variants(self):
        response = client.get("/models")
        models = response.json()["models"]
        assert len(models) == 6

    def test_each_model_has_required_fields(self):
        response = client.get("/models")
        for model in response.json()["models"]:
            assert "id" in model
            assert "description" in model
            assert "max_pieces" in model

    def test_model_ids_are_unique(self):
        response = client.get("/models")
        ids = [m["id"] for m in response.json()["models"]]
        assert len(ids) == len(set(ids))

    def test_max_pieces_values(self):
        response = client.get("/models")
        for model in response.json()["models"]:
            if "two_piece" in model["id"]:
                assert model["max_pieces"] == 2
            else:
                assert model["max_pieces"] == 4


# ---------------------------------------------------------------------------
# 2. Input validation on /predict
# ---------------------------------------------------------------------------

def _make_obj_file(name="piece_0.obj", content="v 0 0 0\nv 1 0 0\nv 0 1 0\nf 1 2 3"):
    return (name, content.encode("utf-8"), "text/plain")


class TestPredictValidation:
    def test_reject_fewer_than_2_pieces(self):
        response = client.post(
            "/predict",
            data={"model": "two_piece_baseline"},
            files=[("pieces", _make_obj_file("piece_0.obj"))],
        )
        assert response.status_code == 400
        assert "At least 2" in response.json()["detail"]

    def test_reject_too_many_pieces_for_two_piece_model(self):
        files = [("pieces", _make_obj_file(f"piece_{i}.obj")) for i in range(3)]
        response = client.post(
            "/predict",
            data={"model": "two_piece_baseline"},
            files=files,
        )
        assert response.status_code == 400
        assert "at most 2" in response.json()["detail"]

    def test_reject_too_many_pieces_for_multi_piece_model(self):
        files = [("pieces", _make_obj_file(f"piece_{i}.obj")) for i in range(5)]
        response = client.post(
            "/predict",
            data={"model": "multi_piece_baseline"},
            files=files,
        )
        assert response.status_code == 400
        assert "at most 4" in response.json()["detail"]

    def test_reject_non_utf8_file(self):
        bad_file = ("bad.obj", b"\x80\x81\x82\xff", "text/plain")
        good_file = _make_obj_file("piece_0.obj")
        response = client.post(
            "/predict",
            data={"model": "two_piece_baseline"},
            files=[("pieces", good_file), ("pieces", bad_file)],
        )
        assert response.status_code == 400
        assert "not valid UTF-8" in response.json()["detail"]

    def test_reject_invalid_model_variant(self):
        files = [("pieces", _make_obj_file(f"piece_{i}.obj")) for i in range(2)]
        response = client.post(
            "/predict",
            data={"model": "nonexistent_model"},
            files=files,
        )
        assert response.status_code == 422


# ---------------------------------------------------------------------------
# 3. Successful inference (mocked)
# ---------------------------------------------------------------------------

class TestPredictSuccess:
    def test_successful_prediction(self):
        mock_run_inference = MagicMock(return_value={
            "pieces": [
                {"name": "piece_0", "assembled_obj": "v 0 0 0\nf 1"},
                {"name": "piece_1", "assembled_obj": "v 1 1 1\nf 1"},
            ],
            "combined_assembled_obj": "v 0 0 0\nv 1 1 1\nf 1 2",
        })

        mock_module = MagicMock()
        mock_module.run_inference = mock_run_inference

        with patch.dict(sys.modules, {"application.inference_server.run_inference": mock_module}), \
             patch("os.path.exists", return_value=True):
            files = [("pieces", _make_obj_file(f"piece_{i}.obj")) for i in range(2)]
            response = client.post(
                "/predict",
                data={"model": "two_piece_baseline"},
                files=files,
            )

        assert response.status_code == 200
        data = response.json()
        assert data["model"] == "two_piece_baseline"
        assert data["num_pieces"] == 2
        assert len(data["pieces"]) == 2
        assert "combined_assembled_obj" in data
