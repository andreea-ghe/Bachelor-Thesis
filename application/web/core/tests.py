"""Tests for the Django web application."""

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

from django.test import Client, TestCase, override_settings

from .fractures import Fracture, _make_id


class FractureModelTest(TestCase):
    """Tests for the Fracture dataclass and helper functions."""

    def test_make_id_deterministic(self):
        """Same inputs must always produce the same ID."""
        id1 = _make_id("two_pieces", "BeerBottle", "abc123", "fractured_0")
        id2 = _make_id("two_pieces", "BeerBottle", "abc123", "fractured_0")
        self.assertEqual(id1, id2)

    def test_make_id_differs_for_different_inputs(self):
        id1 = _make_id("two_pieces", "BeerBottle", "abc123", "fractured_0")
        id2 = _make_id("two_pieces", "BeerBottle", "abc123", "fractured_1")
        self.assertNotEqual(id1, id2)

    def test_make_id_length(self):
        fid = _make_id("two_pieces", "BeerBottle", "abc123", "fractured_0")
        self.assertEqual(len(fid), 12)

    def test_display_name_format(self):
        f = Fracture(
            id="test123",
            category="BeerBottle",
            mesh_hash="abc",
            fracture="fractured_0",
            num_pieces=3,
            piece_type="multi_pieces",
        )
        self.assertEqual(f.display_name, "BeerBottle — fractured_0 (3 pieces)")

    def test_static_prefix_format(self):
        f = Fracture(
            id="test123",
            category="BeerBottle",
            mesh_hash="abc",
            fracture="fractured_0",
            num_pieces=2,
            piece_type="two_pieces",
        )
        self.assertEqual(
            f.static_prefix,
            "core/meshes/two_pieces/BeerBottle/abc/fractured_0",
        )


def _make_fake_fracture(**overrides):
    """Create a Fracture instance with sensible defaults for testing."""
    defaults = dict(
        id="abc123",
        category="BeerBottle",
        mesh_hash="hash1",
        fracture="fractured_0",
        num_pieces=2,
        piece_type="two_pieces",
        obj_paths=[Path("/tmp/piece_0.obj"), Path("/tmp/piece_1.obj")],
    )
    defaults.update(overrides)
    return Fracture(**defaults)


class IndexViewTest(TestCase):
    """Tests for the landing page."""

    @patch("core.views.get_fractures")
    def test_index_returns_200(self, mock_get):
        mock_get.return_value = []
        response = self.client.get("/")
        self.assertEqual(response.status_code, 200)

    @patch("core.views.get_fractures")
    def test_index_passes_fractures_to_template(self, mock_get):
        two = _make_fake_fracture(id="a", piece_type="two_pieces")
        multi = _make_fake_fracture(id="b", piece_type="multi_pieces")
        mock_get.return_value = [two, multi]

        response = self.client.get("/")
        self.assertEqual(len(response.context["two_piece_fractures"]), 1)
        self.assertEqual(len(response.context["multi_piece_fractures"]), 1)


class FracturePiecesViewTest(TestCase):
    """Tests for GET /api/fracture/<id>/."""

    @patch("core.views.get_fracture_by_id")
    def test_valid_fracture_returns_json(self, mock_get):
        mock_get.return_value = _make_fake_fracture()
        response = self.client.get("/api/fracture/abc123/")

        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["id"], "abc123")
        self.assertEqual(data["num_pieces"], 2)
        self.assertEqual(len(data["pieces"]), 2)

    @patch("core.views.get_fracture_by_id")
    def test_unknown_fracture_returns_404(self, mock_get):
        mock_get.return_value = None
        response = self.client.get("/api/fracture/nonexistent/")
        self.assertEqual(response.status_code, 404)

    def test_only_get_allowed(self):
        response = self.client.post("/api/fracture/abc123/")
        self.assertEqual(response.status_code, 405)


class PredictViewTest(TestCase):
    """Tests for POST /api/predict/ and its GPU → precomputed → 503 fallback."""

    @patch("core.views.get_fracture_by_id")
    def test_unknown_fracture_returns_404(self, mock_get):
        mock_get.return_value = None
        response = self.client.post("/api/predict/", {"fracture_id": "bad", "model": "two_piece_baseline"})
        self.assertEqual(response.status_code, 404)

    def test_only_post_allowed(self):
        response = self.client.get("/api/predict/")
        self.assertEqual(response.status_code, 405)

    @patch("core.views.get_fracture_by_id")
    @patch("core.views._lookup_precomputed")
    @override_settings(GPU_POD_URL="")
    def test_precomputed_fallback(self, mock_lookup, mock_fracture):
        mock_fracture.return_value = _make_fake_fracture()
        mock_lookup.return_value = {
            "model": "two_piece_baseline",
            "num_pieces": 2,
            "pieces": [{"name": "piece_0", "assembled_obj": "v 0 0 0"}],
            "combined_assembled_obj": "v 0 0 0",
            "source": "precomputed",
        }

        response = self.client.post("/api/predict/", {"fracture_id": "abc123", "model": "two_piece_baseline"})
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["source"], "precomputed")

    @patch("core.views.get_fracture_by_id")
    @patch("core.views._lookup_precomputed")
    @override_settings(GPU_POD_URL="")
    def test_no_gpu_no_precomputed_returns_503(self, mock_lookup, mock_fracture):
        mock_fracture.return_value = _make_fake_fracture()
        mock_lookup.return_value = None

        response = self.client.post("/api/predict/", {"fracture_id": "abc123", "model": "two_piece_baseline"})
        self.assertEqual(response.status_code, 503)
