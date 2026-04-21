"""Tests for the Flask web server."""

import io
import pytest
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent


class TestServerEndpoints:
    def test_index(self, flask_client):
        resp = flask_client.get("/")
        assert resp.status_code == 200
        assert b"Remesher" in resp.data

    def test_presets(self, flask_client):
        resp = flask_client.get("/api/presets")
        assert resp.status_code == 200
        data = resp.get_json()
        assert "web" in data
        assert "mobile" in data

    def test_progress(self, flask_client):
        resp = flask_client.get("/api/progress")
        assert resp.status_code == 200
        data = resp.get_json()
        assert "stage" in data

    def test_remesh_no_file(self, flask_client):
        resp = flask_client.post("/api/remesh")
        assert resp.status_code == 400

    def test_remesh_unsupported_format(self, flask_client):
        data = {"file": (io.BytesIO(b"fake"), "model.xyz")}
        resp = flask_client.post("/api/remesh", data=data, content_type="multipart/form-data")
        assert resp.status_code == 400

    def test_remesh_with_glb(self, flask_client):
        glb_path = PROJECT_ROOT / "test_fast.glb"
        if not glb_path.exists():
            pytest.skip("test_fast.glb not found")
        with open(glb_path, "rb") as f:
            data = {"file": (f, "test.glb"), "preset": "mobile"}
            resp = flask_client.post("/api/remesh", data=data, content_type="multipart/form-data")
        assert resp.status_code == 200
        assert len(resp.data) > 0


class TestServerValidation:
    def test_max_content_length_configured(self, flask_client):
        from server import app
        assert app.config["MAX_CONTENT_LENGTH"] == 100 * 1024 * 1024
