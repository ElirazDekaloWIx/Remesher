"""Shared test fixtures."""

import os
import pytest
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent


@pytest.fixture
def test_glb_path():
    """Return path to a test GLB file."""
    path = PROJECT_ROOT / "test_fast.glb"
    if not path.exists():
        pytest.skip("test_fast.glb not found")
    return str(path)


@pytest.fixture
def tmp_output(tmp_path):
    """Return a temporary output file path."""
    return str(tmp_path / "output.glb")


@pytest.fixture
def flask_client():
    """Create a Flask test client."""
    import sys
    sys.path.insert(0, str(PROJECT_ROOT))
    from server import app
    app.config["TESTING"] = True
    with app.test_client() as client:
        yield client
