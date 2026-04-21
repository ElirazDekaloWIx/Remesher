"""Tests for UV unwrapping methods."""

import numpy as np
import pytest
import trimesh
from remesher.uv_unwrap import UV_METHODS, unwrap_xatlas, unwrap_lscm, unwrap_arap, apply_uv


@pytest.fixture
def simple_mesh():
    """Create a simple box mesh for testing."""
    return trimesh.creation.box(extents=[1, 1, 1])


class TestUVMethods:
    def test_uv_methods_list(self):
        assert "keep" in UV_METHODS
        assert "xatlas" in UV_METHODS
        assert "lscm" in UV_METHODS
        assert "arap" in UV_METHODS

    def test_xatlas(self, simple_mesh):
        verts, faces, uvs = unwrap_xatlas(simple_mesh)
        assert len(verts) > 0
        assert len(faces) > 0
        assert len(uvs) > 0
        assert uvs.shape[1] == 2
        # UVs should be in [0, 1] range
        assert uvs.min() >= -0.01
        assert uvs.max() <= 1.01

    def test_lscm(self, simple_mesh):
        verts, faces, uvs = unwrap_lscm(simple_mesh)
        assert len(verts) > 0
        assert len(faces) > 0
        assert len(uvs) > 0
        assert uvs.shape[1] == 2

    def test_arap(self, simple_mesh):
        verts, faces, uvs = unwrap_arap(simple_mesh, iterations=3)
        assert len(verts) > 0
        assert len(faces) > 0
        assert len(uvs) > 0
        assert uvs.shape[1] == 2
        # ARAP normalizes to [0, 1]
        assert uvs.min() >= -0.01
        assert uvs.max() <= 1.01


class TestApplyUV:
    def test_keep(self, simple_mesh):
        result = apply_uv(simple_mesh, "keep")
        assert result is simple_mesh  # Same object, not modified

    def test_apply_xatlas(self, simple_mesh):
        result = apply_uv(simple_mesh, "xatlas")
        assert isinstance(result, trimesh.Trimesh)
        assert len(result.faces) > 0

    def test_unknown_method(self, simple_mesh):
        result = apply_uv(simple_mesh, "nonexistent")
        assert result is simple_mesh  # Falls back to keeping original
