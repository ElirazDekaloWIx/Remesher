"""Tests for the retopology pipeline."""

import numpy as np
import pytest
from remesher.retopology import _triangulate_quads, RETOPO_PRESETS


class TestTriangulateQuads:
    def test_single_quad(self):
        quads = [[0, 1, 2, 3]]
        tris = _triangulate_quads(quads)
        assert tris.shape == (2, 3)
        assert tris[0].tolist() == [0, 1, 2]
        assert tris[1].tolist() == [0, 2, 3]

    def test_single_triangle(self):
        tris_in = [[0, 1, 2]]
        tris = _triangulate_quads(tris_in)
        assert tris.shape == (1, 3)

    def test_mixed(self):
        faces = [[0, 1, 2], [3, 4, 5, 6]]
        tris = _triangulate_quads(faces)
        assert tris.shape == (3, 3)

    def test_ngon(self):
        faces = [[0, 1, 2, 3, 4]]  # pentagon
        tris = _triangulate_quads(faces)
        assert tris.shape == (3, 3)

    def test_empty(self):
        tris = _triangulate_quads([])
        assert len(tris) == 0


class TestRetopologyProcessFile:
    """Integration tests — require pyQuadriFlow and xatlas installed."""

    def test_retopo_presets_exist(self):
        assert "web" in RETOPO_PRESETS
        assert "mobile" in RETOPO_PRESETS
        assert all(v > 0 for v in RETOPO_PRESETS.values())
