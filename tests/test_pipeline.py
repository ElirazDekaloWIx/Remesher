"""Tests for the core remeshing pipeline."""

import pytest
from remesher.pipeline import get_target_faces, process_file, PRESETS


class TestGetTargetFaces:
    def test_with_target_faces(self):
        assert get_target_faces(None, 5000, None, 10000) == 5000

    def test_target_faces_capped_at_current(self):
        assert get_target_faces(None, 50000, None, 1000) == 1000

    def test_with_ratio(self):
        assert get_target_faces(None, None, 0.5, 10000) == 5000

    def test_with_ratio_min_one(self):
        assert get_target_faces(None, None, 0.0001, 10) == 1

    def test_with_preset(self):
        assert get_target_faces("mobile", None, None, 100000) == PRESETS["mobile"]

    def test_preset_capped_at_current(self):
        assert get_target_faces("desktop", None, None, 100) == 100

    def test_default_preset(self):
        assert get_target_faces(None, None, None, 100000) == PRESETS["web"]

    def test_target_faces_overrides_preset(self):
        assert get_target_faces("mobile", 8000, None, 100000) == 8000


class TestProcessFile:
    def test_basic_decimation(self, test_glb_path, tmp_output):
        result = process_file(test_glb_path, tmp_output, preset="web")
        assert result.success
        assert result.final_faces > 0
        assert result.final_faces <= result.original_faces
        import os
        assert os.path.exists(tmp_output)

    def test_with_ratio(self, test_glb_path, tmp_output):
        result = process_file(test_glb_path, tmp_output, ratio=0.5)
        assert result.success
        assert result.reduction_pct > 0

    def test_with_target_faces(self, test_glb_path, tmp_output):
        result = process_file(test_glb_path, tmp_output, target_faces=100)
        assert result.success
        assert result.final_faces <= 1000  # QEM may not reach exact target on complex meshes

    def test_invalid_file(self, tmp_output):
        result = process_file("nonexistent.glb", tmp_output)
        assert not result.success
        assert result.error is not None

    def test_progress_callback(self, test_glb_path, tmp_output):
        stages = []
        result = process_file(test_glb_path, tmp_output, preset="web",
                              progress_cb=lambda s: stages.append(s))
        assert result.success
        assert len(stages) > 0
