"""Core remeshing pipeline: clean -> decimate -> validate.

The pymeshlab work runs in a subprocess (`remesher._meshlab_worker`) so a
native segfault on a pathological mesh returns a clean error instead of
killing the Flask server.
"""

import os
import sys
import json
import time
import tempfile
import logging
import subprocess
from pathlib import Path
from dataclasses import dataclass

import numpy as np
import trimesh

logger = logging.getLogger(__name__)

# Web game polygon budgets
PRESETS = {
    "mobile": 3_000,
    "web": 10_000,
    "desktop": 30_000,
    "high": 50_000,
}

SUPPORTED_FORMATS = {".glb", ".gltf", ".obj", ".fbx", ".stl", ".ply", ".off", ".dae"}


@dataclass
class RemeshResult:
    input_path: str
    output_path: str
    original_faces: int
    final_faces: int
    reduction_pct: float
    success: bool
    error: str | None = None


def get_target_faces(preset: str | None, target_faces: int | None, ratio: float | None, current_faces: int) -> int:
    """Determine target face count from user parameters."""
    if target_faces is not None:
        return min(target_faces, current_faces)
    if ratio is not None:
        return max(1, int(current_faces * ratio))
    if preset is not None:
        return min(PRESETS[preset], current_faces)
    return min(PRESETS["web"], current_faces)


class MeshlabWorkerError(RuntimeError):
    """Raised when the pymeshlab subprocess fails (including native crashes)."""


def _run_meshlab_worker(params: dict, timeout_s: int = 600) -> None:
    """Execute the pymeshlab pipeline in an isolated subprocess.

    A native segfault produces a non-zero exit code (often 139/-11 on
    POSIX, 0xC0000005 on Windows) which we surface as a clean exception.
    """
    with tempfile.NamedTemporaryFile(
        "w", suffix=".json", delete=False, encoding="utf-8"
    ) as f:
        json.dump(params, f)
        params_path = f.name
    try:
        proc = subprocess.run(
            [sys.executable, "-m", "remesher._meshlab_worker", params_path],
            capture_output=True,
            text=True,
            timeout=timeout_s,
        )
    finally:
        try:
            os.unlink(params_path)
        except OSError:
            pass

    if proc.stderr:
        for line in proc.stderr.splitlines():
            logger.info(f"  [meshlab] {line}")
    if proc.returncode != 0:
        crash = (
            "native crash" if proc.returncode in (-11, 139, 3221225477) else f"exit {proc.returncode}"
        )
        last_err = (proc.stderr.strip().splitlines() or [""])[-1]
        raise MeshlabWorkerError(
            f"pymeshlab worker failed ({crash}) — {last_err or 'no stderr'}"
        )


def _has_uv(geometry: trimesh.Trimesh) -> bool:
    """Check if a mesh has UV coordinates."""
    if hasattr(geometry.visual, "uv") and geometry.visual.uv is not None:
        return len(geometry.visual.uv) > 0
    return False


def _get_material(geometry: trimesh.Trimesh):
    """Extract material from geometry."""
    if hasattr(geometry.visual, "material"):
        return geometry.visual.material
    return None


def process_single_mesh(
    mesh: trimesh.Trimesh,
    target_faces: int,
    quality_thr: float = 0.3,
    smooth_method: str = "none",
    smooth_iterations: int = 3,
) -> trimesh.Trimesh:
    """Decimate a single trimesh geometry, preserving UVs and material."""
    original_material = _get_material(mesh)
    has_texture = _has_uv(mesh)
    original_face_count = len(mesh.faces)

    if original_face_count <= target_faces:
        logger.info(f"  Mesh already has {original_face_count} faces (target: {target_faces}), skipping")
        return mesh

    t0 = time.time()

    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_in = os.path.join(tmp_dir, "input.obj")
        tmp_out = os.path.join(tmp_dir, "output.obj")

        mesh.export(tmp_in, file_type="obj")

        _run_meshlab_worker({
            "input": tmp_in,
            "output": tmp_out,
            "target_faces": int(target_faces),
            "quality": float(quality_thr),
            "has_texture": bool(has_texture),
            "smooth_method": smooth_method or "none",
            "smooth_iterations": int(smooth_iterations),
        })

        decimated = trimesh.load(tmp_out, process=False, force="mesh")

    # Re-attach original material with new UVs
    if has_texture and hasattr(decimated.visual, "uv") and decimated.visual.uv is not None:
        decimated.visual = trimesh.visual.TextureVisuals(
            uv=decimated.visual.uv,
            material=original_material,
        )
    elif original_material is not None:
        decimated.visual = trimesh.visual.TextureVisuals(material=original_material)

    elapsed = time.time() - t0
    logger.info(f"  {original_face_count} -> {len(decimated.faces)} faces ({elapsed:.1f}s)")
    return decimated


def process_file(
    input_path: str,
    output_path: str,
    preset: str | None = "web",
    target_faces: int | None = None,
    ratio: float | None = None,
    quality_thr: float = 0.3,
    progress_cb=None,
    uv_method: str = "keep",
    texture_size: int = 2048,
    smooth_method: str = "none",
    smooth_iterations: int = 3,
) -> RemeshResult:
    """Process a single 3D file through the remeshing pipeline."""
    input_path = str(Path(input_path).resolve())
    output_path = str(Path(output_path).resolve())

    _cb = progress_cb or (lambda *a: None)
    logger.info(f"Processing: {Path(input_path).name}")
    _cb("Loading file...")

    try:
        loaded = trimesh.load(input_path, process=False)
    except Exception as e:
        return RemeshResult(input_path, output_path, 0, 0, 0, False, str(e))

    try:
        return _process_loaded(
            loaded, input_path, output_path,
            preset, target_faces, ratio, quality_thr, _cb,
            uv_method, texture_size, smooth_method, smooth_iterations,
        )
    except MeshlabWorkerError as e:
        logger.error(f"Decimation failed: {e}")
        return RemeshResult(input_path, output_path, 0, 0, 0, False, f"Decimation failed: {e}")
    except Exception as e:
        logger.exception("Unexpected pipeline error")
        return RemeshResult(input_path, output_path, 0, 0, 0, False, str(e))


def _process_loaded(
    loaded, input_path, output_path,
    preset, target_faces, ratio, quality_thr, _cb,
    uv_method, texture_size, smooth_method, smooth_iterations,
) -> RemeshResult:
    total_original = 0
    total_final = 0

    # Handle scene (multi-mesh, e.g. GLB)
    if isinstance(loaded, trimesh.Scene):
        new_scene = trimesh.Scene()

        geom_items = list(loaded.geometry.items())
        for idx, (name, geom) in enumerate(geom_items):
            if not isinstance(geom, trimesh.Trimesh):
                new_scene.add_geometry(geom, geom_name=name)
                continue

            face_count = len(geom.faces)
            total_original += face_count
            target = get_target_faces(preset, target_faces, ratio, face_count)

            _cb(f"Decimating mesh {idx+1}/{len(geom_items)}: '{name}'")
            logger.info(f"  Mesh '{name}': {face_count} faces -> target {target}")
            decimated = process_single_mesh(geom, target, quality_thr, smooth_method, smooth_iterations)

            # UV re-unwrap if requested
            if uv_method != "keep":
                from .uv_unwrap import apply_uv
                _cb(f"UV unwrap ({uv_method}) mesh {idx+1}/{len(geom_items)}")
                decimated = apply_uv(decimated, uv_method, geom, texture_size)

            total_final += len(decimated.faces)

            # Preserve scene graph transforms
            try:
                node_name = None
                for node in loaded.graph.nodes_geometry:
                    transform, geometry_name = loaded.graph.get(node)
                    if geometry_name == name:
                        node_name = node
                        break
                if node_name:
                    transform, _ = loaded.graph.get(node_name)
                    new_scene.add_geometry(decimated, geom_name=name, node_name=node_name, transform=transform)
                else:
                    new_scene.add_geometry(decimated, geom_name=name)
            except Exception:
                new_scene.add_geometry(decimated, geom_name=name)

        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        new_scene.export(output_path)

    # Handle single mesh
    elif isinstance(loaded, trimesh.Trimesh):
        total_original = len(loaded.faces)
        target = get_target_faces(preset, target_faces, ratio, total_original)

        logger.info(f"  {total_original} faces -> target {target}")
        decimated = process_single_mesh(loaded, target, quality_thr, smooth_method, smooth_iterations)

        # UV re-unwrap if requested
        if uv_method != "keep":
            from .uv_unwrap import apply_uv
            _cb(f"UV unwrap ({uv_method})...")
            decimated = apply_uv(decimated, uv_method, loaded, texture_size)

        total_final = len(decimated.faces)

        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        decimated.export(output_path)
    else:
        return RemeshResult(input_path, output_path, 0, 0, 0, False, "Unsupported mesh type")

    reduction = (1 - total_final / total_original) * 100 if total_original > 0 else 0
    logger.info(f"  Done: {total_original} -> {total_final} faces ({reduction:.1f}% reduction)")

    return RemeshResult(input_path, output_path, total_original, total_final, reduction, True)
