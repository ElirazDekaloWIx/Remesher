"""Core remeshing pipeline: clean -> decimate -> validate."""

import os
import time
import tempfile
import logging
from pathlib import Path
from dataclasses import dataclass

import numpy as np
import trimesh
import pymeshlab

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


def _clean_mesh(ms: pymeshlab.MeshSet):
    """Pre-cleaning steps."""
    ms.meshing_remove_duplicate_vertices()
    ms.meshing_remove_duplicate_faces()
    ms.meshing_remove_null_faces()
    try:
        ms.meshing_repair_non_manifold_edges()
    except Exception:
        pass
    try:
        ms.meshing_repair_non_manifold_vertices()
    except Exception:
        pass


def _decimate_mesh(ms: pymeshlab.MeshSet, target_faces: int, has_texture: bool, quality_thr: float):
    """Run QEM decimation."""
    if has_texture:
        try:
            ms.meshing_decimation_quadric_edge_collapse_with_texture(
                targetfacenum=target_faces,
                qualitythr=quality_thr,
                preserveboundary=True,
                optimalplacement=True,
                planarquadric=True,
            )
            return
        except Exception as e:
            logger.warning(f"Texture-aware decimation failed, falling back to standard: {e}")

    ms.meshing_decimation_quadric_edge_collapse(
        targetfacenum=target_faces,
        qualitythr=quality_thr,
        preserveboundary=True,
        preservenormal=True,
        preservetopology=True,
        optimalplacement=True,
        planarquadric=True,
    )


def _smooth_mesh(ms: pymeshlab.MeshSet, method: str = "laplacian", iterations: int = 3):
    """Apply smoothing to reduce decimation artifacts."""
    if method == "laplacian":
        ms.apply_coord_laplacian_smoothing(
            stepsmoothnum=iterations,
            boundary=True,
            cotangentweight=True,
        )
    elif method == "taubin":
        ms.apply_coord_taubin_smoothing(
            stepsmoothnum=iterations,
            lambda_=0.5,
            mu=-0.53,
        )
    else:
        return
    logger.info(f"  Applied {method} smoothing ({iterations} iterations)")


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

        ms = pymeshlab.MeshSet()
        ms.load_new_mesh(tmp_in)

        _clean_mesh(ms)
        _decimate_mesh(ms, target_faces, has_texture, quality_thr)

        # Apply smoothing after decimation
        if smooth_method and smooth_method != "none":
            _smooth_mesh(ms, smooth_method, smooth_iterations)

        ms.compute_normal_per_vertex()
        ms.compute_normal_per_face()

        ms.save_current_mesh(tmp_out)

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
