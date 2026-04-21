"""Adaptive retopology pipeline with multiple remesh and UV methods."""

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

RETOPO_PRESETS = {
    "mobile": 1_500,
    "web": 5_000,
    "desktop": 15_000,
    "high": 25_000,
}

REMESH_METHODS = ("isotropic", "quadriflow")
UV_METHODS = ("lscm", "xatlas")


@dataclass
class RetopologyResult:
    input_path: str
    output_path: str
    original_faces: int
    final_faces: int
    final_quads: int
    reduction_pct: float
    texture_baked: bool
    success: bool
    error: str | None = None


# ---------------------------------------------------------------------------
# Remesh methods
# ---------------------------------------------------------------------------

def _isotropic_remesh(mesh: trimesh.Trimesh, target_faces: int) -> trimesh.Trimesh:
    """Isotropic explicit remeshing via pymeshlab — robust, uniform triangles."""
    with tempfile.TemporaryDirectory() as tmp:
        tmp_in = os.path.join(tmp, "in.obj")
        tmp_out = os.path.join(tmp, "out.obj")
        mesh.export(tmp_in, file_type="obj")

        ms = pymeshlab.MeshSet()
        ms.load_new_mesh(tmp_in)

        # Pre-clean
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
        try:
            ms.meshing_close_holes(maxholesize=30)
        except Exception:
            pass

        # Compute target edge length from desired face count
        # Approximation: faces ≈ 2 * vertices, and for uniform tris on a surface:
        # edge_length ≈ sqrt(2 * surface_area / (sqrt(3) * target_faces))
        current_area = ms.current_mesh().bounding_box().diagonal() ** 2
        try:
            # Use actual surface area if available
            measures = ms.get_geometric_measures()
            current_area = measures.get("surface_area", current_area)
        except Exception:
            pass

        target_edge = float(np.sqrt(2.0 * current_area / (np.sqrt(3) * max(target_faces, 1))))

        # Clamp to reasonable range relative to bounding box
        diag = ms.current_mesh().bounding_box().diagonal()
        target_edge = max(target_edge, diag * 0.0005)
        target_edge = min(target_edge, diag * 0.5)

        logger.info(f"    Isotropic remesh: target edge length = {target_edge:.6f}")

        ms.meshing_isotropic_explicit_remeshing(
            targetlen=pymeshlab.AbsoluteValue(target_edge),
            iterations=5,
            adaptive=True,
        )

        # Post-clean
        ms.meshing_remove_duplicate_vertices()
        ms.meshing_remove_duplicate_faces()
        ms.meshing_remove_null_faces()
        ms.compute_normal_per_vertex()
        ms.compute_normal_per_face()

        ms.save_current_mesh(tmp_out)
        result = trimesh.load(tmp_out, process=False, force="mesh")

    return result


def _quadriflow_remesh(vertices: np.ndarray, faces: np.ndarray, target_faces: int) -> tuple[np.ndarray, list]:
    """Quad remesh using QuadriFlow with adaptive scaling."""
    from pyQuadriFlow.pyQuadriFlow import pyquadriflow

    result = pyquadriflow(
        target_faces, 42,
        vertices.tolist(), faces.tolist(),
        True, True, True, False, True,
    )
    return np.array(result['vertices'], dtype=np.float64), result['faces']


def _triangulate_quads(quad_faces: list) -> np.ndarray:
    """Convert quad faces to triangles."""
    tris = []
    for face in quad_faces:
        if len(face) == 4:
            tris.append([face[0], face[1], face[2]])
            tris.append([face[0], face[2], face[3]])
        elif len(face) == 3:
            tris.append(face)
        else:
            for i in range(1, len(face) - 1):
                tris.append([face[0], face[i], face[i + 1]])
    return np.array(tris, dtype=np.int32) if tris else np.zeros((0, 3), dtype=np.int32)


# ---------------------------------------------------------------------------
# UV unwrap methods
# ---------------------------------------------------------------------------

def _uv_lscm(mesh_trimesh: trimesh.Trimesh) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """LSCM UV unwrap via pymeshlab — smooth, minimal seams, good for painting."""
    with tempfile.TemporaryDirectory() as tmp:
        tmp_in = os.path.join(tmp, "in.obj")
        tmp_out = os.path.join(tmp, "out.obj")
        mesh_trimesh.export(tmp_in, file_type="obj")

        ms = pymeshlab.MeshSet()
        ms.load_new_mesh(tmp_in)

        # True LSCM parametrization — smooth, conformal UVs with minimal seams
        try:
            ms.compute_texcoord_parametrization_least_squares_conformal_maps()
        except Exception as e:
            logger.warning(f"LSCM failed ({e}), falling back to flat projection")
            ms.compute_texcoord_parametrization_flat_plane_per_wedge(projectionplane="BestDiag")

        ms.save_current_mesh(tmp_out)
        result = trimesh.load(tmp_out, process=False, force="mesh")

    verts = np.array(result.vertices, dtype=np.float32)
    faces = np.array(result.faces, dtype=np.uint32)

    if hasattr(result.visual, "uv") and result.visual.uv is not None and len(result.visual.uv) > 0:
        uvs = np.array(result.visual.uv, dtype=np.float32)
    else:
        # Fallback: generate basic UVs
        uvs = np.zeros((len(verts), 2), dtype=np.float32)

    return verts, faces, uvs


def _uv_xatlas(verts: np.ndarray, faces: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """xatlas UV unwrap — fast, many islands, optimized packing."""
    import xatlas

    verts_f32 = np.ascontiguousarray(verts, dtype=np.float32)
    faces_u32 = np.ascontiguousarray(faces, dtype=np.uint32)
    vmapping, uv_indices, uvs = xatlas.parametrize(verts_f32, faces_u32)

    final_verts = verts[vmapping]
    return final_verts, uv_indices, uvs


# ---------------------------------------------------------------------------
# Texture baking
# ---------------------------------------------------------------------------

def _bake_texture(original_mesh, retopo_mesh, retopo_uvs, texture_size=2048):
    """Bake texture from original to retopo mesh via closest-point sampling."""
    from PIL import Image

    orig_texture = None
    orig_uv = None

    if hasattr(original_mesh.visual, "material"):
        mat = original_mesh.visual.material
        for attr in ("image", "baseColorTexture"):
            img = getattr(mat, attr, None)
            if img is not None:
                orig_texture = np.array(img)
                break

    if hasattr(original_mesh.visual, "uv") and original_mesh.visual.uv is not None:
        orig_uv = original_mesh.visual.uv

    if orig_texture is None or orig_uv is None:
        return None

    tex_h, tex_w = orig_texture.shape[:2]
    ch = orig_texture.shape[2] if orig_texture.ndim > 2 else 3

    retopo_verts = retopo_mesh.vertices
    orig_faces = original_mesh.faces
    orig_verts = original_mesh.vertices

    closest_pts, _, face_ids = trimesh.proximity.closest_point(original_mesh, retopo_verts)

    vertex_colors = np.full((len(retopo_verts), ch), 128, dtype=np.uint8)

    # Vectorized barycentric interpolation
    valid_mask = face_ids < len(orig_faces)
    valid_ids = np.where(valid_mask)[0]

    if len(valid_ids) > 0:
        fids = face_ids[valid_ids]
        tris = orig_faces[fids]
        v0 = orig_verts[tris[:, 0]]
        v1 = orig_verts[tris[:, 1]]
        v2 = orig_verts[tris[:, 2]]
        p = closest_pts[valid_ids]

        e0 = v1 - v0
        e1 = v2 - v0
        ep = p - v0

        d00 = np.einsum('ij,ij->i', e0, e0)
        d01 = np.einsum('ij,ij->i', e0, e1)
        d11 = np.einsum('ij,ij->i', e1, e1)
        dp0 = np.einsum('ij,ij->i', ep, e0)
        dp1 = np.einsum('ij,ij->i', ep, e1)

        denom = d00 * d11 - d01 * d01
        good = np.abs(denom) > 1e-10
        denom_safe = np.where(good, denom, 1.0)

        u = (d11 * dp0 - d01 * dp1) / denom_safe
        v = (d00 * dp1 - d01 * dp0) / denom_safe
        w = 1.0 - u - v

        uv_len = len(orig_uv)
        t0 = np.clip(tris[:, 0], 0, uv_len - 1)
        t1 = np.clip(tris[:, 1], 0, uv_len - 1)
        t2 = np.clip(tris[:, 2], 0, uv_len - 1)

        sample_uv = w[:, None] * orig_uv[t0] + u[:, None] * orig_uv[t1] + v[:, None] * orig_uv[t2]

        sx = np.clip((sample_uv[:, 0] * tex_w).astype(int), 0, tex_w - 1) % tex_w
        sy = np.clip(((1.0 - sample_uv[:, 1]) * tex_h).astype(int), 0, tex_h - 1) % tex_h

        sampled = orig_texture[sy, sx, :ch]
        vertex_colors[valid_ids[good]] = sampled[good]

    # Rasterize into texture
    baked = np.zeros((texture_size, texture_size, ch), dtype=np.float32)
    retopo_faces = retopo_mesh.faces
    ts = texture_size - 1

    all_uvs = retopo_uvs[retopo_faces]
    all_colors = vertex_colors[retopo_faces].astype(np.float32)
    all_px = all_uvs[:, :, 0] * ts
    all_py = (1 - all_uvs[:, :, 1]) * ts

    for fi in range(len(retopo_faces)):
        px = all_px[fi]
        py = all_py[fi]
        colors = all_colors[fi]

        min_x = max(0, int(np.floor(px.min())))
        max_x = min(ts, int(np.ceil(px.max())))
        min_y = max(0, int(np.floor(py.min())))
        max_y = min(ts, int(np.ceil(py.max())))

        if max_x <= min_x or max_y <= min_y:
            continue

        pv0 = np.array([px[0], py[0]])
        pe0 = np.array([px[1] - px[0], py[1] - py[0]])
        pe1 = np.array([px[2] - px[0], py[2] - py[0]])
        dn = pe0[0] * pe1[1] - pe0[1] * pe1[0]
        if abs(dn) < 1e-10:
            continue

        xs = np.arange(min_x, max_x + 1, dtype=np.float64)
        ys = np.arange(min_y, max_y + 1, dtype=np.float64)
        gx, gy = np.meshgrid(xs, ys)
        pts = np.stack([gx.ravel(), gy.ravel()], axis=1)

        ep = pts - pv0
        bu = (pe1[1] * ep[:, 0] - pe1[0] * ep[:, 1]) / dn
        bv = (pe0[0] * ep[:, 1] - pe0[1] * ep[:, 0]) / dn
        bw = 1.0 - bu - bv

        mask = (bw >= -0.01) & (bu >= -0.01) & (bv >= -0.01)
        valid = np.where(mask)[0]
        if len(valid) == 0:
            continue

        c = bw[valid, None] * colors[0] + bu[valid, None] * colors[1] + bv[valid, None] * colors[2]
        coords = pts[valid].astype(int)
        cy = np.clip(coords[:, 1], 0, texture_size - 1)
        cx = np.clip(coords[:, 0], 0, texture_size - 1)
        baked[cy, cx] = np.clip(c, 0, 255)

    return Image.fromarray(baked.astype(np.uint8))


# ---------------------------------------------------------------------------
# Main retopology function
# ---------------------------------------------------------------------------

def retopologize_mesh(mesh, target_faces, texture_size=2048, method="isotropic", uv_method="lscm"):
    """Full retopology pipeline: remesh -> UV unwrap -> texture bake.

    Args:
        method: "isotropic" (pymeshlab, robust) or "quadriflow" (quad-dominant, legacy)
        uv_method: "lscm" (smooth, minimal seams) or "xatlas" (fast, many islands)
    """
    original_material = getattr(mesh.visual, "material", None) if hasattr(mesh.visual, "material") else None
    has_texture = (
        hasattr(mesh.visual, "uv") and mesh.visual.uv is not None
        and len(mesh.visual.uv) > 0 and original_material is not None
    )

    quad_count = 0

    # Step 1: Remesh
    if method == "quadriflow":
        logger.info("  Step 1: QuadriFlow quad remesh...")
        t0 = time.time()
        new_verts, quad_faces = _quadriflow_remesh(
            mesh.vertices.astype(np.float64),
            mesh.faces.astype(np.int32),
            target_faces,
        )
        quad_count = sum(1 for f in quad_faces if len(f) == 4)
        tri_faces = _triangulate_quads(quad_faces)
        remeshed = trimesh.Trimesh(vertices=new_verts, faces=tri_faces, process=False)
        logger.info(f"    -> {len(quad_faces)} faces, {quad_count} quads ({time.time() - t0:.1f}s)")
    else:
        logger.info("  Step 1: Isotropic explicit remesh...")
        t0 = time.time()
        remeshed = _isotropic_remesh(mesh, target_faces)
        logger.info(f"    -> {len(remeshed.faces)} faces ({time.time() - t0:.1f}s)")

    # Step 2: UV unwrap
    if uv_method == "lscm":
        logger.info("  Step 2: LSCM UV unwrap (smooth)...")
        t1 = time.time()
        final_verts, final_faces, final_uvs = _uv_lscm(remeshed)
        result = trimesh.Trimesh(vertices=final_verts, faces=final_faces, process=False)
        logger.info(f"    -> done ({time.time() - t1:.1f}s)")
    else:
        logger.info("  Step 2: xatlas UV unwrap...")
        t1 = time.time()
        final_verts, final_faces, final_uvs = _uv_xatlas(
            np.array(remeshed.vertices), np.array(remeshed.faces)
        )
        result = trimesh.Trimesh(vertices=final_verts, faces=final_faces, process=False)
        logger.info(f"    -> done ({time.time() - t1:.1f}s)")

    # Step 3: Texture bake
    if has_texture:
        logger.info(f"  Step 3: Texture bake ({texture_size}x{texture_size})...")
        t2 = time.time()

        baked_image = _bake_texture(mesh, result, final_uvs, texture_size)

        if baked_image is not None:
            from trimesh.visual.material import PBRMaterial
            result.visual = trimesh.visual.TextureVisuals(
                uv=final_uvs,
                material=PBRMaterial(baseColorTexture=baked_image),
            )
            logger.info(f"    -> done ({time.time() - t2:.1f}s)")
        else:
            logger.info("    -> no texture to bake")
            result.visual = trimesh.visual.TextureVisuals(uv=final_uvs)
    else:
        result.visual = trimesh.visual.TextureVisuals(uv=final_uvs)

    return result, quad_count


# ---------------------------------------------------------------------------
# File-level processing
# ---------------------------------------------------------------------------

def process_file_retopo(
    input_path: str,
    output_path: str,
    preset: str | None = "web",
    target_faces: int | None = None,
    method: str = "isotropic",
    texture_size: int = 2048,
    uv_method: str = "lscm",
    progress_cb=None,
) -> RetopologyResult:
    """Process a single 3D file through the retopology pipeline."""
    input_path = str(Path(input_path).resolve())
    output_path = str(Path(output_path).resolve())
    _cb = progress_cb or (lambda *a: None)
    logger.info(f"Retopology: {Path(input_path).name} [method={method}, uv={uv_method}]")
    _cb("Loading file...")

    try:
        loaded = trimesh.load(input_path, process=False)
    except Exception as e:
        return RetopologyResult(input_path, output_path, 0, 0, 0, 0, False, False, str(e))

    total_target = target_faces or RETOPO_PRESETS.get(preset, RETOPO_PRESETS["web"])

    # Handle scene (multi-mesh)
    if isinstance(loaded, trimesh.Scene):
        geometries = {name: g for name, g in loaded.geometry.items() if isinstance(g, trimesh.Trimesh)}
        if not geometries:
            return RetopologyResult(input_path, output_path, 0, 0, 0, 0, False, False, "No meshes")

        total_original = sum(len(g.faces) for g in geometries.values())
        target_total = min(total_target, total_original)
        total_final = 0
        total_quads = 0
        texture_baked = False
        new_scene = trimesh.Scene()

        geom_items = list(geometries.items())
        for idx, (name, geom) in enumerate(geom_items):
            face_count = len(geom.faces)
            mesh_target = max(10, int(target_total * face_count / total_original))
            _cb(f"Retopologizing mesh {idx+1}/{len(geom_items)}: '{name}'")
            logger.info(f"  Mesh '{name}': {face_count} faces -> target {mesh_target}")

            try:
                result_mesh, quad_count = retopologize_mesh(
                    geom, mesh_target, texture_size, method, uv_method
                )
                total_final += len(result_mesh.faces)
                total_quads += quad_count
                if hasattr(result_mesh.visual, 'uv') and result_mesh.visual.uv is not None:
                    texture_baked = True

                try:
                    node_name = None
                    for node in loaded.graph.nodes_geometry:
                        transform, geometry_name = loaded.graph.get(node)
                        if geometry_name == name:
                            node_name = node
                            break
                    if node_name:
                        transform, _ = loaded.graph.get(node_name)
                        new_scene.add_geometry(result_mesh, geom_name=name, node_name=node_name, transform=transform)
                    else:
                        new_scene.add_geometry(result_mesh, geom_name=name)
                except Exception:
                    new_scene.add_geometry(result_mesh, geom_name=name)
            except Exception as e:
                logger.warning(f"  Mesh '{name}' retopo failed, keeping original: {e}")
                new_scene.add_geometry(geom, geom_name=name)
                total_final += face_count

        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        new_scene.export(output_path)

        reduction = (1 - total_final / total_original) * 100 if total_original > 0 else 0
        logger.info(f"  Done: {total_original} -> {total_final} tris ({total_quads} quads)")

        return RetopologyResult(
            input_path, output_path, total_original, total_final,
            total_quads, reduction, texture_baked, True,
        )

    elif isinstance(loaded, trimesh.Trimesh):
        mesh = loaded
    else:
        return RetopologyResult(input_path, output_path, 0, 0, 0, 0, False, False, "Unsupported")

    original_faces = len(mesh.faces)
    target = min(total_target, original_faces)

    logger.info(f"  {original_faces} faces -> target {target}")
    _cb(f"Retopologizing {original_faces} faces -> {target}")

    try:
        result_mesh, quad_count = retopologize_mesh(mesh, target, texture_size, method, uv_method)
        final_faces = len(result_mesh.faces)

        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        result_mesh.export(output_path)

        reduction = (1 - final_faces / original_faces) * 100 if original_faces > 0 else 0
        logger.info(f"  Done: {original_faces} -> {final_faces} tris ({quad_count} quads)")

        return RetopologyResult(
            input_path, output_path, original_faces, final_faces,
            quad_count, reduction, True, True,
        )
    except Exception as e:
        logger.error(f"  Failed: {e}")
        import traceback; traceback.print_exc()
        return RetopologyResult(input_path, output_path, original_faces, 0, 0, 0, False, False, str(e))
