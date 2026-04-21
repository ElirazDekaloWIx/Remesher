"""UV unwrapping methods: xatlas, LSCM (pymeshlab), ARAP (scipy-based)."""

import os
import tempfile
import logging
import numpy as np
import trimesh
import pymeshlab

logger = logging.getLogger(__name__)

UV_METHODS = ("keep", "xatlas", "lscm", "arap")


def unwrap_xatlas(mesh: trimesh.Trimesh) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """xatlas UV unwrap — fast, automatic seam cutting, packed islands.

    Good for game assets. Many islands but efficient packing.
    """
    import xatlas

    verts = np.ascontiguousarray(mesh.vertices, dtype=np.float32)
    faces = np.ascontiguousarray(mesh.faces, dtype=np.uint32)

    vmapping, new_faces, uvs = xatlas.parametrize(verts, faces)

    new_verts = mesh.vertices[vmapping]
    return new_verts, new_faces, uvs


def unwrap_lscm(mesh: trimesh.Trimesh) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """LSCM UV unwrap via pymeshlab — angle-preserving, smooth flow.

    Excellent for painting on organic shapes. Minimal seams.
    """
    with tempfile.TemporaryDirectory() as tmp:
        tmp_in = os.path.join(tmp, "in.obj")
        tmp_out = os.path.join(tmp, "out.obj")
        mesh.export(tmp_in, file_type="obj")

        ms = pymeshlab.MeshSet()
        ms.load_new_mesh(tmp_in)

        try:
            ms.compute_texcoord_parametrization_least_squares_conformal_maps()
        except Exception:
            # Fallback: harmonic parametrization if LSCM fails
            logger.warning("LSCM failed, trying harmonic parametrization")
            try:
                ms.compute_texcoord_parametrization_harmonic()
            except Exception:
                logger.warning("Harmonic failed too, using flat plane fallback")
                ms.compute_texcoord_parametrization_flat_plane_per_wedge(projectionplane="XY")

        ms.save_current_mesh(tmp_out)
        result = trimesh.load(tmp_out, process=False, force="mesh")

    verts = np.array(result.vertices, dtype=np.float64)
    faces = np.array(result.faces, dtype=np.int32)

    if hasattr(result.visual, "uv") and result.visual.uv is not None and len(result.visual.uv) > 0:
        uvs = np.array(result.visual.uv, dtype=np.float32)
    else:
        logger.warning("LSCM produced no UVs, generating fallback")
        uvs = _fallback_uvs(verts, faces)

    return verts, faces, uvs


def unwrap_arap(mesh: trimesh.Trimesh, iterations: int = 10) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """ARAP (As-Rigid-As-Possible) UV parametrization.

    Best combined angle + area preservation. Ideal for texture painting.
    Uses Tutte embedding initialization + iterative ARAP refinement.
    """
    import scipy.sparse as sp
    from scipy.sparse.linalg import spsolve
    import robust_laplacian

    verts = np.array(mesh.vertices, dtype=np.float64)
    faces = np.array(mesh.faces, dtype=np.int32)

    # Step 1: Compute initial UV via Tutte embedding (boundary -> circle, interior -> harmonic)
    logger.info("    ARAP: computing Tutte embedding initialization...")
    uv_init = _tutte_embedding(verts, faces)

    # Step 2: Iterative ARAP refinement
    logger.info(f"    ARAP: {iterations} refinement iterations...")
    uv = _arap_iterate(verts, faces, uv_init, iterations)

    # Normalize to [0, 1]
    uv_min = uv.min(axis=0)
    uv_max = uv.max(axis=0)
    uv_range = uv_max - uv_min
    uv_range[uv_range < 1e-10] = 1.0
    uvs = ((uv - uv_min) / uv_range).astype(np.float32)

    return verts, faces, uvs


def _fallback_uvs(verts, faces):
    """Simple spherical projection fallback."""
    center = verts.mean(axis=0)
    dirs = verts - center
    norms = np.linalg.norm(dirs, axis=1, keepdims=True)
    norms[norms < 1e-10] = 1.0
    dirs = dirs / norms

    u = 0.5 + np.arctan2(dirs[:, 2], dirs[:, 0]) / (2 * np.pi)
    v = 0.5 + np.arcsin(np.clip(dirs[:, 1], -1, 1)) / np.pi
    return np.column_stack([u, v]).astype(np.float32)


def _tutte_embedding(verts, faces):
    """Tutte embedding: map boundary to circle, solve harmonic for interior."""
    import scipy.sparse as sp
    from scipy.sparse.linalg import spsolve

    n = len(verts)

    # Find boundary edges
    edges = {}
    for f in faces:
        for i in range(3):
            e = (f[i], f[(i + 1) % 3])
            e_sorted = tuple(sorted(e))
            edges[e_sorted] = edges.get(e_sorted, 0) + 1

    boundary_edges = {e for e, count in edges.items() if count == 1}

    if not boundary_edges:
        # Closed mesh — cut along longest edge path to create boundary
        logger.info("    Closed mesh detected, using spherical projection for initialization")
        return _fallback_uvs(verts, faces)[:, :2].astype(np.float64)

    # Build boundary loop
    adj = {}
    for a, b in boundary_edges:
        adj.setdefault(a, []).append(b)
        adj.setdefault(b, []).append(a)

    boundary = []
    start = next(iter(adj))
    current = start
    visited = set()
    while current not in visited:
        visited.add(current)
        boundary.append(current)
        neighbors = adj.get(current, [])
        next_v = None
        for nb in neighbors:
            if nb not in visited:
                next_v = nb
                break
        if next_v is None:
            break
        current = next_v

    boundary = np.array(boundary)
    interior = np.array([i for i in range(n) if i not in set(boundary)])

    # Map boundary to unit circle
    angles = np.linspace(0, 2 * np.pi, len(boundary), endpoint=False)
    bnd_uv = np.column_stack([np.cos(angles), np.sin(angles)])

    # Build cotangent Laplacian
    L = _cotangent_laplacian(verts, faces, n)

    # Solve for interior UVs
    uv = np.zeros((n, 2), dtype=np.float64)
    uv[boundary] = bnd_uv

    if len(interior) == 0:
        return uv

    # L_ii * uv_interior = -L_ib * uv_boundary
    L_ii = L[np.ix_(interior, interior)]
    L_ib = L[np.ix_(interior, boundary)]

    for dim in range(2):
        rhs = -L_ib @ bnd_uv[:, dim]
        uv[interior, dim] = spsolve(L_ii, rhs)

    return uv


def _cotangent_laplacian(verts, faces, n):
    """Build cotangent weight Laplacian matrix."""
    import scipy.sparse as sp

    rows, cols, vals = [], [], []

    for f in faces:
        for i in range(3):
            i0 = f[i]
            i1 = f[(i + 1) % 3]
            i2 = f[(i + 2) % 3]

            e1 = verts[i0] - verts[i2]
            e2 = verts[i1] - verts[i2]

            cos_angle = np.dot(e1, e2)
            sin_angle = np.linalg.norm(np.cross(e1, e2))
            if sin_angle < 1e-10:
                cot = 0.0
            else:
                cot = cos_angle / sin_angle

            w = 0.5 * cot

            rows.extend([i0, i1, i0, i1])
            cols.extend([i1, i0, i0, i1])
            vals.extend([w, w, -w, -w])

    L = sp.coo_matrix((vals, (rows, cols)), shape=(n, n)).tocsc()
    return L


def _arap_iterate(verts_3d, faces, uv_init, iterations):
    """ARAP iterations: alternating local rotation fitting + global solve."""
    import scipy.sparse as sp
    from scipy.sparse.linalg import spsolve

    n = len(verts_3d)
    uv = uv_init.copy()

    # Pre-compute per-face edge data in 3D
    face_edges_3d = []  # (e1, e2) per face in 3D
    for f in faces:
        v0, v1, v2 = verts_3d[f[0]], verts_3d[f[1]], verts_3d[f[2]]
        face_edges_3d.append((v1 - v0, v2 - v0))

    # Cotangent weights
    L = _cotangent_laplacian(verts_3d, faces, n)

    for iteration in range(iterations):
        # Local step: fit best rotation per face
        rotations = {}  # vertex -> accumulated rotation

        for fi, f in enumerate(faces):
            e1_3d, e2_3d = face_edges_3d[fi]

            # Project 3D edges to 2D local frame
            x_axis = e1_3d / (np.linalg.norm(e1_3d) + 1e-10)
            normal = np.cross(e1_3d, e2_3d)
            normal = normal / (np.linalg.norm(normal) + 1e-10)
            y_axis = np.cross(normal, x_axis)

            e1_local = np.array([np.dot(e1_3d, x_axis), np.dot(e1_3d, y_axis)])
            e2_local = np.array([np.dot(e2_3d, x_axis), np.dot(e2_3d, y_axis)])

            # Current UV edges
            uv0, uv1, uv2 = uv[f[0]], uv[f[1]], uv[f[2]]
            e1_uv = uv1 - uv0
            e2_uv = uv2 - uv0

            # Find best rotation via SVD
            S = np.outer(e1_local, e1_uv) + np.outer(e2_local, e2_uv)
            U, _, Vt = np.linalg.svd(S)
            R = (Vt.T @ U.T)
            if np.linalg.det(R) < 0:
                Vt[-1] *= -1
                R = Vt.T @ U.T

            for vi in f:
                if vi not in rotations:
                    rotations[vi] = np.zeros((2, 2))
                rotations[vi] += R

        # Average rotations per vertex
        for vi in rotations:
            R = rotations[vi]
            U, _, Vt = np.linalg.svd(R)
            rotations[vi] = U @ Vt
            if np.linalg.det(rotations[vi]) < 0:
                Vt[-1] *= -1
                rotations[vi] = U @ Vt

        # Global step: solve for new UVs with rotation targets
        rhs = np.zeros((n, 2), dtype=np.float64)
        for fi, f in enumerate(faces):
            e1_3d, e2_3d = face_edges_3d[fi]

            x_axis = e1_3d / (np.linalg.norm(e1_3d) + 1e-10)
            normal = np.cross(e1_3d, e2_3d)
            normal = normal / (np.linalg.norm(normal) + 1e-10)
            y_axis = np.cross(normal, x_axis)

            edges_local = [
                np.array([np.dot(e1_3d, x_axis), np.dot(e1_3d, y_axis)]),
                np.array([np.dot(e2_3d, x_axis), np.dot(e2_3d, y_axis)]),
                np.array([np.dot(e2_3d - e1_3d, x_axis), np.dot(e2_3d - e1_3d, y_axis)]),
            ]

            pairs = [(f[0], f[1], 0), (f[0], f[2], 1), (f[1], f[2], 2)]
            for vi, vj, ei in pairs:
                Ri = rotations.get(vi, np.eye(2))
                Rj = rotations.get(vj, np.eye(2))
                target = 0.5 * (Ri + Rj) @ edges_local[ei]
                rhs[vi] += target
                rhs[vj] -= target

        # Pin one vertex to prevent drift
        pinned = 0
        L_mod = L.tolil()
        L_mod[pinned, :] = 0
        L_mod[pinned, pinned] = 1
        L_mod = L_mod.tocsc()

        for dim in range(2):
            rhs_dim = rhs[:, dim]
            rhs_dim[pinned] = uv[pinned, dim]
            uv[:, dim] = spsolve(L_mod, rhs_dim)

    return uv


def apply_uv(mesh: trimesh.Trimesh, uv_method: str, original_mesh: trimesh.Trimesh = None,
             texture_size: int = 2048) -> trimesh.Trimesh:
    """Apply UV unwrapping to a mesh and optionally bake texture from original.

    Args:
        mesh: The decimated mesh to unwrap
        uv_method: One of "keep", "xatlas", "lscm", "arap"
        original_mesh: Original mesh to bake texture from (optional)
        texture_size: Baked texture resolution

    Returns:
        Mesh with new UVs and optionally baked texture
    """
    if uv_method == "keep":
        return mesh

    logger.info(f"  UV unwrap: {uv_method}...")

    if uv_method == "xatlas":
        new_verts, new_faces, uvs = unwrap_xatlas(mesh)
    elif uv_method == "lscm":
        new_verts, new_faces, uvs = unwrap_lscm(mesh)
    elif uv_method == "arap":
        new_verts, new_faces, uvs = unwrap_arap(mesh)
    else:
        logger.warning(f"Unknown UV method '{uv_method}', keeping original UVs")
        return mesh

    result = trimesh.Trimesh(vertices=new_verts, faces=new_faces, process=False)

    # Bake texture from original if available
    if original_mesh is not None:
        from .retopology import _bake_texture
        from trimesh.visual.material import PBRMaterial

        baked = _bake_texture(original_mesh, result, uvs, texture_size)
        if baked is not None:
            result.visual = trimesh.visual.TextureVisuals(
                uv=uvs,
                material=PBRMaterial(baseColorTexture=baked),
            )
            logger.info(f"    Texture baked ({texture_size}x{texture_size})")
            return result

    result.visual = trimesh.visual.TextureVisuals(uv=uvs)
    return result
