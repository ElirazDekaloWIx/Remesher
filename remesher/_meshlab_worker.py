"""Subprocess worker that runs pymeshlab in isolation.

pymeshlab is a native library that can segfault on pathological inputs
(huge meshes, inconsistent UV coords, non-manifold edges). When that
happens in-process it kills the whole Flask server. This worker runs
the meshlab pipeline as a child process — a crash here returns a
non-zero exit code, and the parent surfaces a clean error.

CLI:
    python -m remesher._meshlab_worker <params.json>

The JSON file describes the work; output.obj path is given inside it.
We use a JSON file (not argv) because Windows command-lines have a
length limit and we want to be future-proof.
"""

from __future__ import annotations

import json
import sys
import logging

logger = logging.getLogger(__name__)


def _has_consistent_tex(ms) -> bool:
    """Heuristic: does every face have valid (non-degenerate) tex coords?

    pymeshlab's `_with_texture` decimator hard-fails (and sometimes
    segfaults) when some faces have texture and others don't. We can't
    fully introspect per-face wedge tex coords cheaply, but we can run
    the consistency-fix filter and trust pymeshlab's own check.
    """
    try:
        # This filter normalizes per-wedge tex coords; if it raises,
        # the mesh is unsalvageable for the texture-aware path.
        ms.compute_texcoord_transfer_wedge_to_vertex()
        return True
    except Exception:
        return False


def _clean(ms):
    ms.meshing_remove_duplicate_vertices()
    ms.meshing_remove_duplicate_faces()
    ms.meshing_remove_null_faces()
    for fn in ("meshing_repair_non_manifold_edges", "meshing_repair_non_manifold_vertices"):
        try:
            getattr(ms, fn)()
        except Exception:
            pass


def _decimate_with_texture(ms, target: int, quality: float):
    ms.meshing_decimation_quadric_edge_collapse_with_texture(
        targetfacenum=target,
        qualitythr=quality,
        preserveboundary=True,
        optimalplacement=True,
        planarquadric=True,
    )


def _decimate_plain(ms, target: int, quality: float):
    ms.meshing_decimation_quadric_edge_collapse(
        targetfacenum=target,
        qualitythr=quality,
        preserveboundary=True,
        preservenormal=True,
        preservetopology=True,
        optimalplacement=True,
        planarquadric=True,
    )


def _smooth(ms, method: str, iterations: int):
    if method == "laplacian":
        ms.apply_coord_laplacian_smoothing(
            stepsmoothnum=iterations, boundary=True, cotangentweight=True,
        )
    elif method == "taubin":
        ms.apply_coord_taubin_smoothing(
            stepsmoothnum=iterations, lambda_=0.5, mu=-0.53,
        )


def run(params: dict) -> int:
    import pymeshlab

    in_path = params["input"]
    out_path = params["output"]
    target = int(params["target_faces"])
    quality = float(params.get("quality", 0.3))
    has_texture = bool(params.get("has_texture", False))
    smooth_method = params.get("smooth_method", "none") or "none"
    smooth_iter = int(params.get("smooth_iterations", 3))

    ms = pymeshlab.MeshSet()
    ms.load_new_mesh(in_path)
    _clean(ms)

    current = ms.current_mesh().face_number()

    # For very large meshes, do a non-texture pre-pass to get under a sane
    # ceiling. The texture-aware decimator is much heavier and crashes
    # more easily on huge inputs.
    PRE_PASS_THRESHOLD = 500_000
    PRE_PASS_TARGET = 200_000
    if current > PRE_PASS_THRESHOLD and target < PRE_PASS_TARGET:
        try:
            _decimate_plain(ms, max(PRE_PASS_TARGET, target), quality)
        except Exception as e:
            print(f"[worker] pre-pass failed: {e}", file=sys.stderr)

    # Texture-aware path is fragile. Validate UV consistency first; if it
    # looks bad, skip it instead of letting pymeshlab segfault.
    used_texture_path = False
    if has_texture:
        if _has_consistent_tex(ms):
            try:
                _decimate_with_texture(ms, target, quality)
                used_texture_path = True
            except Exception as e:
                print(f"[worker] texture-aware decimation failed: {e}", file=sys.stderr)
        else:
            print("[worker] UVs inconsistent — using plain decimation", file=sys.stderr)

    if not used_texture_path:
        _decimate_plain(ms, target, quality)

    if smooth_method != "none":
        try:
            _smooth(ms, smooth_method, smooth_iter)
        except Exception as e:
            print(f"[worker] smoothing failed: {e}", file=sys.stderr)

    try:
        ms.compute_normal_per_vertex()
        ms.compute_normal_per_face()
    except Exception:
        pass

    ms.save_current_mesh(out_path)
    return 0


def main():
    if len(sys.argv) != 2:
        print("usage: python -m remesher._meshlab_worker <params.json>", file=sys.stderr)
        sys.exit(2)

    with open(sys.argv[1], "r", encoding="utf-8") as f:
        params = json.load(f)

    sys.exit(run(params))


if __name__ == "__main__":
    main()
