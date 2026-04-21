"""Web server for the Remesher UI — serves the viewer and handles remesh API calls."""

import os
import sys
import tempfile
import logging
import webbrowser
from pathlib import Path

import io
import threading
from flask import Flask, request, send_file, send_from_directory, jsonify, Response

from remesher.pipeline import process_file, PRESETS, SUPPORTED_FORMATS
from remesher.uv_unwrap import UV_METHODS
from remesher.optimize import optimize_file
from remesher.substance import scan_materials, get_material_info, render_material, render_material_thumbnail, apply_material_to_glb

logging.basicConfig(level=logging.INFO, format="%(message)s")

# Shared progress state for active jobs
_progress_lock = threading.Lock()
_progress_state = {"stage": "", "detail": ""}


def update_progress(stage: str, detail: str = ""):
    with _progress_lock:
        _progress_state["stage"] = stage
        _progress_state["detail"] = detail


app = Flask(__name__, static_folder=None)
app.config["MAX_CONTENT_LENGTH"] = 100 * 1024 * 1024  # 100 MB


@app.errorhandler(413)
def too_large(e):
    return jsonify(error="File too large. Maximum size is 100 MB."), 413


VIEWER_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "viewer")


@app.route("/api/browse-folder", methods=["POST"])
def api_browse_folder():
    """Open a native Windows folder picker dialog and return the selected path."""
    def pick_folder():
        import tkinter as tk
        from tkinter import filedialog
        root = tk.Tk()
        root.withdraw()
        root.attributes("-topmost", True)
        folder = filedialog.askdirectory(title="Select Folder")
        root.destroy()
        return folder

    # Run in a thread to avoid blocking (tkinter must run on main thread on some OS,
    # but on Windows it works from any thread)
    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor(1) as ex:
        folder = ex.submit(pick_folder).result(timeout=120)

    if folder:
        return jsonify(folder=folder)
    return jsonify(folder="")


@app.route("/")
def index():
    return send_from_directory(VIEWER_DIR, "index.html")


@app.route("/viewer")
def viewer_page():
    return send_from_directory(VIEWER_DIR, "viewer.html")


@app.route("/batch")
def batch_page():
    return send_from_directory(VIEWER_DIR, "batch.html")


@app.route("/<path:filename>")
def static_files(filename):
    return send_from_directory(VIEWER_DIR, filename)


@app.route("/api/remesh", methods=["POST"])
def api_remesh():
    if "file" not in request.files:
        return "No file uploaded", 400

    file = request.files["file"]
    ext = Path(file.filename).suffix.lower() if file.filename else ""
    if ext and ext not in SUPPORTED_FORMATS:
        return jsonify(error=f"Unsupported format: {ext}"), 400

    preset = request.form.get("preset", "web")
    target_faces = request.form.get("target_faces", None)
    ratio = request.form.get("ratio", None)
    quality = float(request.form.get("quality", 0.3))
    uv_method = request.form.get("uv_method", "keep")
    texture_size = int(request.form.get("texture_size", 2048))
    smooth_method = request.form.get("smooth_method", "none")
    smooth_iterations = int(request.form.get("smooth_iterations", 3))

    if target_faces:
        target_faces = int(target_faces)
    if ratio:
        ratio = float(ratio)

    if uv_method not in UV_METHODS:
        uv_method = "keep"

    # Validate preset
    if preset not in PRESETS and preset != "custom":
        preset = "web"
    if preset == "custom":
        preset = None

    with tempfile.TemporaryDirectory() as tmp_dir:
        # Save uploaded file
        ext = Path(file.filename).suffix or ".glb"
        input_path = os.path.join(tmp_dir, f"input{ext}")
        file.save(input_path)

        # Output always as GLB
        output_path = os.path.join(tmp_dir, "output.glb")

        update_progress("Starting decimation...")
        result = process_file(
            input_path=input_path,
            output_path=output_path,
            preset=preset,
            target_faces=target_faces,
            ratio=ratio,
            quality_thr=quality,
            progress_cb=update_progress,
            uv_method=uv_method,
            texture_size=texture_size,
            smooth_method=smooth_method,
            smooth_iterations=smooth_iterations,
        )
        update_progress("")

        if not result.success:
            return f"Remesh failed: {result.error}", 500

        # Read into memory before temp dir is cleaned up (Windows file locking)
        with open(output_path, "rb") as f:
            data = f.read()

    return Response(
        data,
        mimetype="model/gltf-binary",
        headers={"Content-Disposition": "attachment; filename=remeshed.glb"},
    )


@app.route("/api/optimize", methods=["POST"])
def api_optimize():
    if "file" not in request.files:
        return "No file uploaded", 400

    file = request.files["file"]
    ext = Path(file.filename).suffix.lower() if file.filename else ""
    if ext not in (".glb",):
        return jsonify(error="Optimize only supports .glb files"), 400

    max_texture_size = int(request.form.get("max_texture_size", 2048))
    jpeg_quality = int(request.form.get("jpeg_quality", 80))

    with tempfile.TemporaryDirectory() as tmp_dir:
        input_path = os.path.join(tmp_dir, "input.glb")
        file.save(input_path)
        output_path = os.path.join(tmp_dir, "output.glb")

        update_progress("Optimizing file weight...")
        result = optimize_file(
            input_path=input_path,
            output_path=output_path,
            max_texture_size=max_texture_size,
            jpeg_quality=jpeg_quality,
            progress_cb=update_progress,
        )
        update_progress("")

        if not result.success:
            return f"Optimize failed: {result.error}", 500

        with open(output_path, "rb") as f:
            data = f.read()

    return Response(
        data,
        mimetype="model/gltf-binary",
        headers={
            "Content-Disposition": "attachment; filename=optimized.glb",
            "X-Original-Size": str(result.original_size),
            "X-Final-Size": str(result.final_size),
            "X-Reduction-Pct": f"{result.reduction_pct:.1f}",
            "X-Textures-Compressed": str(result.textures_compressed),
        },
    )


@app.route("/api/remesh-optimize", methods=["POST"])
def api_remesh_optimize():
    """Remesh then optimize in one step."""
    if "file" not in request.files:
        return "No file uploaded", 400

    file = request.files["file"]
    ext_check = Path(file.filename).suffix.lower() if file.filename else ""
    if ext_check and ext_check not in SUPPORTED_FORMATS:
        return jsonify(error=f"Unsupported format: {ext_check}"), 400

    preset = request.form.get("preset", "web")
    target_faces = request.form.get("target_faces", None)
    ratio = request.form.get("ratio", None)
    quality = float(request.form.get("quality", 0.3))
    uv_method = request.form.get("uv_method", "keep")
    texture_size = int(request.form.get("texture_size", 2048))
    max_texture_size = int(request.form.get("max_texture_size", 2048))
    jpeg_quality = int(request.form.get("jpeg_quality", 80))
    smooth_method = request.form.get("smooth_method", "none")
    smooth_iterations = int(request.form.get("smooth_iterations", 3))

    if target_faces:
        target_faces = int(target_faces)
    if ratio:
        ratio = float(ratio)
    if uv_method not in UV_METHODS:
        uv_method = "keep"
    if preset not in PRESETS and preset != "custom":
        preset = "web"
    if preset == "custom":
        preset = None

    with tempfile.TemporaryDirectory() as tmp_dir:
        ext = Path(file.filename).suffix or ".glb"
        input_path = os.path.join(tmp_dir, f"input{ext}")
        file.save(input_path)

        remeshed_path = os.path.join(tmp_dir, "remeshed.glb")
        output_path = os.path.join(tmp_dir, "output.glb")

        # Step 1: Remesh
        update_progress("Step 1/2: Remeshing...")
        remesh_result = process_file(
            input_path=input_path,
            output_path=remeshed_path,
            preset=preset,
            target_faces=target_faces,
            ratio=ratio,
            quality_thr=quality,
            progress_cb=update_progress,
            uv_method=uv_method,
            texture_size=texture_size,
            smooth_method=smooth_method,
            smooth_iterations=smooth_iterations,
        )

        if not remesh_result.success:
            update_progress("")
            return f"Remesh failed: {remesh_result.error}", 500

        # Step 2: Optimize
        update_progress("Step 2/2: Optimizing weight...")
        opt_result = optimize_file(
            input_path=remeshed_path,
            output_path=output_path,
            max_texture_size=max_texture_size,
            jpeg_quality=jpeg_quality,
            progress_cb=update_progress,
        )
        update_progress("")

        if not opt_result.success:
            # Optimization failed — return the remeshed file anyway
            with open(remeshed_path, "rb") as f:
                data = f.read()
        else:
            with open(output_path, "rb") as f:
                data = f.read()

    return Response(
        data,
        mimetype="model/gltf-binary",
        headers={
            "Content-Disposition": "attachment; filename=remeshed_optimized.glb",
            "X-Original-Faces": str(remesh_result.original_faces),
            "X-Final-Faces": str(remesh_result.final_faces),
            "X-Face-Reduction": f"{remesh_result.reduction_pct:.1f}",
            "X-Original-Size": str(opt_result.original_size if opt_result.success else 0),
            "X-Final-Size": str(opt_result.final_size if opt_result.success else 0),
            "X-Size-Reduction": f"{opt_result.reduction_pct:.1f}" if opt_result.success else "0",
        },
    )


# ---------------------------------------------------------------------------
# Material Assigner endpoints
# ---------------------------------------------------------------------------

# Cache for rendered material thumbnails
_material_thumb_cache = {}


@app.route("/materials")
def materials_page():
    return send_from_directory(VIEWER_DIR, "materials.html")


@app.route("/api/materials/scan", methods=["POST"])
def api_materials_scan():
    data = request.get_json() or {}
    extra_paths = data.get("extra_paths", [])
    materials = scan_materials(extra_paths)
    return jsonify(materials=materials, total=len(materials))


@app.route("/api/materials/info", methods=["POST"])
def api_materials_info():
    data = request.get_json()
    sbsar_path = data.get("path", "")
    if not sbsar_path or not os.path.exists(sbsar_path):
        return jsonify(error="File not found"), 404
    info = get_material_info(sbsar_path)
    return jsonify(info)


@app.route("/api/materials/thumbnail", methods=["POST"])
def api_materials_thumbnail():
    data = request.get_json()
    sbsar_path = data.get("path", "")
    preset = data.get("preset", None)

    if not sbsar_path or not os.path.exists(sbsar_path):
        return jsonify(error="File not found"), 404

    cache_key = f"{sbsar_path}|{preset or ''}"
    if cache_key in _material_thumb_cache:
        return send_file(_material_thumb_cache[cache_key], mimetype="image/jpeg")

    thumb_dir = tempfile.mkdtemp(prefix="mat_thumb_")
    thumb_path = os.path.join(thumb_dir, "thumb.jpg")

    if render_material_thumbnail(sbsar_path, thumb_path, size=256, preset=preset):
        _material_thumb_cache[cache_key] = thumb_path
        return send_file(thumb_path, mimetype="image/jpeg")

    return jsonify(error="Failed to render thumbnail"), 500


@app.route("/api/materials/render-maps", methods=["POST"])
def api_materials_render_maps():
    """Render all PBR maps for a material and return file paths to access them."""
    data = request.get_json()
    sbsar_path = data.get("path", "")
    resolution = int(data.get("resolution", 512))
    preset = data.get("preset", None)

    if not sbsar_path or not os.path.exists(sbsar_path):
        return jsonify(error="File not found"), 404

    cache_key = f"maps|{sbsar_path}|{preset or ''}|{resolution}"
    if cache_key in _material_thumb_cache:
        return jsonify(_material_thumb_cache[cache_key])

    render_dir = tempfile.mkdtemp(prefix="mat_maps_")
    maps = render_material(sbsar_path, render_dir, resolution=resolution, preset=preset)

    if not maps:
        return jsonify(error="Render failed"), 500

    # Build URL map for client to fetch
    url_maps = {}
    for usage, filepath in maps.items():
        if os.path.exists(filepath):
            url_maps[usage] = f"/api/materials/serve-map?path={filepath.replace(os.sep, '/')}"

    _material_thumb_cache[cache_key] = url_maps
    return jsonify(url_maps)


@app.route("/api/materials/serve-map")
def api_materials_serve_map():
    """Serve a rendered material map file."""
    filepath = request.args.get("path", "")
    if not filepath:
        return "Not found", 404
    # Security: resolve the requested path and ensure it lives under the system temp dir
    try:
        resolved = Path(filepath).resolve(strict=True)
        tmp_root = Path(tempfile.gettempdir()).resolve()
        resolved.relative_to(tmp_root)
    except (FileNotFoundError, ValueError):
        return "Access denied", 403
    return send_file(str(resolved), mimetype="image/png")


@app.route("/api/materials/combined-mr", methods=["POST"])
def api_materials_combined_mr():
    """Combine separate roughness + metallic maps into a single glTF-compatible map.

    Output: R=AO (white if none), G=Roughness, B=Metallic
    """
    data = request.get_json()
    roughness_url = data.get("roughness", "")
    metallic_url = data.get("metallic", "")
    ao_url = data.get("ao", "")

    from PIL import Image as PILImage
    import numpy as np

    size = 1024  # Will be resized to match

    def load_channel(path):
        if not path or not os.path.exists(path):
            return None
        return np.array(PILImage.open(path).convert("L"))

    # Extract file paths from URLs
    def url_to_path(url):
        if not url:
            return ""
        if "path=" in url:
            return url.split("path=")[1]
        return ""

    r_path = url_to_path(roughness_url)
    m_path = url_to_path(metallic_url)
    a_path = url_to_path(ao_url)

    r_img = load_channel(r_path)
    m_img = load_channel(m_path)
    a_img = load_channel(a_path)

    # Determine size from first available
    for img in [r_img, m_img, a_img]:
        if img is not None:
            size = img.shape[0]
            break

    if r_img is None:
        r_img = np.full((size, size), 128, dtype=np.uint8)
    if m_img is None:
        m_img = np.zeros((size, size), dtype=np.uint8)
    if a_img is None:
        a_img = np.full((size, size), 255, dtype=np.uint8)

    # Resize to match
    from PIL import Image as PILImage2
    if r_img.shape[0] != size:
        r_img = np.array(PILImage2.fromarray(r_img).resize((size, size)))
    if m_img.shape[0] != size:
        m_img = np.array(PILImage2.fromarray(m_img).resize((size, size)))
    if a_img.shape[0] != size:
        a_img = np.array(PILImage2.fromarray(a_img).resize((size, size)))

    # Combine: R=AO, G=Roughness, B=Metallic (glTF ORM convention)
    combined = np.stack([a_img, r_img, m_img], axis=-1)

    import tempfile as _tmp
    with _tmp.NamedTemporaryFile(suffix=".png", delete=False) as f:
        out_path = f.name
    PILImage2.fromarray(combined).save(out_path, format="PNG")

    return send_file(out_path, mimetype="image/png")


@app.route("/api/materials/apply", methods=["POST"])
def api_materials_apply():
    """Render material and apply to a GLB model."""
    data = request.get_json()
    sbsar_path = data.get("sbsar_path", "")
    model_path = data.get("model_path", "")
    output_path = data.get("output_path", "")
    resolution = int(data.get("resolution", 1024))
    preset = data.get("preset", None)

    if not sbsar_path or not os.path.exists(sbsar_path):
        return jsonify(error="Material not found"), 404
    if not model_path or not os.path.exists(model_path):
        return jsonify(error="Model not found"), 404
    if not output_path:
        return jsonify(error="No output path"), 400

    update_progress("Rendering material maps...")

    with tempfile.TemporaryDirectory() as tmp_dir:
        maps = render_material(sbsar_path, tmp_dir, resolution=resolution, preset=preset)

        if not maps:
            update_progress("")
            return jsonify(error="Failed to render material"), 500

        success = apply_material_to_glb(
            model_path, output_path, maps, progress_cb=update_progress,
        )
        update_progress("")

    if success:
        return jsonify(ok=True, output=output_path)
    return jsonify(error="Failed to apply material"), 500


@app.route("/api/materials/apply-upload", methods=["POST"])
def api_materials_apply_upload():
    """Render material and apply to an uploaded GLB, return the result."""
    if "file" not in request.files:
        return "No file uploaded", 400

    file = request.files["file"]
    sbsar_path = request.form.get("sbsar_path", "")
    resolution = int(request.form.get("resolution", 1024))
    preset = request.form.get("preset", "") or None

    if not sbsar_path or not os.path.exists(sbsar_path):
        return jsonify(error="Material not found"), 404

    with tempfile.TemporaryDirectory() as tmp_dir:
        input_path = os.path.join(tmp_dir, "input.glb")
        file.save(input_path)
        output_path = os.path.join(tmp_dir, "output.glb")

        update_progress("Rendering material maps...")
        maps = render_material(sbsar_path, tmp_dir, resolution=resolution, preset=preset)
        if not maps:
            update_progress("")
            return "Failed to render material", 500

        success = apply_material_to_glb(input_path, output_path, maps, progress_cb=update_progress)
        update_progress("")

        if not success:
            return "Failed to apply material", 500

        with open(output_path, "rb") as f:
            data = f.read()

    return Response(
        data,
        mimetype="model/gltf-binary",
        headers={"Content-Disposition": "attachment; filename=material_applied.glb"},
    )


@app.route("/api/materials/batch-apply", methods=["POST"])
def api_materials_batch_apply():
    """Batch apply: list of {model_path, sbsar_path, output_path, preset, resolution}."""
    data = request.get_json()
    assignments = data.get("assignments", [])
    resolution = int(data.get("resolution", 1024))

    if not assignments:
        return jsonify(error="No assignments"), 400

    with _batch_lock:
        if _batch_state["running"]:
            return jsonify(error="Batch already running"), 409
        _reset_batch_state()
        _batch_state["running"] = True
        _batch_state["total"] = len(assignments)

    # Pre-render unique materials (avoid re-rendering same sbsar)
    def run_material_batch():
        rendered_cache = {}

        for idx, assignment in enumerate(assignments):
            with _batch_lock:
                if _batch_state["cancelled"]:
                    break
                _batch_state["current_file"] = assignment.get("model_name", "")
                _batch_state["completed"] = idx

            sbsar = assignment["sbsar_path"]
            preset = assignment.get("preset")
            model = assignment["model_path"]
            output = assignment["output_path"]
            res = assignment.get("resolution", resolution)

            cache_key = f"{sbsar}|{preset or ''}|{res}"

            update_progress(f"Material batch [{idx+1}/{len(assignments)}]: {Path(model).name}")

            try:
                # Render material (cached)
                if cache_key not in rendered_cache:
                    tmp_dir = tempfile.mkdtemp(prefix="mat_batch_")
                    maps = render_material(sbsar, tmp_dir, resolution=res, preset=preset)
                    rendered_cache[cache_key] = maps
                else:
                    maps = rendered_cache[cache_key]

                if not maps:
                    raise ValueError("Material render failed")

                os.makedirs(os.path.dirname(output) or ".", exist_ok=True)
                success = apply_material_to_glb(model, output, maps, progress_cb=update_progress)

                entry = {
                    "file": Path(model).name,
                    "status": "ok" if success else "error",
                    "error": None if success else "Apply failed",
                }
            except Exception as e:
                entry = {"file": Path(model).name, "status": "error", "error": str(e)}

            with _batch_lock:
                _batch_state["results"].append(entry)

        with _batch_lock:
            _batch_state["completed"] = len(_batch_state["results"])
            _batch_state["current_file"] = ""
            _batch_state["running"] = False
            update_progress("")

    thread = threading.Thread(target=run_material_batch, daemon=True)
    thread.start()

    return jsonify(ok=True, total=len(assignments))


@app.route("/api/gallery/scan", methods=["POST"])
def api_gallery_scan():
    """Scan a folder and return model list with metadata."""
    data = request.get_json()
    folder = data.get("folder", "")

    if not folder or not os.path.isdir(folder):
        return jsonify(error="Invalid folder path"), 400

    folder_path = Path(folder).resolve()
    models = []

    for f in sorted(folder_path.rglob("*")):
        if f.is_file() and f.suffix.lower() in SUPPORTED_FORMATS:
            rel = str(f.relative_to(folder_path)).replace("\\", "/")
            size_bytes = f.stat().st_size
            # Get folder group
            parts = rel.split("/")
            group = "/".join(parts[:-1]) if len(parts) > 1 else ""
            models.append({
                "path": rel,
                "name": f.stem,
                "ext": f.suffix.lower(),
                "group": group,
                "size_mb": round(size_bytes / (1024 * 1024), 2),
            })

    return jsonify(models=models, total=len(models), folder=str(folder_path))


@app.route("/api/gallery/save-state", methods=["POST"])
def api_gallery_save_state():
    """Save gallery visibility state (hidden models) to a JSON file in the folder."""
    import json as _json
    data = request.get_json()
    folder = data.get("folder", "")
    hidden = data.get("hidden", [])

    if not folder or not os.path.isdir(folder):
        return jsonify(error="Invalid folder"), 400

    state_path = os.path.join(folder, ".remesher-gallery.json")
    try:
        with open(state_path, "w", encoding="utf-8") as f:
            _json.dump({"hidden": hidden}, f, ensure_ascii=False, indent=2)
        return jsonify(ok=True)
    except Exception as e:
        return jsonify(error=str(e)), 500


@app.route("/api/gallery/load-state", methods=["POST"])
def api_gallery_load_state():
    """Load gallery visibility state from a JSON file in the folder."""
    import json as _json
    data = request.get_json()
    folder = data.get("folder", "")

    if not folder or not os.path.isdir(folder):
        return jsonify(error="Invalid folder"), 400

    state_path = os.path.join(folder, ".remesher-gallery.json")
    if not os.path.exists(state_path):
        return jsonify(hidden=None)

    try:
        with open(state_path, "r", encoding="utf-8") as f:
            state = _json.load(f)
        return jsonify(hidden=state.get("hidden", []))
    except Exception:
        return jsonify(hidden=None)


@app.route("/api/gallery/file")
def api_gallery_file():
    """Serve a specific model file for the gallery viewer."""
    folder = request.args.get("folder", "")
    file_path = request.args.get("path", "")

    if not folder or not file_path:
        return "Missing params", 400

    full_path = Path(folder) / file_path
    if not full_path.exists() or not full_path.is_file():
        return "File not found", 404

    # Security: ensure the file is under the folder
    try:
        full_path.resolve().relative_to(Path(folder).resolve())
    except ValueError:
        return "Access denied", 403

    return send_file(str(full_path.resolve()))


@app.route("/gallery")
def gallery_page():
    return send_from_directory(VIEWER_DIR, "gallery.html")


@app.route("/api/progress")
def api_progress():
    with _progress_lock:
        return jsonify(_progress_state)


@app.route("/api/presets")
def api_presets():
    return jsonify(PRESETS)


# ---------------------------------------------------------------------------
# Batch automation endpoints
# ---------------------------------------------------------------------------

# Batch job state
_batch_lock = threading.Lock()
_batch_state = {
    "running": False,
    "total": 0,
    "completed": 0,
    "current_file": "",
    "results": [],       # list of {file, status, original_faces, final_faces, error}
    "cancelled": False,
}


def _reset_batch_state():
    _batch_state["running"] = False
    _batch_state["total"] = 0
    _batch_state["completed"] = 0
    _batch_state["current_file"] = ""
    _batch_state["results"] = []
    _batch_state["cancelled"] = False


@app.route("/api/batch/scan", methods=["POST"])
def api_batch_scan():
    """Scan a folder for 3D files recursively."""
    data = request.get_json()
    folder = data.get("folder", "")

    if not folder or not os.path.isdir(folder):
        return jsonify(error="Invalid folder path"), 400

    files = []
    folder_path = Path(folder).resolve()
    for f in sorted(folder_path.rglob("*")):
        if f.is_file() and f.suffix.lower() in SUPPORTED_FORMATS:
            rel = str(f.relative_to(folder_path)).replace("\\", "/")
            size_mb = f.stat().st_size / (1024 * 1024)
            files.append({
                "path": rel,
                "size_mb": round(size_mb, 2),
                "ext": f.suffix.lower(),
            })

    return jsonify(files=files, total=len(files), folder=str(folder_path))


@app.route("/api/batch/start", methods=["POST"])
def api_batch_start():
    """Start batch processing in a background thread."""
    with _batch_lock:
        if _batch_state["running"]:
            return jsonify(error="Batch already running"), 409

    data = request.get_json()
    input_folder = data.get("input_folder", "")
    output_folder = data.get("output_folder", "")
    files = data.get("files", [])  # list of relative paths
    batch_mode = data.get("mode", "remesh")  # "remesh" or "optimize"

    # Remesh params
    preset = data.get("preset", "web")
    target_faces = data.get("target_faces", None)
    ratio = data.get("ratio", None)
    quality = data.get("quality", 0.3)
    uv_method = data.get("uv_method", "keep")
    texture_size = data.get("texture_size", 2048)

    # Optimize params
    max_texture_size = data.get("max_texture_size", 2048)
    jpeg_quality = data.get("jpeg_quality", 80)

    # Smoothing params
    smooth_method = data.get("smooth_method", "none")
    smooth_iterations = int(data.get("smooth_iterations", 3))

    if not input_folder or not os.path.isdir(input_folder):
        return jsonify(error="Invalid input folder"), 400
    if not output_folder:
        return jsonify(error="No output folder specified"), 400
    if not files:
        return jsonify(error="No files to process"), 400

    if target_faces is not None:
        target_faces = int(target_faces)
    if ratio is not None:
        ratio = float(ratio)

    if preset not in PRESETS and preset != "custom":
        preset = "web"
    if preset == "custom":
        preset = None

    if uv_method not in UV_METHODS:
        uv_method = "keep"

    with _batch_lock:
        _reset_batch_state()
        _batch_state["running"] = True
        _batch_state["total"] = len(files)

    def run_batch():
        input_base = Path(input_folder).resolve()
        output_base = Path(output_folder).resolve()

        for idx, rel_path in enumerate(files):
            with _batch_lock:
                if _batch_state["cancelled"]:
                    break
                _batch_state["current_file"] = rel_path
                _batch_state["completed"] = idx

            input_path = str(input_base / rel_path)
            out_rel = Path(rel_path).with_suffix(".glb")
            output_path = str(output_base / out_rel)

            os.makedirs(os.path.dirname(output_path) or str(output_base), exist_ok=True)

            update_progress(f"Batch [{idx+1}/{len(files)}]: {rel_path}")

            try:
                if batch_mode == "optimize":
                    result = optimize_file(
                        input_path=input_path,
                        output_path=output_path,
                        max_texture_size=max_texture_size,
                        jpeg_quality=jpeg_quality,
                        progress_cb=update_progress,
                    )
                    entry = {
                        "file": rel_path,
                        "status": "ok" if result.success else "error",
                        "original_faces": 0,
                        "final_faces": 0,
                        "reduction_pct": round(result.reduction_pct, 1),
                        "original_size": result.original_size,
                        "final_size": result.final_size,
                        "error": result.error,
                    }
                elif batch_mode == "remesh-optimize":
                    # Step 1: Remesh to temp file
                    tmp_remeshed = output_path + ".tmp.glb"
                    remesh_r = process_file(
                        input_path=input_path,
                        output_path=tmp_remeshed,
                        preset=preset,
                        target_faces=target_faces,
                        ratio=ratio,
                        quality_thr=quality,
                        progress_cb=update_progress,
                        uv_method=uv_method,
                        texture_size=texture_size,
                        smooth_method=smooth_method,
                        smooth_iterations=smooth_iterations,
                    )
                    if remesh_r.success:
                        # Step 2: Optimize
                        opt_r = optimize_file(
                            input_path=tmp_remeshed,
                            output_path=output_path,
                            max_texture_size=max_texture_size,
                            jpeg_quality=jpeg_quality,
                            progress_cb=update_progress,
                        )
                        try:
                            os.unlink(tmp_remeshed)
                        except OSError:
                            pass
                        entry = {
                            "file": rel_path,
                            "status": "ok" if opt_r.success else "error",
                            "original_faces": remesh_r.original_faces,
                            "final_faces": remesh_r.final_faces,
                            "reduction_pct": round(remesh_r.reduction_pct, 1),
                            "original_size": opt_r.original_size if opt_r.success else 0,
                            "final_size": opt_r.final_size if opt_r.success else 0,
                            "error": opt_r.error,
                        }
                    else:
                        entry = {
                            "file": rel_path,
                            "status": "error",
                            "original_faces": remesh_r.original_faces,
                            "final_faces": 0,
                            "reduction_pct": 0,
                            "error": remesh_r.error,
                        }
                else:
                    result = process_file(
                        input_path=input_path,
                        output_path=output_path,
                        preset=preset,
                        target_faces=target_faces,
                        ratio=ratio,
                        quality_thr=quality,
                        progress_cb=update_progress,
                        uv_method=uv_method,
                        texture_size=texture_size,
                        smooth_method=smooth_method,
                        smooth_iterations=smooth_iterations,
                    )
                    entry = {
                        "file": rel_path,
                        "status": "ok" if result.success else "error",
                        "original_faces": result.original_faces,
                        "final_faces": result.final_faces,
                        "reduction_pct": round(result.reduction_pct, 1),
                        "error": result.error,
                    }
            except Exception as e:
                entry = {
                    "file": rel_path,
                    "status": "error",
                    "original_faces": 0,
                    "final_faces": 0,
                    "reduction_pct": 0,
                    "error": str(e),
                }

            with _batch_lock:
                _batch_state["results"].append(entry)

        with _batch_lock:
            _batch_state["completed"] = len(_batch_state["results"])
            _batch_state["current_file"] = ""
            _batch_state["running"] = False
            update_progress("")

    thread = threading.Thread(target=run_batch, daemon=True)
    thread.start()

    return jsonify(ok=True, total=len(files))


@app.route("/api/batch/status")
def api_batch_status():
    """Get current batch processing status."""
    with _batch_lock:
        return jsonify(
            running=_batch_state["running"],
            total=_batch_state["total"],
            completed=_batch_state["completed"],
            current_file=_batch_state["current_file"],
            results=_batch_state["results"],
            cancelled=_batch_state["cancelled"],
        )


@app.route("/api/batch/cancel", methods=["POST"])
def api_batch_cancel():
    """Cancel the current batch job."""
    with _batch_lock:
        _batch_state["cancelled"] = True
    return jsonify(ok=True)


def main():
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 5000
    print(f"\n  Remesher UI running at: http://localhost:{port}\n")
    webbrowser.open(f"http://localhost:{port}")

    if os.environ.get("REMESHER_ENV") == "production":
        from waitress import serve
        serve(app, host="0.0.0.0", port=port)
    else:
        app.run(host="0.0.0.0", port=port, debug=False)


if __name__ == "__main__":
    main()
