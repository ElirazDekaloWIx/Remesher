"""Substance 3D .sbsar material integration — scan, render, apply to GLB."""

import os
import io
import json
import subprocess
import tempfile
import logging
from pathlib import Path
from dataclasses import dataclass

import numpy as np
from PIL import Image
import pygltflib

logger = logging.getLogger(__name__)

SBSRENDER = r"C:\Program Files\Adobe\Adobe Substance 3D Designer\sbsrender.exe"

# Known sbsar locations on this system
SBSAR_SEARCH_PATHS = [
    r"C:\Program Files\Adobe\Adobe Dimension\resources\common\scene-assets\materials",
    r"C:\Program Files\Adobe\Adobe Substance 3D Painter\resources\starter_assets",
    r"C:\Program Files\Adobe\Adobe Photoshop 2026\Required\UXP\com.adobe.photoshop-material-filters\parametric_assets\default_parametric_assets",
]


@dataclass
class MaterialInfo:
    path: str
    name: str
    category: str
    presets: list
    inputs: list
    outputs: list


def scan_materials(extra_paths: list = None) -> list[dict]:
    """Scan for .sbsar files in known locations and extra paths."""
    search_paths = SBSAR_SEARCH_PATHS.copy()
    if extra_paths:
        search_paths.extend(extra_paths)

    materials = []
    seen = set()

    for base_path in search_paths:
        base = Path(base_path)
        if not base.exists():
            continue

        for f in sorted(base.rglob("*.sbsar")):
            if f.name in seen:
                continue
            seen.add(f.name)

            # Derive category from folder structure
            try:
                rel = f.relative_to(base)
                parts = rel.parts[:-1]
                category = parts[0] if parts else "Other"
            except ValueError:
                category = "Other"

            # Clean up category name
            category = category.replace("_", " ").replace("-", " ").title()

            materials.append({
                "path": str(f),
                "name": f.stem.replace("_", " ").title(),
                "filename": f.name,
                "category": category,
                "size_kb": round(f.stat().st_size / 1024),
            })

    return materials


def get_material_info(sbsar_path: str) -> dict:
    """Get detailed info about an sbsar file (inputs, outputs, presets)."""
    result = subprocess.run(
        [SBSRENDER, "info", sbsar_path],
        capture_output=True, text=True, timeout=15,
    )

    info = {"presets": [], "inputs": [], "outputs": []}

    for line in result.stdout.strip().split("\n"):
        line = line.strip()
        if line.startswith("PRESET "):
            info["presets"].append(line[7:])
        elif line.startswith("INPUT "):
            parts = line[6:].split(" ", 1)
            if len(parts) == 2:
                info["inputs"].append({"name": parts[0], "type": parts[1]})
        elif line.startswith("OUTPUT "):
            parts = line[7:].split(" ", 1)
            if len(parts) == 2:
                info["outputs"].append({"name": parts[0], "usage": parts[1]})

    return info


def render_material(sbsar_path: str, output_dir: str, resolution: int = 1024,
                    preset: str = None, params: dict = None) -> dict:
    """Render sbsar to texture maps. Returns dict of {usage: filepath}.

    Args:
        resolution: texture size (power of 2: 256, 512, 1024, 2048)
        preset: optional preset name
        params: optional dict of parameter overrides
    """
    os.makedirs(output_dir, exist_ok=True)

    # Resolution to sbsrender size code (log2)
    size_code = max(0, min(12, int(np.log2(resolution))))

    cmd = [
        SBSRENDER, "render", sbsar_path,
        "--output-path", output_dir,
        "--output-name", "{outputNodeName}",
        "--output-format", "png",
        "--set-value", f"$outputsize@{size_code},{size_code}",
    ]

    if preset:
        cmd.extend(["--preset-name", preset])

    if params:
        for key, value in params.items():
            if isinstance(value, (list, tuple)):
                val_str = ",".join(str(v) for v in value)
            else:
                val_str = str(value)
            cmd.extend(["--set-value", f"{key}@{val_str}"])

    logger.info(f"  Rendering material: {Path(sbsar_path).stem}")

    result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)

    if result.returncode != 0:
        logger.error(f"  sbsrender failed: {result.stderr}")
        return {}

    # Parse output JSON to find rendered files
    maps = {}
    try:
        output_data = json.loads(result.stdout)
        for graph in output_data:
            for output in graph.get("outputs", []):
                usages = output.get("usages", [])
                filepath = output.get("value", "")
                if filepath and os.path.exists(filepath):
                    for usage in usages:
                        maps[usage] = filepath
    except json.JSONDecodeError:
        # Fallback: look for known map names in output dir
        for fname in os.listdir(output_dir):
            if not fname.endswith(".png"):
                continue
            name = fname.replace(".png", "").lower()
            if "basecolor" in name or "diffuse" in name:
                maps["baseColor"] = os.path.join(output_dir, fname)
            elif "normal" in name:
                maps["normal"] = os.path.join(output_dir, fname)
            elif "roughness" in name:
                maps["roughness"] = os.path.join(output_dir, fname)
            elif "metallic" in name:
                maps["metallic"] = os.path.join(output_dir, fname)
            elif "height" in name:
                maps["height"] = os.path.join(output_dir, fname)
            elif "occlusion" in name or "ao" in name:
                maps["ambientOcclusion"] = os.path.join(output_dir, fname)

    logger.info(f"    Rendered maps: {list(maps.keys())}")
    return maps


def render_material_thumbnail(sbsar_path: str, output_path: str, size: int = 256,
                              preset: str = None) -> bool:
    """Render just the baseColor map at small size for thumbnail."""
    with tempfile.TemporaryDirectory() as tmp:
        maps = render_material(sbsar_path, tmp, resolution=size, preset=preset)
        bc = maps.get("baseColor") or maps.get("diffuse")
        if bc and os.path.exists(bc):
            img = Image.open(bc).convert("RGB")
            img = img.resize((size, size), Image.LANCZOS)
            img.save(output_path, format="JPEG", quality=85)
            return True
    return False


def apply_material_to_glb(
    input_glb: str,
    output_glb: str,
    texture_maps: dict,
    progress_cb=None,
) -> bool:
    """Apply rendered PBR texture maps to a GLB file.

    Args:
        texture_maps: dict of {usage: filepath} — e.g. {"baseColor": "bc.png", "normal": "n.png", ...}
    """
    _cb = progress_cb or (lambda *a: None)
    _cb("Applying material to model...")

    try:
        gltf = pygltflib.GLTF2().load(input_glb)
    except Exception as e:
        logger.error(f"Failed to load GLB: {e}")
        return False

    blob = bytearray(gltf.binary_blob() or b"")

    def add_image(filepath, mime="image/png"):
        """Add an image to the GLB and return its texture index."""
        with open(filepath, "rb") as f:
            img_data = f.read()

        # Add to buffer
        offset = len(blob)
        blob.extend(img_data)

        # Pad to 4-byte alignment
        while len(blob) % 4 != 0:
            blob.append(0)

        bv_index = len(gltf.bufferViews)
        gltf.bufferViews.append(pygltflib.BufferView(
            buffer=0,
            byteOffset=offset,
            byteLength=len(img_data),
        ))

        img_index = len(gltf.images)
        gltf.images.append(pygltflib.Image(
            bufferView=bv_index,
            mimeType=mime,
        ))

        # Create sampler if none exists
        if not gltf.samplers:
            gltf.samplers.append(pygltflib.Sampler(
                magFilter=pygltflib.LINEAR,
                minFilter=pygltflib.LINEAR_MIPMAP_LINEAR,
                wrapS=pygltflib.REPEAT,
                wrapT=pygltflib.REPEAT,
            ))

        tex_index = len(gltf.textures)
        gltf.textures.append(pygltflib.Texture(
            sampler=0,
            source=img_index,
        ))

        return tex_index

    # glTF spec: metallic (blue) + roughness (green) must be combined
    metallic_roughness_tex = None
    if "roughness" in texture_maps or "metallic" in texture_maps:
        _cb("Combining metallic + roughness maps...")
        r_path = texture_maps.get("roughness")
        m_path = texture_maps.get("metallic")

        if r_path and os.path.exists(r_path):
            r_img = np.array(Image.open(r_path).convert("L"))
        else:
            r_img = np.full((1024, 1024), 128, dtype=np.uint8)

        if m_path and os.path.exists(m_path):
            m_img = np.array(Image.open(m_path).convert("L"))
        else:
            m_img = np.zeros_like(r_img, dtype=np.uint8)

        # Resize to match
        size = max(r_img.shape[0], m_img.shape[0])
        if r_img.shape[0] != size:
            r_img = np.array(Image.fromarray(r_img).resize((size, size), Image.LANCZOS))
        if m_img.shape[0] != size:
            m_img = np.array(Image.fromarray(m_img).resize((size, size), Image.LANCZOS))

        # Combine: R=occlusion(white), G=roughness, B=metallic
        combined = np.stack([
            np.full_like(r_img, 255),  # R: occlusion (unused, white)
            r_img,                      # G: roughness
            m_img,                      # B: metallic
        ], axis=-1)

        combined_img = Image.fromarray(combined)
        import tempfile as _tmp
        with _tmp.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
            combined_path = tmp.name
        combined_img.save(combined_path, format="PNG")
        metallic_roughness_tex = add_image(combined_path)
        os.unlink(combined_path)

    # Add texture maps
    base_color_tex = None
    normal_tex = None
    occlusion_tex = None

    if "baseColor" in texture_maps and os.path.exists(texture_maps["baseColor"]):
        _cb("Adding base color texture...")
        base_color_tex = add_image(texture_maps["baseColor"])
    elif "diffuse" in texture_maps and os.path.exists(texture_maps["diffuse"]):
        base_color_tex = add_image(texture_maps["diffuse"])

    if "normal" in texture_maps and os.path.exists(texture_maps["normal"]):
        _cb("Adding normal map...")
        normal_tex = add_image(texture_maps["normal"])

    if "ambientOcclusion" in texture_maps and os.path.exists(texture_maps["ambientOcclusion"]):
        occlusion_tex = add_image(texture_maps["ambientOcclusion"])

    # Create PBR material
    _cb("Creating PBR material...")
    pbr = pygltflib.PbrMetallicRoughness(
        metallicFactor=1.0,
        roughnessFactor=1.0,
    )

    if base_color_tex is not None:
        pbr.baseColorTexture = pygltflib.TextureInfo(index=base_color_tex)
    if metallic_roughness_tex is not None:
        pbr.metallicRoughnessTexture = pygltflib.TextureInfo(index=metallic_roughness_tex)

    mat = pygltflib.Material(
        name="Substance_Material",
        pbrMetallicRoughness=pbr,
        doubleSided=True,
    )

    if normal_tex is not None:
        mat.normalTexture = pygltflib.NormalTextureInfoClass(index=normal_tex)
    if occlusion_tex is not None:
        mat.occlusionTexture = pygltflib.OcclusionTextureInfoClass(index=occlusion_tex)

    mat_index = len(gltf.materials)
    gltf.materials.append(mat)

    # Assign material to all mesh primitives
    for mesh in gltf.meshes:
        for prim in mesh.primitives:
            prim.material = mat_index

    # Update buffer size and save
    gltf.buffers[0].byteLength = len(blob)
    gltf.set_binary_blob(bytes(blob))

    os.makedirs(os.path.dirname(output_glb) or ".", exist_ok=True)
    gltf.save(output_glb)

    logger.info(f"  Material applied, saved to {output_glb}")
    return True
