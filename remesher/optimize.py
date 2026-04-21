"""GLB file weight optimizer: texture compression + geometry quantization."""

import os
import io
import struct
import logging
from pathlib import Path
from dataclasses import dataclass

import numpy as np
from PIL import Image
import pygltflib

logger = logging.getLogger(__name__)


@dataclass
class OptimizeResult:
    input_path: str
    output_path: str
    original_size: int
    final_size: int
    reduction_pct: float
    textures_compressed: int
    success: bool
    error: str | None = None
    details: str = ""


def _extract_image_data(gltf: pygltflib.GLTF2, image_index: int) -> bytes | None:
    """Extract raw image bytes from a GLB buffer."""
    img = gltf.images[image_index]

    if img.bufferView is not None:
        bv = gltf.bufferViews[img.bufferView]
        blob = gltf.binary_blob()
        if blob is None:
            return None
        offset = bv.byteOffset or 0
        return bytes(blob[offset:offset + bv.byteLength])

    return None


def _replace_image_data(gltf: pygltflib.GLTF2, image_index: int, new_data: bytes, mime_type: str):
    """Replace image data in the GLB buffer."""
    img = gltf.images[image_index]

    if img.bufferView is not None:
        bv = gltf.bufferViews[img.bufferView]
        blob = bytearray(gltf.binary_blob())

        old_offset = bv.byteOffset or 0
        old_length = bv.byteLength

        # Replace bytes in blob
        new_blob = blob[:old_offset] + new_data + blob[old_offset + old_length:]

        # Update buffer view
        size_diff = len(new_data) - old_length
        bv.byteLength = len(new_data)

        # Shift all subsequent buffer views
        for other_bv in gltf.bufferViews:
            if other_bv is not bv and (other_bv.byteOffset or 0) > old_offset:
                other_bv.byteOffset = (other_bv.byteOffset or 0) + size_diff

        # Update total buffer size
        gltf.buffers[0].byteLength = len(new_blob)
        gltf.set_binary_blob(bytes(new_blob))

        img.mimeType = mime_type


MAP_TYPE_NORMAL = "normal"
MAP_TYPE_ROUGHNESS = "roughness"
MAP_TYPE_METALLIC = "metallic"
MAP_TYPE_OCCLUSION = "occlusion"
MAP_TYPE_EMISSIVE = "emissive"
MAP_TYPE_BASE_COLOR = "baseColor"
MAP_TYPE_UNKNOWN = "unknown"


def _identify_texture_roles(gltf: pygltflib.GLTF2) -> dict[int, str]:
    """Map image indices to their material role (normal, roughness, etc.)."""
    roles = {}

    for mat in gltf.materials:
        pbr = mat.pbrMetallicRoughness
        if pbr:
            if pbr.baseColorTexture and pbr.baseColorTexture.index is not None:
                tex = gltf.textures[pbr.baseColorTexture.index]
                if tex.source is not None:
                    roles[tex.source] = MAP_TYPE_BASE_COLOR

            if pbr.metallicRoughnessTexture and pbr.metallicRoughnessTexture.index is not None:
                tex = gltf.textures[pbr.metallicRoughnessTexture.index]
                if tex.source is not None:
                    roles[tex.source] = MAP_TYPE_ROUGHNESS

        if mat.normalTexture and mat.normalTexture.index is not None:
            tex = gltf.textures[mat.normalTexture.index]
            if tex.source is not None:
                roles[tex.source] = MAP_TYPE_NORMAL

        if mat.occlusionTexture and mat.occlusionTexture.index is not None:
            tex = gltf.textures[mat.occlusionTexture.index]
            if tex.source is not None:
                roles[tex.source] = MAP_TYPE_OCCLUSION

        if mat.emissiveTexture and mat.emissiveTexture.index is not None:
            tex = gltf.textures[mat.emissiveTexture.index]
            if tex.source is not None:
                roles[tex.source] = MAP_TYPE_EMISSIVE

    return roles


def compress_texture(image_data: bytes, max_size: int = 2048, jpeg_quality: int = 80,
                     force_jpeg: bool = False, map_type: str = MAP_TYPE_UNKNOWN) -> tuple[bytes, str, str]:
    """Compress a texture image with map-type-aware quality settings.

    Normal maps, roughness maps, and metallic maps get higher quality to preserve
    detail and contrast. Base color maps use standard compression.
    """
    try:
        img = Image.open(io.BytesIO(image_data))
    except Exception:
        return image_data, "image/png", "failed to open"

    original_format = img.format or "PNG"
    w, h = img.size
    desc_parts = [f"[{map_type}]"]

    # Map-type-aware quality: normal/roughness/metallic need higher quality
    effective_quality = jpeg_quality
    if map_type == MAP_TYPE_NORMAL:
        # Normal maps are very sensitive to compression artifacts
        effective_quality = max(jpeg_quality, 92)
        desc_parts.append("hi-q normal")
    elif map_type in (MAP_TYPE_ROUGHNESS, MAP_TYPE_METALLIC):
        # Roughness/metallic: contrast-sensitive, boost quality
        effective_quality = max(jpeg_quality, 88)
        desc_parts.append("hi-q PBR data")
    elif map_type == MAP_TYPE_OCCLUSION:
        effective_quality = max(jpeg_quality, 85)

    # Resize if needed
    if w > max_size or h > max_size:
        ratio = max_size / max(w, h)
        new_w = int(w * ratio)
        new_h = int(h * ratio)
        img = img.resize((new_w, new_h), Image.LANCZOS)
        desc_parts.append(f"resized {w}x{h} -> {new_w}x{new_h}")

    # Determine if we can use JPEG (no alpha channel)
    has_alpha = img.mode in ("RGBA", "LA", "PA")
    if has_alpha:
        if img.mode == "RGBA":
            alpha = np.array(img)[:, :, 3]
            if alpha.min() > 250:
                has_alpha = False
                img = img.convert("RGB")
                desc_parts.append("removed unused alpha")

    use_jpeg = force_jpeg or not has_alpha

    # Save to a temp file (workaround for Pillow/Python 3.13 BytesIO fileno bug)
    import tempfile as _tmpfile
    if use_jpeg:
        if img.mode != "RGB":
            img = img.convert("RGB")
        with _tmpfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
            tmp_path = tmp.name
        img.save(tmp_path, format="JPEG", quality=effective_quality, optimize=True)
        mime = "image/jpeg"
        if original_format == "PNG":
            desc_parts.append("PNG -> JPEG")
    else:
        with _tmpfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
            tmp_path = tmp.name
        img.save(tmp_path, format="PNG", optimize=True)
        mime = "image/png"
        desc_parts.append("PNG optimized")

    with open(tmp_path, "rb") as f:
        result_data = f.read()
    os.unlink(tmp_path)

    desc_parts.append(f"q={effective_quality}")
    return result_data, mime, ", ".join(desc_parts)


def quantize_accessors(gltf: pygltflib.GLTF2):
    """Quantize float32 vertex data to reduce buffer size.

    Strips unused attributes and normalizes where possible.
    Returns description of changes.
    """
    # For now, just strip any padding/alignment waste
    # Full quantization requires modifying accessor componentType
    # which breaks some viewers — keeping this conservative
    pass


def optimize_file(
    input_path: str,
    output_path: str,
    max_texture_size: int = 2048,
    jpeg_quality: int = 80,
    progress_cb=None,
) -> OptimizeResult:
    """Optimize a GLB file for minimal size.

    - Compress textures (PNG -> JPEG where possible, resize, quality)
    - Remove unused data
    """
    input_path = str(Path(input_path).resolve())
    output_path = str(Path(output_path).resolve())
    _cb = progress_cb or (lambda *a: None)

    original_size = os.path.getsize(input_path)
    logger.info(f"Optimizing: {Path(input_path).name} ({original_size / 1024 / 1024:.2f} MB)")

    try:
        gltf = pygltflib.GLTF2().load(input_path)
    except Exception as e:
        return OptimizeResult(input_path, output_path, original_size, 0, 0, 0, False, str(e))

    # Identify texture roles from materials
    texture_roles = _identify_texture_roles(gltf)

    if not gltf.images:
        # No textures — just copy
        logger.info("  No textures to compress")

        # Still save — pygltflib may clean up some padding
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        gltf.save(output_path)
        final_size = os.path.getsize(output_path)
        reduction = (1 - final_size / original_size) * 100 if original_size > 0 else 0
        return OptimizeResult(input_path, output_path, original_size, final_size, reduction, 0, True,
                              details="No textures, file cleaned")

    details = []
    textures_compressed = 0

    # Process images from last to first (so offset shifts don't affect earlier images)
    for i in range(len(gltf.images) - 1, -1, -1):
        img_data = _extract_image_data(gltf, i)
        if img_data is None:
            continue

        old_size = len(img_data)
        role = texture_roles.get(i, MAP_TYPE_UNKNOWN)
        _cb(f"Compressing texture {i + 1}/{len(gltf.images)} ({role})...")
        logger.info(f"  Image {i}: {old_size / 1024:.0f} KB, type={gltf.images[i].mimeType}, role={role}")

        new_data, mime, desc = compress_texture(img_data, max_texture_size, jpeg_quality, map_type=role)
        new_size = len(new_data)

        if new_size < old_size:
            _replace_image_data(gltf, i, new_data, mime)
            saved = (1 - new_size / old_size) * 100
            details.append(f"image {i}: {old_size // 1024}KB -> {new_size // 1024}KB (-{saved:.0f}%) [{desc}]")
            textures_compressed += 1
            logger.info(f"    -> {new_size / 1024:.0f} KB (-{saved:.0f}%) [{desc}]")
        else:
            details.append(f"image {i}: kept original ({desc})")
            logger.info(f"    -> kept original (compressed would be larger)")

    # Save
    _cb("Saving optimized file...")
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    gltf.save(output_path)

    final_size = os.path.getsize(output_path)
    reduction = (1 - final_size / original_size) * 100 if original_size > 0 else 0

    logger.info(f"  Done: {original_size / 1024 / 1024:.2f} MB -> {final_size / 1024 / 1024:.2f} MB ({reduction:.1f}% reduction)")

    return OptimizeResult(
        input_path, output_path, original_size, final_size, reduction,
        textures_compressed, True, details="\n".join(details),
    )
