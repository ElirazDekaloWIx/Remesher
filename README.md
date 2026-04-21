# Remesher

Automatic 3D model optimization for web games. Reduces polygon count while preserving UV maps, textures, and visual quality.

Uses QEM (Quadric Error Metrics) decimation, xatlas/LSCM UV unwrapping, and PBR-aware texture compression.

## Two versions

### 1. Desktop / Server (Python + Flask)

Full-featured pipeline with pymeshlab, QuadriFlow, custom ARAP UV unwrap, Substance 3D material integration, batch processing, and more.

```bash
pip install -r requirements.txt
python server.py
# -> http://localhost:5000
```

CLI:

```bash
remesher single model.glb out.glb --preset web --uv lscm
remesher batch ./models ./optimized --preset web --workers 8
```

### 2. Web (100% client-side)

A standalone HTML page that does remeshing entirely in the browser — no backend, no uploads. Drop it on any static host (or open the file directly).

```
web/index.html
```

Uses:
- **meshoptimizer** (WASM) — QEM decimation
- **gltf-transform** — GLB read/write, texture compression, weld/dedup/prune
- **three.js** — side-by-side preview

Capabilities:
- Remesh with presets (mobile / web / desktop / high) or custom face count
- Quality slider (mapped to meshopt error threshold)
- Texture compression (WebP/JPEG) with size + quality controls
- **Normal smoothing with crease angle** — live-adjustable, splits edges above the threshold
- **Poly Haven material browser** — fetch and apply PBR materials (wood, metal, fabric, stone, ...) directly from the [Poly Haven API](https://api.polyhaven.com)
- **PBR rendering** — HDRI environment maps (Courtyard / Sunset / Sky / Studio), IBL via PMREM, ACES tone mapping, exposure control, shadow-receiving ground plane
- Live dual-panel preview (original vs processed)
- Export optimized GLB

### Credits
HDRI environment maps and PBR materials courtesy of [Poly Haven](https://polyhaven.com) (CC0).

## Presets (target face count)

| Preset  | Faces  |
|---------|--------|
| mobile  | 3,000  |
| web     | 10,000 |
| desktop | 30,000 |
| high    | 50,000 |

## Supported formats

Input: `.glb`, `.gltf`, `.obj`, `.fbx`, `.stl`, `.ply`, `.off`, `.dae` (desktop version)
Web version: `.glb`, `.gltf`
Output: `.glb`
