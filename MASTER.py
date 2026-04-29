"""
MASTER launcher — runs Remesher in its highest-quality configuration.

What this gives you:
- Server-side pipeline (Python + pymeshlab QEM with texture awareness)
- Default preset = 'high' (50,000 faces) — quality bias, not size bias
- UV preservation (uv_method='keep') — keeps original UVs and textures intact
- Strict QEM quality threshold (0.5) for clean topology
- Texture-aware decimation path is preferred when UVs exist

How to use:
    python MASTER.py            # default port 5000
    python MASTER.py 8080       # custom port

Then drag a high-res .glb / .gltf / .obj / .fbx onto the page, choose preset
'high' (already pre-selected via the URL hash), and download the result.
"""

import os
import sys
import webbrowser
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))


def check_deps() -> list[str]:
    missing = []
    for mod in ("trimesh", "pymeshlab", "flask", "PIL", "numpy", "scipy"):
        try:
            __import__(mod)
        except ImportError:
            missing.append(mod)
    return missing


def main():
    missing = check_deps()
    if missing:
        print(f"[MASTER] Missing dependencies: {', '.join(missing)}")
        print(f"[MASTER] Install with: pip install -r {ROOT / 'requirements.txt'}")
        sys.exit(1)

    port = int(sys.argv[1]) if len(sys.argv) > 1 else 5000

    # Quality-biased defaults applied via URL hash so the UI pre-selects them.
    # (The server already accepts these via form fields; the hash just preconfigures the UI.)
    url = (
        f"http://localhost:{port}/"
        f"#preset=high&uv_method=keep&quality=0.5&texture_size=2048&smooth_method=none"
    )

    print()
    print("  Remesher — MASTER (quality configuration)")
    print(f"  Preset: high (50,000 faces) | UV: keep | Quality threshold: 0.5")
    print(f"  Open:   {url}")
    print()

    webbrowser.open(url)

    # Reuse the existing Flask app from server.py (don't re-import main() — it
    # would re-open the browser without our hash).
    from server import app
    if os.environ.get("REMESHER_ENV") == "production":
        from waitress import serve
        serve(app, host="0.0.0.0", port=port)
    else:
        app.run(host="0.0.0.0", port=port, debug=False)


if __name__ == "__main__":
    main()
