"""CLI entry point for the Remesher tool."""

import logging
import sys
from pathlib import Path

import click

from .pipeline import process_file, PRESETS, SUPPORTED_FORMATS
from .batch import process_batch
from .uv_unwrap import UV_METHODS

logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)


@click.group()
@click.version_option("1.0.0")
def cli():
    """Remesher — Automatic 3D model optimization for web games.

    Reduces polygon count while preserving UV maps, textures, and visual quality.
    Uses QEM (Quadric Error Metrics) decimation, the industry standard algorithm.
    Optionally re-unwraps UVs with xatlas, LSCM, or ARAP methods.
    """
    pass


@cli.command()
@click.argument("input_path", type=click.Path(exists=True))
@click.argument("output_path", type=click.Path())
@click.option("--preset", type=click.Choice(list(PRESETS.keys())), default="web",
              help=f"Target quality preset. Budgets: {PRESETS}")
@click.option("--target-faces", "-t", type=int, default=None,
              help="Exact target face count (overrides preset)")
@click.option("--ratio", "-r", type=float, default=None,
              help="Reduction ratio, e.g. 0.5 = keep 50%% of faces (overrides preset)")
@click.option("--quality", "-q", type=float, default=0.3,
              help="Quality threshold 0-1. Lower = higher quality, higher = faster but more degradation (default: 0.3)")
@click.option("--uv", "uv_method", type=click.Choice(list(UV_METHODS)), default="keep",
              help="UV unwrap method: keep (preserve original), xatlas (fast packed), lscm (smooth), arap (best quality)")
@click.option("--texture-size", type=int, default=2048,
              help="Baked texture resolution when re-unwrapping UVs (default: 2048)")
def single(input_path, output_path, preset, target_faces, ratio, quality, uv_method, texture_size):
    """Process a single 3D file.

    Example: remesher single model.glb output.glb --preset web
    Example: remesher single model.glb output.glb --preset web --uv lscm
    """
    result = process_file(input_path, output_path, preset, target_faces, ratio, quality,
                          uv_method=uv_method, texture_size=texture_size)
    if result.success:
        click.echo(f"\n[OK] {result.original_faces:,} -> {result.final_faces:,} faces ({result.reduction_pct:.1f}% reduction)")
        if uv_method != "keep":
            click.echo(f"  UV unwrap: {uv_method}")
        click.echo(f"  Saved to: {result.output_path}")
    else:
        click.echo(f"\n[FAIL] Failed: {result.error}", err=True)
        sys.exit(1)


@cli.command()
@click.argument("input_dir", type=click.Path(exists=True, file_okay=False))
@click.argument("output_dir", type=click.Path())
@click.option("--preset", type=click.Choice(list(PRESETS.keys())), default="web",
              help=f"Target quality preset. Budgets: {PRESETS}")
@click.option("--target-faces", "-t", type=int, default=None,
              help="Exact target face count per mesh (overrides preset)")
@click.option("--ratio", "-r", type=float, default=None,
              help="Reduction ratio, e.g. 0.5 = keep 50%% of faces (overrides preset)")
@click.option("--quality", "-q", type=float, default=0.3,
              help="Quality threshold 0-1 (default: 0.3)")
@click.option("--uv", "uv_method", type=click.Choice(list(UV_METHODS)), default="keep",
              help="UV unwrap method: keep, xatlas, lscm, arap")
@click.option("--workers", "-w", type=int, default=4,
              help="Number of parallel workers (default: 4)")
@click.option("--format", "-f", "output_format", type=str, default=None,
              help="Output format (e.g. glb, obj). Default: same as input")
@click.option("--no-recursive", is_flag=True, help="Don't search subdirectories")
def batch(input_dir, output_dir, preset, target_faces, ratio, quality, uv_method, workers, output_format, no_recursive):
    """Process all 3D files in a directory.

    Example: remesher batch ./models ./optimized --preset web --workers 8
    """
    results = process_batch(
        input_dir, output_dir, preset, target_faces, ratio, quality,
        workers, not no_recursive, output_format,
        uv_method=uv_method,
    )

    succeeded = sum(1 for r in results if r.success)
    failed = sum(1 for r in results if not r.success)

    if failed > 0:
        click.echo(f"\n[WARN] {succeeded} succeeded, {failed} failed")
        sys.exit(1)
    else:
        click.echo(f"\n[OK] All {succeeded} files processed successfully")


@cli.command()
def presets():
    """Show available quality presets and their polygon budgets."""
    click.echo("Available presets (target faces per mesh):\n")
    for name, faces in PRESETS.items():
        click.echo(f"  {name:10s}  {faces:>8,} faces")
    click.echo(f"\nSupported formats: {', '.join(sorted(SUPPORTED_FORMATS))}")
    click.echo(f"\nUV methods: {', '.join(UV_METHODS)}")


def main():
    cli()


if __name__ == "__main__":
    main()
