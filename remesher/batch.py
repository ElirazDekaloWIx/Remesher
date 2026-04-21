"""Batch processing for multiple 3D files."""

import logging
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

from tqdm import tqdm

from .pipeline import process_file, RemeshResult, SUPPORTED_FORMATS
from .retopology import process_file_retopo, RetopologyResult

logger = logging.getLogger(__name__)


def _process_one(args: tuple) -> RemeshResult:
    """Worker function for parallel processing."""
    (input_path, output_path, preset, target_faces, ratio, quality_thr,
     uv_method, texture_size, smooth_method, smooth_iterations) = args
    return process_file(
        input_path, output_path, preset, target_faces, ratio, quality_thr,
        uv_method=uv_method, texture_size=texture_size,
        smooth_method=smooth_method, smooth_iterations=smooth_iterations,
    )


def find_3d_files(input_dir: str, recursive: bool = True) -> list[Path]:
    """Find all supported 3D files in a directory."""
    input_dir = Path(input_dir)
    files = []
    pattern = "**/*" if recursive else "*"
    for f in input_dir.glob(pattern):
        if f.is_file() and f.suffix.lower() in SUPPORTED_FORMATS:
            files.append(f)
    return sorted(files)


def process_batch(
    input_dir: str,
    output_dir: str,
    preset: str | None = "web",
    target_faces: int | None = None,
    ratio: float | None = None,
    quality_thr: float = 0.3,
    workers: int = 4,
    recursive: bool = True,
    output_format: str | None = None,
    uv_method: str = "keep",
    texture_size: int = 2048,
    smooth_method: str = "none",
    smooth_iterations: int = 3,
) -> list[RemeshResult]:
    """Process all 3D files in a directory."""
    input_dir = Path(input_dir).resolve()
    output_dir = Path(output_dir).resolve()

    files = find_3d_files(str(input_dir), recursive)
    if not files:
        logger.warning(f"No supported 3D files found in {input_dir}")
        return []

    logger.info(f"Found {len(files)} files to process")

    # Build task list
    tasks = []
    for f in files:
        rel = f.relative_to(input_dir)
        if output_format:
            out_path = output_dir / rel.with_suffix(f".{output_format.lstrip('.')}")
        else:
            out_path = output_dir / rel
        tasks.append((
            str(f), str(out_path), preset, target_faces, ratio, quality_thr,
            uv_method, texture_size, smooth_method, smooth_iterations,
        ))

    results = []

    if workers <= 1:
        for task in tqdm(tasks, desc="Remeshing"):
            results.append(_process_one(task))
    else:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            futures = {executor.submit(_process_one, t): t[0] for t in tasks}
            with tqdm(total=len(futures), desc="Remeshing") as pbar:
                for future in as_completed(futures):
                    result = future.result()
                    results.append(result)
                    pbar.update(1)
                    if not result.success:
                        logger.error(f"FAILED: {result.input_path} — {result.error}")

    # Summary
    succeeded = [r for r in results if r.success]
    failed = [r for r in results if not r.success]
    total_orig = sum(r.original_faces for r in succeeded)
    total_final = sum(r.final_faces for r in succeeded)
    avg_reduction = (1 - total_final / total_orig) * 100 if total_orig > 0 else 0

    logger.info(f"\n{'='*50}")
    logger.info(f"BATCH COMPLETE: {len(succeeded)}/{len(results)} succeeded")
    logger.info(f"Total faces: {total_orig:,} -> {total_final:,} ({avg_reduction:.1f}% reduction)")
    if failed:
        logger.info(f"Failed files:")
        for r in failed:
            logger.info(f"  {r.input_path}: {r.error}")

    return results


def _process_one_retopo(args: tuple) -> RetopologyResult:
    """Worker function for parallel retopology."""
    input_path, output_path, preset, target_faces, method, texture_size = args
    return process_file_retopo(input_path, output_path, preset, target_faces, method, texture_size)


def process_batch_retopo(
    input_dir: str,
    output_dir: str,
    preset: str | None = "web",
    target_faces: int | None = None,
    method: str = "quadriflow",
    texture_size: int = 2048,
    workers: int = 4,
    recursive: bool = True,
    output_format: str | None = None,
) -> list[RetopologyResult]:
    """Process all 3D files in a directory with retopology."""
    input_dir = Path(input_dir).resolve()
    output_dir = Path(output_dir).resolve()

    files = find_3d_files(str(input_dir), recursive)
    if not files:
        logger.warning(f"No supported 3D files found in {input_dir}")
        return []

    logger.info(f"Found {len(files)} files to retopologize")

    tasks = []
    for f in files:
        rel = f.relative_to(input_dir)
        if output_format:
            out_path = output_dir / rel.with_suffix(f".{output_format.lstrip('.')}")
        else:
            out_path = output_dir / rel
        tasks.append((str(f), str(out_path), preset, target_faces, method, texture_size))

    results = []

    if workers <= 1:
        for task in tqdm(tasks, desc="Retopology"):
            results.append(_process_one_retopo(task))
    else:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            futures = {executor.submit(_process_one_retopo, t): t[0] for t in tasks}
            with tqdm(total=len(futures), desc="Retopology") as pbar:
                for future in as_completed(futures):
                    result = future.result()
                    results.append(result)
                    pbar.update(1)
                    if not result.success:
                        logger.error(f"FAILED: {result.input_path} — {result.error}")

    succeeded = [r for r in results if r.success]
    failed = [r for r in results if not r.success]
    total_orig = sum(r.original_faces for r in succeeded)
    total_final = sum(r.final_faces for r in succeeded)
    avg_reduction = (1 - total_final / total_orig) * 100 if total_orig > 0 else 0

    logger.info(f"\n{'='*50}")
    logger.info(f"BATCH RETOPO COMPLETE: {len(succeeded)}/{len(results)} succeeded")
    logger.info(f"Total faces: {total_orig:,} -> {total_final:,} ({avg_reduction:.1f}% reduction)")
    if failed:
        logger.info(f"Failed files:")
        for r in failed:
            logger.info(f"  {r.input_path}: {r.error}")

    return results
