"""Atomic, cache-aware downloads for unattended external benchmarks."""

from __future__ import annotations

import hashlib
import os
import time
import urllib.request
import zipfile
from pathlib import Path


USER_AGENT = "TokenFeatureDualAxisResearch/0.3 (+unattended benchmark runner)"


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download_file(
    url: str,
    destination: str | Path,
    *,
    timeout: float,
    max_bytes: int | None = None,
) -> Path:
    destination = Path(destination)
    if destination.exists() and (
        max_bytes is None or destination.stat().st_size >= max_bytes
    ):
        return destination
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".part")
    written = 0
    last_error: Exception | None = None
    for attempt in range(3):
        request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
        written = 0
        try:
            with urllib.request.urlopen(
                request, timeout=timeout
            ) as response, temporary.open("wb") as handle:
                while True:
                    remaining = None if max_bytes is None else max_bytes - written
                    if remaining is not None and remaining <= 0:
                        break
                    chunk = response.read(
                        1024 * 1024
                        if remaining is None
                        else min(1024 * 1024, remaining)
                    )
                    if not chunk:
                        break
                    handle.write(chunk)
                    written += len(chunk)
                handle.flush()
                os.fsync(handle.fileno())
            last_error = None
            break
        except Exception as error:
            last_error = error
            if attempt < 2:
                time.sleep(2**attempt)
    if last_error is not None:
        raise RuntimeError(f"download failed after three attempts: {url}") from last_error
    if max_bytes is not None and written < max_bytes:
        raise RuntimeError(
            f"download ended at {written} bytes before requested {max_bytes}: {url}"
        )
    temporary.replace(destination)
    return destination


def safe_extract_zip(archive: str | Path, destination: str | Path) -> Path:
    archive = Path(archive).resolve()
    destination = Path(destination).resolve()
    destination.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive) as handle:
        for member in handle.infolist():
            target = (destination / member.filename).resolve()
            if destination != target and destination not in target.parents:
                raise RuntimeError(f"unsafe path in archive: {member.filename}")
        handle.extractall(destination)
    return destination


def safe_extract_zip_recursive(
    archive: str | Path,
    destination: str | Path,
    *,
    max_depth: int = 3,
    max_nested_archives: int = 64,
) -> Path:
    """Safely extract a ZIP and any ZIP files contained inside it.

    The published CLUTRR bundle has appeared in both flat and nested forms.
    Nested members are extracted beside themselves into a directory named
    after the archive. Bounds prevent an unexpected archive tree from causing
    unbounded work during unattended preflight.
    """

    destination = safe_extract_zip(archive, destination).resolve()
    extracted_archives: set[Path] = set()
    for _ in range(max_depth):
        nested = [
            path.resolve()
            for path in destination.rglob("*.zip")
            if path.resolve() not in extracted_archives
        ]
        if not nested:
            return destination
        if len(extracted_archives) + len(nested) > max_nested_archives:
            raise RuntimeError(
                f"too many nested ZIP archives under {destination}: "
                f"> {max_nested_archives}"
            )
        for nested_archive in nested:
            target = nested_archive.with_suffix("")
            safe_extract_zip(nested_archive, target)
            extracted_archives.add(nested_archive)
    remaining = [
        path
        for path in destination.rglob("*.zip")
        if path.resolve() not in extracted_archives
    ]
    if remaining:
        raise RuntimeError(
            f"nested ZIP depth exceeds {max_depth}: {remaining[0].relative_to(destination)}"
        )
    return destination
