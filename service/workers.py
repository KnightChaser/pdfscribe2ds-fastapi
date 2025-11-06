# service/workers.py
from __future__ import annotations

import shutil
from pathlib import Path

def zip_dir(src: Path, dest_zip_stem: Path) -> Path:
    """
    Zip a directory. dest_zip_stem is the path *without* .zip extension.
    Returns the final .zip Path.

    Args:
        src (Path): Source directory to zip.
        dest_zip_stem (Path): Destination zip file path without .zip extension.

    Returns:
        Path: Path to the created zip file.
    """
    archive = shutil.make_archive(str(dest_zip_stem), "zip", root_dir=src)
    return Path(archive)

