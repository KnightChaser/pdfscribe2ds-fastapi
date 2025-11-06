# service/pipeline.py
from __future__ import annotations

import asyncio
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from fastapi import HTTPException

from caption_pipeline.caption_pipeline import run_caption_pipeline, CaptionRewrite
from ocr_pipeline.pipeline import run_pdf_pipeline
from service.model_manager import get_engines

@asynccontextmanager
async def _admit():
    """
    Acquire admission to GPU for the entire job.
    """
    e = get_engines()
    await e.gate.acquire()

    try:
        yield
    finally:
        e.gate.release()

async def process_pdf(
    tmp_root: Path,
    pdf_bytes: bytes,
    dpi: int,
    rewrite: CaptionRewrite,
    seed: int | None,
) -> Path:
    """
    Asynchronously process a PDF: OCR + Caption.
    After work, return the output directory path.
    Inside the path, there will be per-page subdirectories with results.

    Args:
        tmp_root (Path): Root temporary directory.
        pdf_bytes (bytes): PDF file content in bytes.
        dpi (int): DPI for OCR processing.
        rewrite (CaptionRewrite): Caption rewriting strategy.
        seed (int | None): Random seed for captioning.
    """
    # Set up the directories
    workdir = tmp_root / str(uuid.uuid4())
    pdf_path = workdir / "input.pdf"
    out_dir  = workdir / "output"
    workdir.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)
    pdf_path.write_bytes(pdf_bytes)

    engines = get_engines()

    # NOTE:
    # Entire GPU-critical path (OCR -> caption) under one admission lock
    # After the job, the semaphore is released.
    async with _admit():
        # 1. OCR
        try:
            def _run_ocr():
                run_pdf_pipeline(
                    pdf_path=pdf_path,
                    output_dir=out_dir,
                    ocr_engine=engines.ocr,
                    dpi=dpi,
                )
            await asyncio.to_thread(_run_ocr)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"OCR processing failed: {e!s}")

        # 2. Caption
        try:
            def _run_caption():
                run_caption_pipeline(
                    output_dir=out_dir,
                    caption_model=engines.vl2.cfg.model_name,
                    gpu_mem=engines.vl2.cfg.gpu_memory_utilization,
                    seed=seed,
                    rewrite=rewrite,
                )
            await asyncio.to_thread(_run_caption)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Caption processing failed: {e!s}")

    return out_dir
