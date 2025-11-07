# api/routes.py
from __future__ import annotations

import asyncio
from pathlib import Path
from fastapi import APIRouter, UploadFile, File, HTTPException, Query
from fastapi.responses import FileResponse
from caption_pipeline.caption_pipeline import CaptionRewrite

from service.model_manager import get_engines, engines_busy, try_admit_now, try_admit_with_timeout
from service.pipeline import process_pdf
from service.workers import zip_dir

from api.schemas import HealthResponse, StatusResponse

router = APIRouter()

@router.get("/health", response_model=HealthResponse)
def health():
    """
    Report the health status of the service.
    It will return "ok" as long as the service is running normally.
    """
    e = get_engines()
    return HealthResponse(
        ok=True,
        # TODO: Later, create a configuration schema to ocr_model like vl2_model
        ocr_model=e.ocr.model_name,
        vl2_model=e.vl2.cfg.model_name
    )

@router.get("/models/status", response_model=StatusResponse)
def models_status():
    e = get_engines()
    return StatusResponse(
        ocr_model=e.ocr.model_name,
        vl2_model=e.vl2.cfg.model_name,
        busy=engines_busy()
    )

async def _try_admit(timeout_s: float) -> bool:
    """
    Test if we can acquire the GPU admission within the timeout.

    Args:
        timeout_s (float): Timeout in seconds.

    Returns:
        bool: True if admission acquired, False if timeout.
    """
    e = get_engines()
    try:
        await asyncio.wait_for(e.gate.acquire(), timeout=timeout_s)
        e.gate.release() # immediately release, since it's just a probe
        return True
    except asyncio.TimeoutError:
        return False

@router.post("/process/pdf")
async def process_pdf_endpoint(
    file: UploadFile = File(...),
    dpi: int = 200,
    rewrite_mode: str = "append",
    seed: int | None = None,
    wait_if_busy: bool = Query(
        False, 
        description="If false, reject the request immediately when another job is running"),
    timeout_s: float = Query(
        0.0, ge=0.0, le=600.0, 
        description="Max seconds to wait if busy (only used when wait_if_busy=True)"),
) -> FileResponse:
    """
    Process an uploaded PDF file and return a ZIP archive of the results.
    If the GPU is busy, it may wait or reject the request based on parameters.

    Args:
        file (UploadFile): The uploaded PDF file.
        dpi (int): The DPI to render PDF pages.
        rewrite_mode (str): Caption rewrite mode, either "append" or "replace".
        seed (int | None): Optional random seed for processing.
        wait_if_busy (bool): Whether to wait for GPU availability if busy. (Recommend value is busy)
        timeout_s (float): Maximum seconds to wait if busy.

    Returns:
        FileResponse: A response containing the ZIP archive of processed results.
    """
    if file.content_type != "application/pdf":
        raise HTTPException(status_code=400, detail="Invalid file type. Only PDF files are supported.")
    pdf_bytes = await file.read()
    if not pdf_bytes:
        raise HTTPException(status_code=400, detail="Empty file uploaded.")

    # Admission policy
    if not wait_if_busy:
        ok = await try_admit_now()
        if not ok:
            raise HTTPException(
                status_code=429,
                detail="GPU is busy with another PDF processing request; try again shortly",
                headers={"Retry-After": "15"},
            )
    else:
        ok = await try_admit_with_timeout(timeout_s=timeout_s)
        if not ok:
            raise HTTPException(
                status_code=503,
                detail=f"GPU is still busy after waiting for {timeout_s:.1f} seconds; try again later",
                headers={"Retry-After": "30"},
            )

    rewrite = CaptionRewrite.APPEND if rewrite_mode == "append" else CaptionRewrite.REPLACE
    tmp_root = Path("/tmp/pdfscribe2ds-fastapi")
    tmp_root.mkdir(parents=True, exist_ok=True)

    # Receive the output and archive it into a zip file
    out_dir = await process_pdf(tmp_root, pdf_bytes, dpi, rewrite, seed)
    archive_path = zip_dir(
        src=out_dir, 
        dest_zip_stem=out_dir.parent / "result"
    )

    return FileResponse(
        path=str(archive_path),
        media_type="application/zip",
        filename=f"{(file.filename or 'document').rsplit('.',1)[0]}_markdown.zip",
    )
