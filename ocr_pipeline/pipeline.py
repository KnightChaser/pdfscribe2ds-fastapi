# ocr_pipeline/pipeline.py
from __future__ import annotations

from typing import List, Optional
from pathlib import Path
from typing import List, Optional
import logging
import asyncio

from PIL import Image

from .config import PipelineConfig
from .pdf_loader import pdf_to_images
from .ocr_engine import DeepSeekOCREngine
from .md_rewriter import rewrite_md_with_embeds
import quiet

logger = logging.getLogger("pdfscribe2ds")

def run_pdf_pipeline(
    pdf_path: Path,
    output_dir: Path,
    model_name: str = "deepseek-ai/DeepSeek-OCR", # NOTE: Fixed
    dpi: int = 200,
    num_processes: Optional[int] = None,
    num_threads: Optional[int] = None,
    ocr_engine: Optional[DeepSeekOCREngine] = None,
    cancel_evt: Optional[asyncio.Event] = None,
) -> None:
    """
    Full PDF -> images -> DeepSeek-OCR -> Markdown pipeline.

    Args:
        pdf_path (Path): Path to the input PDF file.
        output_dir (Path): Directory to save output images and markdown files.
        model_name (str): Name of the DeepSeek-OCR model to use.
        dpi (int): Dots per inch for image quality when converting PDF to images.
        num_processes (int, optional): Number of processes for parallel PDF conversion.
        num_threads (int, optional): Number of threads for parallel image saving.
        ocr_engine (DeepSeekOCREngine, optional): Pre-initialized OCR engine to reuse.
        cancel_evt (asyncio.Event, optional): Event to signal cancellation.
    """
    cfg = PipelineConfig(
        pdf_path=pdf_path,
        output_dir=output_dir,
        model_name=model_name,
        dpi=dpi,
    )

    # 1. PDF -> images/{page-001.png, ...}
    images_out_dir = cfg.output_dir / "images"
    logger.info(f"Converting PDF to images in {images_out_dir}...")
    image_paths: List[Path] = pdf_to_images(
        pdf_path=cfg.pdf_path,
        out_dir=images_out_dir,
        dpi=cfg.dpi,
        num_processes=num_processes,
        num_threads=num_threads,
    )
    logger.info("Converted %d pages to images at %s", len(image_paths), images_out_dir)

    # Check for cancellation
    if cancel_evt and cancel_evt.is_set():
        raise asyncio.CancelledError()

    # 2. Prepare OCR engine
    if ocr_engine is not None:
        ocr = ocr_engine
    else:
        with quiet.quiet_stdio():
            ocr = DeepSeekOCREngine(model_name=cfg.model_name)

    # 3. Per page processing
    md_out_dir = cfg.output_dir / "markdown"
    md_out_dir.mkdir(parents=True, exist_ok=True)

    for img_path in image_paths:
        # Check for cancellation
        if cancel_evt and cancel_evt.is_set():
            raise asyncio.CancelledError()

        # OCR image -> raw markdown
        # Rewrite markdown with embedded images
        # Save final markdown file
        try:
            with quiet.quiet_stdio():
                raw_md = ocr.image_to_markdown(img_path)

            page_stem = img_path.stem  # e.g. "page_001"
            assets_dir = md_out_dir / f"{page_stem}_assets"

            # ensure file handle closes immediately
            with Image.open(img_path) as _img:
                img = _img.convert("RGB")

            # re-open image to pass to rewriter
            cleaned_md = rewrite_md_with_embeds(
                text_output=raw_md,
                image=img,
                output_dir=assets_dir,
                base_img_name=page_stem,
            )

            md_file = md_out_dir / f"{page_stem}.md"
            md_file.write_text(cleaned_md, encoding="utf-8")

            logger.info(f"[OK] page {page_stem} --> {md_file}")
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception("Failed to process %s; skipping.", img_path.name)

    logger.info(f"Pipeline finished for {pdf_path} --> {cfg.output_dir}")
