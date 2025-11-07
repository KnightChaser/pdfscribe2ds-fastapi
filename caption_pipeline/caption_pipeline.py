# caption_pipeline/caption_pipeline.py
from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Dict, Optional
from enum import Enum

from PIL import Image
import quiet

from .caption_engine import DeepSeekVL2Captioner, CaptionerConfig

logger = logging.getLogger("pdfscribe2ds")

# Markdown image pattern: ![ALT](PATH)
_IMG_TAG = re.compile(r'!\[(.*?)\]\((.*?)\)')

class CaptionRewrite(str, Enum):
    APPEND = "append"
    REPLACE = "replace"

def _render_caption_block(alt: str, cap: str) -> str:
    """
    Render a caption block in markdown format.

    Args:
        alt (str): The alt text of the image.
        cap (str): The generated caption for the image.

    Returns:
        str: The formatted caption block in markdown.
    """
    prefix = (alt or "Image").strip()

    return f"\n\n*{prefix} - {cap}*\n"

def _resolve_image(md_file: Path, rel_path: str) -> Path:
    """
    Resolve a relative image path to an absolute path based on the markdown file location.

    Args:
        md_file (Path): The markdown file path.
        rel_path (str): The relative image path extracted from the markdown.

    Returns:
        Path: The resolved absolute image path.
    """
    rel_path = rel_path.strip()
    return (md_file.parent / rel_path).resolve()

def caption_markdown_file(
    md_file: Path,
    captioner: DeepSeekVL2Captioner,
    prompt_override: Optional[str] = None,
    rewrite: CaptionRewrite = CaptionRewrite.APPEND,
) -> bool:
    """
    Read a single Markdown file, caption each image tag, and rewrite the file in place.
    Returns True if the file changed, else False.

    Args:
        md_file (Path): The markdown file to process.
        captioner (DeepSeekVL2Captioner): The captioner instance to use for generating captions.
        prompt_override (Optional[str]): Optional prompt to override the default captioning prompt.
        rewrite (CaptionRewrite): Whether to append or replace captions in the markdown.

    Returns:
        bool: True if the file was modified, False otherwise.
    """
    text = md_file.read_text(encoding="utf-8")
    matches = list(_IMG_TAG.finditer(text))
    if not matches:
        logger.info("No images found in %s", md_file.name)
        return False

    # caption each unique image once
    captions_cache: Dict[str, str] = {}

    for m in matches:
        # alt = m.group(1)
        rel = m.group(2)
        if rel in captions_cache:
            continue

        img_path = _resolve_image(md_file, rel)
        if not img_path.exists():
            logger.warning("Image not found: %s (referenced in %s)", img_path, md_file)
            continue

        try:
            with Image.open(img_path) as image:
                with quiet.quiet_stdio():
                    cap = captioner.caption(
                        image.convert("RGB"),
                        page_context=text,
                        prompt_override=prompt_override,
                    )
                captions_cache[rel] = cap
                logger.info("Captioned image: %s", img_path)
        except Exception as e:
            logger.error("Failed to caption image %s: %s", img_path, e)
            continue

    # Replace tags with caption lines
    def repl(m: re.Match) -> str:
        alt = m.group(1)
        rel = m.group(2)
        cap = captions_cache.get(rel)
        if not cap:
            return m.group(0)  # no change
        if rewrite == CaptionRewrite.REPLACE:
            return f"{(alt or 'Image').strip()} (Interpreted and captioned): {cap}"
        # APPEND: keep original + caption
        return f"{m.group(0)}{_render_caption_block(alt, cap)}"
        

    new_text = _IMG_TAG.sub(repl, text)
    if new_text != text:
        md_file.write_text(new_text, encoding="utf-8")
        logger.info("Captioned %s", md_file.name)
        return True
    return False

def run_caption_pipeline(
    output_dir: Path,
    *,
    captioner: Optional[DeepSeekVL2Captioner] = None, # NOTE: Reuse a preloaded engine if provided
    caption_model: str = "deepseek-ai/deepseek-vl2-tiny", # NOTE: Back-compatibility; used only when the captioner is None
    gpu_mem: float = 0.7,
    seed: Optional[int] = None,
    prompt: Optional[str] = None,
    rewrite: CaptionRewrite = CaptionRewrite.APPEND,
) -> None:
    """
    For a finished OCR run (with images/ and markdown/), caption the images referenced
    in every Markdown file by replacing image tags with caption lines.

    Args:
        output_dir (Path): The output directory containing markdown/ subdirectory.
        caption_model (str): The caption model to use.
        gpu_mem (float): The GPU memory utilization fraction.
        seed (Optional[int]): Optional random seed for reproducibility.
        prompt (Optional[str]): Optional prompt to override the default captioning prompt.
    """
    md_dir = output_dir / "markdown"
    if not md_dir.exists():
        raise FileNotFoundError(f"markdown directory not found under: {output_dir}")

    if captioner is None:
        logger.info("Initializing captioner: %s", caption_model)
        with quiet.quiet_stdio():
            captioner = DeepSeekVL2Captioner(
                CaptionerConfig(
                    model_name=caption_model,
                    gpu_memory_utilization=gpu_mem,
                    seed=seed,
                    # TODO:
                    # Hueristically chosen sizes to ensure valid VL2 input.
                    # May need adjustment for different image distributions.
                    min_side=128,
                    max_side=2048,
                )
            )

    changed = 0
    for md_file in sorted(md_dir.glob("*.md")):
        try:
            if caption_markdown_file(md_file, captioner, prompt_override=prompt, rewrite=rewrite):
                changed += 1
        except Exception:
            logger.exception("Failed to process %s", md_file.name)

    logger.info("Caption pipeline finished. %d file(s) updated.", changed)

