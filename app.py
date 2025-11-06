# app.py
from pathlib import Path
import typer
import logging
from rich.logging import RichHandler

from caption_pipeline.caption_pipeline import run_caption_pipeline, CaptionRewrite
from ocr_pipeline.pipeline import run_pdf_pipeline
import quiet

app = typer.Typer(help="PDF --> images --> DeepSeek-OCR --> Markdown --> DeepSeek-VL2 --> Markdown w/ Image captions")

def setup_logging(level: str = "INFO") -> None:
    """
    Set up a dedicated logger that ignores the 'everything is ERROR' setup
    from quiet.py for our app logs.
    """
    logger = logging.getLogger("pdfscribe2ds")
    logger.setLevel(level)

    logger.propagate = False  # Prevent double logging

    # Avoid adding multiple handlers on repeated calls
    if not any(isinstance(h, RichHandler) for h in logger.handlers):
        handler = RichHandler(
            rich_tracebacks=True,
            show_time=True,
            show_path=True
        )
        # RichHandler ignores format's time mostly, keep it simple
        formatter = logging.Formatter("%(message)s")
        handler.setFormatter(formatter)
        logger.addHandler(handler)


@app.command("pdf")
def pdf_to_md(
    pdf_path: Path = typer.Argument(..., exists=True, readable=True, help="Input PDF"),
    output_dir: Path = typer.Option(Path("./output"), help="Where to store images and markdown"),
    model_name: str = typer.Option("deepseek-ai/DeepSeek-OCR", help="DeepSeek-OCR model name"),
    dpi: int = typer.Option(200, help="DPI for pdf2image"),
    num_processes: int = typer.Option(None, help="Number of processes for parallel PDF conversion (default: CPU count // 2)"),
    num_threads: int = typer.Option(None, help="Number of threads for parallel image saving (default: 4)"),
    log_level: str = typer.Option("INFO", help="Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)"),
) -> None:
    """
    Convert a PDF to per-page Markdown files + cropped assets.
    """
    setup_logging(log_level.upper())
    
    run_pdf_pipeline(
        pdf_path=pdf_path,
        output_dir=output_dir,
        model_name=model_name,
        dpi=dpi,
        num_processes=num_processes,
        num_threads=num_threads,
    )

    logging.getLogger("pdfscribe2ds").info("Job completed! PDF processed --> %s", output_dir)

@app.command("caption")
def caption_md(
    output_dir: Path = typer.Argument(..., exists=True, readable=True, file_okay=False, help="Root output dir that contains markdown/ and images/"),
    caption_model: str = typer.Option("deepseek-ai/deepseek-vl2-tiny", help="DeepSeek VL2 model for captioning"),
    gpu_mem: float = typer.Option(0.7, help="GPU memory utilization for vLLM (0-1)"),
    seed: int = typer.Option(None, help="Optional RNG seed"),
    prompt: str = typer.Option(None, help="Override the default caption prompt"),
    log_level: str = typer.Option("INFO", help="Logging level (DEBUG, INFO, WARNING, ERROR)"),
    rewrite_img_tags: str = typer.Option("append", help="How to rewrite image tags: 'append' (default) or 'replace'")
) -> None:
    """
    Caption images referenced in markdown/*.md by replacing image tags with
    'Image N (Interpreted and captioned): <text>'.
    """
    setup_logging(log_level.upper())

    run_caption_pipeline(
        output_dir=output_dir,
        caption_model=caption_model,
        gpu_mem=gpu_mem,
        seed=seed,
        prompt=prompt,
        rewrite=CaptionRewrite.APPEND if rewrite_img_tags.lower() == "append" else CaptionRewrite.REPLACE,
    )

    logging.getLogger("pdfscribe2ds").info("Captioning completed! --> %s", output_dir)

if __name__ == "__main__":
    app()
