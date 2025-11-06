# service/model_manager.py
from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Optional

import quiet
from ocr_pipeline.ocr_engine import DeepSeekOCREngine
from caption_pipeline.caption_engine import DeepSeekVL2Captioner, CaptionerConfig

@dataclass
class Engines:
    ocr: DeepSeekOCREngine
    vl2: DeepSeekVL2Captioner
    gate: asyncio.BoundedSemaphore # admission gate for GPU work

_engines: Optional[Engines] = None

def init_engines(
    ocr_model: str,
    vl2_model: str,
    gpu_mem_ocr: float,
    gpu_mem_vl2: float,
    seed: int | None,
    gpu_slots: int = 1,
) -> None:
    """
    Initialize and warm the engines once per process.

    Args:
        ocr_model (str): Name of the OCR model to load.
        vl2_model (str): Name of the Vision Language Model to load.
        gpu_mem_ocr (float): Fraction of GPU memory to allocate for OCR model.
        gpu_mem_vl2 (float): Fraction of GPU memory to allocate for VL2 model.
        seed (int | None): Random seed for model initialization.
        gpu_slots (int): Number of concurrent GPU jobs allowed.
    """
    global _engines
    if _engines is not None:
        return

    with quiet.quiet_stdio():
        # OCR model
        ocr = DeepSeekOCREngine(
            model_name=ocr_model,
            gpu_memory_utilization=gpu_mem_ocr,
        )

        # Image captioning model (Vision Language Model)
        vl2 = DeepSeekVL2Captioner(
            CaptionerConfig(
                model_name=vl2_model,
                gpu_memory_utilization=gpu_mem_vl2,
                seed=seed,
                min_side=128, # px
                max_side=2048 # px
            )
        )

    # Engine instance to serve requests
    _engines = Engines(
        ocr=ocr,
        vl2=vl2,
        gate=asyncio.BoundedSemaphore(gpu_slots),
    )

def get_engines() -> Engines:
    """
    Retrieve the initialized engines.

    Returns:
        Engines: The initialized OCR and VL2 engines along with the admission gate.
    """
    if _engines is None:
        raise RuntimeError("Engine is not initialized yet.")
    return _engines

def engines_busy() -> bool:
    """
    Check if the engines are currently busy.

    Returns:
        bool: True if the engines are busy, False otherwise.
    """
    e = get_engines()
    return e.gate._value == 0
