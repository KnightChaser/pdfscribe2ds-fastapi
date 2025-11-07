# service/model_manager.py
from __future__ import annotations

import os
import asyncio
from dataclasses import dataclass
from typing import Optional
from contextlib import contextmanager

import quiet
from ocr_pipeline.ocr_engine import DeepSeekOCREngine
from caption_pipeline.caption_engine import DeepSeekVL2Captioner, CaptionerConfig

@contextmanager
def _with_cuda_visible(dev_ids: str):
    """
    Temporarily set CUDA_VISIBLE_DEVICE for engine initialization.

    Args:
        dev_ids (str): Comma-separated GPU device IDs to set.
    """
    old = os.environ.get("CUDA_VISIBLE_DEVICES")
    os.environ["CUDA_VISIBLE_DEVICES"] = dev_ids
    try:
        yield
    finally:
        if old is None:
            os.environ.pop("CUDA_VISIBLE_DEVICES", None)
        else:
            os.environ["CUDA_VISIBLE_DEVICES"] = old

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
    *,
    ocr_device: str = "0",
    vl2_device: str = "0",
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
        # OCR model (Pinning OCR model to GPU0 or choiced device)
        with _with_cuda_visible(ocr_device):
            ocr = DeepSeekOCREngine(
                model_name=ocr_model,
                gpu_memory_utilization=gpu_mem_ocr,
            )

        # Image captioning model (Vision Language Model) (Pinning VL2 model to GPU1 or choiced device)
        with _with_cuda_visible(vl2_device):
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

async def try_admit_now() -> bool:
    """
    Try to admit a new GPU job without waiting.

    Returns:
        bool: True if admitted, False if the gate is full.
    """
    e = get_engines()
    try:
        # NOTE:
        # Non-blocking probe via near-zero timeout
        # If I set timeout=0, it always raises TimeoutError immediately,
        # even though the resource is available. At least we have to give
        # a minimal, trivail amount of time for the event loop to schedule.
        await asyncio.wait_for(e.gate.acquire(), timeout=0.1)
    except asyncio.TimeoutError:
        return False
    else:
        # Immediately release after acquiring
        e.gate.release()
        return True

async def try_admit_with_timeout(timeout_s: float) -> bool:
    """
    Try to admit a new GPU job with a timeout.

    Args:
        timeout_s (float): Timeout in seconds to wait for admission.

    Returns:
        bool: True if admitted within the timeout, False otherwise.
    """
    e = get_engines()
    try:
        await asyncio.wait_for(e.gate.acquire(), timeout=timeout_s)
    except asyncio.TimeoutError:
        return False
    else:
        # Immediately release after acquiring
        e.gate.release()
        return True
