# service/settings.py
from __future__ import annotations
from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    # Models
    model_ocr: str = "deepseek-ai/DeepSeek-OCR"
    model_vl2: str = "deepseek-ai/deepseek-vl2-tiny"

    # GPU memory split
    gpu_mem_ocr: float = 0.70
    gpu_mem_vl2: float = 0.70

    # GPU device pinning
    ocr_device: str = "0"
    vl2_device: str = "1"

    # Admission control
    # NOTE: Not to overwhelm a limited resource with too many concurrent jobs
    gpu_slots: int = 1  # number of concurrent GPU jobs allowed (1 for single GPU)
    seed: int | None = None

    model_config = SettingsConfigDict(env_prefix="", env_file=None, extra="ignore")
