# api/schemas.py
from __future__ import annotations

from pydantic import BaseModel

class HealthResponse(BaseModel):
    ok: bool
    ocr_model: str
    vl2_model: str

class StatusResponse(BaseModel):
    ocr_model: str
    vl2_model: str
    busy: bool
