# api/schemas.py
from __future__ import annotations

from pydantic import BaseModel, Field

class HealthResponse(BaseModel):
    ok: bool
    ocr_model: str
    vl2_model: str

class StatusResponse(BaseModel):
    ocr_model: str
    vl2_model: str
    busy: bool

class ErrorResponse(BaseModel):
    detail: str = Field(..., example="GPU stayed busy for 60.0s; try again later.") # type: ignore
