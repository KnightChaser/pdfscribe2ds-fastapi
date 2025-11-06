# api/main.py
from __future__ import annotations

from fastapi import FastAPI
from contextlib import asynccontextmanager

from api.routes import router
from service.settings import Settings
from service.model_manager import init_engines
import quiet

def create_app() -> FastAPI:
    """
    Create and configure the FastAPI application.
    """
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        """
        When the service starts, initialize the model engines based on the settings.
        By doing so, we don't have to reload or reinitialize the models for each request.
        """
        s = Settings()
        with quiet.quiet_stdio():
            init_engines(
                ocr_model=s.model_ocr,
                vl2_model=s.model_vl2,
                gpu_mem_ocr=s.gpu_mem_ocr,
                gpu_mem_vl2=s.gpu_mem_vl2,
                seed=s.seed,
                gpu_slots=s.gpu_slots,
            )
        yield

    app = FastAPI(title="pdfscribe2ds-fastapi", version="0.1.0", lifespan=lifespan)
    app.include_router(router, prefix="/v1")
    return app

app = create_app()
