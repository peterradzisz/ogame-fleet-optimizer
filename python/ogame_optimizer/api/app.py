"""FastAPI application for the OGame fleet auto-optimizer.

Mounts static + Jinja2 templates for the web UI and exposes API routes.
Configures logging on startup -> logs/ogame-optimizer.log
"""
from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from fastapi.encoders import jsonable_encoder
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import time as _time

from ogame_optimizer.logging_config import setup_logging, get_logger
from ogame_optimizer.api.routes import router as api_router


setup_logging()
_log = get_logger("ogame.api.app")

_WEB_DIR = Path(__file__).parent.parent / "web"
_STATIC_DIR = _WEB_DIR / "static"


def create_app() -> FastAPI:
    _log.info("Creating FastAPI app")
    app = FastAPI(
        title="OGame Fleet Auto-Optimizer",
        version="0.4.0",
        description="HTTP interface for OGame combat simulation and fleet optimization.",
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    if _STATIC_DIR.exists():
        app.mount("/static", StaticFiles(directory=str(_STATIC_DIR)), name="static")
        _log.info("Mounted static dir: %s", _STATIC_DIR)

    app.include_router(api_router)

    @app.middleware("http")
    async def log_requests(request: Request, call_next):
        start = _time.time()
        _log.info("REQ %s %s", request.method, request.url.path)
        try:
            response = await call_next(request)
            elapsed = (_time.time() - start) * 1000
            _log.info("RES %s %s -> %d (%.1fms)", request.method, request.url.path, response.status_code, elapsed)
            return response
        except Exception as e:
            _log.exception("REQ FAILED %s %s: %s", request.method, request.url.path, e)
            raise

    _log.info("App created OK")
    return app


app = create_app()


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    """Log 422 validation errors with full details for debugging."""
    _log.warning("VALIDATION ERROR %s %s", request.method, request.url.path)
    _log.warning("  Errors: %s", exc.errors())
    return JSONResponse(status_code=422, content=jsonable_encoder({"detail": exc.errors()}))


@app.get("/healthz")
def healthz() -> dict:
    return {"status": "ok", "service": "ogame-fleet-optimizer", "version": "0.4.0"}
