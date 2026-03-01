"""FastAPI app factory + lifespan (startup/shutdown)."""

import asyncio
import os
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

from .session_manager import APISessionManager
from . import routes


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup / shutdown lifecycle."""
    # Startup
    manager = APISessionManager()
    routes.session_manager = manager
    routes._start_time = time.time()
    routes._thread_pool = ThreadPoolExecutor(max_workers=4)
    await manager.start_cleanup_loop()

    # Kick off background mission data loading (non-blocking)
    from knowledge.loading_state import get_loading_state
    loading = get_loading_state()
    if not loading.is_ready:
        from knowledge.startup import run_background_load
        loop = asyncio.get_running_loop()
        loop.run_in_executor(routes._thread_pool, run_background_load)

    yield

    # Shutdown
    await manager.stop_cleanup_loop()
    manager.shutdown()
    routes._thread_pool.shutdown(wait=False)


def create_app() -> FastAPI:
    """Build and return the configured FastAPI application."""
    app = FastAPI(
        title="XHelio API",
        description="AI-powered heliophysics data visualization agent",
        version="0.1.0",
        lifespan=lifespan,
    )

    # CORS — restrict origins in production, allow all in development
    cors_origins = os.getenv("CORS_ORIGINS", "").strip()
    app.add_middleware(
        CORSMiddleware,
        allow_origins=cors_origins.split(",") if cors_origins else ["http://localhost:5173", "http://127.0.0.1:5173"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(routes.router)

    # Serve built React frontend in production
    frontend_dist = Path(__file__).parent.parent / "frontend" / "dist"
    if frontend_dist.exists():
        app.mount("/assets", StaticFiles(directory=frontend_dist / "assets"), name="assets")

        @app.get("/{full_path:path}")
        async def serve_frontend(full_path: str):
            file_path = (frontend_dist / full_path).resolve()
            if (file_path.is_relative_to(frontend_dist.resolve())
                    and file_path.exists() and file_path.is_file()):
                return FileResponse(file_path)
            return FileResponse(frontend_dist / "index.html")

    return app
