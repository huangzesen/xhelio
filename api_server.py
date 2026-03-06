#!/usr/bin/env python
"""Entry point for the FastAPI backend server.

Usage:
    python api_server.py [--port 8000] [--host 0.0.0.0]
"""

import argparse

import uvicorn

from api.app import create_app

app = create_app()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="XHelio FastAPI server")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload for development")
    args = parser.parse_args()

    uvicorn.run(
        "api_server:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
    )
