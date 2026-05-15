"""Compatibility entrypoint for KineticsForge.

Render, Procfile, and local launch scripts should all use the same lightweight
physics server so the web UI and API do not drift into separate behavior.
"""
import os

from serve_lite import app


if __name__ == "__main__":
    import uvicorn

    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("serve_lite:app", host="0.0.0.0", port=port, log_level="info")
