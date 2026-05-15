import os
from typing import Optional

from fastapi import Header, HTTPException


def configured_api_token() -> str:
    return os.getenv("KINETICSFORGE_API_TOKEN", "").strip()


def require_api_key(authorization: Optional[str] = Header(default=None)) -> None:
    expected = configured_api_token()
    if not expected:
        return
    if not authorization or not authorization.lower().startswith("bearer "):
        raise HTTPException(status_code=401, detail="Bearer token required")
    supplied = authorization.split(" ", 1)[1].strip()
    if supplied != expected:
        raise HTTPException(status_code=403, detail="Invalid API token")
