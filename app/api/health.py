"""Health + service info."""
from fastapi import APIRouter, Request

from ..config import get_settings

router = APIRouter(tags=["meta"])


@router.get("/health")
async def health() -> dict:
    return {"status": "ok"}


@router.get("/")
async def root(request: Request) -> dict:
    settings = get_settings()
    return {
        "service": settings.app_name,
        "llm_provider": settings.llm_provider,
        "llm_max_calls_per_turn": settings.llm_max_calls_per_turn,
        "endpoints": [
            "POST /auth/token (demo IdP)",
            "POST /chat",
            "GET /chat/{thread_id}/messages",
            "GET /my/requests",
            "GET /manager/requests",
            "POST /manager/requests/{id}/decision",
        ],
    }
