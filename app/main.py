"""FastAPI application factory + lifespan.

The lifespan does four things, once per process:
1. create the schema (demo-grade bootstrap; migrations out of scope)
2. seed the two sample companies if the DB is empty
3. open the LangGraph checkpointer:
     - Postgres (production): AsyncPostgresSaver -> pause/resume survives restarts
     - sqlite (dev/tests):    AsyncSqliteSaver  -> same mechanics on a file
4. build the ChatService around it

Nothing here needs an LLM key at startup; without one, the deterministic
offline assistant takes over and every flow still works.
"""
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from .agent_service import ChatService
from .api import auth as auth_api
from .api import chat as chat_api
from .api import health as health_api
from .api import manager as manager_api
from .config import get_settings
from .db import SessionLocal, create_all, engine
from .seed import seed_if_empty

log = logging.getLogger("shift.main")


def _psycopg_uri(asyncpg_uri: str) -> str:
    """postgresql+asyncpg://... -> postgresql://... (the checkpointer uses psycopg3)."""
    return asyncpg_uri.replace("postgresql+asyncpg://", "postgresql://")


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    await create_all()
    async with SessionLocal() as session:
        await seed_if_empty(session)

    saver_cm = None
    sqlite_conn = None
    if settings.database_url.startswith("postgres"):
        from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver

        saver_cm = AsyncPostgresSaver.from_conn_string(_psycopg_uri(settings.database_url))
        checkpointer = await saver_cm.__aenter__()
        await checkpointer.setup()
    else:
        from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
        import aiosqlite

        path = settings.database_url.split("///", 1)[-1] or "./dev.db"
        sqlite_conn = await aiosqlite.connect(path)
        checkpointer = AsyncSqliteSaver(sqlite_conn)
        await checkpointer.setup()

    try:
        app.state.chat_service = ChatService(settings=settings, checkpointer=checkpointer)
        log.info("chat service ready (provider=%s)", settings.llm_provider)
        yield
    finally:
        if saver_cm is not None:
            await saver_cm.__aexit__(None, None, None)
        if sqlite_conn is not None:
            await sqlite_conn.close()
        await engine.dispose()


def create_app() -> FastAPI:
    settings = get_settings()
    app = FastAPI(title=settings.app_name, lifespan=lifespan)

    app.include_router(health_api.router)
    app.include_router(auth_api.router)
    app.include_router(chat_api.router)
    app.include_router(manager_api.router)

    @app.exception_handler(Exception)
    async def unhandled(request: Request, exc: Exception) -> JSONResponse:
        # Stack traces go to the server log, never to the client.
        log.exception("unhandled error on %s", request.url.path)
        return JSONResponse(
            status_code=500,
            content={"error": "internal_error",
                     "message": "Something went wrong on our side. Please try again in a moment."},
        )

    return app


app = create_app()
