import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.core.config import settings

logging.basicConfig(
    level=logging.DEBUG if settings.DEBUG else logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # ---------- startup ----------
    logger.info("Creating database tables …")
    from app.db.base import Base
    from app.db.session import engine
    import app.models

    Base.metadata.create_all(bind=engine)
    logger.info("Database tables ready.")

    logger.info("Loading ML model …")
    from app.ml.load_model import load_model

    load_model()
    logger.info("ML model loaded.")

    yield
    logger.info("Shutting down.")


app = FastAPI(title=settings.APP_NAME, lifespan=lifespan)

from app.api.router import api_router

app.include_router(api_router, prefix="/api/v1")


@app.get("/health", tags=["health"])
def health():
    return {"status": "ok"}
