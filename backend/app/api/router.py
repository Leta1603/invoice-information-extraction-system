from fastapi import APIRouter

from app.api.routes import documents

api_router = APIRouter()
api_router.include_router(documents.router, prefix="/documents", tags=["documents"])
