# Import all ORM models so SQLAlchemy registers them with the metadata.
from app.models.user import User  # noqa: F401
from app.models.document import Document, DocumentStatus  # noqa: F401
from app.models.extracted_data import ExtractedData  # noqa: F401

__all__ = ["User", "Document", "DocumentStatus", "ExtractedData"]
