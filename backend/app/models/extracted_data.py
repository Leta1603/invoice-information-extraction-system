from datetime import datetime

from sqlalchemy import DateTime, ForeignKey, Integer, JSON, String, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.base import Base


class ExtractedData(Base):
    __tablename__ = "extracted_data"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    document_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("documents.id"), nullable=False, unique=True, index=True
    )
    seller: Mapped[str | None] = mapped_column(Text, nullable=True)
    invoice_number: Mapped[str | None] = mapped_column(String(255), nullable=True)
    invoice_date: Mapped[str | None] = mapped_column(String(50), nullable=True)
    total_amount: Mapped[str | None] = mapped_column(String(100), nullable=True)
    item_description: Mapped[str | None] = mapped_column(Text, nullable=True)
    confidence: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime, default=datetime.utcnow, nullable=False
    )

    document: Mapped["Document"] = relationship(  # noqa: F821
        "Document", back_populates="extracted_data"
    )
