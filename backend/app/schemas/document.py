from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, ConfigDict


class ExtractedDataSchema(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    seller: Optional[str] = None
    invoice_number: Optional[str] = None
    invoice_date: Optional[str] = None
    total_amount: Optional[str] = None
    item_description: Optional[str] = None
    confidence: Optional[dict] = None


class DocumentResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    filename: str
    status: str
    extracted_data: Optional[ExtractedDataSchema] = None


class UploadResponse(BaseModel):
    document_id: int
    status: str
    extracted_data: ExtractedDataSchema
