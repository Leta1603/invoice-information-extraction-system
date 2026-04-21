"""
POST /documents/upload   — upload image/PDF, run ML, save to DB + MinIO
GET  /documents/{id}     — retrieve document + extracted data
POST /documents/{id}/confirm — confirm, export to Google Sheets
DELETE /documents/{id}   — cancel / delete
"""
from __future__ import annotations

import io
import logging
import uuid
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile, status
from sqlalchemy.orm import Session

from app.db.session import get_db
from app.models.document import Document, DocumentStatus
from app.models.extracted_data import ExtractedData
from app.models.user import User
from app.schemas.document import DocumentResponse, ExtractedDataSchema, UploadResponse
from app.services.sheets import append_invoice
from app.services.storage import delete_file, upload_file
from app.ml.pipeline import process_document

from PIL import Image

logger = logging.getLogger(__name__)
router = APIRouter()

_ALLOWED_TYPES = frozenset(
    {"image/jpeg", "image/png", "image/webp", "image/tiff", "application/pdf"}
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get_or_create_user(
    db: Session, telegram_id: int, username: Optional[str]
) -> User:
    user = db.query(User).filter(User.telegram_id == telegram_id).first()
    if not user:
        user = User(telegram_id=telegram_id, username=username)
        db.add(user)
        db.commit()
        db.refresh(user)
    return user


def _bytes_to_image(file_bytes: bytes, content_type: str) -> Image.Image:
    if content_type == "application/pdf":
        try:
            import fitz  # PyMuPDF
        except ImportError:
            raise HTTPException(
                status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
                detail="PDF support requires PyMuPDF (pip install PyMuPDF).",
            )
        doc = fitz.open(stream=file_bytes, filetype="pdf")
        page = doc[0]
        mat = fitz.Matrix(2.0, 2.0)  # 2× scale → better OCR
        pix = page.get_pixmap(matrix=mat)
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        doc.close()
        return img

    return Image.open(io.BytesIO(file_bytes)).convert("RGB")


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@router.post(
    "/upload",
    response_model=UploadResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Upload and process an invoice document",
)
async def upload_document(
    file: UploadFile = File(..., description="Image (JPEG/PNG/WEBP) or PDF"),
    telegram_id: int = Form(..., description="Telegram user ID"),
    username: Optional[str] = Form(None, description="Telegram username"),
    db: Session = Depends(get_db),
):
    # Validate MIME type
    content_type = file.content_type or ""
    if content_type not in _ALLOWED_TYPES:
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail=f"Unsupported file type '{content_type}'. "
                   f"Allowed: {sorted(_ALLOWED_TYPES)}",
        )

    file_bytes = await file.read()
    if not file_bytes:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Empty file.")

    # User
    user = _get_or_create_user(db, telegram_id, username)

    # Build S3 key
    ext = Path(file.filename or "document").suffix or ".jpg"
    s3_key = f"{telegram_id}/{uuid.uuid4()}{ext}"

    # Upload to MinIO
    try:
        upload_file(file_bytes, s3_key, content_type)
    except Exception as exc:
        logger.error("MinIO upload failed: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Storage service unavailable: {exc}",
        )

    # Create Document record
    document = Document(
        user_id=user.id,
        filename=file.filename or "document",
        s3_key=s3_key,
        status=DocumentStatus.processing,
    )
    db.add(document)
    db.commit()
    db.refresh(document)

    # ML inference
    try:
        image = _bytes_to_image(file_bytes, content_type)
        extracted = process_document(image)
    except HTTPException:
        raise
    except Exception as exc:
        logger.error("ML pipeline failed for document %d: %s", document.id, exc)
        document.status = DocumentStatus.failed
        document.error_message = str(exc)
        db.commit()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Processing failed: {exc}",
        )

    # Persist extracted data
    ext_data = ExtractedData(
        document_id=document.id,
        seller=extracted.get("seller"),
        invoice_number=extracted.get("invoice_number"),
        invoice_date=extracted.get("invoice_date"),
        total_amount=extracted.get("total_amount"),
        item_description=extracted.get("item_description"),
        confidence=extracted.get("confidence"),
    )
    db.add(ext_data)
    document.status = DocumentStatus.processed
    db.commit()
    db.refresh(ext_data)

    return UploadResponse(
        document_id=document.id,
        status=document.status.value,
        extracted_data=ExtractedDataSchema.model_validate(ext_data),
    )


@router.get(
    "/{document_id}",
    response_model=DocumentResponse,
    summary="Get document with extracted data",
)
def get_document(document_id: int, db: Session = Depends(get_db)):
    document = db.query(Document).filter(Document.id == document_id).first()
    if not document:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Document not found.")
    return DocumentResponse.model_validate(document)


@router.post(
    "/{document_id}/confirm",
    response_model=DocumentResponse,
    summary="Confirm document and export to Google Sheets",
)
def confirm_document(document_id: int, db: Session = Depends(get_db)):
    document = db.query(Document).filter(Document.id == document_id).first()
    if not document:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Document not found.")

    if document.status != DocumentStatus.processed:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Cannot confirm document with status '{document.status.value}'. "
                   "Only 'processed' documents can be confirmed.",
        )

    # Export to Google Sheets (non-critical — log errors, don't fail)
    if document.extracted_data:
        append_invoice(
            {
                "document_id": document.id,
                "seller": document.extracted_data.seller,
                "invoice_number": document.extracted_data.invoice_number,
                "invoice_date": document.extracted_data.invoice_date,
                "total_amount": document.extracted_data.total_amount,
                "item_description": document.extracted_data.item_description,
            }
        )

    document.status = DocumentStatus.confirmed
    db.commit()
    db.refresh(document)
    return DocumentResponse.model_validate(document)


@router.delete(
    "/{document_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Cancel / delete a document",
)
def delete_document(document_id: int, db: Session = Depends(get_db)):
    document = db.query(Document).filter(Document.id == document_id).first()
    if not document:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Document not found.")

    if document.s3_key:
        try:
            delete_file(document.s3_key)
        except Exception as exc:
            logger.warning("Failed to delete from MinIO (key=%s): %s", document.s3_key, exc)

    db.delete(document)
    db.commit()
