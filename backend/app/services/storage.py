"""MinIO storage service via boto3."""
from __future__ import annotations

import io
import logging

import boto3
from botocore.exceptions import ClientError

from app.core.config import settings

logger = logging.getLogger(__name__)


def _client():
    return boto3.client(
        "s3",
        endpoint_url=settings.S3_ENDPOINT,
        aws_access_key_id=settings.S3_ACCESS_KEY,
        aws_secret_access_key=settings.S3_SECRET_KEY,
    )


def _ensure_bucket(client) -> None:
    try:
        client.head_bucket(Bucket=settings.S3_BUCKET)
    except ClientError:
        client.create_bucket(Bucket=settings.S3_BUCKET)
        logger.info("Created bucket: %s", settings.S3_BUCKET)


def upload_file(
    file_bytes: bytes,
    s3_key: str,
    content_type: str = "application/octet-stream",
) -> str:
    """Upload bytes to MinIO. Returns the s3_key."""
    client = _client()
    _ensure_bucket(client)
    client.upload_fileobj(
        io.BytesIO(file_bytes),
        settings.S3_BUCKET,
        s3_key,
        ExtraArgs={"ContentType": content_type},
    )
    logger.info("Uploaded s3://%s/%s", settings.S3_BUCKET, s3_key)
    return s3_key


def download_file(s3_key: str) -> bytes:
    """Download file from MinIO. Returns raw bytes."""
    client = _client()
    buf = io.BytesIO()
    client.download_fileobj(settings.S3_BUCKET, s3_key, buf)
    buf.seek(0)
    return buf.read()


def delete_file(s3_key: str) -> None:
    """Delete a file from MinIO."""
    client = _client()
    client.delete_object(Bucket=settings.S3_BUCKET, Key=s3_key)
    logger.info("Deleted s3://%s/%s", settings.S3_BUCKET, s3_key)
