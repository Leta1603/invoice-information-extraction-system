from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict

_BASE_DIR = Path(__file__).parent.parent  # backend/app/


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # Application
    APP_NAME: str = "Invoice Diploma API"
    DEBUG: bool = False

    # ML Model
    MODEL_PATH: str = str(_BASE_DIR / "models" / "best_fcos_detector.pt")
    DEVICE: str = "cpu"

    # PostgreSQL
    DATABASE_URL: str = "postgresql+psycopg2://postgres:postgres@localhost:5432/invoicedb"

    # MinIO / S3
    S3_ENDPOINT: str = "http://localhost:9000"
    S3_ACCESS_KEY: str = "minioadmin"
    S3_SECRET_KEY: str = "minioadmin"
    S3_BUCKET: str = "invoices"

    # Google Sheets
    GOOGLE_SHEETS_CREDENTIALS: str = str(_BASE_DIR / "credentials.json")
    GOOGLE_SHEET_NAME: str = "Invoice Data"


settings = Settings()
