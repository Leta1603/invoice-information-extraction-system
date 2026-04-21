# Invoice Information Extraction System

An end-to-end system for automated invoice field extraction using a custom FCOS object detector and Tesseract OCR. Users send invoice images via a Telegram bot; the backend detects and extracts key fields, stores results, and exports confirmed data to Google Sheets.

---

## Architecture

```
Telegram Bot (aiogram 3)
        │  HTTP
        ▼
FastAPI Backend (port 8000)
   ├── FCOS Detector  ──►  Tesseract OCR  ──►  Field Cleaning
   ├── PostgreSQL 16         (structured data)
   ├── MinIO S3              (original images)
   └── Google Sheets         (confirmed invoices)
```

### Extracted fields

| Field | Description |
|---|---|
| `seller` | Vendor / company name |
| `invoice_number` | Invoice / bill number |
| `invoice_date` | Issue date (normalised to DD.MM.YYYY) |
| `total_amount` | Total sum with currency |
| `item_description` | Goods or services listed |

---

## ML Model

Custom **FCOS** (Fully Convolutional One-Stage Object Detector) trained from scratch on invoice images.

| Component | Details |
|---|---|
| Backbone | 5-stage lightweight CNN, base channels = 32 |
| Neck | FPN — 3 levels (P3/P4/P5), 128 channels, strides [8, 16, 32] |
| Head | Shared 4-layer ConvBN head → cls / centerness / bbox branches |
| Classes | 5 (seller, invoice_number, invoice_date, total_amount, item_description) |
| Input | 512 × 512, normalised with mean = 0.5, std = 0.5 |
| Score | 0.7 · cls_prob + 0.3 · centerness_prob |
| Weights | `backend/app/models/best_fcos_detector.pt` |

---

## Tech Stack

| Layer | Technology |
|---|---|
| Bot | Python 3.11, aiogram 3 |
| Backend | Python 3.11, FastAPI, Uvicorn |
| ML | PyTorch (CPU), Tesseract OCR via pytesseract |
| Database | PostgreSQL 16, SQLAlchemy 2, Alembic |
| Object storage | MinIO (S3-compatible) via boto3 |
| Spreadsheet export | Google Sheets via gspread + service account |
| Containerisation | Docker, Docker Compose |

---

## Project Structure

```
.
├── backend/
│   ├── app/
│   │   ├── api/          # FastAPI routes
│   │   ├── core/         # Config (pydantic-settings)
│   │   ├── db/           # SQLAlchemy session & base
│   │   ├── ml/           # FCOS model, inference pipeline
│   │   ├── models/       # ORM models (User, Document, ExtractedData)
│   │   ├── schemas/      # Pydantic schemas
│   │   └── services/     # MinIO storage, Google Sheets
│   ├── Dockerfile
│   └── requirements.txt
├── bot/
│   ├── app/
│   │   └── handlers/     # Telegram message handlers
│   ├── Dockerfile
│   └── requirements.txt
├── data/                 # Dataset splits (train/val/test)
├── docker-compose.yml
└── .env                  # Environment variables (not committed)
```

---

## Getting Started

### Prerequisites

- Docker and Docker Compose
- Telegram bot token ([@BotFather](https://t.me/BotFather))
- Google Cloud service account JSON with Sheets API enabled

### 1. Clone the repository

```bash
git clone https://github.com/Leta1603/invoice-information-extraction-system.git
cd invoice-information-extraction-system
```

### 2. Configure environment

Copy the template and fill in your values:

```bash
cp .env.example .env
```

| Variable | Description |
|---|---|
| `BOT_TOKEN` | Telegram bot token |
| `BACKEND_URL` | Backend URL visible to the bot container (e.g. `http://backend:8000`) |
| `DATABASE_URL` | PostgreSQL connection string |
| `POSTGRES_USER` / `POSTGRES_PASSWORD` / `POSTGRES_DB` | Database credentials |
| `S3_ENDPOINT` / `S3_ACCESS_KEY` / `S3_SECRET_KEY` / `S3_BUCKET` | MinIO config |
| `GOOGLE_SHEETS_CREDENTIALS` | Path to service account JSON inside the container |
| `GOOGLE_SHEET_NAME` | Spreadsheet name (must be shared with the service account) |
| `DEVICE` | `cpu` or `cuda` |

### 3. Add credentials

Place your Google service account file at:

```
backend/app/credentials.json
```

Share your Google Spreadsheet with the service account e-mail address (Editor access).

### 4. Run

```bash
docker compose up --build -d
```

Services started:

| Service | URL |
|---|---|
| FastAPI backend | http://localhost:8000 |
| API docs (Swagger) | http://localhost:8000/docs |
| MinIO console | http://localhost:9001 |
| PostgreSQL | localhost:5432 |

---

## API Endpoints

| Method | Path | Description |
|---|---|---|
| `POST` | `/api/v1/documents/upload` | Upload image, run ML pipeline, save to DB & MinIO |
| `GET` | `/api/v1/documents/{id}` | Retrieve document with extracted fields |
| `POST` | `/api/v1/documents/{id}/confirm` | Confirm data and export row to Google Sheets |
| `DELETE` | `/api/v1/documents/{id}` | Cancel and delete document |
| `GET` | `/health` | Health check |

### Accepted image formats

`image/jpeg`, `image/png`, `image/webp`, `image/tiff`, `application/pdf`

---

## Telegram Bot Usage

1. Send `/start` to begin.
2. Send an invoice **photo** or **file** (JPEG / PNG / WebP, max 20 MB).
3. The bot returns extracted fields.
4. Tap **✅ Подтвердить** to save to Google Sheets or **❌ Отменить** to discard.
5. Send `/help` or tap **📖 Справка** for usage tips.

---

## Google Sheets Export

Confirmed invoices are appended to the configured spreadsheet with the following columns:

| Invoice № | Date | Seller | Total | Items |
|---|---|---|---|---|

Dates are normalised to `DD.MM.YYYY` regardless of the original format on the invoice.

---

## Development

### Run backend locally (without Docker)

```bash
cd backend
pip install -r requirements.txt
uvicorn app.main:app --reload
```

### Run bot locally

```bash
cd bot
pip install -r requirements.txt
python -m app.main
```

---

## Dataset

The model was trained on a custom annotated dataset of invoice images in COCO format.
Dataset splits are stored in `data/` (indices only; raw images are not committed).

Training details and exploratory analysis: [`Dataset.ipynb`](Dataset.ipynb)
