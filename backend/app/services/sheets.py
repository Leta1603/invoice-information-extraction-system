"""Google Sheets integration via gspread + service account."""
from __future__ import annotations

import logging
import re
from datetime import datetime
from typing import Optional

from app.core.config import settings

# Target format for all dates written to the sheet
_DATE_FORMAT_OUT = "%d.%m.%Y"

# Russian month names → month number
_RU_MONTHS = {
    "января": 1, "февраля": 2, "марта": 3, "апреля": 4,
    "мая": 5, "июня": 6, "июля": 7, "августа": 8,
    "сентября": 9, "октября": 10, "ноября": 11, "декабря": 12,
    "янв": 1, "фев": 2, "мар": 3, "апр": 4,
    "май": 5, "июн": 6, "июл": 7, "авг": 8,
    "сен": 9, "окт": 10, "ноя": 11, "дек": 12,
}

_STRPTIME_FMTS = [
    "%d.%m.%Y", "%d/%m/%Y", "%Y-%m-%d", "%d-%m-%Y",
    "%d.%m.%y", "%d/%m/%y", "%Y/%m/%d",
    "%b %d, %Y", "%b %d %Y", "%B %d, %Y", "%B %d %Y",  # Jan 11, 2023 / January 11 2023
    "%b%d,%Y", "%b%d %Y", # Feb11,2023 / Feb11 2023
]


def _normalize_date(raw: str) -> str:
    """
    Try to parse *raw* into a date and return it as DD.MM.YYYY.
    If parsing fails, return *raw* unchanged.
    """
    if not raw:
        return raw

    cleaned = raw.strip().rstrip("г.").strip()

    # Try standard strptime patterns
    for fmt in _STRPTIME_FMTS:
        try:
            return datetime.strptime(cleaned, fmt).strftime(_DATE_FORMAT_OUT)
        except ValueError:
            continue

    m = re.match(
        r"(\d{1,2})\s+([а-яёА-ЯЁ]+)\.?\s+(\d{2,4})",
        cleaned,
        re.IGNORECASE,
    )
    if m:
        day, month_str, year = int(m.group(1)), m.group(2).lower(), int(m.group(3))
        if month_str in _RU_MONTHS:
            if year < 100:
                year += 2000
            try:
                return datetime(year, _RU_MONTHS[month_str], day).strftime(_DATE_FORMAT_OUT)
            except ValueError:
                pass

    return raw 

logger = logging.getLogger(__name__)

_SCOPES = [
    "https://www.googleapis.com/auth/spreadsheets",
    "https://www.googleapis.com/auth/drive",
]


def _get_client():
    import gspread
    from google.oauth2.service_account import Credentials

    creds = Credentials.from_service_account_file(
        settings.GOOGLE_SHEETS_CREDENTIALS, scopes=_SCOPES
    )
    return gspread.authorize(creds)


def append_invoice(data: dict) -> bool:
    """
    Append a row to the configured Google Sheet.
    Returns True on success, False on any error (logs the reason).

    Expected keys in *data*:
        document_id, seller, invoice_number, invoice_date,
        total_amount, item_description
    """
    try:
        client = _get_client()
        sheet = client.open(settings.GOOGLE_SHEET_NAME).sheet1

        row = [
            data.get("invoice_number", ""),
            _normalize_date(data.get("invoice_date", "")),
            data.get("seller", ""),
            data.get("total_amount", ""),
            data.get("item_description", ""),
        ]
        sheet.append_row(row, value_input_option="USER_ENTERED")
        logger.info("Appended row to Google Sheet '%s'", settings.GOOGLE_SHEET_NAME)
        return True

    except FileNotFoundError:
        logger.warning(
            "Google Sheets credentials not found at '%s'. Skipping.",
            settings.GOOGLE_SHEETS_CREDENTIALS,
        )
        return False
    except Exception as exc:
        logger.error("Failed to append to Google Sheets: %s", exc)
        return False
