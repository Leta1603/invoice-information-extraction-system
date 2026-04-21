"""
Document handling flow:
  photo / document → upload to backend → show extracted data → confirm / cancel
"""
from __future__ import annotations

import logging
from collections import deque
from typing import Optional

import aiohttp
from aiogram import Bot, F, Router
from aiogram.filters import Command, CommandStart
from aiogram.types import (
    CallbackQuery,
    InlineKeyboardButton,
    InlineKeyboardMarkup,
    KeyboardButton,
    Message,
    ReplyKeyboardMarkup,
)
from aiogram.utils.keyboard import InlineKeyboardBuilder

from app.config import settings

logger = logging.getLogger(__name__)
router = Router()

_ALLOWED_MIME = frozenset({"image/jpeg", "image/png", "image/webp"})

# File size limit (20 MB — Telegram bot API download limit)
_MAX_FILE_SIZE = 20 * 1024 * 1024

# Users currently being processed (prevents parallel submissions)
_processing_users: set[int] = set()

# Media group IDs already accepted for processing (deduplicates album sends)
# deque with maxlen caps memory usage
_seen_media_groups: deque[str] = deque(maxlen=256)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _main_keyboard() -> ReplyKeyboardMarkup:
    return ReplyKeyboardMarkup(
        keyboard=[[KeyboardButton(text="📖 Справка")]],
        resize_keyboard=True,
    )


def _is_empty_result(extracted: dict) -> bool:
    fields = ["seller", "invoice_number", "invoice_date", "total_amount", "item_description"]
    return all(not extracted.get(f) for f in fields)


def _confirm_keyboard(document_id: int) -> InlineKeyboardMarkup:
    builder = InlineKeyboardBuilder()
    builder.row(
        InlineKeyboardButton(
            text="✅ Подтвердить", callback_data=f"confirm:{document_id}"
        ),
        InlineKeyboardButton(
            text="❌ Отменить", callback_data=f"cancel:{document_id}"
        ),
    )
    return builder.as_markup()


def _format_result(extracted: dict) -> str:
    seller = extracted.get("seller") or "—"
    invoice_number = extracted.get("invoice_number") or "—"
    invoice_date = extracted.get("invoice_date") or "—"
    total_amount = extracted.get("total_amount") or "—"
    item_description = extracted.get("item_description") or "—"

    lines = [
        "📄 <b>Результат обработки счёта</b>",
        "",
        "🏢 <b>Продавец</b>",
        seller,
        "",
        "🔢 <b>Номер счёта</b>",
        invoice_number,
        "",
        "📅 <b>Дата</b>",
        invoice_date,
        "",
        "💰 <b>Сумма</b>",
        total_amount,
        "",
        "📝 <b>Описание позиций</b>",
        item_description,
    ]

    return "\n".join(lines)


async def _send_to_backend(
    file_bytes: bytes,
    filename: str,
    content_type: str,
    telegram_id: int,
    username: Optional[str],
) -> dict:
    """POST /api/v1/documents/upload and return parsed JSON."""
    form = aiohttp.FormData()
    form.add_field("file", file_bytes, filename=filename, content_type=content_type)
    form.add_field("telegram_id", str(telegram_id))
    form.add_field("username", username or "")

    async with aiohttp.ClientSession() as session:
        async with session.post(
            f"{settings.API_BASE_URL}/api/v1/documents/upload",
            data=form,
            timeout=aiohttp.ClientTimeout(total=180),
        ) as resp:
            body = await resp.json(content_type=None)
            if resp.status != 201:
                detail = body.get("detail", resp.reason)
                raise RuntimeError(f"API {resp.status}: {detail}")
            return body


# ---------------------------------------------------------------------------
# Command handlers
# ---------------------------------------------------------------------------

@router.message(CommandStart())
async def cmd_start(message: Message) -> None:
    await message.answer(
        "👋 <b>Привет!</b> Я помогу извлечь данные из счетов.\n\n"
        "Отправьте мне <b>одно фото</b> счёта, и я автоматически распознаю:\n\n"
        "  • Продавца\n"
        "  • Номер счёта\n"
        "  • Дату\n"
        "  • Сумму\n"
        "  • Описание позиций\n\n"
        "⚠️ Отправляйте по одному фото за раз.",
        reply_markup=_main_keyboard(),
    )


@router.message(Command("help"))
async def cmd_help(message: Message) -> None:
    await message.answer(
        "📖 <b>Как пользоваться:</b>\n\n"
        "1. Отправьте фото счёта\n"
        "2. Дождитесь результата распознавания\n"
        "3. Проверьте данные\n"
        "4. Нажмите <b>«Подтвердить»</b> для сохранения "
        "или <b>«Отменить»</b>\n\n"
        "📌 <b>Поддерживаемые форматы:</b> JPEG, PNG, WEBP\n"
        "📌 <b>Макс. размер файла:</b> 20 МБ\n"
        "📌 Обрабатывается по одному счёту за раз\n"
        "📌 Для лучшего результата отправляйте чёткие фото без бликов и обрезки"
    )


# ---------------------------------------------------------------------------
# File handlers
# ---------------------------------------------------------------------------

@router.message(F.text == "📖 Справка")
async def handle_help_button(message: Message) -> None:
    await cmd_help(message)


@router.message(F.photo)
async def handle_photo(message: Message, bot: Bot) -> None:
    # Deduplicate photos sent as an album (media group)
    if message.media_group_id is not None:
        if message.media_group_id in _seen_media_groups:
            return  # silently skip extra photos from the same album
        _seen_media_groups.append(message.media_group_id)
        await message.answer(
            "⚠️ Вы отправили несколько фото сразу.\n"
            "Я обработаю только первое — отправляйте по одному счёту за раз."
        )

    user_id = message.from_user.id
    if user_id in _processing_users:
        await message.answer("⏳ Подождите — я ещё обрабатываю предыдущий счёт.")
        return

    _processing_users.add(user_id)
    status_msg = await message.answer("⏳ Обрабатываю изображение…")
    try:
        photo = message.photo[-1]  # largest available
        file_info = await bot.get_file(photo.file_id)
        file_io = await bot.download_file(file_info.file_path)
        file_bytes = file_io.read()

        await _process_and_respond(
            message=message,
            status_msg=status_msg,
            file_bytes=file_bytes,
            filename="photo.jpg",
            content_type="image/jpeg",
        )
    finally:
        _processing_users.discard(user_id)


@router.message(F.document)
async def handle_document(message: Message, bot: Bot) -> None:
    doc = message.document

    # Validate MIME type
    if doc.mime_type not in _ALLOWED_MIME:
        await message.answer(
            "⚠️ Неподдерживаемый формат файла.\n\n"
            "Отправьте изображение в формате JPEG, PNG или WEBP."
        )
        return

    # Validate file size
    if doc.file_size and doc.file_size > _MAX_FILE_SIZE:
        await message.answer(
            "⚠️ Файл слишком большой (макс. 20 МБ).\n"
            "Попробуйте сжать изображение или отправить фото."
        )
        return

    user_id = message.from_user.id
    if user_id in _processing_users:
        await message.answer("⏳ Подождите — я ещё обрабатываю предыдущий счёт.")
        return

    _processing_users.add(user_id)
    status_msg = await message.answer("⏳ Обрабатываю документ…")
    try:
        file_info = await bot.get_file(doc.file_id)
        file_io = await bot.download_file(file_info.file_path)
        file_bytes = file_io.read()

        if not file_bytes:
            await status_msg.edit_text("⚠️ Файл пустой. Отправьте другой документ.")
            return

        await _process_and_respond(
            message=message,
            status_msg=status_msg,
            file_bytes=file_bytes,
            filename=doc.file_name or "document",
            content_type=doc.mime_type,
        )
    finally:
        _processing_users.discard(user_id)


@router.message()
async def handle_unknown(message: Message) -> None:
    """Catch-all for text messages and unsupported content."""
    await message.answer(
        "🤔 Я понимаю только фото счетов.\n"
        "Отправьте изображение (JPEG, PNG, WEBP).\n\n"
        "Нажмите кнопку «📖 Справка» для подробностей."
    )


async def _process_and_respond(
    message: Message,
    status_msg: Message,
    file_bytes: bytes,
    filename: str,
    content_type: str,
) -> None:
    telegram_id = message.from_user.id
    username = message.from_user.username

    try:
        result = await _send_to_backend(
            file_bytes, filename, content_type, telegram_id, username
        )
    except aiohttp.ClientError as exc:
        logger.error("HTTP error contacting backend: %s", exc)
        await status_msg.edit_text(
            "❌ Не удалось связаться с сервером. Попробуйте позже."
        )
        return
    except RuntimeError as exc:
        logger.error("Backend error: %s", exc)
        await status_msg.edit_text(
            "❌ Не удалось обработать файл. Попробуйте другое изображение."
        )
        return
    except Exception as exc:
        logger.exception("Unexpected error: %s", exc)
        await status_msg.edit_text(
            "❌ Произошла ошибка. Попробуйте позже."
        )
        return

    document_id: int = result["document_id"]
    extracted: dict = result.get("extracted_data", {})

    # Remove the "processing" message
    try:
        await status_msg.delete()
    except Exception:
        pass

    # Nothing recognized — inform and stop
    if _is_empty_result(extracted):
        await message.answer(
            "😔 Не удалось распознать данные счёта.\n\n"
            "Попробуйте:\n"
            "  • Сфотографировать чётче, без бликов\n"
            "  • Убедиться, что весь счёт попал в кадр\n"
            "  • Отправить файл изображения вместо фото"
        )
        return

    # Send result as a new message (preserved, not edited later)
    text = _format_result(extracted)
    await message.answer(text)

    # Send confirmation buttons as a separate message
    keyboard = _confirm_keyboard(document_id)
    await message.answer(
        "👆 Данные верны?",
        reply_markup=keyboard,
    )


# ---------------------------------------------------------------------------
# Callback handlers
# ---------------------------------------------------------------------------

@router.callback_query(F.data.startswith("confirm:"))
async def on_confirm(callback: CallbackQuery) -> None:
    document_id = int(callback.data.split(":", 1)[1])

    # Immediately show progress so user knows it's working
    await callback.message.edit_text("⏳ Сохраняю данные…")
    await callback.answer()

    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{settings.API_BASE_URL}/api/v1/documents/{document_id}/confirm",
                timeout=aiohttp.ClientTimeout(total=30),
            ) as resp:
                if resp.status == 200:
                    await callback.message.edit_text(
                        "✅ Данные подтверждены и сохранены!"
                    )
                else:
                    body = await resp.json(content_type=None)
                    detail = body.get("detail", resp.reason)
                    await callback.message.edit_text(
                        f"⚠️ Не удалось подтвердить: {detail}"
                    )
    except Exception as exc:
        logger.error("Confirm error: %s", exc)
        await callback.message.edit_text(
            "⚠️ Ошибка при подтверждении. Попробуйте позже."
        )


@router.callback_query(F.data.startswith("cancel:"))
async def on_cancel(callback: CallbackQuery) -> None:
    document_id = int(callback.data.split(":", 1)[1])

    try:
        async with aiohttp.ClientSession() as session:
            async with session.delete(
                f"{settings.API_BASE_URL}/api/v1/documents/{document_id}",
                timeout=aiohttp.ClientTimeout(total=30),
            ) as resp:
                logger.info(
                    "Deleted document %d, status=%d", document_id, resp.status
                )
    except Exception as exc:
        logger.warning("Delete error (non-critical): %s", exc)

    await callback.message.edit_text("❌ Отменено.")
    await callback.answer()
