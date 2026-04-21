"""Bot entry point."""
import asyncio
import logging

from aiogram import Bot, Dispatcher
from aiogram.client.default import DefaultBotProperties
from aiogram.enums import ParseMode

from app.config import settings
from app.handlers.document_handler import router as doc_router

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)


async def _wait_for_backend(api_base: str, retries: int = 30, delay: float = 3.0) -> None:
    """Poll /health until the backend is reachable."""
    import aiohttp

    for attempt in range(1, retries + 1):
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    f"{api_base}/health", timeout=aiohttp.ClientTimeout(total=5)
                ) as resp:
                    if resp.status == 200:
                        logger.info("Backend is ready.")
                        return
        except Exception:
            pass
        logger.info("Waiting for backend … attempt %d/%d", attempt, retries)
        await asyncio.sleep(delay)
    raise RuntimeError(f"Backend at {api_base} did not become ready in time.")


async def main() -> None:
    await _wait_for_backend(settings.API_BASE_URL)

    bot = Bot(
        token=settings.BOT_TOKEN,
        default=DefaultBotProperties(parse_mode=ParseMode.HTML),
    )
    dp = Dispatcher()
    dp.include_router(doc_router)

    logger.info("Starting polling …")
    await dp.start_polling(bot)


if __name__ == "__main__":
    asyncio.run(main())
