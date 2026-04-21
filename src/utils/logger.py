"""
Módulo 0.4 — Logger global via loguru.
"""
import sys
from pathlib import Path
from loguru import logger

_configured = False


def setup_logger(log_file: str = "logs/pipeline.log") -> None:
    global _configured
    if _configured:
        return

    log_path = Path(__file__).parent.parent.parent / log_file
    log_path.parent.mkdir(parents=True, exist_ok=True)

    logger.remove()
    logger.add(
        sys.stderr,
        colorize=True,
        format="[{time:YYYY-MM-DD HH:mm:ss}] [{level}] {module}:{line} — {message}",
        level="DEBUG",
    )
    logger.add(
        str(log_path),
        rotation="10 MB",
        retention="7 days",
        format="[{time:YYYY-MM-DD HH:mm:ss}] [{level}] {module}:{line} — {message}",
        level="DEBUG",
        encoding="utf-8",
    )
    _configured = True


setup_logger()

__all__ = ["logger"]
