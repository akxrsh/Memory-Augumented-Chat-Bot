import logging
import sys
from pathlib import Path

def setup_logger(name: str = "app") -> logging.Logger:
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger

    logger.setLevel(logging.INFO)
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    # Console Handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # Ensure data directory exists
    log_dir = Path("data/logs")
    log_dir.mkdir(parents=True, exist_ok=True)

    # File Handler
    file_handler = logging.FileHandler(log_dir / f"{name}.log", encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return logger

logger = setup_logger("backend")
