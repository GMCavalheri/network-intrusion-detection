"""Shared helpers for the dataset download scripts."""

import logging
import os
import sys
from logging.handlers import RotatingFileHandler

import requests

LOG_DIR = os.environ.get("LOG_DIR", "/opt/logs")
DATA_DIR = os.environ.get("DATA_DIR", "data")


def get_logger(name: str) -> logging.Logger:
    """Console + rotating-file logger, same shape as spark_jobs/common.py's
    so every component in this project logs identically."""
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger

    logger.setLevel(logging.INFO)
    fmt = logging.Formatter(
        "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
    )

    console = logging.StreamHandler(sys.stdout)
    console.setFormatter(fmt)
    logger.addHandler(console)

    try:
        os.makedirs(LOG_DIR, exist_ok=True)
        file_handler = RotatingFileHandler(
            os.path.join(LOG_DIR, f"{name}.log"), maxBytes=5_000_000, backupCount=3
        )
        file_handler.setFormatter(fmt)
        logger.addHandler(file_handler)
    except OSError:
        pass  # LOG_DIR not writable/mounted in this environment - console-only is fine

    logger.propagate = False
    return logger


def download_file(url: str, dest_path: str, logger: logging.Logger, session=None, chunk_size: int = 1 << 20) -> int:
    """Streams `url` to `dest_path`, skipping the request entirely if a
    non-empty file already exists there (re-run the download scripts freely
    without re-fetching hundreds of MB). Returns the byte count written (0 if
    skipped). Raises requests.HTTPError on a non-2xx response.
    """
    if os.path.exists(dest_path) and os.path.getsize(dest_path) > 0:
        logger.info("Already downloaded, skipping: %s", dest_path)
        return 0

    os.makedirs(os.path.dirname(dest_path), exist_ok=True)
    http = session or requests
    logger.info("Downloading %s -> %s", url, dest_path)

    tmp_path = dest_path + ".part"
    written = 0
    with http.get(url, stream=True, timeout=60) as resp:
        resp.raise_for_status()
        with open(tmp_path, "wb") as f:
            for chunk in resp.iter_content(chunk_size=chunk_size):
                if chunk:
                    f.write(chunk)
                    written += len(chunk)
    os.replace(tmp_path, dest_path)
    logger.info("Downloaded %d bytes -> %s", written, dest_path)
    return written
