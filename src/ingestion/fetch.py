"""Fetch the raw YC company dataset and save it untouched with a provenance manifest."""
from __future__ import annotations

import hashlib
import json
import logging
import time
from datetime import datetime, timezone
from pathlib import Path

import requests

from src import config

log = logging.getLogger(__name__)


def _get(urls: list[str], timeout: int = 90, retries: int = 3) -> tuple[bytes, str]:
    """Try each URL in order, retrying with backoff. Returns (content, url_used)."""
    last_err: Exception | None = None
    for url in urls:
        for attempt in range(1, retries + 1):
            try:
                resp = requests.get(url, timeout=timeout)
                resp.raise_for_status()
                return resp.content, url
            except requests.RequestException as exc:  # network/HTTP errors only
                last_err = exc
                log.warning("Fetch failed (%s, attempt %d/%d): %s", url, attempt, retries, exc)
                time.sleep(2 * attempt)
    raise RuntimeError(f"All sources failed; last error: {last_err}")


def fetch_raw(force: bool = False) -> Path:
    """Download raw company JSON + metadata. Skips if a local raw file already exists (unless force)."""
    config.RAW_DIR.mkdir(parents=True, exist_ok=True)
    if config.RAW_COMPANIES.exists() and config.FETCH_MANIFEST.exists() and not force:
        log.info("Raw data already present at %s (use force=True to refetch)", config.RAW_COMPANIES)
        return config.RAW_COMPANIES

    content, url = _get(config.SOURCE_URLS)
    records = json.loads(content)
    if not isinstance(records, list) or not records:
        raise ValueError("Source returned an empty or non-list payload")
    config.RAW_COMPANIES.write_bytes(content)

    try:
        meta_bytes, _ = _get(config.META_URLS, timeout=30, retries=2)
        config.RAW_META.write_bytes(meta_bytes)
    except RuntimeError as exc:
        log.warning("Could not fetch meta.json (non-fatal): %s", exc)

    manifest = {
        "source": "yc-oss/api (unofficial; mirrors the YC public company directory's Algolia index)",
        "url": url,
        "retrieved_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "sha256": hashlib.sha256(content).hexdigest(),
        "bytes": len(content),
        "n_records": len(records),
        "licence_note": "Factual public company data; check yc-oss/api and ycombinator.com terms before redistribution.",
    }
    config.FETCH_MANIFEST.write_text(json.dumps(manifest, indent=2))
    log.info("Saved %d records (%d bytes) from %s", len(records), len(content), url)
    return config.RAW_COMPANIES
