"""Retries with exponential backoff and jitter for transient network errors.

Quick tunnels, ntfy.sh and the Kaggle API fail now and then: a dropped connection,
a 502 while cloudflared reconnects, a 524 from the edge. These are retried; client
errors (4xx) are not.
"""

import logging
import random
import time
import urllib.error
import urllib.request

ATTEMPTS = 4
DELAYS = (1.0, 2.0, 4.0, 8.0)  # seconds before retry 1, 2, 3, ... (x 0.5-1.5 jitter)
RETRY_STATUS = {429, 502, 503, 504, 520, 521, 522, 523, 524, 530}
log = logging.getLogger("kis.retry")


def backoff(attempt: int) -> float:
    return DELAYS[min(attempt, len(DELAYS) - 1)] * random.uniform(0.5, 1.5)


def transient(error: Exception) -> bool:
    if isinstance(error, urllib.error.HTTPError):
        return error.code in RETRY_STATUS
    return isinstance(error, OSError)  # URLError, timeouts, connection resets


def urlopen(req: urllib.request.Request | str, timeout: float = 30, what: str = "request"):
    """urllib.request.urlopen, retried on transient errors."""
    for attempt in range(ATTEMPTS):
        try:
            return urllib.request.urlopen(req, timeout=timeout)
        except Exception as e:
            if attempt == ATTEMPTS - 1 or not transient(e):
                raise
            delay = backoff(attempt)
            log.warning("%s failed (%s); retry %d/%d in %.1fs", what, e, attempt + 1, ATTEMPTS - 1, delay)
            time.sleep(delay)
    raise AssertionError("unreachable")
