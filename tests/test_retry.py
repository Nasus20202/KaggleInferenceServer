import urllib.error
from email.message import Message

import pytest

from kis import retry


def test_retries_transient_errors_then_succeeds(monkeypatch):
    calls = []

    def flaky(req, timeout):
        calls.append(req)
        if len(calls) < 3:
            raise urllib.error.URLError("connection reset")
        return "ok"

    monkeypatch.setattr(retry.urllib.request, "urlopen", flaky)
    assert retry.urlopen("http://x") == "ok" and len(calls) == 3


def test_gives_up_and_does_not_retry_client_errors(monkeypatch):
    calls = []

    def fail(req, timeout):
        calls.append(req)
        raise urllib.error.HTTPError("http://x", 404, "not found", Message(), None)

    monkeypatch.setattr(retry.urllib.request, "urlopen", fail)
    with pytest.raises(urllib.error.HTTPError):
        retry.urlopen("http://x")
    assert len(calls) == 1
    monkeypatch.setattr(retry.urllib.request, "urlopen", lambda req, timeout: calls.append(req) or 1 / 0)
    with pytest.raises(ZeroDivisionError):
        retry.urlopen("http://x")
