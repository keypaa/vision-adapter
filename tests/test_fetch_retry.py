"""429 rate-limit handling in _fetch_range (HF CDN throttles parallel Range).

A 429 must honor Retry-After and retry on its own budget (not consume the
generic 3-attempt budget): index builds with 8 workers hit 429 storms that
3x(0.5-2s) backoffs cannot survive.
"""
import io
import time
import urllib.error
import urllib.request

import pytest

import vision_adapter.data.stream as st


def _http429(url, retry_after="7"):
    return urllib.error.HTTPError(url, 429, "Too Many Requests",
                                  {"Retry-After": retry_after}, None)


def test_429_honors_retry_after_then_succeeds(monkeypatch):
    calls = {"n": 0}
    sleeps: list[float] = []

    def fake_urlopen(req, timeout=None):
        calls["n"] += 1
        if calls["n"] < 3:
            raise _http429(req.full_url)
        return io.BytesIO(b"data")

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    monkeypatch.setattr(time, "sleep", lambda s: sleeps.append(s))

    assert st._fetch_range("http://x/y.parquet", 0, 3) == b"data"
    assert calls["n"] == 3
    assert len(sleeps) == 2 and all(s >= 7 for s in sleeps)


def test_persistent_429_raises_after_budget(monkeypatch):
    calls = {"n": 0}

    def fake_urlopen(req, timeout=None):
        calls["n"] += 1
        raise _http429(req.full_url, retry_after="1")

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    monkeypatch.setattr(time, "sleep", lambda s: None)

    with pytest.raises(urllib.error.HTTPError):
        st._fetch_range("http://x/y.parquet", 0, 3)
    assert calls["n"] == 1 + st._RATE_LIMIT_RETRIES


def test_remote_size_retries_429(monkeypatch):
    calls = {"n": 0}

    class _Resp:
        headers = {"Content-Length": "123"}

    def fake_urlopen(req, timeout=None):
        calls["n"] += 1
        if calls["n"] < 3:
            raise urllib.error.HTTPError(req.full_url, 429, "Slow Down",
                                         {"Retry-After": "2"}, None)
        return _Resp()

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    monkeypatch.setattr(time, "sleep", lambda s: None)

    assert st._remote_size("http://x/y.parquet") == 123
    assert calls["n"] == 3


def test_non_429_behavior_unchanged(monkeypatch):
    calls = {"n": 0}

    def fake_urlopen(req, timeout=None):
        calls["n"] += 1
        raise urllib.error.HTTPError(req.full_url, 500, "boom", {}, None)

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    monkeypatch.setattr(time, "sleep", lambda s: None)

    with pytest.raises(urllib.error.HTTPError):
        st._fetch_range("http://x/y.parquet", 0, 3, retries=3)
    assert calls["n"] == 3
