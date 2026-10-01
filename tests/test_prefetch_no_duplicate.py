"""The row-group prefetch must never cause a second fetch of the same span.

Audit (two independent passes, 2026-10-01): the prefetch thread fetches
rg[idx+1] into the disk cache, but the main thread then calls
`rs.load_span` for that same row group without waiting. When the fetch is
slower than the compute it was meant to hide — 1170 s for a 1.8 GiB group
against ~8 s steps — the cache file is not there yet, so the main thread
re-downloads the identical span on its own connection. Slow row groups are
fetched twice, concurrently, which is the worst possible place to double the
traffic.

After the whole-shard path was opened off-Modal this matters less (a cached
shard is read locally, no Range at all), but the Range path is still the
fallback whenever hf_hub_download fails.

Run: python -m pytest tests/test_prefetch_no_duplicate.py -q
"""
import inspect
import time

from vision_adapter.data import stream as st


class _SlowFuture:
    """A future that never completes quickly — the race this guards."""

    def __init__(self, blob=b"x"):
        self._blob = blob

    def done(self):
        return False

    def result(self, timeout=None):
        time.sleep(0.2)
        return self._blob


def test_drain_waits_for_a_pending_prefetch():
    from vision_adapter.data.stream import drain_row_group_prefetch

    waited = {"n": 0}

    class _Fut:
        def done(self):
            return False

        def result(self):
            waited["n"] += 1
            return b"x"

    drain_row_group_prefetch(_Fut())
    assert waited["n"] == 1, "a pending prefetch must be waited on"


def test_drain_is_a_noop_when_nothing_is_pending():
    from vision_adapter.data.stream import drain_row_group_prefetch

    drain_row_group_prefetch(None)          # must not raise


def test_drain_returns_the_blob_so_the_main_thread_reuses_it():
    """Draining returns the fetched span; the caller should not re-fetch."""
    from vision_adapter.data.stream import drain_row_group_prefetch

    got = drain_row_group_prefetch(_SlowFuture(b"payload"))
    assert got == b"payload"


def test_the_row_group_loop_actually_drains():
    """The pin that matters: the loop must call it, or the race returns."""
    src = inspect.getsource(st.EmbStreamDataset.__iter__)
    assert "drain_row_group_prefetch" in src, (
        "the row-group loop never drains its pending prefetch — slow groups "
        "get fetched twice"
    )
