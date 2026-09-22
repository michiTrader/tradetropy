"""Tests for the Sesh streaming hooks (supports_streaming / create_feed)."""

import pytest

from tradetropy.connectors.ccxt import SeshCCXTLive, SeshCCXTSim
from tradetropy.streaming.ccxt_pro import CCXTProFeed


class _MiniEx:
    """Minimal ccxt-like exchange: empty `has` so the session's sync() is a
    no-op and no network is touched during construction."""

    id = "mockex"
    has: dict = {}

    def __init__(self):
        self.sandbox = False
        self.demo = False

    def set_sandbox_mode(self, flag):
        self.sandbox = flag

    def enable_demo_trading(self, flag=True):
        self.demo = flag

    def load_markets(self, *a, **k):
        return {}


# =====
# SeshCCXTLive enables streaming
# =====
class TestCCXTLiveStreaming:
    def test_supports_streaming_true(self):
        sesh = SeshCCXTLive(_MiniEx())
        assert sesh.supports_streaming is True

    def test_create_feed_returns_ccxtprofeed(self):
        sesh = SeshCCXTLive(_MiniEx())
        feed = sesh.create_feed(timeframe_ms=300_000, book_limit=10)
        assert isinstance(feed, CCXTProFeed)
        # Built from the exchange id (fresh ccxt.pro instance on connect()).
        assert feed._exchange_arg == "mockex"
        assert feed._timeframe_ms == 300_000
        assert feed._book_limit == 10

    def test_create_feed_propagates_sandbox(self):
        sesh = SeshCCXTLive(_MiniEx(), sandbox=True)
        feed = sesh.create_feed()
        assert feed._sandbox is True

    def test_create_feed_propagates_demo(self):
        sesh = SeshCCXTLive(_MiniEx(), demo=True)
        feed = sesh.create_feed()
        assert feed._demo is True
        assert feed._sandbox is False

    def test_create_feed_default_params(self):
        sesh = SeshCCXTLive(_MiniEx())
        feed = sesh.create_feed()
        assert feed._timeframe_ms == 60_000
        assert feed._book_limit == 20
        assert feed._sandbox is False
        assert feed._demo is False


# =====
# Non-streaming sessions keep the defaults
# =====
class TestDefaults:
    def test_sim_session_not_streaming(self):
        sesh = SeshCCXTSim(feed_type="tick")
        assert sesh.supports_streaming is False
        with pytest.raises(NotImplementedError):
            sesh.create_feed()
