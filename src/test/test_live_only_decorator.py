import pytest
from tradetropy.models import live_only, Strategy

pytestmark = pytest.mark.unit


# ══════════════════════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════════════════════


class _StrategyBase(Strategy):
    def __init__(self, run_mode="backtest"):
        Strategy.__init__(self)
        self._run_mode = run_mode


class _NoRunMode:
    """Class that has NO _run_mode — for TypeError testing."""


# ══════════════════════════════════════════════════════════════════════════════
# TEST: @live_only no args → None in backtest, executes in live
# ══════════════════════════════════════════════════════════════════════════════

def test_no_args_backtest_returns_none():
    class StrategyCls(_StrategyBase):
        @live_only
        def notify(self, msg: str):
            return f"sent: {msg}"

    e = StrategyCls(run_mode="backtest")
    assert e.notify("hello") is None


def test_no_args_live_executes():
    class StrategyCls(_StrategyBase):
        @live_only
        def notify(self, msg: str):
            return f"sent: {msg}"

    e = StrategyCls(run_mode="live")
    assert e.notify("hello") == "sent: hello"


# ══════════════════════════════════════════════════════════════════════════════
# TEST: @live_only(default=X) → returns X in backtest
# ══════════════════════════════════════════════════════════════════════════════

def test_default_value():
    class StrategyCls(_StrategyBase):
        @live_only(default=False)
        def has_connection(self) -> bool:
            return True

    e = StrategyCls(run_mode="backtest")
    assert e.has_connection() is False


def test_default_value_in_live_not_affected():
    class StrategyCls(_StrategyBase):
        @live_only(default=False)
        def has_connection(self) -> bool:
            return True

    e = StrategyCls(run_mode="live")
    assert e.has_connection() is True


# ══════════════════════════════════════════════════════════════════════════════
# TEST: @live_only(default=fn) → calls fn() in backtest
# ══════════════════════════════════════════════════════════════════════════════

def test_default_callable():
    class StrategyCls(_StrategyBase):
        @live_only(default=lambda: [])
        def get_positions(self):
            return ["pos1", "pos2"]

    e = StrategyCls(run_mode="backtest")
    assert e.get_positions() == []


def test_default_callable_always_returns_new_instance():
    class StrategyCls(_StrategyBase):
        @live_only(default=list)
        def get_positions(self):
            return ["pos1"]

    e = StrategyCls(run_mode="backtest")
    r1 = e.get_positions()
    r2 = e.get_positions()
    assert r1 == []
    assert r2 == []
    assert r1 is not r2


# ══════════════════════════════════════════════════════════════════════════════
# TEST: @live_only(raises=True) → NotImplementedError
# ══════════════════════════════════════════════════════════════════════════════

def test_raises_true():
    class StrategyCls(_StrategyBase):
        @live_only(raises=True)
        def execute_order(self, sym: str, vol: int):
            return "ok"

    e = StrategyCls(run_mode="backtest")
    with pytest.raises(NotImplementedError) as exc:
        e.execute_order("BTCUSDT", 1)
    assert "execute_order" in str(exc.value)
    assert "backtest" in str(exc.value)


def test_raises_true_not_affected_in_live():
    class StrategyCls(_StrategyBase):
        @live_only(raises=True)
        def execute_order(self, sym: str, vol: int):
            return "ok"

    e = StrategyCls(run_mode="live")
    assert e.execute_order("BTCUSDT", 1) == "ok"


# ══════════════════════════════════════════════════════════════════════════════
# TEST: @live_only(raises="msg") → NotImplementedError with custom msg
# ══════════════════════════════════════════════════════════════════════════════

def test_raises_msg_custom():
    class StrategyCls(_StrategyBase):
        @live_only(raises="No se puede enviar orden de emergencia en backtest")
        def emergency_order(self, sym: str):
            return "ok"

    e = StrategyCls(run_mode="backtest")
    with pytest.raises(NotImplementedError) as exc:
        e.emergency_order("BTCUSDT")
    assert "No se puede enviar orden de emergencia en backtest" in str(exc.value)


# ══════════════════════════════════════════════════════════════════════════════
# TEST: Introspection — __wrapped__ points to the original
# ══════════════════════════════════════════════════════════════════════════════

def test_wrapped_access_from_class():
    class StrategyCls(_StrategyBase):
        @live_only
        def notify(self, msg):
            return f"sent: {msg}"

    wrapped = StrategyCls.notify.__wrapped__
    assert wrapped is not None
    e = StrategyCls(run_mode="live")
    assert wrapped(e, "msg") == "sent: msg"


def test_wrapped_access_from_instance():
    class StrategyCls(_StrategyBase):
        @live_only
        def notify(self, msg):
            return f"sent: {msg}"

    e = StrategyCls(run_mode="live")
    wrapped = e.notify.__wrapped__
    assert wrapped is not None
    assert wrapped(e, "msg") == "sent: msg"


# ══════════════════════════════════════════════════════════════════════════════
# TEST: functools.wraps — __name__, __doc__ preserved
# ══════════════════════════════════════════════════════════════════════════════

def test_name_preserved():
    class StrategyCls(_StrategyBase):
        @live_only
        def notify_telegram(self, msg: str):
            ...

    assert StrategyCls.notify_telegram.__name__ == "notify_telegram"


def test_doc_preserved():
    class StrategyCls(_StrategyBase):
        @live_only
        def notify_telegram(self, msg: str):
            """Sends a notification via Telegram."""

    assert StrategyCls.notify_telegram.__doc__ == "Sends a notification via Telegram."


def test_name_preserved_from_instance():
    class StrategyCls(_StrategyBase):
        @live_only
        def notify_telegram(self, msg: str):
            ...

    e = StrategyCls(run_mode="live")
    assert e.notify_telegram.__name__ == "notify_telegram"


# ══════════════════════════════════════════════════════════════════════════════
# TEST: Usage outside Strategy → clear TypeError
# ══════════════════════════════════════════════════════════════════════════════

def test_typeerror_without_run_mode():
    class StrategyCls(_StrategyBase):
        @live_only
        def notify(self, msg):
            ...

    obj = _NoRunMode()
    # Bind the descriptor manually to simulate incorrect usage
    descriptor = StrategyCls.notify
    bound = descriptor.__get__(obj)
    with pytest.raises(TypeError) as exc:
        bound("hello")
    assert "has no '_run_mode' attribute" in str(exc.value)


# ══════════════════════════════════════════════════════════════════════════════
# TEST: optimize/pool → returns default silently
# ══════════════════════════════════════════════════════════════════════════════

@pytest.mark.parametrize("run_mode", ["optimize", "pool"])
def test_optimize_pool_returns_default(run_mode):
    class StrategyCls(_StrategyBase):
        @live_only(default=False)
        def has_connection(self):
            return True

    e = StrategyCls(run_mode=run_mode)
    assert e.has_connection() is False


@pytest.mark.parametrize("run_mode", ["optimize", "pool"])
def test_optimize_pool_without_default_is_none(run_mode):
    class StrategyCls(_StrategyBase):
        @live_only
        def notify(self, msg):
            return "ok"

    e = StrategyCls(run_mode=run_mode)
    assert e.notify("msg") is None


# ══════════════════════════════════════════════════════════════════════════════
# TEST: @live_only(default=None) explicit
# ══════════════════════════════════════════════════════════════════════════════

def test_default_none_explicit():
    class StrategyCls(_StrategyBase):
        @live_only(default=None)
        def algo(self):
            return "value"

    e = StrategyCls(run_mode="backtest")
    assert e.algo() is None


# ══════════════════════════════════════════════════════════════════════════════
# TEST: positional and keyword arguments are passed correctly
# ══════════════════════════════════════════════════════════════════════════════

def test_positional_arguments():
    class StrategyCls(_StrategyBase):
        @live_only
        def suma(self, a, b):
            return a + b

    e = StrategyCls(run_mode="live")
    assert e.suma(2, 3) == 5


def test_keyword_arguments():
    class StrategyCls(_StrategyBase):
        @live_only
        def greet(self, *, name):
            return f"hello {name}"

    e = StrategyCls(run_mode="live")
    assert e.greet(name="world") == "hello world"
