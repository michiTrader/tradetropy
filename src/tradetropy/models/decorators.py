from __future__ import annotations

import functools
from typing import Callable


__all__ = ['live_only']


_SENTINEL = object()


class _LiveOnlyDescriptor:
    """
    Descriptor that inspects Strategy._run_mode at call time.

    Enforces that decorated methods only execute in live trading mode.
    In other modes (backtest, optimize, pool), returns a default value,
    calls a default function, or raises NotImplementedError based on
    configuration.

    Args:
        fn: Decorated function/method
        default: Value to return if not in live mode (default None)
        raises: True to raise NotImplementedError; str to use custom message
    """

    def __init__(self, fn: Callable, default: object, raises: bool | str):
        self.__wrapped__ = fn
        self._default = default
        self._raises = raises
        self.__name__ = fn.__name__
        self.__qualname__ = fn.__qualname__
        self.__doc__ = fn.__doc__
        self.__module__ = fn.__module__
        self.__annotations__ = getattr(fn, '__annotations__', {})

    def __get__(self, obj, objtype=None):
        if obj is None:
            return self

        def wrapper(*args, **kwargs):
            return self._run(obj, *args, **kwargs)

        wrapper.__wrapped__ = self.__wrapped__
        functools.update_wrapper(wrapper, self.__wrapped__)
        return wrapper

    def __repr__(self) -> str:
        return f'<live_only {self.__qualname__}>'

    def _run(self, obj, *args, **kwargs):
        """
        Execute wrapped function or return default based on run mode.

        Args:
            obj: Strategy instance
            *args: Function arguments
            **kwargs: Function keyword arguments

        Returns:
            Function result if in live mode; default value otherwise

        Raises:
            TypeError: If _run_mode attribute not found
            NotImplementedError: If raises is True and not in live mode
        """
        run_mode = getattr(obj, '_run_mode', None)
        if run_mode is None:
            raise TypeError(
                '@live_only only works on Strategy methods. '
                f"'{type(obj).__name__}' has no '_run_mode' attribute."
            )

        if run_mode == 'live':
            return self.__wrapped__(obj, *args, **kwargs)

        if self._raises:
            msg = (
                self._raises
                if isinstance(self._raises, str)
                else f'{self.__name__} not supported in mode {run_mode!r}'
            )
            raise NotImplementedError(msg)

        default = self._default
        if default is _SENTINEL:
            return None
        if callable(default):
            return default()
        return default


def live_only(fn=None, *, default=_SENTINEL, raises=False):
    """
    Decorator restricting method execution to live trading mode.

    In backtesting, pool, and optimize modes, the decorated method
    either returns a default value, calls a default function, or
    raises NotImplementedError.

    Args:
        fn: Function to decorate (when used as @live_only)
        default: Value/callable to return in non-live modes
        raises: True to raise NotImplementedError; string for custom message

    Example:
        class Strategy(Strategy):
            @live_only(default=None)
            def cancel_all_orders(self):
                '''Cancel orders - no-op in backtest.'''
                exchange.cancel_all()

            @live_only(raises='Order cancellation not available in backtest')
            def urgent_cancel(self):
                exchange.cancel_urgent()
    """
    if fn is not None:
        return _LiveOnlyDescriptor(fn, _SENTINEL, False)

    def decorator(fn):
        return _LiveOnlyDescriptor(fn, default, raises)

    return decorator
