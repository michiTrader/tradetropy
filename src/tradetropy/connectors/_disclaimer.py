"""
_disclaimer.py
==============
Risk disclaimer shared by ALL live trading sessions
(SeshMT5Live, SeshBybitLive, SeshCCXTLive, …).

Internal use:
    from tradetropy.connectors._disclaimer import emit_live_disclaimer
    emit_live_disclaimer()   # logs a one-line notice ONCE per process

The full text is also exported as `tradetropy.LIVE_DISCLAIMER` so users can
display it in their own applications or documentation.

Set the environment variable TRADETROPY_SUPPRESS_DISCLAIMER=1 to opt out of
the runtime notice entirely (the full text remains available via
`tradetropy.LIVE_DISCLAIMER` regardless).
"""

from __future__ import annotations

import logging
import os

_logger = logging.getLogger("tradetropy.connectors")

#: Risk disclaimer shown when starting any live trading session.
LIVE_DISCLAIMER = """\
⚠️  RISK DISCLAIMER — LIVE TRADING
────────────────────────────────────────────────────────────────────
1. Software is provided "AS IS", WITHOUT WARRANTY of any kind (either
   express or implied). Use is at YOUR OWN RISK.
2. Trading financial instruments involves RISK OF LOSS of capital. You
   may lose part or all of your funds.
3. ALWAYS TEST FIRST on a DEMO / TESTNET / PAPER account and verify
   that orders, positions, and balances behave as expected BEFORE
   trading with real money.
4. Broker/exchange APIs MAY CHANGE WITHOUT NOTICE. An exchange update
   may break this connector or alter its behavior. Verify functionality
   periodically.
5. Authors and contributors are NOT RESPONSIBLE for losses, damages,
   execution failures, latency, disconnections, or errors arising from
   the use of this software.
6. This is NOT financial, investment, or legal advice.
7. You are responsible for complying with the laws and regulations of
   your jurisdiction and the terms of service of your broker/exchange.
────────────────────────────────────────────────────────────────────"""

#: One-line notice logged at the start of a live session. Kept short on
#: purpose to avoid cluttering the console; the full text lives in
#: LIVE_DISCLAIMER above.
_SHORT_NOTICE = (
    "RISK DISCLAIMER - LIVE TRADING: use at your own risk, test on "
    "DEMO/TESTNET first. Full text: tradetropy.LIVE_DISCLAIMER. Set "
    "TRADETROPY_SUPPRESS_DISCLAIMER=1 to silence this notice."
)

# Process flag: the notice is logged once to avoid spam.
_DISCLAIMER_EMITTED = False


def emit_live_disclaimer(force: bool = False) -> None:
    """
    Log a one-line risk notice once per process at WARNING level.

    The full disclaimer text is available at `tradetropy.LIVE_DISCLAIMER` and
    is not printed to the console by default to keep it clean; only this
    short notice is logged, pointing to the full text.

    Set the environment variable TRADETROPY_SUPPRESS_DISCLAIMER=1 to skip the
    notice entirely.

    Args:
        force: If True, logs even if already shown or suppressed (useful
            in tests).

    Returns:
        None
    """
    global _DISCLAIMER_EMITTED
    if _DISCLAIMER_EMITTED and not force:
        return
    if os.environ.get("TRADETROPY_SUPPRESS_DISCLAIMER") == "1" and not force:
        _DISCLAIMER_EMITTED = True
        return
    _DISCLAIMER_EMITTED = True
    _logger.warning(_SHORT_NOTICE)


def _reset_disclaimer_flag() -> None:
    """
    Reset the disclaimer flag (tests only).

    Returns:
        None
    """
    global _DISCLAIMER_EMITTED
    _DISCLAIMER_EMITTED = False
