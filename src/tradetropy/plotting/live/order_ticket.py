"""
order_ticket.py
===============
UI Order Ticket panel for the manual PaperEngine.

Renders an explicit two-column Buy / Sell ticket (Market / Limit / Stop / Stop
Limit, with Size and TP/SL brackets). Every action is routed to the engine
through ``engine.submit_command`` so the order executes on the single engine
thread (never mutating broker state from the Bokeh IOLoop), consistent with the
framework's single-writer model.

    from tradetropy.plotting.live.order_ticket import build_order_ticket
    panel = build_order_ticket(engine.submit_command, symbol="BTCUSDT")
"""

from __future__ import annotations

from typing import Callable


# Order-type label -> routing token consumed by BaseEngine.place_order().
_TYPE_TOKENS = {
    "Market":     "market",
    "Limit":      "limit",
    "Stop":       "stop",
    "Stop Limit": "stop_limit",
}


def _panel_css() -> str:
    return """
:host {
    background: rgba(182, 192, 215, 0.85) !important;
    border: 1px solid rgba(255,255,255,0.08) !important;
    border-radius: 8px !important;
    padding: 8px !important;
    box-sizing: border-box !important;
}
"""


def _action_btn_css(bg: str, hover: str) -> str:
    return f"""
:host .bk-btn {{
    background: {bg} !important;
    color: #fff !important;
    border: none !important;
    border-radius: 5px !important;
    font-weight: 700 !important;
    font-size: 12px !important;
    cursor: pointer !important;
}}
:host(:hover) .bk-btn {{ background: {hover} !important; }}
"""


def _build_side(side: str, submit: Callable, symbol: str,
                price_step: float = 0.01, vol_step: float = 0.01):
    """
    Build one Buy or Sell column of the ticket.

    Args:
        side (str): 'buy' or 'sell'.
        submit (Callable): engine.submit_command (enqueues a callable(engine)).
        symbol (str): Trading symbol for the routed orders.
        price_step (float): Minimum price increment (tick_size) for price spinners.
        vol_step (float): Minimum volume increment (volume_step) for size spinner.

    Returns:
        tuple: (column_layout, widgets_dict) - widgets_dict exposes the inner
            widgets and the on-click handler for testing.
    """
    from bokeh.models import Select, Spinner, Button, Div, InlineStyleSheet
    from bokeh.layouts import column as _bk_column

    is_buy = side == "buy"
    title_color = "#3B82F6" if is_buy else "#EF5350"
    btn_bg = "#2563EB" if is_buy else "#DC2626"
    btn_hover = "#1D4ED8" if is_buy else "#B91C1C"
    label = "BUY" if is_buy else "SELL"

    header = Div(
        text=f"<b style='color:{title_color}'>{label}</b>",
        width=120, height=18,
    )
    type_sel = Select(
        title="Type", value="Market",
        options=list(_TYPE_TOKENS.keys()), width=120,
    )
    size_in = Spinner(title="Size", low=0.0, step=vol_step, value=vol_step, width=120)
    price_in = Spinner(title="Price", value=None, step=price_step, width=120, visible=False)
    limit_in = Spinner(title="Limit Price", value=None, step=price_step, width=120, visible=False)
    tp_in = Spinner(title="Take Profit", value=None, step=price_step, width=120)
    sl_in = Spinner(title="Stop Loss", value=None, step=price_step, width=120)

    action_btn = Button(
        label=label, width=120, height=30, button_type="light",
        stylesheets=[InlineStyleSheet(css=_action_btn_css(btn_bg, btn_hover))],
    )

    def _on_type_change(attr, old, new):
        # Price field shown for any non-market order; limit field only for
        # Stop Limit (the limit price placed once the stop triggers).
        token = _TYPE_TOKENS.get(new, "market")
        price_in.visible = token != "market"
        limit_in.visible = token == "stop_limit"

    type_sel.on_change("value", _on_type_change)

    def _on_submit(_=None):
        token = _TYPE_TOKENS.get(type_sel.value, "market")
        volume = float(size_in.value or 0.0)
        price = price_in.value if token != "market" else None
        limit_price = limit_in.value if token == "stop_limit" else None
        sl = float(sl_in.value) if sl_in.value else 0.0
        tp = float(tp_in.value) if tp_in.value else 0.0

        submit(lambda eng: eng.place_order(
            side=side,
            order_type=token,
            volume=volume,
            price=price,
            limit_price=limit_price,
            sl=sl,
            tp=tp,
            symbol=symbol,
            comment="ticket",
        ))

    action_btn.on_click(_on_submit)

    col = _bk_column(
        header, type_sel, size_in, price_in, limit_in, tp_in, sl_in, action_btn,
        width=140,
        styles={"padding": "4px", "gap": "4px"},
    )
    widgets = dict(
        type=type_sel, size=size_in, price=price_in, limit=limit_in,
        tp=tp_in, sl=sl_in, button=action_btn,
        on_submit=_on_submit, on_type_change=_on_type_change,
    )
    return col, widgets


def build_order_ticket(submit: Callable, symbol: str, doc=None,
                       price_step: float = 0.01, vol_step: float = 0.01):
    """
    Build the two-column Buy / Sell Order Ticket panel.

    Args:
        submit (Callable): engine.submit_command - enqueues callable(engine) to
            run on the engine thread (thread-safe order routing).
        symbol (str): Trading symbol the ticket places orders for.
        doc: Bokeh Document (unused; reserved for future scheduling needs).
        price_step (float): Minimum price increment (tick_size) for price spinners.
        vol_step (float): Minimum volume increment (volume_step) for size spinner.

    Returns:
        tuple: (panel_layout, sides) where sides = {'buy': widgets, 'sell':
            widgets} exposing the inner widgets/handlers (useful for tests).
    """
    from bokeh.models import Div, InlineStyleSheet
    from bokeh.layouts import column as _bk_column, row as _bk_row

    title = Div(
        text="<b style='color:#000000'>Order Ticket</b>",
        width=260, height=20,
    )
    buy_col, buy_w = _build_side("buy", submit, symbol, price_step, vol_step)
    sell_col, sell_w = _build_side("sell", submit, symbol, price_step, vol_step)
    cols = _bk_row(buy_col, sell_col, styles={"gap": "8px"})

    panel = _bk_column(
        title, cols,
        width=300,
        stylesheets=[InlineStyleSheet(css=_panel_css())],
        styles={"gap": "6px"},
    )
    return panel, {"buy": buy_w, "sell": sell_w}


# =============================================================================
# LIVE POSITION MODIFIER
# =============================================================================

def build_position_modifier(submit: Callable, symbol: str, fetch_position: Callable,
                            price_step: float = 0.01):
    """
    Build the Live Position Modifier widget.

    Shows the current open position (average price + size), lets the user attach
    or adjust TP/SL on the fly, and close the position partially or fully. All
    actions route through ``submit`` (engine.submit_command) to the engine
    thread. ``refresh()`` (returned in the widgets dict) is called by the chart
    updater each tick to reflect the live position state.

    Args:
        submit (Callable): engine.submit_command.
        symbol (str): Trading symbol.
        fetch_position (Callable): Returns the current net position as a dict
            {'ticket', 'avg_price', 'size', 'side'} or None when flat. Read on
            the IOLoop, so it must be cheap and non-mutating.
        price_step (float): Minimum price increment (tick_size) for TP/SL spinners.

    Returns:
        tuple: (panel_layout, widgets_dict). widgets_dict exposes the inner
            widgets plus 'refresh', 'on_set', 'on_close_half', 'on_close_all'.
    """
    from bokeh.models import Spinner, Button, Div, InlineStyleSheet
    from bokeh.layouts import column as _bk_column, row as _bk_row

    title = Div(text="<b style='color:#000000'>Position</b>", width=260, height=20)
    info = Div(text="<span style='color:#5F6A79'>Flat - no open position</span>",
               width=260, height=20)
    tp_in = Spinner(title="Take Profit", value=None, step=price_step, width=120)
    sl_in = Spinner(title="Stop Loss", value=None, step=price_step, width=120)

    btn_set = Button(label="Set TP/SL", width=120, height=26, button_type="light",
                     stylesheets=[InlineStyleSheet(css=_action_btn_css("#2563EB", "#1D4ED8"))])
    btn_half = Button(label="Close 50%", width=120, height=26, button_type="light",
                      stylesheets=[InlineStyleSheet(css=_action_btn_css("#D97706", "#B45309"))])
    btn_all = Button(label="Close All", width=120, height=26, button_type="light",
                     stylesheets=[InlineStyleSheet(css=_action_btn_css("#DC2626", "#B91C1C"))])

    # Last seen position snapshot (read by the close/modify handlers).
    state = {"pos": None}

    def refresh():
        """Update the panel from the current position; called each tick."""
        try:
            pos = fetch_position()
        except Exception:
            pos = None
        state["pos"] = pos
        if not pos:
            info.text = "<span style='color:#5F6A79'>Flat - no open position</span>"
            btn_set.disabled = True
            btn_half.disabled = True
            btn_all.disabled = True
            return
        side = str(pos.get("side", "")).upper()
        color = "#3B82F6" if side == "BUY" else "#EF5350"
        avg = float(pos.get("avg_price", 0.0))
        size = float(pos.get("size", 0.0))
        info.text = (
            f"<span style='color:{color}'><b>{side}</b></span> "
            f"<span style='color:#E2E8F0'>Size {size:g} @ Avg {avg:.2f}</span>"
        )
        btn_set.disabled = False
        btn_half.disabled = False
        btn_all.disabled = False

    def _ticket():
        pos = state["pos"]
        return int(pos["ticket"]) if pos and pos.get("ticket") is not None else None

    def _on_set(_=None):
        ticket = _ticket()
        if ticket is None:
            return
        sl = float(sl_in.value) if sl_in.value else 0.0
        tp = float(tp_in.value) if tp_in.value else 0.0
        submit(lambda eng: eng.modify_position(ticket, sl=sl, tp=tp))

    def _on_close_half(_=None):
        pos = state["pos"]
        if not pos:
            return
        ticket = int(pos["ticket"])
        half = abs(float(pos.get("size", 0.0))) / 2.0
        if half <= 0:
            return
        submit(lambda eng: eng.close_position(ticket, half))

    def _on_close_all(_=None):
        ticket = _ticket()
        if ticket is None:
            return
        submit(lambda eng: eng.close_position(ticket, None))

    btn_set.on_click(_on_set)
    btn_half.on_click(_on_close_half)
    btn_all.on_click(_on_close_all)

    refresh()

    panel = _bk_column(
        title, info, _bk_row(tp_in, sl_in, styles={"gap": "8px"}),
        btn_set, _bk_row(btn_half, btn_all, styles={"gap": "8px"}),
        width=300,
        stylesheets=[InlineStyleSheet(css=_panel_css())],
        styles={"gap": "6px"},
    )
    widgets = dict(
        info=info, tp=tp_in, sl=sl_in,
        btn_set=btn_set, btn_half=btn_half, btn_all=btn_all,
        refresh=refresh, on_set=_on_set,
        on_close_half=_on_close_half, on_close_all=_on_close_all,
    )
    return panel, widgets


# =============================================================================
# PENDING ORDERS PANEL
# =============================================================================

def build_orders_panel(submit: Callable, symbol: str, fetch_orders: Callable,
                       cancel_order: Callable):
    """
    Build the Pending Orders panel showing active limit/stop orders.

    Each order is displayed with its type, side, price, volume, and a cancel
    button. The panel refreshes on each tick via the returned ``refresh()``.

    Args:
        submit (Callable): engine.submit_command.
        symbol (str): Trading symbol.
        fetch_orders (Callable): Returns a list of pending Order objects from the
            broker. Read on the IOLoop, must be cheap and non-mutating.
        cancel_order (Callable): engine.cancel_order — enqueues order deletion.

    Returns:
        tuple: (panel_layout, widgets_dict). widgets_dict exposes 'refresh'.
    """
    from bokeh.models import Div, InlineStyleSheet
    from bokeh.layouts import column as _bk_column

    title = Div(
        text="<b style='color:#000000'>Pending Orders</b>",
        width=260, height=20,
    )
    container = Div(
        text="<span style='color:#5F6A79'>No pending orders</span>",
        width=260, height=20,
    )

    state = {"orders": []}

    def refresh():
        """Update the panel from the current pending orders; called each tick."""
        try:
            orders = fetch_orders()
        except Exception:
            orders = []
        from tradetropy.core.broker import OrderType, OrderState
        pending = [
            o for o in orders
            if getattr(o, "state", None) == OrderState.ORDER_STATE_PLACED
        ]
        state["orders"] = pending
        if not pending:
            container.text = (
                "<span style='color:#5F6A79'>No pending orders</span>"
            )
            return
        rows = []
        for o in pending:
            ticket = getattr(o, "ticket", 0)
            otype = getattr(o, "type", None)
            price = float(getattr(o, "price", 0) or 0)
            vol = float(getattr(o, "volume", 0) or 0)
            is_buy = otype in (
                OrderType.ORDER_TYPE_BUY_LIMIT,
                OrderType.ORDER_TYPE_BUY_STOP,
                OrderType.ORDER_TYPE_BUY_STOP_LIMIT,
            ) if otype is not None else True
            side = "B" if is_buy else "S"
            side_color = "#3B82F6" if is_buy else "#EF5350"
            type_label = {
                OrderType.ORDER_TYPE_BUY_LIMIT: "LMT",
                OrderType.ORDER_TYPE_SELL_LIMIT: "LMT",
                OrderType.ORDER_TYPE_BUY_STOP: "STP",
                OrderType.ORDER_TYPE_SELL_STOP: "STP",
                OrderType.ORDER_TYPE_BUY_STOP_LIMIT: "STP LMT",
                OrderType.ORDER_TYPE_SELL_STOP_LIMIT: "STP LMT",
            }.get(otype, "ORD")
            rows.append(
                f"<div style='display:flex;align-items:center;gap:4px;"
                f"margin:2px 0;font-size:11px;'>"
                f"<span style='color:{side_color};font-weight:700;"
                f"width:14px;'>{side}</span>"
                f"<span style='color:#94A3B8;width:36px;'>{type_label}</span>"
                f"<span style='color:#E2E8F0;flex:1;'>{vol:g} @ {price:g}</span>"
                f"<span class='cancel-btn' data-ticket=\"{ticket}\" "
                f"style='color:#EF5350;cursor:pointer;font-weight:700;"
                f"padding:0 4px;' title='Cancel'>&#10005;</span>"
                f"</div>"
            )
        container.text = "".join(rows)

    refresh()

    panel = _bk_column(
        title, container,
        width=260,
        stylesheets=[InlineStyleSheet(css=_panel_css())],
        styles={"gap": "4px"},
    )
    widgets = dict(refresh=refresh)
    return panel, widgets
