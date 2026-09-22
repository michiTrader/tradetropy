import numpy as np


def _deals_to_trades(deals: list) -> list:
    from tradetropy.core.broker import OrderType as _OT
    from datetime import timezone

    by_pos: dict = {}
    for d in deals:
        pid = getattr(d, "position_id", None) or getattr(d, "ticket", 0)
        by_pos.setdefault(pid, []).append(d)

    trades = []
    for pid, pos_deals in by_pos.items():
        pos_deals_sorted = sorted(pos_deals, key=lambda d: _deal_ts_ms(d))

        if len(pos_deals_sorted) < 2:
            continue

        entry_deal = pos_deals_sorted[0]
        entry_ts_ms = _deal_ts_ms(entry_deal)
        entry_price = float(getattr(entry_deal, "price", 0.0) or 0.0)
        size = abs(float(getattr(entry_deal, "volume", 0.0) or 0.0))

        entry_type = getattr(entry_deal, "type", None)
        if entry_type is not None:
            try:
                direction = "buy" if entry_type == _OT.ORDER_TYPE_BUY else "sell"
            except Exception:
                direction = "buy"
        else:
            direction = "buy"

        for exit_deal in pos_deals_sorted[1:]:
            exit_ts_ms = _deal_ts_ms(exit_deal)
            exit_price = float(getattr(exit_deal, "price", 0.0) or 0.0)
            pnl        = float(getattr(exit_deal, "profit", 0.0) or 0.0)

            trades.append({
                "entry_ts_ms":  entry_ts_ms,
                "exit_ts_ms":   exit_ts_ms,
                "entry_price":  entry_price,
                "exit_price":   exit_price,
                "pnl":          pnl,
                "direction":    direction,
                "size":         size,
            })

    return trades


def _deal_ts_ms(deal) -> int:
    t = getattr(deal, "time", None)
    if t is None:
        return 0
    try:
        from datetime import timezone
        import datetime as _dt
        if isinstance(t, (int, float)):
            return int(t * 1000) if t < 1e12 else int(t)
        if isinstance(t, _dt.datetime):
            return int(t.timestamp() * 1000)
    except Exception:
        pass
    return 0


def _trades_df_to_list(trades_df) -> list:
    import pandas as _pd

    tc = trades_df[trades_df["exit_time"].notna()].copy()
    if tc.empty:
        return []

    def _ts_ms(series):
        return (
            _pd.to_datetime(series, utc=True)
            .dt.tz_localize(None)
            .astype("datetime64[ms]")
            .astype("int64")
        )

    entry_ms = _ts_ms(tc["entry_time"])
    exit_ms  = _ts_ms(tc["exit_time"])

    result = []
    for i in range(len(tc)):
        result.append({
            "entry_ts_ms":  int(entry_ms.iloc[i]),
            "exit_ts_ms":   int(exit_ms.iloc[i]),
            "entry_price":  float(tc["entry_price"].iloc[i]),
            "exit_price":   float(tc["exit_price"].iloc[i]),
            "pnl":          float(tc["pnl_net"].iloc[i]),
            "direction":    str(tc["direction"].iloc[i]).lower(),
            "size":         abs(float(tc["size"].iloc[i])) if "size" in tc.columns else 0.0,
        })
    return result


def _empty_markers_source():
    from bokeh.models import ColumnDataSource
    return ColumnDataSource(dict(
        ts          = np.array([], dtype="datetime64[ms]"),
        price       = np.array([], dtype=np.float64),
        color       = np.array([], dtype=object),
        marker      = np.array([], dtype=object),
        angle       = np.array([], dtype=np.float64),
        tooltip_ts  = np.array([], dtype="datetime64[ms]"),
        tooltip_price = np.array([], dtype=np.float64),
        tooltip_pnl = np.array([], dtype=np.float64),
        tooltip_dir = np.array([], dtype=object),
        tooltip_tag = np.array([], dtype=object),
    ))


def _build_open_markers_data_live(positions: list, theme: dict) -> dict:
    import math

    open_color = theme.get("text", "#E6EDF3")

    empty = dict(
        ts            = np.array([], dtype="datetime64[ms]"),
        price         = np.array([], dtype=np.float64),
        color         = np.array([], dtype=object),
        marker        = np.array([], dtype=object),
        angle         = np.array([], dtype=np.float64),
        tooltip_ts    = np.array([], dtype="datetime64[ms]"),
        tooltip_price = np.array([], dtype=np.float64),
        tooltip_dir   = np.array([], dtype=object),
        tooltip_tag   = np.array([], dtype=object),
    )

    if not positions:
        return empty

    PI = math.pi
    ts_list, price_list, color_list, marker_list, angle_list = [], [], [], [], []
    ttag_list, tprice_list, tdir_list, ttag2_list = [], [], [], []

    for pos in positions:
        is_buy = pos["direction"] == "buy"
        angle  = 0.0 if is_buy else PI
        ts_ms  = pos["entry_ts_ms"]

        ts_list.append(np.datetime64(ts_ms, "ms"))
        price_list.append(pos["entry_price"])
        color_list.append(open_color)
        marker_list.append("triangle")
        angle_list.append(angle)
        ttag_list.append(np.datetime64(ts_ms, "ms"))
        tprice_list.append(pos["entry_price"])
        tdir_list.append(pos["direction"])
        ttag2_list.append("Abierta")

    return dict(
        ts            = np.array(ts_list,    dtype="datetime64[ms]"),
        price         = np.array(price_list, dtype=np.float64),
        color         = np.array(color_list, dtype=object),
        marker        = np.array(marker_list, dtype=object),
        angle         = np.array(angle_list, dtype=np.float64),
        tooltip_ts    = np.array(ttag_list,   dtype="datetime64[ms]"),
        tooltip_price = np.array(tprice_list, dtype=np.float64),
        tooltip_dir   = np.array(tdir_list,   dtype=object),
        tooltip_tag   = np.array(ttag2_list,  dtype=object),
    )


def _build_markers_data_live(trades: list, theme: dict) -> dict:
    import math

    win_color  = theme.get("trade_win",  "#007F5F")
    loss_color = theme.get("trade_loss", "#AD1D2B")

    if not trades:
        return dict(
            ts            = np.array([], dtype="datetime64[ms]"),
            price         = np.array([], dtype=np.float64),
            color         = np.array([], dtype=object),
            marker        = np.array([], dtype=object),
            angle         = np.array([], dtype=np.float64),
            tooltip_ts    = np.array([], dtype="datetime64[ms]"),
            tooltip_price = np.array([], dtype=np.float64),
            tooltip_pnl   = np.array([], dtype=np.float64),
            tooltip_dir   = np.array([], dtype=object),
            tooltip_tag   = np.array([], dtype=object),
        )

    ts_list     = []
    price_list  = []
    color_list  = []
    marker_list = []
    angle_list  = []
    ttag_list   = []
    tprice_list = []
    tpnl_list   = []
    tdir_list   = []
    ttag2_list  = []

    PI = math.pi

    for t in trades:
        color = win_color if t["pnl"] > 0 else loss_color
        is_buy = t["direction"] == "buy"

        entry_angle = 0.0 if is_buy else PI
        ts_list.append(np.datetime64(t["entry_ts_ms"], "ms"))
        price_list.append(t["entry_price"])
        color_list.append(color)
        marker_list.append("triangle")
        angle_list.append(entry_angle)
        ttag_list.append(np.datetime64(t["entry_ts_ms"], "ms"))
        tprice_list.append(t["entry_price"])
        tpnl_list.append(t["pnl"])
        tdir_list.append(t["direction"])
        ttag2_list.append("Entrada")

        exit_angle = PI if is_buy else 0.0
        ts_list.append(np.datetime64(t["exit_ts_ms"], "ms"))
        price_list.append(t["exit_price"])
        color_list.append(color)
        marker_list.append("triangle")
        angle_list.append(exit_angle)
        ttag_list.append(np.datetime64(t["exit_ts_ms"], "ms"))
        tprice_list.append(t["exit_price"])
        tpnl_list.append(t["pnl"])
        tdir_list.append(t["direction"])
        ttag2_list.append("Salida")

    return dict(
        ts            = np.array(ts_list,    dtype="datetime64[ms]"),
        price         = np.array(price_list, dtype=np.float64),
        color         = np.array(color_list, dtype=object),
        marker        = np.array(marker_list, dtype=object),
        angle         = np.array(angle_list, dtype=np.float64),
        tooltip_ts    = np.array(ttag_list,   dtype="datetime64[ms]"),
        tooltip_price = np.array(tprice_list, dtype=np.float64),
        tooltip_pnl   = np.array(tpnl_list,   dtype=np.float64),
        tooltip_dir   = np.array(tdir_list,   dtype=object),
        tooltip_tag   = np.array(ttag2_list,  dtype=object),
    )


def _build_trades_data_live(
    trades: list,
    theme: dict,
    interval_ms: int,
) -> dict:
    win_color  = theme.get("trade_win",  "#007F5F")
    loss_color = theme.get("trade_loss", "#AD1D2B")

    # The connecting lines below build Python list-of-lists (MultiLine xs/ys),
    # which Bokeh serializes to the websocket as plain JSON - a non-finite entry
    # or exit price would raise "nan not JSON compliant" on push. Drop any trade
    # with a non-finite price so the overlay never crashes the live update.
    trades = [
        t for t in trades
        if np.isfinite(t.get("entry_price", np.nan))
        and np.isfinite(t.get("exit_price", np.nan))
    ]

    if not trades:
        return dict(
            lines_xs    = [],
            lines_ys    = [],
            trade_color = np.array([], dtype=object),
            entry_ts    = np.array([], dtype="datetime64[ms]"),
            exit_ts     = np.array([], dtype="datetime64[ms]"),
            entry_price = np.array([], dtype=np.float64),
            exit_price  = np.array([], dtype=np.float64),
            pnl         = np.array([], dtype=np.float64),
            direction   = np.array([], dtype=object),
            size        = np.array([], dtype=np.float64),
        )

    n = len(trades)
    entry_ts_arr    = np.array([t["entry_ts_ms"] for t in trades], dtype=np.int64)
    exit_ts_arr     = np.array([t["exit_ts_ms"]  for t in trades], dtype=np.int64)
    entry_price_arr = np.array([t["entry_price"] for t in trades], dtype=np.float64)
    exit_price_arr  = np.array([t["exit_price"]  for t in trades], dtype=np.float64)
    pnl_arr         = np.array([t["pnl"]         for t in trades], dtype=np.float64)
    direction_arr   = np.array([t["direction"]   for t in trades], dtype=object)
    size_arr        = np.array([t.get("size", 0.0) for t in trades], dtype=np.float64)

    lines_xs = [[int(e), int(x)] for e, x in zip(entry_ts_arr, exit_ts_arr)]
    lines_ys = [[float(ep), float(xp)]
                for ep, xp in zip(entry_price_arr, exit_price_arr)]

    colors = np.array(
        [win_color if p > 0 else loss_color for p in pnl_arr],
        dtype=object,
    )

    return dict(
        lines_xs    = lines_xs,
        lines_ys    = lines_ys,
        trade_color = colors,
        entry_ts    = entry_ts_arr.astype("datetime64[ms]"),
        exit_ts     = exit_ts_arr.astype("datetime64[ms]"),
        entry_price = entry_price_arr,
        exit_price  = exit_price_arr,
        pnl         = pnl_arr,
        direction   = direction_arr,
        size        = size_arr,
    )


def _empty_tpsl_data() -> dict:
    """Empty data dict for the TP/SL horizontal-line source (flat / no levels)."""
    return dict(
        y     = np.array([], dtype=np.float64),
        color = np.array([], dtype=object),
        text  = np.array([], dtype=object),
    )


def _build_tpsl_data_live(positions: list, theme: dict) -> dict:
    """
    Build the data-driven TP/SL horizontal-line source from open positions.

    Emits one row per non-zero stop-loss / take-profit level found across the
    open positions: SL rows are colored with the loss color, TP rows with the
    win color, and each carries a short text label ('SL <price>' / 'TP
    <price>'). Returns an empty source when there are no levels (flat or no
    brackets set).

    Args:
        positions (list): Open-position dicts (each may carry 'sl' / 'tp').
        theme (dict): Active plot theme (provides the win/loss colors).

    Returns:
        dict: Columns 'y' (level price), 'color' (line/text color) and 'text'
            (label) ready to assign to the TP/SL ColumnDataSource.
    """
    tp_color = theme.get("trade_win",  "#007F5F")
    sl_color = theme.get("trade_loss", "#AD1D2B")

    ys, colors, texts = [], [], []
    for pos in positions:
        sl = float(pos.get("sl", 0.0) or 0.0)
        tp = float(pos.get("tp", 0.0) or 0.0)
        if sl > 0:
            ys.append(sl)
            colors.append(sl_color)
            texts.append(f"SL {sl:.2f}")
        if tp > 0:
            ys.append(tp)
            colors.append(tp_color)
            texts.append(f"TP {tp:.2f}")

    if not ys:
        return _empty_tpsl_data()

    return dict(
        y     = np.array(ys,     dtype=np.float64),
        color = np.array(colors, dtype=object),
        text  = np.array(texts,  dtype=object),
    )


def _empty_pending_orders_data() -> dict:
    """Empty data dict for the pending-orders horizontal-line source."""
    return dict(
        y     = np.array([], dtype=np.float64),
        color = np.array([], dtype=object),
        text  = np.array([], dtype=object),
    )


def _build_pending_orders_data_live(orders: list, theme: dict) -> dict:
    """
    Build horizontal-line data for pending limit/stop orders.

    Each pending order emits one row with its price level, a distinguishing
    color, and a label like 'LMT BUY 1.0 @ 105.50'.

    Args:
        orders (list): Pending Order dataclass instances from broker.get_orders().
        theme (dict): Active plot theme.

    Returns:
        dict: Columns 'y', 'color', 'text' for the pending-orders source.
    """
    order_color = theme.get("pending_order", "#F59E0B")

    from tradetropy.core.broker import OrderType

    ys, colors, texts = [], [], []
    for o in orders:
        price = float(getattr(o, "price", 0.0) or 0.0)
        if price <= 0:
            continue
        vol = float(getattr(o, "volume", 0.0) or 0.0)
        otype = getattr(o, "type", None)
        is_buy = otype in (
            OrderType.ORDER_TYPE_BUY_LIMIT,
            OrderType.ORDER_TYPE_BUY_STOP,
            OrderType.ORDER_TYPE_BUY_STOP_LIMIT,
        ) if otype is not None else True
        side = "BUY" if is_buy else "SELL"
        type_label = {
            OrderType.ORDER_TYPE_BUY_LIMIT: "LMT",
            OrderType.ORDER_TYPE_SELL_LIMIT: "LMT",
            OrderType.ORDER_TYPE_BUY_STOP: "STP",
            OrderType.ORDER_TYPE_SELL_STOP: "STP",
            OrderType.ORDER_TYPE_BUY_STOP_LIMIT: "STP LMT",
            OrderType.ORDER_TYPE_SELL_STOP_LIMIT: "STP LMT",
        }.get(otype, "ORD")
        ys.append(price)
        colors.append(order_color)
        texts.append(f"{type_label} {side} {vol:g} @ {price:g}")

    if not ys:
        return _empty_pending_orders_data()

    return dict(
        y     = np.array(ys,     dtype=np.float64),
        color = np.array(colors, dtype=object),
        text  = np.array(texts,  dtype=object),
    )
