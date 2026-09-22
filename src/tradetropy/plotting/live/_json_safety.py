"""
NaN-safe websocket serialization guard for the live Bokeh server.

Bokeh serializes each outgoing protocol message with ``serialize_json``, whose
``PayloadEncoder`` runs ``json`` with ``allow_nan=False``. If any non-finite
float (``NaN`` / ``Inf``) survives into the message payload - e.g. a model
property or a data value that slipped past Bokeh's own Serializer - the push
raises::

    ValueError: Out of range float values are not JSON compliant: nan

That exception fires inside Tornado's send coroutine, so it cannot be caught by
the chart's own update callback; it just spams the console and drops that push.

This module installs a thin wrapper around the ``serialize_json`` used by the
protocol message: the fast path is unchanged, and ONLY when a message fails with
the non-finite-float error do we (1) log the JSON path to the offending value
once (for diagnosis) and (2) retry with the non-finite floats replaced by
``null``. ``null`` renders as a gap - the same visual meaning as ``NaN`` - so no
information is lost and the live chart stays alive.

The patch is idempotent and scoped to when a LiveChart is used.
"""

from __future__ import annotations

import math
import warnings

_INSTALLED = False
_WARNED_PATHS: set[str] = set()


def _is_bad_float(x) -> bool:
    return isinstance(x, float) and not math.isfinite(x)


def _sanitize(obj):
    """Return a copy of ``obj`` with non-finite floats replaced by None."""
    if _is_bad_float(obj):
        return None
    if isinstance(obj, dict):
        return {k: _sanitize(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_sanitize(v) for v in obj]
    if isinstance(obj, tuple):
        return tuple(_sanitize(v) for v in obj)
    return obj


def _find_path(obj, path: str = "$"):
    """Return the JSON-ish path to the first non-finite float, or None."""
    if _is_bad_float(obj):
        return path
    if isinstance(obj, dict):
        for k, v in obj.items():
            hit = _find_path(v, f"{path}.{k}")
            if hit is not None:
                return hit
    elif isinstance(obj, (list, tuple)):
        for i, v in enumerate(obj):
            hit = _find_path(v, f"{path}[{i}]")
            if hit is not None:
                return hit
    return None


def _describe(content, path: str) -> str:
    """
    Best-effort human description of the offending value's source.

    A live push is a Bokeh patch-doc message whose ``events`` are model updates.
    A positional path like ``$.events[0].data.entries[0][1][0]`` does not name
    the culprit, so this walks the same event and pulls the event ``kind``, the
    target model id and - for a ColumnData change - the actual COLUMN NAME
    (``data.entries[<i>][0]``). Knowing the column name maps the NaN straight to
    the indicator/glyph that produced it. Returns '' if it cannot be resolved.

    Args:
        content: The message content (dict) being serialized.
        path (str): The path returned by ``_find_path``.

    Returns:
        str: A ``" (kind=..., model=..., column=...)"`` suffix, or ''.
    """
    import re

    if not isinstance(content, dict):
        return ""
    m = re.match(r"\$\.events\[(\d+)\]", path or "")
    events = content.get("events")
    if not (m and isinstance(events, list)):
        return ""
    idx = int(m.group(1))
    if idx >= len(events) or not isinstance(events[idx], dict):
        return ""
    ev = events[idx]

    parts = [f"kind={ev.get('kind', '?')}"]
    model = ev.get("model") or ev.get("column_source")
    if isinstance(model, dict) and model.get("id") is not None:
        parts.append(f"model={model['id']}")

    em = re.search(r"entries\[(\d+)\]", path or "")
    data = ev.get("data")
    if em and isinstance(data, dict):
        entries = data.get("entries")
        e_idx = int(em.group(1))
        if isinstance(entries, list) and e_idx < len(entries):
            entry = entries[e_idx]
            if isinstance(entry, (list, tuple)) and entry:
                parts.append(f"column={entry[0]!r}")

    return " (" + ", ".join(parts) + ")"


def install_nan_safe_serialization() -> None:
    """
    Patch the live protocol's ``serialize_json`` to survive non-finite floats.

    Idempotent. Wraps the name bound in ``bokeh.protocol.message`` (the actual
    call site) so an already-imported reference is replaced.
    """
    global _INSTALLED
    if _INSTALLED:
        return

    try:
        import bokeh.protocol.message as _msg
        from bokeh.core.serialization import Serialized
    except Exception:
        return

    _orig = _msg.serialize_json

    def _safe_serialize_json(obj, *args, **kwargs):
        try:
            return _orig(obj, *args, **kwargs)
        except ValueError as exc:
            if "JSON compliant" not in str(exc):
                raise
            content = obj.content if isinstance(obj, Serialized) else obj
            path = _find_path(content)
            if path is not None and path not in _WARNED_PATHS:
                _WARNED_PATHS.add(path)
                warnings.warn(
                    "Live chart push carried a non-finite float at "
                    f"{path}{_describe(content, path)}; replaced with null to "
                    "keep the stream alive. This is a data/plot bug worth "
                    "reporting.",
                    stacklevel=2,
                )
            if isinstance(obj, Serialized):
                fixed = Serialized(content=_sanitize(content), buffers=obj.buffers)
            else:
                fixed = _sanitize(content)
            return _orig(fixed, *args, **kwargs)

    _msg.serialize_json = _safe_serialize_json
    _INSTALLED = True
