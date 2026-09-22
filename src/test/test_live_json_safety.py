"""
The live server push must survive a non-finite float in a message payload.

Bokeh's ``serialize_json`` runs json with ``allow_nan=False``; a stray NaN/Inf
would crash the websocket send coroutine (uncatchable by the chart's own update
callback). ``install_nan_safe_serialization`` retries such a payload with the
non-finite floats replaced by ``null`` so the stream stays alive.
"""

import json
import math

import pytest

from bokeh.core.serialization import Serialized


def test_original_serialize_json_rejects_nan():
    import bokeh.core.json_encoder as je
    with pytest.raises(ValueError):
        je.serialize_json(Serialized(content={"x": float("nan")}, buffers=None))


def test_guard_makes_push_nan_safe():
    from tradetropy.plotting.live._json_safety import install_nan_safe_serialization
    install_nan_safe_serialization()
    import bokeh.protocol.message as msg

    # A payload that would otherwise crash the send: NaN nested in the content.
    payload = Serialized(
        content={"events": [{"kind": "ModelChanged", "new": float("nan")},
                            {"vals": [1.0, float("inf"), 3.0]}]},
        buffers=None,
    )
    with pytest.warns(UserWarning, match="non-finite float"):
        out = msg.serialize_json(payload)
    # Valid JSON, non-finite floats became null (no NaN/Infinity tokens).
    parsed = json.loads(out)
    assert parsed["events"][0]["new"] is None
    assert parsed["events"][1]["vals"] == [1.0, None, 3.0]


def test_guard_is_idempotent_and_transparent_for_finite():
    from tradetropy.plotting.live._json_safety import install_nan_safe_serialization
    install_nan_safe_serialization()
    install_nan_safe_serialization()  # second call is a no-op
    import bokeh.protocol.message as msg

    good = Serialized(content={"a": 1, "b": [2.0, 3.0], "c": "x"}, buffers=None)
    assert json.loads(msg.serialize_json(good)) == {"a": 1, "b": [2.0, 3.0], "c": "x"}


def test_path_finder_locates_nan():
    from tradetropy.plotting.live._json_safety import _find_path
    content = {"roots": [{"attrs": {"y_range": {"start": float("nan")}}}]}
    assert _find_path(content) == "$.roots[0].attrs.y_range.start"
