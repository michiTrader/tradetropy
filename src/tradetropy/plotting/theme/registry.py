"""
Theme registry.

Maps theme names to their token dicts and exposes the default indicator color
cycle. ``THEMES`` is the single source of truth previously held inline in
``plotting/_util.py``; that module now re-exports from here for compatibility.
"""

from __future__ import annotations

from tradetropy.plotting.theme.light import LIGHT
from tradetropy.plotting.theme.dark import DARK

# name -> token dict
THEMES: dict[str, dict] = {
    "light": LIGHT,
    "dark": DARK,
}

# Default color cycle for indicators that do not declare an explicit color.
INDICATOR_COLORS: list[str] = [
    "#F59E0B", "#8B5CF6", "#EC4899", "#06B6D4",
    "#84CC16", "#EF4444", "#3B82F6", "#F97316",
    "#10B981", "#A855F7",
]


def get_theme(name: str) -> dict:
    """
    Return the token dict for a theme name.

    Args:
        name (str): Theme name ('light' or 'dark').

    Returns:
        dict: The theme token dict.

    Raises:
        KeyError: If the theme name is not registered.
    """
    return THEMES[name]
