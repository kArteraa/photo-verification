"""Directional advisory vocabulary for soft requirements.

Each signed measurement key maps to a pair of corrective phrases: the first
is used for negative values, the second for positive ones.  The phrase tells
the user what to do, not how the value is oriented: a positive yaw (face turned
towards the left edge of the image) yields «влево», i.e. turn to your left.
"""

from __future__ import annotations

from collections.abc import Mapping

from app.core.report import Number

SIGNED_KEYS: dict[str, tuple[str, str]] = {
    "yaw": ("вправо", "влево"),
    "pitch": ("поднимите подбородок", "опустите подбородок"),
    "roll": ("по часовой стрелке", "против часовой стрелки"),
    "dx": ("правее", "левее"),
    "dy": ("ниже", "выше"),
    "asymmetry": ("слева", "справа"),
}


def hint_placeholders(keys: tuple[str, ...]) -> set[str]:
    """Placeholders that :func:`hint_context` can supply for the given measurer keys."""
    names: set[str] = set()
    for key in keys:
        if key in SIGNED_KEYS:
            names.update((f"{key}_dir", f"{key}_abs"))
    return names


def hint_context(values: Mapping[str, Number]) -> dict[str, Number | str]:
    """Derive ``<key>_dir`` and ``<key>_abs`` entries for every signed key present."""
    context: dict[str, Number | str] = {}
    for key, (negative, positive) in SIGNED_KEYS.items():
        if key in values:
            value = values[key]
            context[f"{key}_dir"] = negative if value < 0 else positive
            context[f"{key}_abs"] = abs(value)
    return context
