"""Small helpers shared by the kit widgets."""

from __future__ import annotations

import json
import math

from PySide6.QtCore import QEasingCurve, QVariantAnimation
from PySide6.QtWidgets import QWidget

DURATION_MS = 180
QWIDGETSIZE_MAX = 16777215


def make_animation(parent, duration: int = DURATION_MS) -> QVariantAnimation:
    """An OutCubic QVariantAnimation (values are set by the caller)."""
    animation = QVariantAnimation(parent)
    animation.setDuration(duration)
    animation.setEasingCurve(QEasingCurve.Type.OutCubic)
    return animation


def set_style_property(widget: QWidget, name: str, value) -> None:
    """Set a dynamic property used by stylesheet selectors and re-polish the widget."""
    if widget.property(name) == value:
        return
    widget.setProperty(name, value)
    style = widget.style()
    style.unpolish(widget)
    style.polish(widget)
    widget.update()


def to_bool(value) -> bool | None:
    """A stored bool (``True`` or the strings ``"true"``/``"false"``), else None."""
    if isinstance(value, bool):
        return value
    if isinstance(value, str) and value.lower() in ("true", "false"):
        return value.lower() == "true"
    return None


def to_float(value) -> float | None:
    """A finite float from a number (not a bool), else None."""
    if isinstance(value, bool) or not isinstance(value, int | float):
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def json_object(value) -> dict | None:
    """A stored JSON object (a dict or its JSON text), else None."""
    if isinstance(value, dict):
        return value
    if not isinstance(value, str):
        return None
    try:
        data = json.loads(value)
    except ValueError:
        return None
    return data if isinstance(data, dict) else None
