"""Line icons, tinted to the current theme.

The SVGs are Lucide icons (https://lucide.dev, lucide-static 1.50.0, ISC licence, see
``LICENSE-lucide.txt``): 24x24, stroke 2, round caps, ``stroke="currentColor"``.

``icon(name)`` renders an SVG in a colour (default: the palette's text colour) into pixmaps at
1x and 2x and caches the result. Buttons and actions given their icon with :func:`set_icon` are
re-tinted by :func:`refresh`, which the theme calls whenever its colours change.

Lucide has outline icons only. A *filled* icon (``fill=``) draws the outline glyph on a rounded
tile of the fill colour, which works for open shapes too (e.g. ``activity``); the SVGs stay as
they are.
"""

from __future__ import annotations

import weakref
from functools import cache
from pathlib import Path

from PySide6.QtCore import QByteArray, QRectF, Qt
from PySide6.QtGui import QColor, QGuiApplication, QIcon, QImage, QPainter, QPalette, QPixmap
from PySide6.QtSvg import QSvgRenderer
from shiboken6 import isValid

_DIR = Path(__file__).resolve().parent
_SIZES = (16, 20, 24)  # logical sizes rendered; QIcon picks or scales the nearest
_SCALES = (1.0, 2.0)
TILE_RADIUS = 0.26  # corner radius of a filled icon's tile, of its side
TILE_GLYPH = 0.74  # size of the glyph on the tile, of its side

ColorSpec = QColor | str | None

_cache: dict[tuple, QIcon] = {}
_bound: weakref.WeakKeyDictionary = weakref.WeakKeyDictionary()


def names() -> list[str]:
    """All available icon names."""
    return sorted(p.stem for p in _DIR.glob("*.svg"))


@cache
def svg_data(name: str) -> bytes:
    """The SVG source of *name*; raises KeyError for unknown icons."""
    path = _DIR / f"{name}.svg"
    if not path.is_file():
        raise KeyError(f"no icon named {name!r}")
    return path.read_bytes()


def _palette_color(group: QPalette.ColorGroup) -> QColor:
    return QGuiApplication.palette().color(group, QPalette.ColorRole.WindowText)


def resolve_color(color: ColorSpec) -> QColor:
    """A colour argument as a QColor: None (palette text), a theme token name, or a colour."""
    if color is None:
        return _palette_color(QPalette.ColorGroup.Active)
    if isinstance(color, QColor):
        return QColor(color)
    from mag_opt_detective.gui.theme import current_tokens  # theme imports this module

    tokens = current_tokens()
    if color in tokens:
        return tokens[color]
    resolved = QColor(color)
    if not resolved.isValid():
        raise ValueError(f"not a colour or theme token: {color!r}")
    return resolved


def _renderer(name: str, tint: QColor) -> QSvgRenderer:
    data = svg_data(name).replace(b"currentColor", tint.name().encode())
    return QSvgRenderer(QByteArray(data))


def _render(renderer: QSvgRenderer, size: int, scale: float, opacity: float) -> QPixmap:
    side = max(1, round(size * scale))
    image = QImage(side, side, QImage.Format.Format_ARGB32_Premultiplied)
    image.fill(Qt.GlobalColor.transparent)
    painter = QPainter(image)
    painter.setRenderHint(QPainter.RenderHint.Antialiasing)
    painter.setOpacity(opacity)
    renderer.render(painter, QRectF(0, 0, side, side))
    painter.end()
    result = QPixmap.fromImage(image)
    result.setDevicePixelRatio(scale)
    return result


def _render_tile(renderer: QSvgRenderer, size: int, scale: float, fill: QColor) -> QPixmap:
    """The glyph of *renderer* centred on a rounded tile of colour *fill*."""
    side = max(1, round(size * scale))
    image = QImage(side, side, QImage.Format.Format_ARGB32_Premultiplied)
    image.fill(Qt.GlobalColor.transparent)
    painter = QPainter(image)
    painter.setRenderHint(QPainter.RenderHint.Antialiasing)
    painter.setPen(Qt.PenStyle.NoPen)
    painter.setBrush(fill)
    radius = side * TILE_RADIUS
    painter.drawRoundedRect(QRectF(0, 0, side, side), radius, radius)
    glyph = side * TILE_GLYPH
    offset = (side - glyph) / 2
    renderer.render(painter, QRectF(offset, offset, glyph, glyph))
    painter.end()
    result = QPixmap.fromImage(image)
    result.setDevicePixelRatio(scale)
    return result


def pixmap(name: str, size: int, color: ColorSpec = None, scale: float = 1.0) -> QPixmap:
    """*name* rendered at *size* logical pixels and device pixel ratio *scale*."""
    tint = resolve_color(color)
    return _render(_renderer(name, tint), size, scale, tint.alphaF())


def _add(
    target: QIcon,
    name: str,
    color: QColor,
    mode: QIcon.Mode,
    state: QIcon.State,
    fill: QColor | None = None,
) -> None:
    renderer = _renderer(name, color)
    for size in _SIZES:
        for scale in _SCALES:
            if fill is None:
                rendered = _render(renderer, size, scale, color.alphaF())
            else:
                rendered = _render_tile(renderer, size, scale, fill)
            target.addPixmap(rendered, mode, state)


def icon(
    name: str, color: ColorSpec = None, on_color: ColorSpec = None, fill: ColorSpec = None
) -> QIcon:
    """A tinted icon. *on_color* (optional) tints the checked state, e.g. ``"accent"``.

    *fill* (optional) makes a filled icon: the glyph in *color* (*on_color* when checked) on a
    rounded tile of *fill*, e.g. ``icon(name, "accent-fg", fill="accent")``; disabled, the
    tile takes the disabled colour.

    Colours are None (the palette text colour), a theme token name such as ``"err"``, or
    anything QColor accepts. Disabled pixmaps use the palette's disabled text colour.
    """
    normal = resolve_color(color)
    on = resolve_color(on_color) if on_color is not None else None
    tile = resolve_color(fill) if fill is not None else None
    disabled = _palette_color(QPalette.ColorGroup.Disabled)
    key = (name, *(c if c is None else c.rgba() for c in (normal, on, disabled, tile)))
    cached = _cache.get(key)
    if cached is not None:
        return cached
    result = QIcon()
    tints = [(QIcon.State.Off, normal)] + ([(QIcon.State.On, on)] if on is not None else [])
    for state, tint in tints:
        if tile is None:  # outline; disabled: the glyph in the disabled colour
            _add(result, name, tint, QIcon.Mode.Normal, state)
            _add(result, name, disabled, QIcon.Mode.Disabled, state)
        else:  # filled; disabled: the tile in the disabled colour
            _add(result, name, tint, QIcon.Mode.Normal, state, tile)
            _add(result, name, tint, QIcon.Mode.Disabled, state, disabled)
    _cache[key] = result
    return result


def set_icon(
    target, name: str, color: ColorSpec = None, on_color: ColorSpec = None, fill: ColorSpec = None
) -> None:
    """``target.setIcon(icon(...))`` and remember it, so :func:`refresh` re-tints it."""
    target.setIcon(icon(name, color, on_color, fill))
    _bound[target] = (name, color, on_color, fill)


def clear_cache() -> None:
    """Forget all rendered icons (the next :func:`icon` call renders again)."""
    _cache.clear()


def refresh() -> None:
    """Clear the cache and re-tint every icon set with :func:`set_icon`."""
    clear_cache()
    for target, args in list(_bound.items()):
        if isValid(target):
            target.setIcon(icon(*args))
        else:
            _bound.pop(target, None)
