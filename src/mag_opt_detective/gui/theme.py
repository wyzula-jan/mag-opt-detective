"""Light and dark design tokens, the application palette and a short stylesheet.

The tokens are the CSS variables of the approved redesign mockup. :class:`Theme` turns them into
a Fusion ``QPalette`` plus a few stylesheet rules for what a palette cannot express. Kit widgets
opt into those rules with a ``kit`` dynamic property (``kit="segmented"``, ``"tool"``, ...).
"""

from __future__ import annotations

import re

from PySide6.QtCore import QObject, Qt, Signal
from PySide6.QtGui import QColor, QGuiApplication, QPalette
from PySide6.QtWidgets import QApplication

from mag_opt_detective.gui import icons

SCHEMES = ("system", "light", "dark")

LIGHT: dict[str, str] = {
    "bg": "#f3f1f5",
    "fg": "#1d1824",
    "muted": "#605869",
    "faint": "#8a8294",
    "line": "#ddd7e4",
    "line-strong": "#c4bccd",
    "win": "#fbfafc",
    "surface": "#ffffff",
    "sunken": "#efecf3",
    "hover": "#e7e2ed",
    "accent": "#7a2a8c",
    "accent-fg": "#ffffff",
    "accent-soft": "#f2e8f5",
    "ok": "#24764a",
    "warn": "#9a5b08",
    "err": "#b3362a",
    "warn-soft": "#fbf1df",
    "err-soft": "#fbe9e6",
    "plot-bg": "#ffffff",
    "plot-well": "#e9e6ed",
    "plot-fg": "#3b3443",
    "plot-grid": "rgba(59, 52, 67, 0.09)",
    "toast-bg": "#2a2331",
    "toast-fg": "#f6f2f9",
}

DARK: dict[str, str] = {
    "bg": "#0e0c11",
    "fg": "#ece7f1",
    "muted": "#a59dad",
    "faint": "#7d7587",
    "line": "#2c2633",
    "line-strong": "#40384a",
    "win": "#17141b",
    "surface": "#1e1a23",
    "sunken": "#121015",
    "hover": "#27222d",
    "accent": "#fe9f6d",
    "accent-fg": "#21141c",
    "accent-soft": "#3a2621",
    "ok": "#5cc68b",
    "warn": "#e3a640",
    "err": "#f2725f",
    "warn-soft": "#33270f",
    "err-soft": "#3a1d1a",
    "plot-bg": "#121015",
    "plot-well": "#1c1921",
    "plot-fg": "#cbc4d3",
    "plot-grid": "rgba(220, 210, 235, 0.08)",
    "toast-bg": "#ece7f1",
    "toast-fg": "#1d1824",
}

_RGBA = re.compile(r"rgba\(\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*,\s*([\d.]+)\s*\)")


def parse_color(text: str) -> QColor:
    """A token value (``#rrggbb`` or CSS ``rgba(r, g, b, a)``) as a QColor."""
    match = _RGBA.fullmatch(text.strip())
    if match:
        r, g, b, a = match.groups()
        return QColor(int(r), int(g), int(b), round(float(a) * 255))
    color = QColor(text)
    if not color.isValid():
        raise ValueError(f"not a colour: {text!r}")
    return color


def tokens_for(dark: bool) -> dict[str, QColor]:
    """The light or dark token set as QColors."""
    return {name: parse_color(value) for name, value in (DARK if dark else LIGHT).items()}


def mix(a: QColor, b: QColor, amount: float) -> QColor:
    """``amount`` of *a* over *b*, like CSS ``color-mix(in srgb, a amount, b)``; opaque."""
    t = min(max(amount, 0.0), 1.0)
    return QColor.fromRgbF(
        a.redF() * t + b.redF() * (1 - t),
        a.greenF() * t + b.greenF() * (1 - t),
        a.blueF() * t + b.blueF() * (1 - t),
    )


def _over(color: QColor, background: QColor) -> QColor:
    """A translucent colour composited over an opaque background."""
    return mix(color, background, color.alphaF())


def _rgba(color: QColor, alpha: float) -> str:
    return f"rgba({color.red()}, {color.green()}, {color.blue()}, {round(alpha * 255)})"


def build_palette(t: dict[str, QColor]) -> QPalette:
    """A QPalette from the tokens (all colour groups, plus dimmed disabled colours)."""
    p = QPalette()
    roles = {
        QPalette.ColorRole.Window: "win",
        QPalette.ColorRole.WindowText: "fg",
        QPalette.ColorRole.Base: "surface",
        QPalette.ColorRole.AlternateBase: "sunken",
        QPalette.ColorRole.ToolTipBase: "toast-bg",
        QPalette.ColorRole.ToolTipText: "toast-fg",
        QPalette.ColorRole.PlaceholderText: "faint",
        QPalette.ColorRole.Text: "fg",
        QPalette.ColorRole.Button: "surface",
        QPalette.ColorRole.ButtonText: "fg",
        QPalette.ColorRole.BrightText: "accent-fg",
        QPalette.ColorRole.Midlight: "line",  # the kit widgets draw lines with these two
        QPalette.ColorRole.Mid: "line-strong",
        QPalette.ColorRole.Highlight: "accent",
        QPalette.ColorRole.HighlightedText: "accent-fg",
        QPalette.ColorRole.Link: "accent",
        QPalette.ColorRole.LinkVisited: "accent",
        QPalette.ColorRole.Accent: "accent",
    }
    dark = t["bg"].lightness() < t["fg"].lightness()
    # bevels (qDrawShade* frames): Light above Button, Dark and Shadow below it
    roles[QPalette.ColorRole.Light] = "line-strong" if dark else "surface"
    roles[QPalette.ColorRole.Dark] = "sunken" if dark else "faint"
    roles[QPalette.ColorRole.Shadow] = "bg" if dark else "fg"
    for role, name in roles.items():
        p.setColor(role, t[name])
    disabled = QPalette.ColorGroup.Disabled
    for role in (
        QPalette.ColorRole.WindowText,
        QPalette.ColorRole.Text,
        QPalette.ColorRole.ButtonText,
    ):
        p.setColor(disabled, role, t["faint"])
    p.setColor(disabled, QPalette.ColorRole.Base, t["sunken"])
    p.setColor(disabled, QPalette.ColorRole.Button, t["sunken"])
    p.setColor(disabled, QPalette.ColorRole.Highlight, t["line-strong"])
    p.setColor(disabled, QPalette.ColorRole.Accent, t["line-strong"])
    p.setColor(disabled, QPalette.ColorRole.HighlightedText, t["fg"])
    return p


def build_stylesheet(t: dict[str, QColor]) -> str:
    """The few rules the palette cannot express; they only match ``kit`` properties."""
    c = {name: color.name() for name, color in t.items()}
    accent_ring = _rgba(t["accent"], 0.35)
    err_border = mix(t["err"], t["line"], 0.45).name()
    warn_border = mix(t["warn"], t["line"], 0.45).name()
    info_border = mix(t["accent"], t["line"], 0.45).name()
    chip_border = mix(t["accent"], t["surface"], 0.4).name()
    return f"""
*[kit="segmented"] {{
    background: {c["sunken"]}; border: 1px solid {c["line"]}; border-radius: 7px;
}}
QToolButton[kit="segment"] {{
    background: transparent; border: 1px solid transparent; border-radius: 5px;
    color: {c["muted"]}; padding: 3px 9px; font-weight: 500;
}}
*[kit="segmented"][kitSize="sm"] QToolButton[kit="segment"] {{ padding: 2px 8px; }}
*[kit="segmented"][kitSize="xs"] QToolButton[kit="segment"] {{ padding: 1px 7px; }}
QToolButton[kit="segment"]:hover {{ color: {c["fg"]}; }}
QToolButton[kit="segment"]:checked {{
    background: {c["surface"]}; border-color: {c["line-strong"]}; color: {c["fg"]};
}}
QToolButton[kit="segment"]:focus {{ border-color: {c["accent"]}; }}
QToolButton[kit="segment"]:disabled {{ color: {c["faint"]}; }}
QToolButton[kit="tool"] {{
    background: transparent; border: 1px solid transparent; border-radius: 6px; padding: 5px;
    color: {c["muted"]};
}}
QToolButton[kit="tool"]:hover {{ background: {c["hover"]}; color: {c["fg"]}; }}
QToolButton[kit="tool"]:checked {{
    background: {c["accent-soft"]}; color: {c["accent"]}; border-color: {accent_ring};
}}
QToolButton[kit="tool"]:focus {{ border-color: {c["accent"]}; }}
QToolButton[kit="chip"] {{
    background: {c["surface"]}; border: 1px solid {c["line-strong"]}; border-radius: 6px;
    padding: 0px 9px; min-height: 24px; color: {c["muted"]}; font-weight: 500;
}}
QToolButton[kit="chip"]:hover {{ color: {c["fg"]}; }}
QToolButton[kit="chip"]:checked {{
    background: {c["accent-soft"]}; color: {c["accent"]}; border-color: {chip_border};
}}
QToolButton[kit="chip"]:focus {{ border-color: {c["accent"]}; }}
QToolButton[kit="chip"]:disabled {{
    background: {c["win"]}; color: {c["line-strong"]}; border-color: {c["line"]};
}}
QPushButton[kit="button"], QToolButton[kit="button"] {{
    background: {c["surface"]}; border: 1px solid {c["line-strong"]}; border-radius: 6px;
    padding: 4px 10px; color: {c["fg"]};
}}
QToolButton[kit="button"] {{ padding: 3px 8px; }}
QPushButton[kit="button"]:hover, QToolButton[kit="button"]:hover {{
    background: {c["hover"]};
}}
QPushButton[kit="button"]:pressed, QToolButton[kit="button"]:pressed {{
    background: {c["sunken"]};
}}
QToolButton[kit="button"]:checked {{
    background: {c["accent-soft"]}; color: {c["accent"]}; border-color: {chip_border};
}}
QPushButton[kit="button"]:focus, QToolButton[kit="button"]:focus {{
    border-color: {c["accent"]};
}}
QPushButton[kit="button"]:disabled, QToolButton[kit="button"]:disabled {{
    background: {c["win"]}; color: {c["faint"]}; border-color: {c["line"]};
}}
QToolButton[kit="rail"] {{
    background: transparent; border: none; border-radius: 7px; padding: 7px 2px 6px;
    color: {c["muted"]}; font-weight: 500;
}}
QToolButton[kit="rail"]:hover {{ background: {c["hover"]}; color: {c["fg"]}; }}
QToolButton[kit="rail"]:checked {{ background: {c["accent-soft"]}; color: {c["accent"]}; }}
QToolButton[kit="rail"]:focus {{ border: 1px solid {c["accent"]}; }}
QPushButton[kit="primary"] {{
    background: {c["accent"]}; border: 1px solid {c["accent"]}; border-radius: 6px;
    color: {c["accent-fg"]}; font-weight: 600; padding: 4px 10px;
}}
QPushButton[kit="primary"]:focus {{ border-color: {c["fg"]}; }}
QPushButton[kit="primary"]:disabled {{ background: {c["line-strong"]}; border-color: {c["line"]}; }}
QToolButton[kit="section-header"] {{
    background: transparent; border: none; padding: 2px 0px; color: {c["fg"]}; font-weight: 600;
}}
QToolButton[kit="section-header"]:focus {{ color: {c["accent"]}; }}
QFrame[kit="infobar"] {{
    background: {c["err-soft"]}; border: 1px solid {err_border}; border-radius: 8px;
}}
QFrame[kit="infobar"][level="warning"] {{
    background: {c["warn-soft"]}; border-color: {warn_border};
}}
QFrame[kit="infobar"][level="info"] {{
    background: {c["accent-soft"]}; border-color: {info_border};
}}
QLabel[kit="muted"], QLabel[kit="note"] {{ color: {c["muted"]}; }}
QLabel[kit="note"][error="true"] {{ color: {c["err"]}; }}
*[kit="range"] QAbstractSpinBox {{
    background: {c["surface"]}; border: 1px solid {c["line-strong"]}; border-radius: 5px;
    padding: 2px 4px; color: {c["fg"]};
}}
*[kit="range"] QAbstractSpinBox:focus {{ border-color: {c["accent"]}; }}
QAbstractSpinBox[invalid="true"], *[kit="range"] QAbstractSpinBox[invalid="true"] {{
    background: {c["err-soft"]}; border: 1px solid {c["err"]}; border-radius: 5px;
    padding: 2px 4px; color: {c["fg"]};
}}
"""


_active: Theme | None = None


def system_is_dark() -> bool:
    """Whether the OS asks for dark colours (Qt reports Unknown on some platforms: light)."""
    return QGuiApplication.styleHints().colorScheme() == Qt.ColorScheme.Dark


def current_theme() -> Theme | None:
    """The theme applied last with :meth:`Theme.apply`, if any."""
    return _active


def current_tokens() -> dict[str, QColor]:
    """Tokens of the applied theme, or the set that matches the application palette."""
    if _active is not None:
        return _active.tokens()
    window = QGuiApplication.palette().color(QPalette.ColorRole.Window)
    return tokens_for(window.lightness() < 128)


class Theme(QObject):
    """The application look: ``system`` (follows the OS), ``light`` or ``dark``.

    ``changed`` fires whenever the effective colours may differ (scheme switched, OS scheme
    changed while following it, or :meth:`apply`); cached icons are re-tinted first. Use one
    Theme per application: only the theme applied last carries an OS scheme change into the
    application palette.
    """

    changed = Signal()

    def __init__(self, scheme: str = "system", parent: QObject | None = None):
        super().__init__(parent)
        if scheme not in SCHEMES:
            raise ValueError(f"unknown scheme {scheme!r}; use one of {SCHEMES}")
        self._scheme = scheme
        self._app: QApplication | None = None
        self._requesting = False
        self.changed.connect(icons.refresh)  # first slot, so later slots see fresh icons
        QGuiApplication.styleHints().colorSchemeChanged.connect(self._on_system_scheme)

    # --- scheme ------------------------------------------------------------------------
    def scheme(self) -> str:
        return self._scheme

    def set_scheme(self, scheme: str) -> None:
        if scheme not in SCHEMES:
            raise ValueError(f"unknown scheme {scheme!r}; use one of {SCHEMES}")
        if scheme == self._scheme:
            return
        self._scheme = scheme
        self._request_platform_scheme()
        if self._app is not None:
            self._apply_to(self._app)
        self.changed.emit()

    def is_dark(self) -> bool:
        if self._scheme != "system":
            return self._scheme == "dark"
        return system_is_dark()

    def _on_system_scheme(self, _scheme=None) -> None:
        if self._scheme != "system" or self._requesting:
            return
        if self._app is not None and _active is self:
            self._apply_to(self._app)
        self.changed.emit()

    def _request_platform_scheme(self) -> None:
        """Ask the platform for matching window frames (ignored where unsupported)."""
        wanted = {"light": Qt.ColorScheme.Light, "dark": Qt.ColorScheme.Dark}
        self._requesting = True  # the hint may emit colorSchemeChanged now; we apply anyway
        try:
            QGuiApplication.styleHints().setColorScheme(
                wanted.get(self._scheme, Qt.ColorScheme.Unknown)
            )
        finally:
            self._requesting = False

    # --- colours -----------------------------------------------------------------------
    def tokens(self) -> dict[str, QColor]:
        """The active token set (a fresh dict of QColors)."""
        return tokens_for(self.is_dark())

    def palette(self) -> QPalette:
        return build_palette(self.tokens())

    def stylesheet(self) -> str:
        return build_stylesheet(self.tokens())

    def plot_colors(self) -> dict[str, str]:
        """Plot colours as opaque ``#rrggbb`` strings (keys match ``PlotColors``)."""
        t = self.tokens()
        return {
            "background": t["plot-bg"].name(),
            "foreground": t["plot-fg"].name(),
            "grid": _over(t["plot-grid"], t["plot-bg"]).name(),
            "crosshair": t["faint"].name(),
            "accent": t["accent"].name(),
            "well": t["plot-well"].name(),
        }

    # --- applying ----------------------------------------------------------------------
    def apply(self, app: QApplication | None = None) -> None:
        """Use this theme for *app* (default: the running application) from now on."""
        app = app or QApplication.instance()
        if app is None:
            raise RuntimeError("Theme.apply() needs a QApplication")
        self._app = app
        self._request_platform_scheme()
        self._apply_to(app)
        self.changed.emit()

    def _apply_to(self, app: QApplication) -> None:
        global _active
        _active = self
        QApplication.setStyle("Fusion")  # the base style can't be read back under a stylesheet
        tokens = self.tokens()
        app.setPalette(build_palette(tokens))
        app.setStyleSheet(build_stylesheet(tokens))
