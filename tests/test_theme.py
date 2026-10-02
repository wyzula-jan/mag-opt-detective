import re
from pathlib import Path
from types import SimpleNamespace

import pytest
from PySide6.QtCore import Qt
from PySide6.QtGui import QColor, QGuiApplication, QPalette
from PySide6.QtWidgets import QApplication, QToolButton

from mag_opt_detective.gui import icons, theme
from mag_opt_detective.gui.theme import DARK, LIGHT, Theme, parse_color

TOKENS = {
    *("bg", "fg", "muted", "faint", "line", "line-strong", "win", "surface", "sunken", "hover"),
    *("accent", "accent-fg", "accent-soft", "ok", "warn", "err", "warn-soft", "err-soft"),
    *("plot-bg", "plot-well", "plot-fg", "plot-grid"),
}
ICONS = (
    *("folder-open", "play", "download", "upload", "sun", "moon", "contrast", "activity"),
    *("layers", "sliders-horizontal", "library-big", "chart-scatter", "move", "zoom-in"),
    *("crosshair", "scan", "image", "terminal", "chevron-down", "chevron-right", "x"),
    *("triangle-alert", "info", "check", "plus", "trash-2", "save", "git-merge", "file-text"),
    *("panel-left", "panel-right", "palette", "eye", "eye-off", "undo-2", "redo-2"),
    *("wand-sparkles", "sigma", "pin"),
)


@pytest.fixture
def app(qapp):
    """The application, with its look restored after the test applied a theme."""
    palette, sheet, style = QPalette(qapp.palette()), qapp.styleSheet(), qapp.style().name()
    yield qapp
    qapp.setStyleSheet(sheet)
    qapp.setPalette(palette)
    if qapp.style().name() != style:
        QApplication.setStyle(style)
    QGuiApplication.styleHints().setColorScheme(Qt.ColorScheme.Unknown)
    theme._active = None
    icons.clear_cache()


def window_color(app) -> str:
    return app.palette().color(QPalette.ColorRole.Window).name()


def test_tokens_match_the_mockup():
    assert set(LIGHT) == set(DARK)
    assert set(LIGHT) >= TOKENS
    assert (LIGHT["accent"], DARK["accent"]) == ("#7a2a8c", "#fe9f6d")
    assert (LIGHT["err-soft"], DARK["plot-bg"]) == ("#fbe9e6", "#121015")


def test_parse_color_reads_css_rgba():
    grid = parse_color(LIGHT["plot-grid"])
    assert (grid.red(), grid.green(), grid.blue(), grid.alpha()) == (59, 52, 67, 23)
    assert parse_color("#7a2a8c") == QColor("#7a2a8c")
    with pytest.raises(ValueError):
        parse_color("not a colour")


def test_light_and_dark_palettes(app, qtbot):
    t = Theme("light")
    with qtbot.waitSignal(t.changed):
        t.apply(app)
    assert window_color(app) == LIGHT["win"]
    assert app.palette().color(QPalette.ColorRole.Highlight).name() == LIGHT["accent"]
    assert app.palette().color(QPalette.ColorRole.Base).name() == LIGHT["surface"]
    assert 'kit="segmented"' in app.styleSheet()
    assert not t.is_dark()

    with qtbot.waitSignal(t.changed):
        t.set_scheme("dark")
    assert t.is_dark()
    assert t.scheme() == "dark"
    assert window_color(app) == DARK["win"]
    assert app.palette().color(QPalette.ColorRole.Text).name() == DARK["fg"]
    assert DARK["err-soft"] in app.styleSheet()
    assert theme.current_theme() is t
    assert theme.current_tokens()["accent"].name() == DARK["accent"]
    app.setStyleSheet("")  # under a stylesheet, style() is the stylesheet wrapper
    assert app.style().name().lower() == "fusion"


@pytest.mark.parametrize("dark", [False, True])
def test_palette_bevel_roles_are_ordered(dark):
    p = theme.build_palette(theme.tokens_for(dark))
    roles = (QPalette.ColorRole.Light, QPalette.ColorRole.Button, QPalette.ColorRole.Dark)
    light, button, darker = (p.color(role).lightness() for role in roles)
    shadow = p.color(QPalette.ColorRole.Shadow).lightness()
    assert light >= button > darker >= shadow


def test_switching_to_system_applies_once(app, monkeypatch):
    t = Theme("dark")
    t.apply(app)
    hints = QGuiApplication.styleHints()

    class Gui:  # a platform that reports a requested scheme at once, as macOS does
        @staticmethod
        def styleHints():
            return SimpleNamespace(
                setColorScheme=hints.colorSchemeChanged.emit, colorScheme=hints.colorScheme
            )

    monkeypatch.setattr(theme, "QGuiApplication", Gui)
    applied, emitted = [], []
    monkeypatch.setattr(t, "_apply_to", applied.append)
    t.changed.connect(lambda: emitted.append(True))
    t.set_scheme("system")
    t.apply(app)
    assert len(applied) == len(emitted) == 2


def test_stylesheet_only_matches_kit_properties():
    selectors = re.findall(r"^([^{}\n]+)\{", theme.build_stylesheet(theme.tokens_for(False)), re.M)
    assert selectors
    for rule in selectors:
        for selector in rule.split(","):
            assert "[kit=" in selector or "[invalid=" in selector, selector


def test_system_scheme_is_followed_and_overridable(app, qtbot, monkeypatch):
    dark = False
    monkeypatch.setattr(theme, "system_is_dark", lambda: dark)
    t = Theme()
    assert t.scheme() == "system"
    t.apply(app)
    assert window_color(app) == LIGHT["win"]

    dark = True
    hints = QGuiApplication.styleHints()
    with qtbot.waitSignal(t.changed):
        hints.colorSchemeChanged.emit(Qt.ColorScheme.Dark)
    assert t.is_dark()
    assert window_color(app) == DARK["win"]

    t.set_scheme("light")  # an explicit choice ignores the OS
    with qtbot.assertNotEmitted(t.changed):
        hints.colorSchemeChanged.emit(Qt.ColorScheme.Dark)
    assert window_color(app) == LIGHT["win"]

    with pytest.raises(ValueError):
        t.set_scheme("sepia")
    with pytest.raises(ValueError):
        Theme("sepia")


def test_plot_colors_are_opaque_hex_from_the_tokens():
    light, dark = Theme("light").plot_colors(), Theme("dark").plot_colors()
    keys = {"background", "foreground", "grid", "crosshair", "accent", "well"}
    assert set(light) == set(dark) == keys
    for colors in (light, dark):
        assert all(re.fullmatch(r"#[0-9a-f]{6}", value) for value in colors.values())
    assert light["background"] == LIGHT["plot-bg"]
    assert dark["well"] == DARK["plot-well"]
    assert dark["accent"] == DARK["accent"]
    # the translucent grid is composited over the plot background
    assert light["grid"] not in (LIGHT["plot-bg"], LIGHT["plot-fg"])
    assert QColor(light["grid"]).lightness() < QColor(light["background"]).lightness()


def test_current_tokens_follow_the_palette_without_a_theme(app):
    theme._active = None
    app.setPalette(theme.build_palette(theme.tokens_for(dark=True)))
    assert theme.current_tokens()["win"].name() == DARK["win"]
    app.setPalette(theme.build_palette(theme.tokens_for(dark=False)))
    assert theme.current_tokens()["win"].name() == LIGHT["win"]


def test_every_needed_icon_exists_and_renders(qapp):
    assert set(ICONS) <= set(icons.names())
    for name in ICONS:
        icon = icons.icon(name)
        assert not icon.isNull(), name
        assert not icon.pixmap(16, 16).isNull(), name
        assert b'stroke="currentColor"' in icons.svg_data(name)
    licence = Path(icons.__file__).with_name("LICENSE-lucide.txt")
    assert "ISC License" in licence.read_text()
    with pytest.raises(KeyError):
        icons.icon("no-such-icon")


def _ink(pixmap) -> QColor:
    """The colour of the most opaque pixel."""
    image = pixmap.toImage()
    best = max(
        (image.pixelColor(x, y) for x in range(image.width()) for y in range(image.height())),
        key=QColor.alpha,
    )
    best.setAlpha(255)
    return best


def test_icons_are_tinted_cached_and_hidpi(qapp):
    icons.clear_cache()
    red = icons.pixmap("x", 24, "#ff0000")
    assert _ink(red).name() == "#ff0000"
    hidpi = icons.pixmap("x", 24, "#ff0000", scale=2.0)
    assert (hidpi.width(), hidpi.devicePixelRatio()) == (48, 2.0)
    assert icons.icon("x", "#00ff00").cacheKey() == icons.icon("x", "#00ff00").cacheKey()
    assert icons.icon("x", "#00ff00").cacheKey() != icons.icon("x", "#0000ff").cacheKey()
    with pytest.raises(ValueError):
        icons.icon("x", "no-such-token")


def test_theme_change_retints_bound_icons(app, qtbot):
    t = Theme("light")
    t.apply(app)
    button = QToolButton()
    qtbot.addWidget(button)
    icons.set_icon(button, "x")
    icons.set_icon(button, "check", "accent")  # the last binding wins
    assert _ink(button.icon().pixmap(24, 24)).name() == LIGHT["accent"]
    t.set_scheme("dark")
    assert _ink(button.icon().pixmap(24, 24)).name() == DARK["accent"]
