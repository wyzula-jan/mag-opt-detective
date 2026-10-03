"""Screenshots of the app for the README and the docs site, in light and dark.

    python docs/make_screenshots.py                # every image of the site and the README
    python docs/make_screenshots.py --scene fit    # only some scenes (repeat --scene)
    python docs/make_screenshots.py OUT --all      # every scene, also the extra ones, into OUT

The scenes are listed in docs/screenshot_list.py. Each is drawn offscreen twice, by the same
steps in a new window each time, as ``<scene>-light.webp`` and ``<scene>-dark.webp`` in
docs/site/images/; the README's are copied into docs/images/, so git keeps each picture
once. The site and the README show the one that matches the reader's appearance
(``<picture>`` with ``prefers-color-scheme``).

The sweep is a made-up Landau fan (no measurement data), the random numbers are seeded and
the clock is fixed, so the images change only when the app does. Each image is 1400 × 900,
the export window 1120 × 720 (the same shape), saved as lossless WebP (pixel for pixel the
window, 90 to 150 kB).
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
import tempfile
import time
from collections.abc import Callable
from datetime import datetime
from io import BytesIO
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
from PIL import Image
from PySide6.QtCore import QBuffer, QEventLoop, QIODevice, QTimer
from PySide6.QtWidgets import QApplication, QWidget
from screenshot_list import EXTRA, README, SCHEMES, SITE, SUFFIX, file_names

from mag_opt_detective.core.models import dirac_interband
from mag_opt_detective.gui import controller as controller_module
from mag_opt_detective.gui.controller import SweepFiles
from mag_opt_detective.gui.main_window import MainWindow
from mag_opt_detective.gui.theme import Theme

IMAGES = Path(__file__).resolve().parent / "images"
SITE_IMAGES = Path(__file__).resolve().parent / "site" / "images"
SIZE = (1400, 900)
EXPORT_SIZE = (1120, 720)  # the export window, a little wider than its own: as SIZE, 14 : 9
MAX_BYTES = 200_000  # as tests/test_site.py
CM1_PER_MEV = 8.0656
FIELDS = np.arange(0.25, 16.01, 0.25)
VELOCITY, GAP = 5.4, 12.0  # 10^5 m/s and meV of the made-up Dirac material


class FixedClock(datetime):
    """``datetime.now()`` at a fixed time, so the status bar reads the same in every run."""

    @classmethod
    def now(cls, tz=None):
        return cls(2026, 1, 15, 9, 30)


# ---------------------------------------------------------------------- data
def write_sweep(folder: Path) -> SweepFiles:
    """A Landau fan in OPUS-macro text files: zero field before and after, 0.25 … 16 T."""
    x = np.linspace(350.0, 3200.0, 1400)  # cm-1
    base = 1.0 + 0.25 * np.sin(x / 300.0) + 0.1 * np.cos(x / 41.0)
    rng = np.random.default_rng(1)

    def write(name: str, y: np.ndarray) -> str:
        path = folder / name
        np.savetxt(path, np.column_stack([x, y]), delimiter="\t", fmt="%.8f")
        return str(path)

    zero = (
        write("Demo_4p2K_Sam1_a00p000T_a00p000T.txt", base),
        write("Demo_4p2K_Sam1_a00p000T_a16p000T.txt", 1.02 * base),
    )
    field = []
    for b in FIELDS:
        lines = dirac_interband(np.array([b]), VELOCITY, GAP, 6)[:, 0] * CM1_PER_MEV
        y = np.ones_like(x)
        for i, e in enumerate(lines):
            y -= 0.06 / (i + 1) ** 0.5 * np.exp(-(((x - e) / (18 + 3 * i)) ** 2))
        y *= 1 + 0.003 * rng.standard_normal(x.size)
        whole, frac = divmod(round(b * 1000), 1000)
        name = f"Demo_4p2K_Sam1_a{whole:02d}p{frac:03d}T.txt"
        field.append(write(name, (1 + 0.01 * b) * base * y))
    return SweepFiles(zero, tuple(field))


def pick_points(window: MainWindow) -> None:
    """Three picked curves along the first Landau-level transitions, with a little scatter."""
    c = window.controller
    rng = np.random.default_rng(3)
    for n, (curve, step, start) in enumerate((("LL 1", 3, 6), ("LL 2", 4, 20), ("CR", 6, 30))):
        if c.points is None or curve not in c.points.names:
            c.add_curve(curve)
        c.set_curve(curve)
        b = FIELDS[start::step]
        e = dirac_interband(b, VELOCITY, GAP, 3)[n] * CM1_PER_MEV + rng.normal(0, 4, b.size)
        c.record_points(b, e, unit="cm-1")
    c.set_curve("LL 2")


def show_dirac_model(window: MainWindow) -> None:
    """The Models section's first (Dirac) model with the made-up material's parameters."""
    from mag_opt_detective.gui.inspector import model_state as ms

    models = window.inspector["models"].body_layout().itemAt(0).widget().models
    dirac = next(e for e in models.entries if e.kind == ms.DIRAC)
    p = ms.params(dirac)
    p["velocity"].value, p["delta"].value = VELOCITY, GAP
    dirac.model.n_lines = 3
    models.set_visible(dirac, True)


def wait(condition: Callable[[], bool], timeout: float = 30.0) -> bool:
    end = time.monotonic() + timeout
    while not condition() and time.monotonic() < end:
        loop = QEventLoop()
        QTimer.singleShot(20, loop.quit)
        loop.exec()
    return condition()


def settle(app: QApplication) -> None:
    for _ in range(5):
        app.processEvents()
    wait(lambda: False, 0.3)  # slides, debounced redraws


def step(name: str, action: Callable[[], object]) -> None:
    """Run a part of a scene; one the app no longer supports is reported and skipped."""
    try:
        action()
    except Exception as exc:  # the screenshot is still worth having
        print(f"  skipped {name}: {type(exc).__name__}: {exc}", file=sys.stderr)


# ---------------------------------------------------------------------- scenes
def main_window_scene(window: MainWindow) -> QWidget:
    """The Sample panel, the map with its legend and the inspector."""
    window.show_panel("sample")
    step("legend", window.plot_area.legend_button.click)
    return window


def points_scene(window: MainWindow) -> QWidget:
    """Picking in meV with slim colour scales: the Colour section shows the histogram, and
    View is folded so that the Models header is not cut by the bottom of the inspector."""
    step("slim colour scale", lambda: window.plot_area.scale_style_button.setChecked(True))
    step("unit", lambda: window.toolbar.unit.set_value("meV"))
    window.show_panel("points")
    step("pick tool", lambda: window.tools.set_active("pick"))
    window.inspector["view"].set_expanded(False, animate=False)
    return window


def stacked_scene(window: MainWindow) -> QWidget:
    window.show_panel("processing")
    step("stacked tab", lambda: window.plot_area.tabs.setCurrentIndex(1))
    return window


def library_scene(window: MainWindow) -> QWidget:
    """Two library maps ready to merge by field, the limits of the first one open."""
    c = window.controller

    def two_halves() -> None:
        low = c.save_current_map("Demo_Sam1_0-8T")
        c.update_entry(low, field_cut=(None, 8.0))
        high = c.save_current_map("Demo_Sam1_8-16T")
        c.update_entry(high, field_cut=(8.25, None))
        window.panels["library"].rows[low.key].expand_button.click()

    window.show_panel("library")
    step("library maps", two_halves)
    return window


def reference_scene(window: MainWindow) -> QWidget:
    window.show_panel("reference")
    return window


def models_scene(window: MainWindow) -> QWidget:
    """The Models section on its own: the other inspector sections folded."""
    window.show_panel("points")
    for name, section in window.inspector.items():
        section.set_expanded(name == "models", animate=False)
    return window


def processing_scene(window: MainWindow) -> QWidget:
    """An energy window and a baseline region, processed, with their guides on the map."""
    c = window.controller
    c.set_processing(energy_cut=(420.0, 3100.0), baseline=(2650.0, 2950.0))
    window.commands["process"].trigger()
    window.show_panel("processing")
    return window


def derivative_scene(window: MainWindow) -> QWidget:
    """The 1st derivative along the field per tesla in meV, bipolar with symmetric levels,
    slim colour scales, the side panel closed and only the Colour section open (as in
    :func:`points_scene`)."""
    from mag_opt_detective.core.processing import Axis

    step("unit", lambda: window.toolbar.unit.set_value("meV"))
    window.controller.set_selection(order=1, axis=Axis.FIELD, physical=True)
    window.controller.set_view(colormap="bipolar")
    step("slim colour scale", lambda: window.plot_area.scale_style_button.setChecked(True))
    window.side_panel.set_open(False, animate=False)
    for name, section in window.inspector.items():
        section.set_expanded(name == "colour", animate=False)
    return window


def autopick_scene(window: MainWindow) -> QWidget:
    """Auto-pick in Detect mode: the lines found in a box, LL 2's line chosen."""
    window.show_panel("points")
    tool = window.autopick
    window.tools.set_active("autopick")
    tool.bar.mode.set_value("detect")
    tool.bar.feature.set_value("min")
    tool.detect_in((3.0, 12.0), (800.0, 1700.0))
    return window


def fit_scene(window: MainWindow) -> QWidget:
    """The Dirac model fitted to the three picked curves: the results with their errors."""
    from PySide6.QtWidgets import QScrollArea

    from mag_opt_detective.gui.inspector import model_state as ms

    window.show_panel("points")
    for name, section in window.inspector.items():
        section.set_expanded(name == "models", animate=False)
    models = window.inspector["models"].body_layout().itemAt(0).widget().models
    dirac = next(e for e in models.entries if e.kind == ms.DIRAC)
    p = ms.params(dirac)
    p["velocity"].value, p["delta"].value = 5.0, 9.0  # a start away from the made-up material
    models.cards[dirac].refresh()
    models.draw()
    card = models.cards[dirac]
    card.fit_button.click()
    card.fit_area.fit_button.click()
    wait(lambda: models.result(dirac) is not None)
    settle(QApplication.instance())
    scroll = window.inspector_panel.findChild(QScrollArea)
    scroll.ensureWidgetVisible(card.fit_area, 0, 0)
    return window


def export_scene(window: MainWindow) -> QWidget:
    """The export window, at the shape of the other screenshots."""
    window.commands["export_figure"].trigger()
    dialog = window.export_dialog
    dialog.resize(*EXPORT_SIZE)
    settle(QApplication.instance())
    wait(lambda: dialog.preview.image() is not None and dialog.is_idle())
    return dialog


# name -> scene, in the order of screenshot_list
SCENES: dict[str, Callable[[MainWindow], QWidget]] = {
    "main-window": main_window_scene,
    "points": points_scene,
    "export-window": export_scene,
    "processing": processing_scene,
    "stacked": stacked_scene,
    "derivative": derivative_scene,
    "autopick": autopick_scene,
    "models": models_scene,
    "fit": fit_scene,
    "library": library_scene,
    "reference": reference_scene,
}
assert tuple(SCENES) == SITE + EXTRA, "SCENES and screenshot_list.py name other scenes"


def render(app, theme, sweep: SweepFiles, scene, scheme: str, path: Path) -> None:
    theme.set_scheme(scheme)
    window = MainWindow(theme=theme)
    try:
        window.resize(*SIZE)
        window.controller.set_processing(sample_files=sweep)
        window.show()
        settle(app)
        window.commands["process"].trigger()
        step("picked points", lambda: pick_points(window))
        step("Dirac model", lambda: show_dirac_model(window))
        widget = scene(window)
        settle(app)
        save_image(widget, path)
    finally:
        dialog = getattr(window, "export_dialog", None)
        if dialog is not None:
            dialog.close()
        window.close()
        window.deleteLater()
        app.processEvents()


def save_image(widget: QWidget, path: Path) -> None:
    """Grab *widget* and write it as a lossless WebP at the smallest size (slow but exact)."""
    buffer = QBuffer()
    buffer.open(QIODevice.OpenModeFlag.WriteOnly)
    widget.grab().save(buffer, "PNG")
    image = Image.open(BytesIO(bytes(buffer.data()))).convert("RGB")
    image.save(path, "WEBP", lossless=True, quality=100, method=6)
    size = path.stat().st_size
    note = "" if size <= MAX_BYTES else f" (over {MAX_BYTES // 1000} kB)"
    print(f"{path} {image.width}x{image.height} {size // 1000} kB{note}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("out", nargs="?", type=Path, help="folder (default: the site's)")
    choice = parser.add_mutually_exclusive_group()
    choice.add_argument("--all", action="store_true", help="every scene, also the extra ones")
    choice.add_argument(
        "--scene", action="append", choices=SCENES, help="only this scene (repeatable)"
    )
    args = parser.parse_args(argv)
    scenes = args.scene or list(SCENES if args.all else SITE)
    if args.out is None and not set(scenes) <= set(SITE):
        parser.error("the site shows only its own scenes: give a folder for the others")
    out = args.out or SITE_IMAGES
    out.mkdir(parents=True, exist_ok=True)
    controller_module.datetime = FixedClock  # "Processed 09:30"
    app = QApplication.instance() or QApplication(sys.argv[:1])
    app.setApplicationName("Magneto-Optical Detective")
    theme = Theme("light")
    theme.apply(app)
    with tempfile.TemporaryDirectory(prefix="mag-opt-docs-") as tmp:
        sweep = write_sweep(Path(tmp))
        for name in scenes:
            for scheme in SCHEMES:
                path = out / f"{name}-{scheme}{SUFFIX}"
                render(app, theme, sweep, SCENES[name], scheme, path)
    if args.out is None:  # the README's copies of the site's images
        IMAGES.mkdir(exist_ok=True)
        for name in file_names(scene for scene in README if scene in scenes):
            shutil.copyfile(out / name, IMAGES / name)
            print(f"{IMAGES / name} (copy)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
