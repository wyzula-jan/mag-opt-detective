"""Screenshots of the app for the README, drawn offscreen from a synthetic measurement.

    python docs/make_screenshots.py              # the README images, into docs/images/
    python docs/make_screenshots.py OUT --all    # every scene in light and dark, into OUT

The sweep is a made-up Landau fan (no measurement data), the random numbers are seeded and
the clock is fixed, so the images change only when the app does. Each image is 1400 × 900
(the export window: its own size), saved as an optimised RGB PNG (under 400 kB).
"""

from __future__ import annotations

import argparse
import os
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

from mag_opt_detective.core.models import dirac_interband
from mag_opt_detective.gui import controller as controller_module
from mag_opt_detective.gui.controller import SweepFiles
from mag_opt_detective.gui.main_window import MainWindow
from mag_opt_detective.gui.theme import Theme

IMAGES = Path(__file__).resolve().parent / "images"
SIZE = (1400, 900)
MAX_BYTES = 400_000
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
    window.show_panel("sample")
    return window


def points_scene(window: MainWindow) -> QWidget:
    step("slim colour scale", lambda: window.plot_area.scale_style_button.setChecked(True))
    step("unit", lambda: window.toolbar.unit.set_value("meV"))
    window.show_panel("points")
    step("pick tool", lambda: window.tools.set_active("pick"))
    return window


def stacked_scene(window: MainWindow) -> QWidget:
    window.show_panel("processing")
    step("stacked tab", lambda: window.plot_area.tabs.setCurrentIndex(1))
    return window


def library_scene(window: MainWindow) -> QWidget:
    step("library map", lambda: window.controller.save_current_map())
    window.show_panel("library")
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


def export_scene(window: MainWindow) -> QWidget:
    window.commands["export_figure"].trigger()
    dialog = window.export_dialog
    wait(lambda: dialog.preview.image() is not None and dialog.is_idle())
    return dialog


# name -> (scene, colour scheme in the README)
SCENES: dict[str, tuple[Callable[[MainWindow], QWidget], str]] = {
    "main-window": (main_window_scene, "light"),
    "points-dark": (points_scene, "dark"),
    "export-window": (export_scene, "light"),
}
# more for the documentation (--all)
EXTRA_SCENES = {
    "reference": reference_scene,
    "stacked": stacked_scene,
    "library": library_scene,
    "models": models_scene,
}


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
        save_png(widget, path)
    finally:
        dialog = getattr(window, "export_dialog", None)
        if dialog is not None:
            dialog.close()
        window.close()
        window.deleteLater()
        app.processEvents()


def save_png(widget: QWidget, path: Path) -> None:
    """Grab *widget* and write it as an optimised RGB PNG."""
    buffer = QBuffer()
    buffer.open(QIODevice.OpenModeFlag.WriteOnly)
    widget.grab().save(buffer, "PNG")
    image = Image.open(BytesIO(bytes(buffer.data()))).convert("RGB")
    image.save(path, optimize=True)
    size = path.stat().st_size
    note = "" if size <= MAX_BYTES else f" (over {MAX_BYTES // 1000} kB)"
    print(f"{path} {image.width}x{image.height} {size // 1000} kB{note}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("out", nargs="?", type=Path, default=IMAGES)
    parser.add_argument("--all", action="store_true", help="every scene, light and dark")
    args = parser.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)
    controller_module.datetime = FixedClock  # "Processed 09:30"
    app = QApplication.instance() or QApplication(sys.argv[:1])
    app.setApplicationName("Magneto-Optical Detective")
    theme = Theme("light")
    theme.apply(app)
    with tempfile.TemporaryDirectory(prefix="mag-opt-docs-") as tmp:
        sweep = write_sweep(Path(tmp))
        if args.all:
            scenes = {name: scene for name, (scene, _scheme) in SCENES.items()}
            for name, scene in {**scenes, **EXTRA_SCENES}.items():
                base = name.removesuffix("-dark")
                for scheme in ("light", "dark"):
                    render(app, theme, sweep, scene, scheme, args.out / f"{base}-{scheme}.png")
        else:
            for name, (scene, scheme) in SCENES.items():
                render(app, theme, sweep, scene, scheme, args.out / f"{name}.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
