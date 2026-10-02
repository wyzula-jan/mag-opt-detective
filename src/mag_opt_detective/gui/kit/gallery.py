"""Developer gallery of the kit widgets: ``python -m mag_opt_detective.gui.kit.gallery``.

``--scheme light|dark|system`` picks the theme, ``--screenshot out.png`` renders the window
(offscreen works: ``QT_QPA_PLATFORM=offscreen``) and exits.
"""

from __future__ import annotations

import argparse
import sys

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QApplication,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QPushButton,
    QScrollArea,
    QSplitter,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from mag_opt_detective.gui import icons
from mag_opt_detective.gui.kit import (
    CollapsibleSection,
    InfoBar,
    RangeControl,
    RangeSlider,
    SegmentedControl,
    SlidePanel,
    Switch,
)
from mag_opt_detective.gui.theme import SCHEMES, Theme


def _segmented(options, *, size="md", expand=False, name="") -> SegmentedControl:
    control = SegmentedControl(size=size, expand=expand)
    for option in options:
        control.add_option(*option)
    control.setAccessibleName(name)
    return control


def _label(text: str, muted: bool = False) -> QLabel:
    label = QLabel(text)
    label.setWordWrap(True)
    if muted:
        label.setProperty("kit", "muted")
    return label


def _tool(name: str, tip: str, kit: str = "tool", checkable: bool = False) -> QToolButton:
    button = QToolButton()
    button.setProperty("kit", kit)
    button.setCheckable(checkable)
    button.setToolTip(tip)
    button.setAccessibleName(tip)
    icons.set_icon(button, name, "muted", on_color="accent")
    return button


class Gallery(QMainWindow):
    """Every kit widget in a layout like the redesign mockup."""

    def __init__(self, theme: Theme):
        super().__init__()
        self.theme = theme
        self.setWindowTitle("Kit gallery")

        self.left = SlidePanel(self._side_panel(), 280)
        self.inspector = SlidePanel(self._inspector(), 300)
        self.log = SlidePanel(self._log(), 150)
        self.stage = QSplitter(Qt.Orientation.Vertical)
        self.stage.addWidget(self._stage())
        self.stage.addWidget(self.log)
        self.stage.setStretchFactor(0, 1)
        self.body = QSplitter(Qt.Orientation.Horizontal)
        self.body.addWidget(self.left)
        self.body.addWidget(self.stage)
        self.body.addWidget(self.inspector)
        self.body.setStretchFactor(1, 1)

        central = QWidget()
        layout = QVBoxLayout(central)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        layout.addWidget(self._toolbar())
        row = QHBoxLayout()
        row.setSpacing(0)
        row.addWidget(self._rail())
        row.addWidget(self.body, 1)
        layout.addLayout(row, 1)
        self.setCentralWidget(central)

    # --- areas -------------------------------------------------------------------------
    def _toolbar(self) -> QWidget:
        bar = QWidget()
        rows = QVBoxLayout(bar)
        rows.setContentsMargins(10, 8, 10, 8)
        rows.setSpacing(8)
        layout = QHBoxLayout()
        layout.setSpacing(12)
        rows.addLayout(layout)
        open_button = QPushButton("Open sweep…")
        icons.set_icon(open_button, "folder-open")
        process = QPushButton("Process")
        process.setProperty("kit", "primary")
        icons.set_icon(process, "play", "accent-fg")
        layout.addWidget(open_button)
        layout.addWidget(process)
        layout.addWidget(_label("Plot", muted=True))
        kinds = [
            ("ratio", "R(B)/R(0)"),
            ("data", "Data"),
            ("average", "R(B)/R(B-AVR)"),
            ("step", "R(B)/R(B-ΔB)"),
        ]
        layout.addWidget(_segmented(kinds, name="Plot kind"))
        layout.addStretch(1)
        scheme = SegmentedControl()
        scheme.setAccessibleName("Appearance")
        for value, icon_name in zip(SCHEMES, ("contrast", "sun", "moon"), strict=True):
            scheme.add_option(value, "", f"Appearance: {value}", icon_name)
        scheme.set_value(self.theme.scheme())
        scheme.valueChanged.connect(self.theme.set_scheme)
        layout.addWidget(scheme)
        layout = QHBoxLayout()
        layout.setSpacing(12)
        rows.addLayout(layout)
        layout.addWidget(_label("Derivative", muted=True))
        order = _segmented([("0", "Off"), ("1", "1st"), ("2", "2nd")], name="Derivative")
        axis = _segmented([("E", "d/dE"), ("B", "d/dB")], name="Derivative axis")
        order.valueChanged.connect(lambda v: [axis.set_option_enabled(o, v != "0") for o in "EB"])
        order.set_value("1")
        layout.addWidget(order)
        layout.addWidget(axis)
        layout.addWidget(_label("Unit", muted=True))
        layout.addWidget(
            _segmented([("cm-1", "cm⁻¹"), ("meV", "meV"), ("THz", "THz")], name="Unit")
        )
        layout.addStretch(1)
        return bar

    def _rail(self) -> QWidget:
        rail = QWidget()
        layout = QVBoxLayout(rail)
        layout.setContentsMargins(6, 8, 6, 8)
        layout.setSpacing(4)
        names = [
            ("activity", "Sample"),
            ("layers", "Reference"),
            ("sliders-horizontal", "Process"),
            ("library-big", "Library"),
            ("chart-scatter", "Points"),
        ]
        for i, (icon_name, text) in enumerate(names):
            button = _tool(icon_name, text, kit="rail", checkable=True)
            button.setText(text)
            button.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextUnderIcon)
            button.setChecked(i == 0)
            if i == 0:
                button.toggled.connect(lambda on: self.left.set_open(on))
                self.left.openChanged.connect(button.setChecked)
            layout.addWidget(button)
        layout.addStretch(1)
        return rail

    def _side_panel(self) -> QWidget:
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(14, 12, 14, 12)
        layout.setSpacing(12)
        title = _label("<b>Sample</b>")
        layout.addWidget(title)
        layout.addWidget(_label("64 spectra · 0.25 – 16 T · text export", muted=True))
        layout.addWidget(
            _segmented(
                [("names", "From file names"), ("custom", "Custom range")],
                size="sm",
                expand=True,
                name="Field values",
            )
        )
        smooth = Switch("Smooth the reference")
        smooth.setChecked(True)
        layout.addWidget(smooth)
        layout.addWidget(Switch("Energy window"))
        off = Switch("Disabled switch")
        off.setEnabled(False)
        layout.addWidget(off)
        layout.addWidget(_label("RangeSlider (muted = Auto)", muted=True))
        slider = RangeSlider()
        slider.setAccessibleName("Example range")
        slider.set_extent(0, 100)
        slider.set_values(20, 70)
        layout.addWidget(slider)
        muted = RangeSlider()
        muted.setAccessibleName("Muted range")
        muted.set_extent(0, 100)
        muted.set_values(0, 100)
        muted.set_muted(True)
        layout.addWidget(muted)
        layout.addStretch(1)
        return panel

    def _stage(self) -> QWidget:
        stage = QWidget()
        layout = QVBoxLayout(stage)
        layout.setContentsMargins(12, 8, 12, 12)
        layout.setSpacing(10)
        tools = QHBoxLayout()
        tools.setSpacing(2)
        for name, tip in [("move", "Pan and zoom"), ("zoom-in", "Box zoom"), ("crosshair", "Pick")]:
            button = _tool(name, tip, checkable=True)
            button.setChecked(name == "move")
            tools.addWidget(button)
        tools.addSpacing(8)
        for name, tip in [("scan", "Fit to data"), ("image", "Save image")]:
            tools.addWidget(_tool(name, tip))
        panels = _tool("panel-right", "Inspector", checkable=True)
        panels.setChecked(True)
        panels.toggled.connect(lambda on: self.inspector.set_open(on))
        self.inspector.openChanged.connect(panels.setChecked)
        log = _tool("terminal", "Log", checkable=True)
        log.toggled.connect(lambda on: self.log.set_open(on))
        self.log.openChanged.connect(log.setChecked)
        tools.addStretch(1)
        tools.addWidget(panels)
        tools.addWidget(log)
        layout.addLayout(tools)

        for level, title, text, action in [
            ("error", "Can't process: the reference sweep has no files.", "Add them.", "Open"),
            ("warning", "Fields are not evenly spaced.", "Two steps differ from 0.25 T.", None),
            ("info", "Saved 64 spectra.", "", None),
        ]:
            bar = InfoBar()
            bar.show_message(level, title, text, action, (lambda: None) if action else None)
            layout.addWidget(bar)

        grid = QGridLayout()
        grid.setSpacing(4)
        for i, name in enumerate(icons.names()):
            label = QLabel()
            label.setPixmap(icons.pixmap(name, 20, None, self.devicePixelRatioF()))
            label.setToolTip(name)
            grid.addWidget(label, i // 13, i % 13)
        layout.addLayout(grid)
        layout.addStretch(1)
        return stage

    def _inspector(self) -> QWidget:
        content = QWidget()
        layout = QVBoxLayout(content)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        head = QWidget()
        head_layout = QVBoxLayout(head)
        head_layout.setContentsMargins(14, 12, 14, 10)
        head_layout.addWidget(_label("<b>Map</b>"))
        head_layout.addWidget(_label("Settings for the plot on screen", muted=True))
        layout.addWidget(head)

        view = CollapsibleSection("View")
        field = RangeControl("Field <i>B</i>", "T", name="Field")
        field.set_extent(0.25, 16)
        field.set_range(0.25, 16)
        energy = RangeControl("Energy <i>E</i>", "cm⁻¹", name="Energy")
        energy.set_extent(100, 4000, "Data 100 – 4000 cm⁻¹ · shared with Stacked")
        energy.set_range(400, 2200)
        energy.set_auto(False)
        intensity = RangeControl("Intensity", "", name="Intensity")
        intensity.set_extent(0.8, 1.3)
        intensity.set_range(0.9, 1.2)
        intensity.set_auto(False)
        intensity.lo_spin.setValue(1.25)  # shows the inline error
        for control in (field, energy, intensity):
            view.body_layout().addWidget(control)
        layout.addWidget(view)

        sub = _label("Ratio · 0th", muted=True)
        colour = CollapsibleSection("Colour", trailing=sub)
        colour.body_layout().addWidget(
            _segmented(
                [("auto", "Auto"), ("fixed", "Fixed"), ("sym", "Symmetric")],
                size="sm",
                expand=True,
                name="Colour range",
            )
        )
        colour.body_layout().addWidget(Switch("Slim colour bar"))
        layout.addWidget(colour)
        traces = CollapsibleSection("Traces", expanded=False)
        traces.body_layout().addWidget(_label("Offset, every n-th spectrum", muted=True))
        layout.addWidget(traces)
        layout.addStretch(1)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QScrollArea.Shape.NoFrame)
        scroll.setWidget(content)
        return scroll

    def _log(self) -> QWidget:
        log = QWidget()
        layout = QVBoxLayout(log)
        layout.setContentsMargins(12, 6, 12, 6)
        layout.addWidget(_label("<b>Log</b>"))
        layout.addWidget(_label("12:04:31  Processed 64 spectra in 0.4 s", muted=True))
        layout.addStretch(1)
        return log


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scheme", choices=SCHEMES, default="system")
    parser.add_argument("--size", default="1400x900", help="window size, e.g. 1100x800")
    parser.add_argument("--screenshot", help="save a PNG of the window and exit")
    parser.add_argument("--log-open", action="store_true", help="start with the log open")
    args = parser.parse_args(argv)

    app = QApplication.instance() or QApplication(sys.argv[:1])
    theme = Theme(args.scheme)
    theme.apply(app)
    window = Gallery(theme)
    width, height = (int(v) for v in args.size.lower().split("x"))
    window.resize(width, height)
    window.log.set_open(args.log_open, animate=False)
    window.show()
    app.processEvents()
    if args.screenshot:
        app.processEvents()
        window.grab().save(args.screenshot)
        return 0
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
