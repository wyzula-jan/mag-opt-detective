"""Main window: wires the panels to the processing core."""

from __future__ import annotations

import functools
import logging
import platform
from pathlib import Path

import numpy as np
import pyqtgraph as pg
import scipy
from PySide6 import __version__ as pyside_version
from PySide6.QtCore import QSettings, Qt, qVersion
from PySide6.QtGui import QAction, QKeySequence
from PySide6.QtWidgets import QMainWindow, QMessageBox, QSplitter

from mag_opt_detective import __version__
from mag_opt_detective.core.models import dirac_interband
from mag_opt_detective.core.pipeline import (
    PlotKind,
    ProcessOptions,
    ProcessResult,
    ReferenceMode,
    process,
)
from mag_opt_detective.core.points import PointTable
from mag_opt_detective.core.processing import (
    average_maps,
    crop_energy,
    crop_field,
    merge_energy,
    merge_field,
)
from mag_opt_detective.core.readers import Measurement, load_measurement
from mag_opt_detective.core.spectra import FieldMap, load_tsv, save_tsv
from mag_opt_detective.core.units import Unit, convert
from mag_opt_detective.gui.console import QtLogHandler
from mag_opt_detective.gui.data_panel import DataPanel
from mag_opt_detective.gui.measurement_tab import MeasurementTab
from mag_opt_detective.gui.plot_panel import PlotPanel
from mag_opt_detective.gui.settings import Persistence
from mag_opt_detective.gui.tools_tab import PointMode, level_key
from mag_opt_detective.gui.widgets import (
    IMAGE_FILTER,
    last_dir,
    open_file,
    save_file,
    set_last_dir,
)

logger = logging.getLogger("mag_opt_detective")

EXPORT_NAMES = {
    PlotKind.RATIO: "Ratio",
    PlotKind.DATA: "Data",
    PlotKind.AVERAGE: "Ratio_AVR",
    PlotKind.STEP: "Ratio_Step",
}
ORDER_SUFFIX = {0: "", 1: "_1stDer", 2: "_2ndDer"}
PER_UNIT_SUFFIX = "_perUnit"

SHORTCUTS = [
    ("Ctrl+F", "Process"),
    ("Ctrl+E", "Export current plot"),
    ("Ctrl+Shift+E", "Save the visible plot as an image"),
    ("Ctrl+L / Ctrl+Shift+L", "Load sample field / zero-field files"),
    ("Ctrl+R / Ctrl+Shift+R", "Load reference field / zero-field files"),
    ("Ctrl+1 / 2 / 3 / 4", "Plot R(B)/R(0) / Data / R(B)/R(B-AVR) / R(B)/R(B-ΔB)"),
    ("Alt+1 / 2 / 3", "No / 1st / 2nd derivative"),
]


def user_action(title: str):
    """Run a slot, logging and reporting expected errors instead of crashing."""

    def decorator(method):
        @functools.wraps(method)
        def wrapper(self, *args, **kwargs):
            try:
                return method(self, *args, **kwargs)
            except (ValueError, OSError, KeyError) as exc:
                self.report_error(title, str(exc))
            except Exception as exc:
                logger.exception("%s failed", title)
                self.report_error(title, f"Unexpected error: {exc!r}")
            return None

        return wrapper

    return decorator


class MainWindow(QMainWindow):
    """The application window.

    *settings*: where choices are remembered between sessions; None keeps nothing
    (used by tests). The application passes the user's settings store.
    """

    def __init__(self, parent=None, settings: QSettings | None = None):
        super().__init__(parent)
        self.setWindowTitle(f"Magneto-Optical Detective {__version__}")
        self.resize(1400, 900)

        self.result: ProcessResult | None = None
        self.points: PointTable | None = None
        self.slots: dict[int, FieldMap] = {}

        self.data_panel = DataPanel()
        self.plot_panel = PlotPanel()
        self.main_splitter = QSplitter(Qt.Orientation.Horizontal)
        self.main_splitter.addWidget(self.data_panel)
        self.main_splitter.addWidget(self.plot_panel)
        self.main_splitter.setStretchFactor(0, 0)
        self.main_splitter.setStretchFactor(1, 1)
        self.main_splitter.setSizes([460, 940])
        self.setCentralWidget(self.main_splitter)
        self.statusBar()

        self._log_handler = QtLogHandler(self.data_panel.console)
        logger.addHandler(self._log_handler)
        logger.setLevel(logging.INFO)

        self.limits_page = self.data_panel.tools.limits_page
        self.corrections = self.data_panel.tools.corrections_page
        self.models_page = self.data_panel.tools.models_page
        self.point_model = self.corrections.model

        self.persistence = Persistence(settings) if settings is not None else None
        self.geometry_restored = False
        self._create_actions()
        self._connect()
        if self.persistence is not None:
            self._bind_settings()
            self.restore_settings()

    # ------------------------------------------------------------------ setup
    def _action(self, text: str, slot, shortcut: str | QKeySequence | None = None) -> QAction:
        action = QAction(text, self)
        if shortcut is not None:
            action.setShortcut(QKeySequence(shortcut))
        action.triggered.connect(lambda _checked=False: slot())
        self.addAction(action)
        return action

    def _create_actions(self) -> None:
        dp = self.data_panel
        self.process_action = self._action("&Process", self.process_data, "Ctrl+F")
        self.export_action = self._action("&Export Current Plot…", self.export_current, "Ctrl+E")
        self.image_action = self._action("Save Plot &Image…", self.save_image, "Ctrl+Shift+E")
        load_actions = [
            self._action("Load Sample Field…", dp.sample.load_field_dialog, "Ctrl+L"),
            self._action("Load Sample Zero Field…", dp.sample.load_zero_dialog, "Ctrl+Shift+L"),
            self._action("Load Reference Field…", dp.reference.load_field_dialog, "Ctrl+R"),
            self._action(
                "Load Reference Zero Field…", dp.reference.load_zero_dialog, "Ctrl+Shift+R"
            ),
        ]
        quit_action = self._action("&Quit", self.close, QKeySequence.StandardKey.Quit)

        file_menu = self.menuBar().addMenu("&File")
        for action in load_actions:
            file_menu.addAction(action)
        file_menu.addSeparator()
        file_menu.addAction(self.process_action)
        file_menu.addAction(self.export_action)
        file_menu.addAction(self.image_action)
        file_menu.addSeparator()
        file_menu.addAction(quit_action)

        view_menu = self.menuBar().addMenu("&View")
        view_menu.addAction(self._action("&Center Window", self.center_on_screen))
        reset = self._action("&Reset Settings", self.reset_settings)
        reset.setEnabled(self.persistence is not None)
        view_menu.addAction(reset)

        help_menu = self.menuBar().addMenu("&Help")
        help_menu.addAction(self._action("&Shortcuts", self.show_shortcuts))
        help_menu.addAction(self._action("&About", self.show_about))

    def _connect(self) -> None:
        dp, pp = self.data_panel, self.plot_panel
        dp.process_button.clicked.connect(self.process_action.trigger)
        dp.export_button.clicked.connect(self.export_action.trigger)

        pp.selectionChanged.connect(self.replot)
        pp.referenceSelectionChanged.connect(self.plot_reference)
        self.limits_page.changed.connect(self.replot_all)
        self.models_page.changed.connect(self.update_model_overlay)
        pp.color_map.pointClicked.connect(self.on_point_clicked)
        pp.color_map.levelsEdited.connect(self.on_levels_edited)
        pp.reference_map.levelsEdited.connect(self.on_reference_levels_edited)

        c = self.corrections
        c.show_all_button.toggled.connect(self.update_point_markers)
        c.column_name.editingFinished.connect(self.update_point_markers)
        c.drop_button.clicked.connect(self.drop_curve)
        c.load_button.clicked.connect(self.load_points)
        c.export_button.clicked.connect(self.export_points)

        pt = dp.processed
        pt.loadRequested.connect(self.load_slot)
        pt.saveRequested.connect(self.save_slot)
        pt.plotRequested.connect(self.plot_slot)
        pt.mergeEnergyRequested.connect(self.merge_slots)
        pt.mergeFieldRequested.connect(self.merge_slots_by_field)
        pt.averageRequested.connect(self.average_slots)

    # ------------------------------------------------------------------ settings
    def _bind_settings(self) -> None:
        bind = self.persistence.bind
        dp, pp = self.data_panel, self.plot_panel
        bind("data/field_source", dp.field_source)
        bind("data/unit", dp.unit)
        bind("data/export_type_suffix", dp.export_type_suffix)
        for name, tab in (("sample", dp.sample), ("reference", dp.reference)):
            box = tab.field_range
            for part in ("start", "step", "end"):
                bind(f"{name}/field_{part}", getattr(box, part))
        ref = dp.reference
        for attr in ("ref_none", "ref_separate", "ref_self", "smooth", "sg_window", "sg_poly"):
            bind(f"reference/{attr}", getattr(ref, attr))

        bind("tools/page", dp.tools.page_combo)
        lp = self.limits_page
        for attr in (
            "field_auto", "field_custom", "field_min", "field_max",
            "energy_auto", "energy_custom", "energy_cut", "energy_min", "energy_max",
            "level_auto", "level_custom",
            "stacked_auto", "stacked_custom", "stacked_min", "stacked_max",
        ):  # fmt: skip
            bind(f"limits/{attr}", getattr(lp, attr))
        for key, (lo, hi) in lp.level_edits.items():
            bind(f"limits/levels_{key}_min", lo)
            bind(f"limits/levels_{key}_max", hi)
        c = self.corrections
        for attr in ("baseline_off", "baseline_on", "baseline_min", "baseline_max", "column_name"):
            bind(f"corrections/{attr}", getattr(c, attr))
        mp = self.models_page
        bind("models/show_dirac", mp.show_dirac)
        bind("models/velocity", mp.velocity.spin)
        bind("models/delta", mp.delta.spin)
        bind("models/n_lines", mp.n_lines)
        bind("processed/full_energy", dp.processed.full_energy)
        bind("processed/auto_field", dp.processed.auto_field)

        for kind, button in pp.kind_buttons.items():
            bind(f"plot/kind_{kind}", button)
        for order, button in pp.order_buttons.items():
            bind(f"plot/order_{order}", button)
        bind("plot/axis", pp.axis_combo)
        bind("plot/per_unit", pp.per_unit)
        bind("plot/colours", pp.cmap_combo)
        bind("plot/stacked_enabled", pp.stacked_enabled)
        bind("plot/offset", pp.offset)
        bind("plot/reference_ratio", pp.ref_ratio)
        bind("plot/reference_data", pp.ref_data)

    def restore_settings(self) -> None:
        p = self.persistence
        p.restore()
        if (geometry := p.bytes_value("window/geometry")) is not None:
            self.geometry_restored = self.restoreGeometry(geometry)
        for key, splitter in (
            ("window/main_splitter", self.main_splitter),
            ("window/data_splitter", self.data_panel.splitter),
        ):
            if (state := p.bytes_value(key)) is not None:
                splitter.restoreState(state)
        folder = p.value("files/last_dir")
        if isinstance(folder, str) and Path(folder).is_dir():
            set_last_dir(folder)

    def save_settings(self) -> None:
        p = self.persistence
        if p is None:
            return
        p.set_value("window/geometry", self.saveGeometry())
        p.set_value("window/main_splitter", self.main_splitter.saveState())
        p.set_value("window/data_splitter", self.data_panel.splitter.saveState())
        p.set_value("files/last_dir", last_dir())
        p.save()

    def reset_settings(self) -> None:
        if self.persistence is None:
            return
        self.persistence.reset()
        self.main_splitter.setSizes([460, 940])
        self.resize(1400, 900)
        self.center_on_screen()
        logger.info("Settings reset to the defaults.")

    # ------------------------------------------------------------------ helpers
    def report_error(self, title: str, message: str) -> None:
        logger.error("%s: %s", title, message)
        QMessageBox.warning(self, title, message)

    def _load(self, tab: MeasurementTab, energy_cut) -> Measurement:
        dp = self.data_panel
        return load_measurement(
            tab.zero_paths(),
            tab.field_paths(),
            field=tab.field_range.field() if dp.custom_field() else None,
            unit=dp.energy_unit(),
            energy_limits=energy_cut,
        )

    def _set_result(self, result: ProcessResult) -> None:
        self.result = result
        self._init_points_if_requested(result.ratio.field)
        self.replot_all()

    # ------------------------------------------------------------------ processing
    @user_action("Process")
    def process_data(self) -> None:
        dp = self.data_panel
        limits = self.limits_page.limits()
        ref_tab = dp.reference
        mode = ref_tab.reference_mode()
        options = ProcessOptions(
            reference_mode=mode,
            smooth_reference=ref_tab.smooth.isChecked(),
            sg_window=ref_tab.sg_window.value(),
            sg_poly=ref_tab.sg_poly.value(),
            baseline_region=self.corrections.baseline_region(),
        )
        logger.info("-" * 40)
        sample = self._load(dp.sample, limits.energy_cut)
        spectra = sample.spectra
        logger.info(
            "Sample: %d spectra, B = %g … %g T, %d zero-field file(s), E = %.4g … %.4g %s",
            spectra.field.size,
            spectra.field.min(),
            spectra.field.max(),
            sample.zero.shape[1],
            spectra.energy[0],
            spectra.energy[-1],
            spectra.unit,
        )
        reference = None
        if mode is ReferenceMode.SEPARATE:
            reference = self._load(ref_tab, limits.energy_cut)
            logger.info(
                "Reference: %d spectra interpolated onto the sample field",
                reference.spectra.field.size,
            )
        elif mode is ReferenceMode.SELF:
            logger.info("Using the data itself as reference.")
        if options.smooth_reference and mode is not ReferenceMode.NONE:
            logger.info(
                "Reference smoothed (SG window %d, order %d).", options.sg_window, options.sg_poly
            )
        result = process(sample, reference, options)
        if options.baseline_region is not None:
            lo, hi = options.baseline_region
            logger.info("Baseline corrected in range %g – %g %s.", lo, hi, spectra.unit)
        self._set_result(result)

    # ------------------------------------------------------------------ plotting
    def replot_all(self) -> None:
        self.replot()
        self.plot_reference()

    @user_action("Plot")
    def replot(self) -> None:
        if self.result is None:
            return
        pp = self.plot_panel
        limits = self.limits_page.limits()
        kind, order, physical = pp.kind(), pp.order(), pp.physical()
        fmap = self.result.get(kind, order, pp.axis(), physical=physical)
        # the derivative limits in Tools are per data point; per-unit maps autoscale
        levels = None if physical and order else limits.levels_for(kind, order)
        pp.color_map.set_map(
            fmap,
            levels=levels,
            cmap=pp.colormap(order),
            x_range=limits.field_range,
            y_range=limits.energy_view,
        )
        self.update_point_markers()
        self.update_model_overlay()
        if pp.stacked_enabled.isChecked():
            pp.stacked.set_map(
                fmap, pp.offset.value(), y_range=limits.stacked_range, x_range=limits.energy_view
            )
        else:
            pp.stacked.clear_map()

    def update_model_overlay(self) -> None:
        cmap = self.plot_panel.color_map
        model = self.models_page.dirac()
        if model is None or self.result is None:
            cmap.set_model_curves(np.array([]), None)
            return
        shown = self.result.ratio
        lo = max(0.0, float(shown.field.min()))
        field = np.linspace(lo, float(shown.field.max()), 300)
        lines_mev = dirac_interband(field, model.velocity, model.delta, model.n_lines)
        cmap.set_model_curves(field, convert(lines_mev, Unit.MEV, shown.unit))

    @user_action("Plot reference")
    def plot_reference(self) -> None:
        pp = self.plot_panel
        result = self.result
        if result is None or result.reference_data is None or result.reference_ratio is None:
            pp.reference_map.clear_map()
            return
        limits = self.limits_page.limits()
        kind = pp.reference_kind()
        fmap = result.reference_data if kind is PlotKind.DATA else result.reference_ratio
        pp.reference_map.set_map(
            fmap,
            levels=limits.levels_for(kind),
            cmap=pp.colormap(0),
            x_range=limits.field_range,
            y_range=limits.energy_view,
        )

    def on_levels_edited(self, lo: float, hi: float) -> None:
        pp = self.plot_panel
        if pp.physical() and pp.order():
            return  # per-unit derivatives are always autoscaled
        self._store_levels(level_key(pp.kind(), pp.order()), lo, hi)

    def on_reference_levels_edited(self, lo: float, hi: float) -> None:
        self._store_levels(level_key(self.plot_panel.reference_kind()), lo, hi)

    def _store_levels(self, key: str, lo: float, hi: float) -> None:
        self.limits_page.set_levels(key, lo, hi)
        logger.info(
            "Colour range of %s set to %.4g … %.4g", self.limits_page.level_label(key), lo, hi
        )

    # ------------------------------------------------------------------ export
    @user_action("Export")
    def export_current(self) -> None:
        if self.result is None:
            raise ValueError("nothing to export - process data first")
        pp = self.plot_panel
        kind, order, physical = pp.kind(), pp.order(), pp.physical()
        fmap = self.result.get(kind, order, pp.axis(), physical=physical)
        name = (
            EXPORT_NAMES[kind]
            + ORDER_SUFFIX[order]
            + (PER_UNIT_SUFFIX if physical and order else "")
        )
        path = save_file(self, "Export current plot")
        if not path:
            return
        out = Path(path)
        if not out.suffix:
            out = out.with_suffix(".csv")
        if self.data_panel.export_type_suffix.isChecked():
            out = out.with_name(f"{out.stem}_{name}{out.suffix}")
        save_tsv(fmap, out)
        logger.info("Exported %s to %s", name, out)

    @user_action("Save image")
    def save_image(self) -> None:
        if self.result is None:
            raise ValueError("nothing to save - process data first")
        view = self.plot_panel.current_view()
        path = save_file(self, "Save plot image", IMAGE_FILTER)
        if not path:
            return
        out = Path(path) if Path(path).suffix else Path(path).with_suffix(".png")
        view.export_image(out)
        logger.info("Saved image %s", out)

    # ------------------------------------------------------------------ points
    def _init_points_if_requested(self, field: np.ndarray) -> None:
        c = self.corrections
        if not c.init_table.isChecked() and self.points is not None:
            return
        self.points = PointTable(field)
        name = c.column_name.text().strip()
        if name:
            self.points.add_column(name)
        self.point_model.set_table(self.points)
        c.init_table.setChecked(False)
        logger.info("New point extraction table initialized.")

    @user_action("Pick point")
    def on_point_clicked(self, b: float, energy: float) -> None:
        mode = self.corrections.point_mode()
        if mode is PointMode.OFF or self.points is None:
            return
        name = self.corrections.curve_name()
        if mode is PointMode.RECORD:
            row = self.points.set_nearest(name, b, energy)
            logger.info("%s: B = %g T -> E = %.4g", name, self.points.field[row], energy)
        else:
            row = self.points.clear_nearest(name, b)
            logger.info("%s: point at B = %g T removed", name, self.points.field[row])
        self.point_model.refresh()
        self.update_point_markers()

    def update_point_markers(self) -> None:
        cmap = self.plot_panel.color_map
        if self.points is None:
            cmap.set_points(None)
            return
        name = self.corrections.column_name.text().strip()
        current = self.points.points(name) if name in self.points.names else None
        others = []
        if self.corrections.show_all_button.isChecked():
            names = self.points.names
            for i, other in enumerate(names):
                b, e = self.points.points(other)
                others.append((b, e, pg.intColor(i, hues=max(len(names), 1))))
        cmap.set_points(current, others)

    @user_action("Drop curve")
    def drop_curve(self) -> None:
        if self.points is None:
            return
        name = self.corrections.curve_name()
        self.points.drop(name)
        self.point_model.refresh()
        self.update_point_markers()
        logger.info("Curve %r dropped.", name)

    @user_action("Load points")
    def load_points(self) -> None:
        path = open_file(self, "Load points")
        if not path:
            return
        self.points = PointTable.load_tsv(path)
        self.point_model.set_table(self.points)
        self.corrections.init_table.setChecked(False)
        if self.points.names:
            self.corrections.column_name.setText(self.points.names[0])
        self.update_point_markers()
        logger.info("Loaded points %s (%s)", Path(path).name, ", ".join(self.points.names))

    @user_action("Export points")
    def export_points(self) -> None:
        if self.points is None:
            raise ValueError("no points to export")
        path = save_file(self, "Export points")
        if not path:
            return
        out = Path(path) if Path(path).suffix else Path(path).with_suffix(".csv")
        self.points.save_tsv(out)
        logger.info("Exported points to %s", out)

    # ------------------------------------------------------------------ processed slots
    @user_action("Load slot")
    def load_slot(self, slot: int) -> None:
        path = open_file(self, f"Load processed table into slot {slot}")
        if not path:
            return
        unit = self.data_panel.energy_unit()
        fmap = load_tsv(path, default_unit=unit)
        if fmap.unit != unit:
            logger.info("Converted %s from %s to %s.", Path(path).name, fmap.unit, unit)
            fmap = fmap.replace(energy=convert(fmap.energy, fmap.unit, unit), unit=unit)
        if not self.data_panel.processed.auto_field.isChecked():
            field = self.data_panel.sample.field_range.field()
            if field.size != fmap.field.size:
                raise ValueError(
                    f"custom field range has {field.size} values, the table has {fmap.field.size}"
                )
            fmap = fmap.replace(field=field)
        self.slots[slot] = fmap
        self.data_panel.processed.set_slot_name(slot, Path(path).name)
        logger.info("Slot %d: loaded %s", slot, Path(path).name)

    @user_action("Save slot")
    def save_slot(self, slot: int) -> None:
        if self.result is None:
            raise ValueError("nothing to save - process data first")
        self.slots[slot] = self.result.ratio
        self.data_panel.processed.set_slot_name(slot, f"Saved R(B)/R(0) [{slot}]")
        logger.info("Slot %d: current R(B)/R(0) saved", slot)

    @user_action("Plot slot")
    def plot_slot(self, slot: int) -> None:
        if slot not in self.slots:
            raise ValueError(f"slot {slot} is empty")
        processed = self.data_panel.processed
        fmap = self.slots[slot]
        if not processed.full_energy.isChecked():
            fmap = crop_energy(fmap, *processed.energy_range(slot))
        logger.info("-" * 40)
        logger.info("Plotting slot %d: %s", slot, processed.slot_name(slot))
        self._set_result(ProcessResult.from_map(fmap, self.corrections.baseline_region()))

    def _used_slots(self) -> list[int]:
        used = self.data_panel.processed.used_slots(self.slots)
        if not used:
            raise ValueError("no slot selected - load slots and tick them in the Use column")
        return used

    @user_action("Merge by energy")
    def merge_slots(self) -> None:
        processed = self.data_panel.processed
        used = self._used_slots()
        merged = merge_energy([(self.slots[i], *processed.energy_range(i)) for i in used])
        logger.info("-" * 40)
        logger.info("Merged slots %s by energy; energy re-gridded to a uniform step.", used)
        self._set_result(ProcessResult.from_map(merged, self.corrections.baseline_region()))

    @user_action("Merge by field")
    def merge_slots_by_field(self) -> None:
        processed = self.data_panel.processed
        used = self._used_slots()
        merged = merge_field([(self.slots[i], *processed.field_range(i)) for i in used])
        logger.info("-" * 40)
        logger.info(
            "Merged slots %s by field: %d fields, B = %g … %g T.",
            used,
            merged.field.size,
            merged.field[0],
            merged.field[-1],
        )
        self._set_result(ProcessResult.from_map(merged, self.corrections.baseline_region()))

    @user_action("Average")
    def average_slots(self) -> None:
        processed = self.data_panel.processed
        used = self._used_slots()
        maps = [
            crop_field(
                crop_energy(self.slots[i], *processed.energy_range(i)),
                *processed.field_range(i),
            )
            for i in used
        ]
        averaged = average_maps(maps)
        logger.info("-" * 40)
        logger.info("Averaged slots %s (%d datasets).", used, len(used))
        self._set_result(ProcessResult.from_map(averaged, self.corrections.baseline_region()))

    # ------------------------------------------------------------------ window
    def center_on_screen(self) -> None:
        screen = self.screen()
        if screen is None:
            return
        frame = self.frameGeometry()
        frame.moveCenter(screen.availableGeometry().center())
        self.move(frame.topLeft())

    def show_shortcuts(self) -> None:
        rows = "".join(f"<tr><td><b>{k}</b></td><td>{v}</td></tr>" for k, v in SHORTCUTS)
        QMessageBox.information(self, "Shortcuts", f"<table cellspacing='6'>{rows}</table>")

    def show_about(self) -> None:
        QMessageBox.about(
            self,
            "About",
            f"<b>Magneto-Optical Detective {__version__}</b><br>"
            f"Python {platform.python_version()}, Qt {qVersion()}, PySide6 {pyside_version}<br>"
            f"numpy {np.__version__}, scipy {scipy.__version__}, pyqtgraph {pg.__version__}",
        )

    def closeEvent(self, event) -> None:
        self.save_settings()
        logger.removeHandler(self._log_handler)
        super().closeEvent(event)
