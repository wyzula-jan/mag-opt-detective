"""Application state and the actions on it, without widgets.

:class:`AppController` keeps what the window shows: the processed result, the picked points,
the library slots, the display energy unit, the view (:class:`ViewState`) and the processing
options (:class:`ProcessingState`). Data, slots, points, the energy window and the baseline
region are kept in cm^-1; the display unit is applied only when showing or exporting, so a
unit switch never reprocesses anything. Area modules (panels, inspector, plot area) change the
state through this class and redraw on its signals.
"""

from __future__ import annotations

import contextlib
import dataclasses
import functools
import logging
from collections.abc import Callable, Iterator, Mapping
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import numpy as np
from PySide6.QtCore import QObject, Signal

from mag_opt_detective.core.pipeline import (
    PlotKind,
    ProcessOptions,
    ProcessResult,
    ReferenceMode,
    process,
)
from mag_opt_detective.core.points import PointTable
from mag_opt_detective.core.processing import (
    Axis,
    average_maps,
    crop_energy,
    crop_field,
    merge_energy,
    merge_field,
)
from mag_opt_detective.core.readers import Measurement, load_measurement, sort_paths
from mag_opt_detective.core.spectra import FieldMap, energy_mask, load_tsv
from mag_opt_detective.core.units import (
    Range,
    Unit,
    convert_levels,
    convert_range,
    from_cm1,
    to_cm1,
)
from mag_opt_detective.gui.plots.base import robust_levels

logger = logging.getLogger("mag_opt_detective")

EXPECTED_ERRORS = (ValueError, OSError, KeyError)

KIND_LABELS = {
    PlotKind.RATIO: "R(B)/R(0)",
    PlotKind.DATA: "Data",
    PlotKind.AVERAGE: "R(B)/R(B-AVR)",
    PlotKind.STEP: "R(B)/R(B-ΔB)",
}
EXPORT_NAMES = {
    PlotKind.RATIO: "Ratio",
    PlotKind.DATA: "Data",
    PlotKind.AVERAGE: "Ratio_AVR",
    PlotKind.STEP: "Ratio_Step",
}
ORDER_SUFFIX = {0: "", 1: "_1stDer", 2: "_2ndDer"}
PER_UNIT_SUFFIX = "_perUnit"
AUTO_COLOURS = "Auto"

Curve = tuple[np.ndarray, np.ndarray]


# ---------------------------------------------------------------------- errors
def tag_panel(exc: BaseException, panel: str) -> BaseException:
    """Remember on *exc* which rail panel fixes it (the error bar offers to open it)."""
    if getattr(exc, "panel", None) is None:
        with contextlib.suppress(AttributeError, TypeError):
            exc.panel = panel  # type: ignore[attr-defined]
    return exc


@contextlib.contextmanager
def in_panel(panel: str) -> Iterator[None]:
    """Tag expected errors raised inside the block with *panel* (see :func:`tag_panel`)."""
    try:
        yield
    except EXPECTED_ERRORS as exc:
        tag_panel(exc, panel)
        raise


def panel_error(message: str, panel: str) -> ValueError:
    return tag_panel(ValueError(message), panel)  # type: ignore[return-value]


def user_action(title: str):
    """Run a handler ``f(window, ...)``, reporting errors instead of raising them.

    Expected errors (:data:`EXPECTED_ERRORS`) go to ``window.report_error`` with the panel that
    fixes them; anything else is logged with its traceback and reported as unexpected.
    """

    def decorator(func):
        @functools.wraps(func)
        def wrapper(window, *args, **kwargs):
            try:
                return func(window, *args, **kwargs)
            except EXPECTED_ERRORS as exc:
                window.report_error(title, str(exc), panel=getattr(exc, "panel", None))
            except Exception as exc:
                logger.exception("%s failed", title)
                window.report_error(title, f"Unexpected error: {exc!r}", expected=False)
            return None

        return wrapper

    return decorator


def fmt(value: float | None) -> str:
    """A number as typed by people (six significant digits); None is an open end."""
    return "open" if value is None else f"{value:.6g}"


# ---------------------------------------------------------------------- state
def level_key(kind: PlotKind | str, order: int = 0, axis: Axis = Axis.ENERGY, physical=False):
    """Key of the colour levels of a plot.

    Maps share a key per kind; derivatives per data point share one key per order (as the old
    Plot dimensions page did); derivatives per unit get their own key per order and axis.
    """
    if not order:
        return str(PlotKind(kind))
    if physical:
        return f"der{order}_{'E' if axis == Axis.ENERGY else 'B'}_unit"
    return f"der{order}"


def parse_level_key(key: str) -> tuple[int, bool, bool]:
    """``(order, axis_is_energy, physical)`` of a :func:`level_key`."""
    if not key.startswith("der"):
        return 0, True, False
    order = int(key[3])
    physical = key.endswith("_unit")
    return order, not (physical and key[5] == "B"), physical


@dataclass(frozen=True)
class PlotSelection:
    """Which map is shown: kind, derivative order and axis, per unit or per point."""

    kind: PlotKind = PlotKind.RATIO
    order: int = 0
    axis: Axis = Axis.ENERGY
    physical: bool = False
    reference_kind: PlotKind = PlotKind.RATIO

    @property
    def level_key(self) -> str:
        return level_key(self.kind, self.order, self.axis, self.physical)

    @property
    def per_unit(self) -> bool:
        """A derivative divided by the real step (only meaningful with an order)."""
        return bool(self.physical and self.order)

    def export_name(self) -> str:
        return (
            EXPORT_NAMES[self.kind]
            + ORDER_SUFFIX[self.order]
            + (PER_UNIT_SUFFIX if self.per_unit else "")
        )

    def description(self, unit: Unit) -> str:
        text = KIND_LABELS[self.kind]
        if self.order:
            nth = {1: "1st", 2: "2nd"}[self.order]
            d = "d/dE" if self.axis == Axis.ENERGY else "d/dB"
            per = (
                ("per " + (str(unit) if self.axis == Axis.ENERGY else "T"))
                if self.physical
                else ("per point")
            )
            text += f" · {nth} derivative {d} {per}"
        return text


@dataclass(frozen=True)
class ViewState:
    """How the maps are shown. Energies and the levels of per-unit E-derivatives are in the
    display unit; everything else does not depend on it."""

    field_range: Range | None = None  # T; None: fit the data
    energy_range: Range | None = None  # display unit; None: fit the data
    levels: Mapping[str, tuple[float, float]] = field(default_factory=dict)
    custom_levels: bool = True  # False: every map autoscales
    stacked_range: Range | None = None
    stacked_offset: float = 0.01
    colormap: str = AUTO_COLOURS

    def levels_for(self, key: str) -> tuple[float, float] | None:
        if not self.custom_levels:
            return None
        return self.levels.get(key)

    def colormap_for(self, order: int) -> str:
        if self.colormap == AUTO_COLOURS:
            return "magma" if order == 0 else "grey"
        return self.colormap

    def converted(self, src: Unit, dst: Unit) -> ViewState:
        """The same view in unit *dst*: energy range and per-unit E-derivative levels."""
        levels = {}
        for key, value in self.levels.items():
            order, energy, physical = parse_level_key(key)
            levels[key] = convert_levels(value, src, dst, order, energy, physical)
        return dataclasses.replace(
            self, energy_range=convert_range(self.energy_range, src, dst), levels=levels
        )


@dataclass(frozen=True)
class SweepFiles:
    """Zero-field and in-field files of one measurement, sorted by field."""

    zero: tuple[str, ...] = ()
    field: tuple[str, ...] = ()


@dataclass(frozen=True)
class FieldRange:
    """Start / step / end of a custom field range in T (None: empty or not a number)."""

    start: float | None = 0.25
    step: float | None = 0.25
    end: float | None = 16.0

    def values(self) -> np.ndarray:
        for name, value in (("start", self.start), ("step", self.step), ("end", self.end)):
            if value is None:
                raise ValueError(f"custom field range: enter a number for the {name}")
        start, step, end = self.start, self.step, self.end
        if step <= 0:
            raise ValueError("field step must be positive")
        if end < start:
            raise ValueError("end field must not be smaller than start field")
        n = round((end - start) / step) + 1
        return start + step * np.arange(n)


@dataclass(frozen=True)
class ProcessingState:
    """Everything Process uses; energies in cm^-1 (an end may be None: open or empty)."""

    sample_files: SweepFiles = SweepFiles()
    reference_files: SweepFiles = SweepFiles()
    custom_field: bool = False
    sample_field: FieldRange = FieldRange()
    reference_field: FieldRange = FieldRange()
    reference_mode: ReferenceMode = ReferenceMode.NONE
    smooth: bool = False
    sg_window: int = 11
    sg_poly: int = 2
    energy_cut: Range | None = None  # None: keep every energy
    baseline: Range | None = None  # None: no baseline correction

    def effective(self) -> ProcessingState:
        """The same state with the options that cannot change the result reset (smoothing
        without a reference, the reference sweep in another mode, an unused field range)."""
        default = ProcessingState()
        separate = self.reference_mode is ReferenceMode.SEPARATE
        changes: dict[str, object] = {}
        if not self.custom_field:
            changes["sample_field"] = default.sample_field
        if not separate:
            changes["reference_files"] = default.reference_files
        if not (separate and self.custom_field):
            changes["reference_field"] = default.reference_field
        if self.reference_mode is ReferenceMode.NONE:
            changes["smooth"] = default.smooth
        if self.reference_mode is ReferenceMode.NONE or not self.smooth:
            changes["sg_window"], changes["sg_poly"] = default.sg_window, default.sg_poly
        return dataclasses.replace(self, **changes)


@dataclass(frozen=True)
class FigureState:
    """What is shown on the map, in the display unit (for figure export)."""

    fmap: FieldMap
    unit: Unit
    selection: PlotSelection
    description: str
    colormap: str
    levels: tuple[float, float]
    fixed_levels: bool
    field_range: Range | None
    energy_range: Range | None
    points: dict[str, Curve]
    curve: str
    overlays: dict[str, list[Curve]]


# ---------------------------------------------------------------------- controller
class AppController(QObject):
    """State of the window and the actions on it (no widgets)."""

    resultChanged = Signal()
    unitChanged = Signal(object, object)  # old Unit, new Unit
    viewChanged = Signal()
    selectionChanged = Signal()
    processingChanged = Signal()
    changedSinceProcess = Signal(bool)
    pointsChanged = Signal()
    slotsChanged = Signal()
    restored = Signal()  # settings were restored; re-derive what is shown from stored state

    def __init__(self, parent: QObject | None = None):
        super().__init__(parent)
        self.result: ProcessResult | None = None
        self.points: PointTable | None = None
        self.slots: dict[int, FieldMap] = {}
        self.processed_at: datetime | None = None  # of the last Process
        self.result_source = ""  # "process" or "library": where the shown result comes from
        self._unit = Unit.CM1
        self._view = ViewState()
        self._selection = PlotSelection()
        self._processing = ProcessingState()
        self._processed_with: ProcessingState | None = None
        self._changed = False
        self._restoring = 0
        self.curve = "LL 1"
        self.new_table = True  # start a new point table on the next result
        self._overlays: dict[str, Callable[[Unit], list[Curve]]] = {}

    # --- unit --------------------------------------------------------------------------
    @property
    def unit(self) -> Unit:
        return self._unit

    def set_unit(self, unit: Unit | str) -> None:
        """Show everything in *unit* at once; the data are not processed again.

        While settings are restored (:meth:`restoring`) the view is not converted: restored
        values are already in the restored unit, whatever order they come back in.
        """
        unit = Unit(unit)
        old = self._unit
        if unit is old:
            return
        self._unit = unit
        if not self._restoring:
            self._view = self._view.converted(old, unit)
        logger.debug("Energy unit %s -> %s", old, unit)
        self.unitChanged.emit(old, unit)

    @contextlib.contextmanager
    def restoring(self) -> Iterator[None]:
        """Restore settings inside the block: no conversions; :attr:`restored` fires after."""
        self._restoring += 1
        try:
            yield
        finally:
            self._restoring -= 1
        if not self._restoring:
            self.restored.emit()

    def is_restoring(self) -> bool:
        return bool(self._restoring)

    # --- view and selection ------------------------------------------------------------
    @property
    def view(self) -> ViewState:
        return self._view

    def set_view(self, view: ViewState | None = None, **changes) -> None:
        new = dataclasses.replace(view or self._view, **changes)
        if new != self._view:
            self._view = new
            self.viewChanged.emit()

    def set_levels(self, key: str, lo: float, hi: float) -> None:
        """Fix the colour levels of *key* (display unit) and use fixed levels."""
        levels = {**self._view.levels, key: (float(lo), float(hi))}
        self.set_view(levels=levels, custom_levels=True)

    @property
    def selection(self) -> PlotSelection:
        return self._selection

    def set_selection(self, **changes) -> None:
        new = dataclasses.replace(self._selection, **changes)
        if new != self._selection:
            self._selection = new
            self.selectionChanged.emit()

    def current_levels(self) -> tuple[float, float] | None:
        """Fixed levels of the map shown, or None to autoscale."""
        return self._view.levels_for(self._selection.level_key)

    def current_map(self) -> FieldMap:
        """The map shown (or exported), in the display unit."""
        if self.result is None:
            raise ValueError("nothing to show - process data first")
        s = self._selection
        return self.result.get(s.kind, s.order, s.axis, physical=s.physical, unit=self._unit)

    def reference_map(self) -> FieldMap | None:
        result = self.result
        if result is None or result.reference_data is None or result.reference_ratio is None:
            return None
        kind = self._selection.reference_kind
        fmap = result.reference_data if kind is PlotKind.DATA else result.reference_ratio
        return fmap.to_unit(self._unit)

    def description(self) -> str:
        return self._selection.description(self._unit)

    # --- processing --------------------------------------------------------------------
    @property
    def processing(self) -> ProcessingState:
        return self._processing

    def set_processing(self, **changes) -> None:
        if "sample_files" in changes:
            changes["sample_files"] = _sorted_files(changes["sample_files"])
        if "reference_files" in changes:
            changes["reference_files"] = _sorted_files(changes["reference_files"])
        new = dataclasses.replace(self._processing, **changes)
        if new != self._processing:
            self._processing = new
            self.processingChanged.emit()
            self._update_changed()

    def changed_since_process(self) -> bool:
        return self._changed

    def _update_changed(self) -> None:
        used = self._processed_with
        changed = used is not None and used.effective() != self._processing.effective()
        if changed != self._changed:
            self._changed = changed
            self.changedSinceProcess.emit(changed)

    def _range_error(self, what: str, rng: Range, panel: str) -> ValueError:
        lo, hi = convert_range(rng, Unit.CM1, self._unit)
        return panel_error(
            f"{what}: the first value must be below the second; it is "
            f"{fmt(lo)} – {fmt(hi)} {self._unit}",
            panel,
        )

    def check_energy_range(
        self, what: str, rng: Range | None, energy: np.ndarray, panel: str
    ) -> None:
        """Raise (in the display unit) if the cm^-1 range *rng* is reversed or holds no data."""
        if rng is None:
            return
        lo, hi = rng
        if lo is not None and hi is not None and lo >= hi:
            raise self._range_error(what, rng, panel)
        if not energy_mask(energy, lo, hi).any():
            shown = convert_range(rng, Unit.CM1, self._unit)
            data = from_cm1(np.array([energy.min(), energy.max()]), self._unit)
            raise panel_error(
                f"{what} {fmt(shown[0])} – {fmt(shown[1])} {self._unit} contains no data "
                f"(the data span {fmt(data[0])} – {fmt(data[1])} {self._unit})",
                panel,
            )

    def _baseline(self, panel: str = "processing") -> tuple[float, float] | None:
        region = self._processing.baseline
        if region is None:
            return None
        if region[0] is None or region[1] is None:
            raise panel_error("baseline region: enter both limits", panel)
        if region[0] >= region[1]:
            raise self._range_error("baseline region", region, panel)
        return region[0], region[1]

    def _load(self, files: SweepFiles, field_range: FieldRange, panel: str) -> Measurement:
        with in_panel(panel):
            field_values = field_range.values() if self._processing.custom_field else None
            return load_measurement(files.zero, files.field, field=field_values)

    def _cut(self, measurement: Measurement) -> Measurement:
        cut = self._processing.energy_cut
        if cut is None:
            return measurement
        spectra = measurement.spectra
        self.check_energy_range("energy window", cut, spectra.energy, "processing")
        mask = energy_mask(spectra.energy, *cut)
        return Measurement(spectra=crop_energy(spectra, *cut), zero=measurement.zero[mask])

    def process(self) -> ProcessResult:
        """Load the sweep(s), process them and show the result."""
        p = self._processing
        unit = self._unit
        baseline = self._baseline()
        logger.info("-" * 40)
        sample = self._cut(self._load(p.sample_files, p.sample_field, "sample"))
        spectra = sample.spectra
        if baseline is not None:
            self.check_energy_range("baseline region", baseline, spectra.energy, "processing")
        e_lo, e_hi = from_cm1(spectra.energy[[0, -1]], unit)
        logger.info(
            "Sample: %d spectra, B = %g … %g T, %d zero-field file(s), E = %.4g … %.4g %s",
            spectra.field.size,
            spectra.field.min(),
            spectra.field.max(),
            sample.zero.shape[1],
            e_lo,
            e_hi,
            unit,
        )
        reference = None
        mode = p.reference_mode
        if mode is ReferenceMode.SEPARATE:
            reference = self._cut(self._load(p.reference_files, p.reference_field, "reference"))
            logger.info(
                "Reference: %d spectra interpolated onto the sample field",
                reference.spectra.field.size,
            )
        elif mode is ReferenceMode.SELF:
            logger.info("Using the data itself as reference.")
        options = ProcessOptions(
            reference_mode=mode,
            smooth_reference=p.smooth,
            sg_window=p.sg_window,
            sg_poly=p.sg_poly,
            baseline_region=baseline,
        )
        if options.smooth_reference and mode is not ReferenceMode.NONE:
            logger.info("Reference smoothed (SG window %d, order %d).", p.sg_window, p.sg_poly)
        with in_panel("reference" if mode is not ReferenceMode.NONE else "processing"):
            result = process(sample, reference, options)
        if baseline is not None:
            lo, hi = from_cm1(np.array(baseline), unit)
            logger.info("Baseline corrected in range %.6g – %.6g %s.", lo, hi, unit)
        self.set_result(result)
        return result

    def set_result(self, result: ProcessResult, processed: bool = True) -> None:
        """Show *result* (cm^-1).

        *processed*: made by :meth:`process` from the current options, which then count as
        applied; False for library maps, which leave the Process state as it was.
        """
        self.result = result
        self.result_source = "process" if processed else "library"
        if processed:
            self.processed_at = datetime.now()
            self._processed_with = self._processing
        self._init_points_if_requested(result.ratio.field)
        self.resultChanged.emit()
        self._update_changed()

    def from_map(self, fmap: FieldMap) -> ProcessResult:
        """Show an already processed map (library), baseline-corrected like a processed one."""
        baseline = self._baseline(panel="processing")
        if baseline is not None:
            self.check_energy_range("baseline region", baseline, fmap.energy, "processing")
        result = ProcessResult.from_map(fmap, baseline)
        self.set_result(result, processed=False)
        return result

    # --- figure ------------------------------------------------------------------------
    def set_overlay(self, name: str, curves: Callable[[Unit], list[Curve]] | None) -> None:
        """Register (or with None remove) a source of overlay curves for :meth:`figure_state`."""
        if curves is None:
            self._overlays.pop(name, None)
        else:
            self._overlays[name] = curves

    def figure_state(self) -> FigureState:
        """A snapshot of the map as shown, in the display unit."""
        fmap = self.current_map()
        fixed = self.current_levels()
        unit = self._unit
        points: dict[str, Curve] = {}
        if self.points is not None:
            for name in self.points.names:
                b, e = self.points.points(name)
                points[name] = (b, from_cm1(e, unit))
        view = self._view
        return FigureState(
            fmap=fmap,
            unit=unit,
            selection=self._selection,
            description=self.description(),
            colormap=view.colormap_for(self._selection.order),
            levels=fixed if fixed is not None else robust_levels(fmap.values),
            fixed_levels=fixed is not None,
            field_range=view.field_range,
            energy_range=view.energy_range,
            points=points,
            curve=self.curve,
            overlays={name: source(unit) for name, source in self._overlays.items()},
        )

    # --- points ------------------------------------------------------------------------
    def set_curve(self, name: str) -> None:
        name = name.strip()
        if name != self.curve:
            self.curve = name
            self.pointsChanged.emit()

    def curve_name(self) -> str:
        if not self.curve:
            raise panel_error("enter a curve name for the picked points", "points")
        return self.curve

    def set_new_table(self, new: bool) -> None:
        if bool(new) != self.new_table:
            self.new_table = bool(new)
            self.pointsChanged.emit()

    def _init_points_if_requested(self, field_values: np.ndarray) -> None:
        if not self.new_table and self.points is not None:
            return
        self.points = PointTable(field_values)
        if self.curve:
            self.points.add_column(self.curve)
        self.new_table = False
        logger.info("New point extraction table initialized.")
        self.pointsChanged.emit()

    def record_point(self, b: float, energy: float) -> int:
        """Record a point clicked at *energy* in the display unit (kept in cm^-1)."""
        if self.points is None:
            raise panel_error("no point table - process data first", "points")
        name = self.curve_name()
        row = self.points.set_nearest(name, b, float(to_cm1(energy, self._unit)))
        logger.info("%s: B = %g T -> E = %.4g %s", name, self.points.field[row], energy, self._unit)
        self.pointsChanged.emit()
        return row

    def remove_point(self, b: float) -> int:
        if self.points is None:
            raise panel_error("no point table - process data first", "points")
        name = self.curve_name()
        row = self.points.clear_nearest(name, b)
        logger.info("%s: point at B = %g T removed", name, self.points.field[row])
        self.pointsChanged.emit()
        return row

    def drop_curve(self) -> None:
        if self.points is None:
            return
        name = self.curve_name()
        self.points.drop(name)
        logger.info("Curve %r dropped.", name)
        self.pointsChanged.emit()

    def load_points(self, path: str | Path) -> PointTable:
        """Read a point table; one without a unit in its header is in the display unit."""
        with in_panel("points"):
            table = PointTable.load_tsv(path, default_unit=self._unit)
        self.points = table
        self.new_table = False
        if table.names:
            self.curve = table.names[0]
        logger.info("Loaded points %s (%s)", Path(path).name, ", ".join(table.names))
        self.pointsChanged.emit()
        return table

    def save_points(self, path: str | Path) -> None:
        if self.points is None:
            raise panel_error("no points to export", "points")
        self.points.save_tsv(path, unit=self._unit)
        logger.info("Exported points to %s", path)

    # --- library slots -----------------------------------------------------------------
    def load_slot(self, slot: int, path: str | Path, field_values: np.ndarray | None = None):
        """Read a processed table into *slot* (kept in cm^-1); returns its file unit.

        A table without a unit in its header is in the display unit. *field_values* replace
        the field read from the header.
        """
        with in_panel("library"):
            fmap = load_tsv(path, default_unit=self._unit)
            file_unit = fmap.unit
            fmap = fmap.to_unit(Unit.CM1)
            if field_values is not None:
                if field_values.size != fmap.field.size:
                    raise ValueError(
                        f"custom field range has {field_values.size} values, "
                        f"the table has {fmap.field.size}"
                    )
                fmap = fmap.replace(field=field_values)
        self.slots[slot] = fmap
        logger.info("Slot %d: loaded %s (energy in %s)", slot, Path(path).name, file_unit)
        self.slotsChanged.emit()
        return file_unit

    def save_slot(self, slot: int) -> None:
        if self.result is None:
            raise panel_error("nothing to save - process data first", "library")
        self.slots[slot] = self.result.ratio
        logger.info("Slot %d: current R(B)/R(0) saved", slot)
        self.slotsChanged.emit()

    def slot(self, slot: int) -> FieldMap:
        if slot not in self.slots:
            raise panel_error(f"slot {slot} is empty", "library")
        return self.slots[slot]

    def _slot_energy(self, slot: int, rng: Range | None) -> FieldMap:
        fmap = self.slot(slot)
        if rng is None or rng == (None, None):
            return fmap
        self.check_energy_range(f"slot {slot}: E range", rng, fmap.energy, "library")
        return crop_energy(fmap, *rng)

    def plot_slot(self, slot: int, energy_range: Range | None = None) -> ProcessResult:
        """Show *slot*, cut to *energy_range* (cm^-1) if given."""
        fmap = self._slot_energy(slot, energy_range)
        logger.info("-" * 40)
        logger.info("Plotting slot %d", slot)
        return self.from_map(fmap)

    def merge_by_energy(self, parts: list[tuple[int, Range | None]]) -> ProcessResult:
        """Join slots measured in different spectral ranges, each cut to its range (cm^-1)."""
        if not parts:
            raise panel_error("no slot selected - load slots and tick them", "library")
        for slot, rng in parts:
            if rng is not None:
                self.check_energy_range(
                    f"slot {slot}: E range", rng, self.slot(slot).energy, "library"
                )
        with in_panel("library"):
            merged = merge_energy([(self.slot(i), *(rng or (None, None))) for i, rng in parts])
        logger.info("-" * 40)
        logger.info(
            "Merged slots %s by energy; energy re-gridded to a uniform step.",
            [i for i, _ in parts],
        )
        return self.from_map(merged)

    def merge_by_field(self, parts: list[tuple[int, Range | None]]) -> ProcessResult:
        """Join slots measured over different field ranges, each cut to its range (T)."""
        if not parts:
            raise panel_error("no slot selected - load slots and tick them", "library")
        with in_panel("library"):
            merged = merge_field([(self.slot(i), *(rng or (None, None))) for i, rng in parts])
        logger.info("-" * 40)
        logger.info(
            "Merged slots %s by field: %d fields, B = %g … %g T.",
            [i for i, _ in parts],
            merged.field.size,
            merged.field[0],
            merged.field[-1],
        )
        return self.from_map(merged)

    def average(self, parts: list[tuple[int, Range | None, Range | None]]) -> ProcessResult:
        """Average slots, each cut to its energy (cm^-1) and field range."""
        if not parts:
            raise panel_error("no slot selected - load slots and tick them", "library")
        maps = []
        for slot, e_range, b_range in parts:
            fmap = self._slot_energy(slot, e_range)
            with in_panel("library"):
                maps.append(crop_field(fmap, *(b_range or (None, None))))
        with in_panel("library"):
            averaged = average_maps(maps)
        logger.info("-" * 40)
        logger.info("Averaged slots %s (%d datasets).", [i for i, *_ in parts], len(parts))
        return self.from_map(averaged)


def _sorted_files(files: SweepFiles) -> SweepFiles:
    return SweepFiles(zero=tuple(sort_paths(files.zero)), field=tuple(sort_paths(files.field)))
