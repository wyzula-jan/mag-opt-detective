"""Application state and the actions on it, without widgets.

:class:`AppController` keeps what the window shows: the processed result, the picked points,
the library maps, the display energy unit, the view (:class:`ViewState`) and the processing
options (:class:`ProcessingState`). Data, maps, points, the energy window and the baseline
region are kept in cm^-1; the display unit is applied only when showing or exporting, so a
unit switch never reprocesses anything. Area modules (panels, inspector, plot area) change the
state through this class and redraw on its signals. Point edits go through
:meth:`AppController.point_edit`, which makes them undoable (``points_undo``).
"""

from __future__ import annotations

import contextlib
import dataclasses
import functools
import itertools
import logging
import math
import re
from collections.abc import Callable, Hashable, Iterator, Mapping
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import NamedTuple

import numpy as np
from PySide6.QtCore import QObject, Signal

from mag_opt_detective.core.fitting import Assignment, FitResult, Model, Observation, fit
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
    energy_seams,
    merge_energy,
    merge_field,
)
from mag_opt_detective.core.readers import (
    Measurement,
    SpectrumCache,
    load_measurement,
    read_spectrum,
    sort_paths,
)
from mag_opt_detective.core.spectra import FieldMap, energy_mask, load_tsv, sample_at
from mag_opt_detective.core.units import (
    Range,
    Unit,
    convert_levels,
    convert_range,
    derivative_scale,
    from_cm1,
    to_cm1,
)
from mag_opt_detective.gui.display import format_number, format_range, unit_text
from mag_opt_detective.gui.plots.base import robust_levels
from mag_opt_detective.gui.points_undo import PointsState, PointsUndoStack

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
ORDINALS = {1: "1st", 2: "2nd"}
AUTO_COLOURS = "Auto"
AUTO_TRACE_COLOURS = "viridis"  # traces coloured by field with the Auto colour map

# colour level modes: 1-99 % of the map, fixed levels, or symmetric about the centre of a kind
AUTO_LEVELS, FIXED_LEVELS, SYMMETRIC_LEVELS = "auto", "fixed", "sym"
LEVEL_MODES = (AUTO_LEVELS, FIXED_LEVELS, SYMMETRIC_LEVELS)
DEFAULT_LEVELS = {
    str(PlotKind.RATIO): (0.9, 1.1),
    str(PlotKind.AVERAGE): (0.9, 1.1),
    str(PlotKind.STEP): (0.98, 1.02),
}

# the ViewState ranges shown on the x and y axes of each plot view
VIEW_RANGES = {
    "map": ("field_range", "energy_range"),
    "stacked": ("energy_range", "stacked_range"),
    "reference": ("field_range", "energy_range"),
}

CURVE_NAME = re.compile(r"[A-Za-z0-9_ .+-]{1,32}")  # names of picked-point curves
MAX_FIELD_VALUES = 100_000  # of a custom field range

# the state of the sample's or the reference's data (AppController.data_state): none loaded,
# loaded but not processed as they are, or processed by the last Process as they are
DATA_EMPTY, DATA_CHANGED, DATA_CURRENT = "empty", "changed", "current"
DATA_PARTS = ("sample", "reference")
NO_REFERENCE_FILES = "Reference sweep has no files – shown without reference."
LIBRARY_KEEPS_BASELINE = "the library map shown keeps the baseline it was plotted with"

# what the plot shows (AppController.showing) and how ticked library maps are combined
SHOWS_PROCESS, SHOWS_MAPS, SHOWS_PRODUCT = "process", "maps", "product"
COMBINE_METHODS = {"energy": "Merge by energy", "field": "Merge by field", "average": "Average"}
PRODUCT_PREFIX = {"energy": "Merged by energy", "field": "Merged by field", "average": "Average"}
DEFAULT_OPACITY = 0.5  # of the library maps drawn above the bottom one
FIELD_TOL = 1e-6  # T: fields closer than this are the same field

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
                hint = {"hint": exc.hint} if getattr(exc, "hint", None) else {}  # a remedy
                window.report_error(title, str(exc), panel=getattr(exc, "panel", None), **hint)
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
class LevelKey(NamedTuple):
    """What a :func:`level_key` stands for."""

    kind: PlotKind
    order: int = 0
    axis: Axis = Axis.ENERGY
    physical: bool = False  # a derivative per unit (only with an order)

    @property
    def axis_is_energy(self) -> bool:
        return self.axis == Axis.ENERGY


_DERIVATIVE_KEY = re.compile(r"(?P<kind>.+?)_der(?P<order>[12])_(?P<axis>[EB])(?P<unit>_unit)?")


def level_key(kind: PlotKind | str, order: int = 0, axis: Axis = Axis.ENERGY, physical=False):
    """Key of the colour levels of a plot: one per kind, derivative order, axis and per unit.

    A map without a derivative is keyed by its kind (``"Ratio"``); derivatives add the order
    and axis (``"Ratio_der1_E"``) and ``"_unit"`` when divided by the real step.
    """
    kind = PlotKind(kind)
    if not order:
        return str(kind)
    axis_name = "E" if axis == Axis.ENERGY else "B"
    return f"{kind}_der{order}_{axis_name}" + ("_unit" if physical else "")


def parse_level_key(key: str) -> LevelKey:
    """The plot of a :func:`level_key`; ValueError for anything else."""
    match = _DERIVATIVE_KEY.fullmatch(key)
    if match is None:
        return LevelKey(PlotKind(key))
    axis = Axis.ENERGY if match["axis"] == "E" else Axis.FIELD
    return LevelKey(PlotKind(match["kind"]), int(match["order"]), axis, bool(match["unit"]))


def level_label(key: str) -> str:
    """What the levels of *key* are remembered for, e.g.
    ``"R(B)/R(0) · 1st derivative d/dE per unit"``."""
    k = parse_level_key(key)
    text = KIND_LABELS[k.kind]
    if k.order:
        per = ("per unit" if k.axis_is_energy else "per T") if k.physical else "per point"
        text += f" · {ORDINALS[k.order]} derivative d/d{'E' if k.axis_is_energy else 'B'} {per}"
    return text


def symmetric_centre(key: str) -> float | None:
    """Centre of symmetric levels: 0 for derivatives, 1 for ratios, None for raw data."""
    k = parse_level_key(key)
    if k.order:
        return 0.0
    return None if k.kind is PlotKind.DATA else 1.0


def default_level_mode(key: str) -> str:
    """Derivatives start symmetric about 0, ratios at fixed levels, raw data on Auto."""
    if parse_level_key(key).order:
        return SYMMETRIC_LEVELS
    return FIXED_LEVELS if key in DEFAULT_LEVELS else AUTO_LEVELS


def symmetric_levels(levels: tuple[float, float], centre: float) -> tuple[float, float]:
    """The smallest levels symmetric about *centre* that cover *levels*."""
    half = max(abs(levels[0] - centre), abs(levels[1] - centre))
    if not (math.isfinite(half) and half > 0):
        half = 1e-6
    return centre - half, centre + half


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
                ("per " + (unit_text(unit) if self.axis == Axis.ENERGY else "T"))
                if self.physical
                else ("per point")
            )
            text += f" · {nth} derivative {d} {per}"
        return text


@dataclass(frozen=True)
class ViewState:
    """How the maps are shown. Energies and the intensities of per-unit E-derivatives (their
    levels, the stacked range and offset) are in the display unit; the rest does not depend on
    it. Each colour level key (:func:`level_key`) has a mode (``LEVEL_MODES``) and, when fixed
    or symmetric, its levels."""

    field_range: Range | None = None  # T; None: fit the data
    energy_range: Range | None = None  # display unit; None: fit the data
    levels: Mapping[str, tuple[float, float]] = field(default_factory=lambda: dict(DEFAULT_LEVELS))
    level_modes: Mapping[str, str] = field(default_factory=dict)  # missing: default_level_mode
    stacked_range: Range | None = None  # intensity of the stacked plot; None: fit the data
    stacked_offset: float = 0.01
    stacked_every: int = 1  # show every n-th spectrum
    stacked_by_field: bool = True  # colour the spectra by field (else one hue each)
    colormap: str = AUTO_COLOURS

    def level_mode(self, key: str) -> str:
        return self.level_modes.get(key, default_level_mode(key))

    def levels_for(self, key: str) -> tuple[float, float] | None:
        """The kept levels of *key*, or None to autoscale (Auto, or nothing kept yet)."""
        if self.level_mode(key) == AUTO_LEVELS:
            return None
        return self.levels.get(key)

    def levels_in_effect(self, key: str, values: np.ndarray) -> tuple[float, float]:
        """Levels a map of *values* is drawn with: the kept ones, else the 1st-99th
        percentile (symmetric about the centre of the kind in the symmetric mode)."""
        kept = self.levels_for(key)
        if kept is not None:
            return kept
        auto = robust_levels(values)
        centre = symmetric_centre(key)
        if self.level_mode(key) == SYMMETRIC_LEVELS and centre is not None:
            return symmetric_levels(auto, centre)
        return auto

    def colormap_for(self, order: int) -> str:
        if self.colormap == AUTO_COLOURS:
            return "magma" if order == 0 else "grey"
        return self.colormap

    def trace_colormap(self) -> str:
        """Colour map of the spectra coloured by field (Auto: viridis)."""
        return AUTO_TRACE_COLOURS if self.colormap == AUTO_COLOURS else self.colormap

    def converted(self, src: Unit, dst: Unit, selection: PlotSelection | None = None) -> ViewState:
        """The same view in unit *dst*: energy range and per-unit E-derivative levels, and with
        the *selection* shown, the stacked intensity range and offset of such a derivative.

        The offset is shared by all maps; :meth:`AppController.set_selection` scales it back
        when another map is shown, so it is the same per cm^-1 for every map."""
        levels = {}
        for key, value in self.levels.items():
            k = parse_level_key(key)
            levels[key] = convert_levels(value, src, dst, k.order, k.axis_is_energy, k.physical)
        changes: dict[str, object] = {}
        if selection is not None:
            s = selection
            factor = derivative_scale(dst, s.order, s.axis == Axis.ENERGY, s.physical)
            factor /= derivative_scale(src, s.order, s.axis == Axis.ENERGY, s.physical)
            if self.stacked_range is not None and None not in self.stacked_range:
                lo, hi = self.stacked_range
                changes["stacked_range"] = (lo * factor, hi * factor)
            changes["stacked_offset"] = self.stacked_offset * factor
        return dataclasses.replace(
            self, energy_range=convert_range(self.energy_range, src, dst), levels=levels, **changes
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

    def count(self) -> int:
        """How many values the range has (ValueError for an empty or reversed range)."""
        for name, value in (("start", self.start), ("step", self.step), ("end", self.end)):
            if value is None:
                raise ValueError(f"custom field range: enter a number for the {name}")
        start, step, end = self.start, self.step, self.end
        if not (math.isfinite(start) and math.isfinite(step) and math.isfinite(end)):
            raise ValueError("custom field range: enter finite numbers")
        if step <= 0:
            raise ValueError("field step must be positive")
        if end < start:
            raise ValueError("end field must not be smaller than start field")
        return round((end - start) / step) + 1

    def values(self, expected: int | None = None) -> np.ndarray:
        """The field values; ValueError before they are made when there are not *expected*
        of them (one per file) or more than MAX_FIELD_VALUES."""
        n = self.count()
        if expected is not None and n != expected:
            raise ValueError(f"custom field range has {n} values but {expected} files are loaded")
        if n > MAX_FIELD_VALUES:
            raise ValueError(f"custom field range has {n} values; at most {MAX_FIELD_VALUES}")
        return self.start + self.step * np.arange(n)


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

    def reference_used(self) -> ReferenceMode:
        """The reference Process applies: a separate sweep without files is none (Process
        shows the sample alone, see :meth:`AppController.process`)."""
        if self.reference_mode is ReferenceMode.SEPARATE and self.reference_files == SweepFiles():
            return ReferenceMode.NONE
        return self.reference_mode

    def effective(self) -> ProcessingState:
        """The same state with the options that cannot change the result reset (smoothing
        without a reference, the reference sweep in another mode, an unused field range);
        a separate sweep without files counts as no reference."""
        default = ProcessingState()
        mode = self.reference_used()
        separate = mode is ReferenceMode.SEPARATE
        changes: dict[str, object] = {"reference_mode": mode}
        if not self.custom_field:
            changes["sample_field"] = default.sample_field
        if not separate:
            changes["reference_files"] = default.reference_files
        if not (separate and self.custom_field):
            changes["reference_field"] = default.reference_field
        if mode is ReferenceMode.NONE:
            changes["smooth"] = default.smooth
        if mode is ReferenceMode.NONE or not self.smooth:
            changes["sg_window"], changes["sg_poly"] = default.sg_window, default.sg_poly
        return dataclasses.replace(self, **changes)

    def data_options(self, part: str) -> tuple:
        """The effective options that make the *part* ("sample" or "reference") of a result:
        the sample's files and fields; the reference's mode, files, fields and smoothing."""
        e = self.effective()
        if part == "sample":
            return e.sample_files, e.custom_field, e.sample_field
        if part == "reference":
            custom = e.custom_field and e.reference_mode is ReferenceMode.SEPARATE
            fields = e.reference_files, custom, e.reference_field
            return e.reference_mode, *fields, e.smooth, e.sg_window, e.sg_poly
        raise ValueError(f"no data part {part!r}; use one of {DATA_PARTS}")


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


@dataclass(frozen=True)
class PartRecord:
    """A map that went into a combined one: its name, kind and cut limits (cm^-1 and T)."""

    name: str
    kind: str
    energy_cut: Range = (None, None)
    field_cut: Range = (None, None)
    provenance: Provenance | None = None


@dataclass(frozen=True)
class Provenance:
    """How a library map was made: by Process, loaded from a table or combined from maps.

    *method* is ``"process"``, ``"table"`` or a :data:`COMBINE_METHODS` key. Energies are in
    cm^-1; :meth:`lines` describes it in a unit. *facts* are what combining did, as
    ``(what, lo, hi)`` (see :func:`fact_text`).
    """

    method: str
    made: datetime
    source: str = ""  # table: its path; process: the name of the sweep
    parts: tuple[PartRecord, ...] = ()
    facts: tuple[tuple[str, float, float], ...] = ()
    files: tuple[int, int] = (0, 0)  # process: in-field and zero-field files
    reference: str = ""  # process: the reference used
    window: Range | None = None  # process: the energy window
    baseline: Range | None = None  # process: the baseline region

    def lines(self, unit: Unit, indent: str = "") -> list[str]:
        """The provenance as lines of text, energies in *unit* (nested parts indented)."""
        stamp = self.made.strftime("%Y-%m-%d %H:%M")
        if self.method == "process":
            n_field, n_zero = self.files
            lines = [f"Processed sweep {self.source}, {stamp}"]
            lines.append(f"{n_field} in-field and {n_zero} zero-field files")
            lines.append(f"Reference: {self.reference or 'none'}")
            if self.window is not None:
                lines.append(f"Energy window {energy_range_text(self.window, unit)}")
            if self.baseline is not None:
                lines.append(f"Baseline region {energy_range_text(self.baseline, unit)}")
        elif self.method == "table":
            lines = [f"Loaded from {self.source}, {stamp}"]
        else:
            lines = [f"{COMBINE_METHODS[self.method]} of {len(self.parts)} maps, {stamp}"]
            for part in self.parts:
                cuts = cut_text(part.energy_cut, part.field_cut, unit)
                lines.append(f"- {part.name} ({part.kind}{', ' + cuts if cuts else ''})")
                if part.provenance is not None:
                    lines += part.provenance.lines(unit, "  ")
            lines += [fact_text(fact, unit)[0] for fact in self.facts]
        return [indent + line for line in lines]


@dataclass(eq=False)
class LibraryEntry:
    """A processed map in the library (cm^-1) with its tick and cut limits.

    *energy_cut* is in cm^-1 and *field_cut* in T; a None end keeps everything on that side.
    A ticked entry (*used*) is on the plot and in the maps combined. *key* identifies the
    entry while it is in the library.
    """

    name: str
    fmap: FieldMap
    kind: str = KIND_LABELS[PlotKind.RATIO]
    used: bool = False
    energy_cut: Range = (None, None)
    field_cut: Range = (None, None)
    source: str = ""  # the file it was loaded from ("" for a saved map)
    key: int = 0
    provenance: Provenance | None = None


@dataclass(eq=False)
class Product:
    """A map combined from library maps (cm^-1), until it is saved into the library."""

    name: str
    fmap: FieldMap
    kind: str
    provenance: Provenance
    saved: int | None = None  # key of the library entry it was saved as


@dataclass(frozen=True)
class OverlayLayer:
    """A ticked library map below the top one: as drawn (display unit), or why it is not."""

    entry: LibraryEntry
    fmap: FieldMap | None
    problem: str = ""


@dataclass(frozen=True)
class CombinePreview:
    """What combining *entries* with *method* gives: the map (cm^-1) and what was done to
    make it (*facts*, see :func:`fact_text`), or why it cannot be made (*problem*)."""

    method: str
    entries: tuple[LibraryEntry, ...]
    fmap: FieldMap | None = None
    facts: tuple[tuple[str, float, float], ...] = ()
    problem: str = ""


# ---------------------------------------------------------------------- controller
class AppController(QObject):
    """State of the window and the actions on it (no widgets)."""

    resultChanged = Signal()
    unitChanged = Signal(object, object)  # old Unit, new Unit
    viewChanged = Signal()
    rangesChanged = Signal()  # only plot ranges changed (set_ranges): no redraw needed
    selectionChanged = Signal()
    processingChanged = Signal()
    changedSinceProcess = Signal(bool)
    dataStateChanged = Signal(str, str)  # "sample" or "reference", its new data_state
    # Process found no reference files in Separate mode and showed the sample without a
    # reference (a note, not an error; the log has it as a warning): the note
    referenceMissing = Signal(str)
    pointsChanged = Signal()
    libraryChanged = Signal()  # library entries added or removed
    entryChanged = Signal(int)  # key of a library entry whose tick or limits changed
    restored = Signal()  # settings were restored; re-derive what is shown from stored state

    def __init__(self, parent: QObject | None = None):
        super().__init__(parent)
        self.result: ProcessResult | None = None
        self.points: PointTable | None = None
        self.library: list[LibraryEntry] = []  # processed maps, cm^-1
        self._next_key = 0
        self._library_name = ""  # of the library map shown
        self.processed_at: datetime | None = None  # of the last Process
        self.result_source = ""  # "process" or "library": where the shown result comes from
        self._unit = Unit.CM1
        self._view = ViewState()
        self._selection = PlotSelection()
        self._processing = ProcessingState()
        self._processed_with: ProcessingState | None = None
        self._changed = False
        self._live_baseline = False  # the baseline region is applied as it changes
        self._spectra: SpectrumCache | None = None  # while a folder is watched
        self._data_states = {part: DATA_EMPTY for part in DATA_PARTS}
        self._restoring = 0
        self.curve = "LL 1"
        self.new_table = False  # start a new point table with the next result
        self.points_undo = PointsUndoStack(self._restore_points, self)
        self._point_edits = 0  # depth of nested point_edit blocks
        self._overlays: dict[str, Callable[[Unit], list[Curve]]] = {}
        # what the library shows (see the library section)
        self._plotted: list[int] = []  # keys of the ticked library maps, bottom to top
        self._library_shows = SHOWS_MAPS  # what a library result is: the ticked maps or product
        self._shown_top: tuple | None = None  # the ticked map shown on top, with its cut
        self._kept_result: ProcessResult | None = None  # the processed sweep, while hidden
        self.product: Product | None = None
        self._full_range = True  # library maps are plotted without their cut limits
        self._opacity = DEFAULT_OPACITY  # of the library maps drawn above the bottom one
        self._layers: tuple[tuple, list[OverlayLayer]] | None = None  # overlay_layers() cache

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
            self._view = self._view.converted(old, unit, self._selection)
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

    def set_ranges(self, **ranges: Range | None) -> None:
        """Change plot ranges (``field_range``, ``energy_range``, ``stacked_range``; None fits
        the data) without a redraw: :attr:`rangesChanged` fires instead of viewChanged."""
        unknown = set(ranges) - {"field_range", "energy_range", "stacked_range"}
        if unknown:
            raise TypeError(f"not a plot range: {', '.join(sorted(unknown))}")
        ranges = {k: None if v is None else (float(v[0]), float(v[1])) for k, v in ranges.items()}
        new = dataclasses.replace(self._view, **ranges)
        if new != self._view:
            self._view = new
            self.rangesChanged.emit()

    def fit_ranges(self, view: str) -> None:
        """Fit the axes of plot *view* ("map", "stacked", "reference") to the data again."""
        self._view = dataclasses.replace(self._view, **dict.fromkeys(VIEW_RANGES[view]))
        self.rangesChanged.emit()  # also when already fitted: the plot may have moved

    def derivative_scale(self) -> float:
        """Factor of the shown map's intensities over the same map per cm^-1 (a per-unit energy
        derivative scales with the unit; anything else is 1)."""
        return _selection_scale(self._unit, self._selection)

    def set_levels(self, key: str, lo: float, hi: float, mode: str | None = None) -> None:
        """Keep the colour levels of *key* (display unit).

        *mode* None: symmetric levels stay symmetric when one end was moved (it is mirrored)
        or the levels are centred already; anything else fixes them.
        """
        lo, hi = float(lo), float(hi)
        view = self._view
        centre = symmetric_centre(key)
        if mode is None:
            mode = FIXED_LEVELS
            if view.level_mode(key) == SYMMETRIC_LEVELS and centre is not None:
                olds = [view.levels_for(key)]
                if olds[0] is None:  # nothing kept: the levels of the map(s) drawn with key
                    s = self._selection
                    olds = [
                        self.current_levels() if key == s.level_key else None,
                        self.reference_levels() if key == level_key(s.reference_kind) else None,
                    ]
                for old in olds:
                    mirrored = _mirrored(lo, hi, old, centre)
                    if mirrored is not None:
                        mode, (lo, hi) = SYMMETRIC_LEVELS, mirrored
                        break
        if mode not in LEVEL_MODES:
            raise ValueError(f"unknown level mode {mode!r}")
        if not (math.isfinite(lo) and math.isfinite(hi) and lo < hi):
            raise ValueError(f"invalid colour range {lo:g} – {hi:g}")
        if mode == SYMMETRIC_LEVELS and centre is not None:
            lo, hi = symmetric_levels((lo, hi), centre)
        self.set_view(
            levels={**view.levels, key: (lo, hi)}, level_modes={**view.level_modes, key: mode}
        )

    def set_level_mode(
        self, key: str, mode: str, levels: tuple[float, float] | None = None
    ) -> None:
        """Use *mode* for *key*, keeping *levels* (display unit) if given."""
        if mode not in LEVEL_MODES:
            raise ValueError(f"unknown level mode {mode!r}")
        view = self._view
        kept = dict(view.levels)
        if levels is not None:
            kept[key] = (float(levels[0]), float(levels[1]))
        self.set_view(levels=kept, level_modes={**view.level_modes, key: mode})

    @property
    def selection(self) -> PlotSelection:
        return self._selection

    def set_selection(self, **changes) -> None:
        """Show another map. The stacked offset keeps its size per cm^-1: it is rescaled
        when a per-unit energy derivative is shown or left in another unit, so a unit switch
        made on one never changes the offset of the other maps (see :meth:`ViewState.converted`).
        """
        new = dataclasses.replace(self._selection, **changes)
        old = self._selection
        if new != old:
            self._selection = new
            if not self._restoring:
                factor = _selection_scale(self._unit, new) / _selection_scale(self._unit, old)
                if factor != 1.0:
                    offset = self._view.stacked_offset * factor
                    self._view = dataclasses.replace(self._view, stacked_offset=offset)
            self.selectionChanged.emit()

    def current_levels(self) -> tuple[float, float] | None:
        """Levels of the map shown, or None to autoscale (1st-99th percentile)."""
        return self._levels_to_draw(self._selection.level_key, self.current_map)

    def reference_levels(self) -> tuple[float, float] | None:
        """Levels of the reference map, or None to autoscale (1st-99th percentile)."""
        key = level_key(self._selection.reference_kind)
        return self._levels_to_draw(key, self.reference_map)

    def _levels_to_draw(
        self, key: str, get_map: Callable[[], FieldMap | None]
    ) -> tuple[float, float] | None:
        levels = self._view.levels_for(key)
        symmetric = self._view.level_mode(key) == SYMMETRIC_LEVELS
        if levels is None and symmetric and self.result is not None:
            fmap = get_map()
            if fmap is not None:
                levels = self._view.levels_in_effect(key, fmap.values)
        return levels

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

    def processed_options(self) -> ProcessingState | None:
        """The processing options of the last Process (with the baseline Live applied since),
        None before the first."""
        return self._processed_with

    def _update_changed(self) -> None:
        used = self._processed_with
        if used is not None and self._baseline_is_live():  # applied at once (apply_baseline)
            used = dataclasses.replace(used, baseline=self._processing.baseline)
        changed = used is not None and used.effective() != self._processing.effective()
        if changed != self._changed:
            self._changed = changed
            self.changedSinceProcess.emit(changed)
        for part in DATA_PARTS:
            state = self.data_state(part)
            if state != self._data_states[part]:
                self._data_states[part] = state
                self.dataStateChanged.emit(part, state)

    def data_state(self, part: str) -> str:
        """State of the *part*'s ("sample" or "reference") data: :data:`DATA_EMPTY` without
        files, :data:`DATA_CURRENT` when the last Process used them and their options as they
        are, :data:`DATA_CHANGED` when they were not processed yet or changed since.

        The reference follows its sweep in Separate mode, the sample in Self mode (the sample
        is its own reference) and is empty in None mode. Library maps leave the states as they
        are (they describe the last Process), and so does the unit.
        """
        p = self._processing
        loaded = p.sample_files
        if part == "reference":
            mode = p.reference_mode
            if mode is ReferenceMode.NONE:
                return DATA_EMPTY
            loaded = p.reference_files if mode is ReferenceMode.SEPARATE else loaded
        elif part != "sample":
            raise ValueError(f"no data part {part!r}; use one of {DATA_PARTS}")
        if loaded == SweepFiles():
            return DATA_EMPTY
        used = self._processed_with
        if used is None:
            return DATA_CHANGED
        parts = [part]
        if part == "reference" and p.reference_mode is ReferenceMode.SELF:
            parts.append("sample")  # the sample is the reference
        same = all(used.data_options(each) == p.data_options(each) for each in parts)
        return DATA_CURRENT if same else DATA_CHANGED

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
            field_values = None
            if self._processing.custom_field:
                field_values = field_range.values(expected=len(files.field) or None)
            return load_measurement(
                files.zero, files.field, field=field_values, read=self.read_spectrum
            )

    def _cut(self, measurement: Measurement) -> Measurement:
        cut = self._processing.energy_cut
        if cut is None:
            return measurement
        spectra = measurement.spectra
        self.check_energy_range("energy window", cut, spectra.energy, "processing")
        mask = energy_mask(spectra.energy, *cut)
        return Measurement(spectra=crop_energy(spectra, *cut), zero=measurement.zero[mask])

    def process(self, quiet: bool = False) -> ProcessResult:
        """Load the sweep(s), process them and show the result.

        A separate reference sweep without files is left out: the sample is processed as
        without a reference, and :attr:`referenceMissing` says so after the result is shown
        (unless *quiet*: the repeated updates of a watched folder, :meth:`process_update`).
        """
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
        mode = p.reference_used()
        missing = mode is not p.reference_mode  # a separate sweep without files
        if missing and not quiet:
            logger.warning(NO_REFERENCE_FILES)
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
        # after resultChanged, whose listeners close the bar a note shows in
        if missing and not quiet:
            self.referenceMissing.emit(NO_REFERENCE_FILES)
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
            self._kept_result = None  # the sweep a library map hid is out of date
        self._init_points_if_requested(result.ratio.field)
        self.resultChanged.emit()
        self._update_changed()

    def from_map(self, fmap: FieldMap) -> ProcessResult:
        """Show an already processed map (library), baseline-corrected like a processed one."""
        self.settle_baseline()  # the processed map leaves with the region it counts as having
        baseline = self._baseline(panel="processing")
        if baseline is not None:
            self.check_energy_range("baseline region", baseline, fmap.energy, "processing")
            if not _has_values(fmap, baseline):
                lo, hi = convert_range(baseline, Unit.CM1, self._unit)
                raise panel_error(
                    f"baseline region {fmt(lo)} – {fmt(hi)} {self._unit} contains no data "
                    "(it lies in a gap of the map)",
                    "processing",
                )
        result = ProcessResult.from_map(fmap, baseline)
        self.set_result(result, processed=False)
        return result

    # --- watched folder (gui/watch.py) ------------------------------------------------
    # While a folder is watched, the spectra read are kept (SpectrumCache: by path, size and
    # modification time), so an update reads only the new or changed files; Process reads
    # through the same cache then. process_update processes as Process does.
    def set_spectrum_cache(self, on: bool) -> None:
        """Keep the spectra read (*on*), or read every file each time (off: they are freed)."""
        if not on:
            self._spectra = None
        elif self._spectra is None:
            self._spectra = SpectrumCache()

    def spectrum_cache(self) -> SpectrumCache | None:
        return self._spectra

    def read_spectrum(self, path: str | Path) -> tuple[np.ndarray, np.ndarray]:
        """Read one spectrum (cm^-1), through the cache while there is one."""
        cache = self._spectra
        return read_spectrum(path) if cache is None else cache.read(path)

    def process_update(self, first: bool = False) -> ProcessResult:
        """Process again for an update of a watched folder, exactly as :meth:`process` does
        with the current options; points, view and levels stay. Only the *first* update of a
        watch gives the note about missing reference files. The cache then keeps only the
        files in use."""
        result = self.process(quiet=not first)
        if self._spectra is not None:
            p = self._processing
            files = (p.sample_files, p.reference_files)
            self._spectra.retain(path for f in files for path in (*f.zero, *f.field))
        return result

    # --- live baseline -----------------------------------------------------------------
    # With Live on, the baseline region is applied to the maps of the last Process as it
    # changes (apply_baseline, which the Processing panel calls at most about ten times a
    # second while the region is dragged), so changing it never makes the map out of date.
    # Nothing is loaded or processed again; a library map keeps the baseline it was plotted with.
    def live_baseline(self) -> bool:
        return self._live_baseline

    def set_live_baseline(self, live: bool) -> None:
        """Count the baseline region as applied whenever :meth:`apply_baseline` can apply it."""
        if bool(live) != self._live_baseline:
            self._live_baseline = bool(live)
            self._update_changed()

    def can_apply_baseline(self) -> bool:
        """A map made by Process is shown (a library map keeps its baseline)."""
        return self.result is not None and self.result_source == "process"

    def baseline_is_live(self) -> bool:
        """The baseline region of the processing options counts as applied to the map shown
        (Live applies it at once): Live is on, the map is processed and the region is off or
        can be applied to it."""
        return self._baseline_is_live()

    def _baseline_is_live(self) -> bool:
        """Live is on and the region can be applied to the map shown."""
        if not (self._live_baseline and self.can_apply_baseline()):
            return False
        try:
            region = self._baseline()
        except ValueError:
            return False
        return region is None or bool(energy_mask(self.result.ratio.energy, *region).any())

    def settle_baseline(self) -> bool:
        """Apply a live region that counts as applied but is still waiting (the panel applies
        changes at most about ten times a second), before the map shown is kept or replaced or
        Live is turned off. Returns whether the map changed. Nothing is applied while settings
        are restored (:meth:`restoring`): restored values are applied after it, if at all."""
        if self._restoring:
            return False
        if self._baseline_is_live() and self._baseline() != self.result.baseline_region:
            return self.apply_baseline()
        return False

    def apply_baseline(self) -> bool:
        """Apply the baseline region of the processing options (none when it is off) to the
        map of the last Process: only the normalisation is computed
        (:meth:`ProcessResult.with_baseline`). The points, view ranges and colour levels stay,
        and the region counts as processed. Returns whether the map changed.

        ValueError (in the display unit) without a processed map or for a region that cannot
        be used; :attr:`referenceMissing` is never emitted.
        """
        result = self.result
        if result is None or not self.can_apply_baseline():
            why = "no map is processed yet" if result is None else LIBRARY_KEEPS_BASELINE
            raise panel_error(f"baseline region not applied: {why}", "processing")
        region = self._baseline()
        if region is not None:
            self.check_energy_range("baseline region", region, result.ratio.energy, "processing")
        changed = region != result.baseline_region
        if changed:
            self.result = result.with_baseline(region)
            logger.debug("Baseline re-applied: %s cm-1", region)
            self.resultChanged.emit()
        if self._processed_with is not None:
            baseline = self._processing.baseline
            self._processed_with = dataclasses.replace(self._processed_with, baseline=baseline)
        self._update_changed()
        return changed

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

    # --- models and fitting (Models section) -------------------------------------------
    # Models register their curves with set_overlay; overlaysChanged tells listeners (e.g. an
    # export preview) that registered curves changed. Fits use the picked points (cm^-1).
    overlaysChanged = Signal()

    def notify_overlays(self) -> None:
        """Announce that overlay curves registered with :meth:`set_overlay` changed."""
        self.overlaysChanged.emit()

    def picked_curves(self) -> dict[str, int]:
        """The curves with picked points and how many each has, in table order."""
        if self.points is None:
            return {}
        counts = {
            name: int(np.isfinite(self.points.column(name)).sum()) for name in self.points.names
        }
        return {name: n for name, n in counts.items() if n}

    def fit_observations(
        self, mapping: Mapping[str, int | None], unit: Unit | str
    ) -> list[Observation]:
        """The picked curves of *mapping* (curve -> branch; None skips it), energies in *unit*."""
        if not self.picked_curves():
            raise panel_error("there are no picked points to fit; pick some first", "points")
        observations = []
        for name, branch in mapping.items():
            if branch is None or self.points is None or name not in self.points.names:
                continue
            field_values, energy = self.points.points(name)
            if field_values.size:
                observations.append(Observation(field_values, from_cm1(energy, unit), branch, name))
        if not observations:
            raise ValueError("assign at least one picked curve to a branch")
        return observations

    def prepare_fit(self, model: Model, mapping: Mapping[str, int | None]) -> list[Observation]:
        """The observations to fit *model* to; ValueError when they cannot fit it (no points,
        no curve assigned, every parameter fixed, not more points than free parameters)."""
        observations = self.fit_observations(mapping, model.unit)
        n_points = sum(obs.field.size for obs in observations)
        n_free = sum(not p.fixed for p in model.params)
        if n_free == 0:
            raise ValueError("every parameter is fixed, so there is nothing to fit")
        if n_points <= n_free:
            raise ValueError(
                f"{n_points} points cannot fit {n_free} free parameters; "
                "pick more points or fix parameters"
            )
        return observations

    @staticmethod
    def run_fit(
        model: Model, observations: list[Observation], assignment: Assignment | str
    ) -> FitResult:
        """:func:`fit`, with a ValueError when it does not converge (no Qt: a worker thread
        may run it on a copy of the model)."""
        result = fit(model, observations, assignment)
        if not result.success:
            raise ValueError(f"the fit did not converge ({result.message})")
        if not all(math.isfinite(v) for v in result.values.values()):
            raise ValueError("the fit gave values that are not finite")
        return result

    @staticmethod
    def log_fit(result: FitResult, assignment: Assignment | str, unit: Unit | str) -> None:
        n_points = sum(r.size for r in result.residuals)
        logger.info(
            "Fit (%s, %d points): chi2 = %.4g %s^2, dof = %d",
            Assignment(assignment).value,
            n_points,
            result.chi2,
            unit,
            result.dof,
        )

    # --- points ------------------------------------------------------------------------
    @contextlib.contextmanager
    def point_edit(self, text: str, merge: Hashable | None = None) -> Iterator[None]:
        """Change the points inside the block as one undo step called *text*.

        Blocks nest (the outermost one makes the step), and :attr:`pointsChanged` fires once at
        the end if anything changed. Edits that raise half-way still become a step. Steps with
        the same *merge* key in a row are merged (e.g. the letters of a typed name).
        """
        outer = not self._point_edits
        before = self.points_undo.capture(self.points, self.curve) if outer else None
        self._point_edits += 1
        try:
            yield
        finally:
            self._point_edits -= 1
            if before is not None:
                after = self.points_undo.capture(self.points, self.curve)
                if self.points_undo.record(text, before, after, merge):
                    self.pointsChanged.emit()

    def _restore_points(self, state: PointsState) -> None:
        """Put back an undone or redone table, with rows for the fields of the map shown
        (a table from before a map on another grid lacks them)."""
        table = state.table()
        if table is not None and self.result is not None:
            table = table.with_fields(self.result.ratio.field)
        self.points = table
        self.curve = state.curve
        self.pointsChanged.emit()

    def curve_names(self) -> list[str]:
        """The curves: the table's columns, then the current curve if it has no column yet."""
        names = self.points.names if self.points is not None else []
        if self.curve and self.curve not in names:
            names.append(self.curve)
        return names

    def set_curve(self, name: str) -> None:
        """Make *name* the current curve, the one clicks record into (not an undo step)."""
        name = name.strip()
        if name != self.curve:
            self.curve = name
            self.pointsChanged.emit()

    def curve_name(self) -> str:
        if not self.curve:
            raise panel_error("enter a curve name for the picked points", "points")
        return self.curve

    def curve_name_problem(self, name: str, old: str | None = None) -> str | None:
        """Why *name* cannot name a curve (renaming *old*), or None if it can."""
        name = name.strip()
        if problem := _curve_name_rule(name):
            return problem
        if name != old and name in self.curve_names():
            return f"a curve named {name!r} exists already"
        return None

    def add_curve(self, name: str | None = None) -> str:
        """Add an empty curve (by default the first free "LL n") and make it current."""
        if self.points is None:
            raise panel_error("no point table - process data first", "points")
        names = self.curve_names()
        if name is None:
            n = len(names) + 1
            while f"LL {n}" in names:
                n += 1
            name = f"LL {n}"
        name = name.strip()
        if problem := self.curve_name_problem(name):
            raise panel_error(problem, "points")
        with self.point_edit(f"New curve {name}"):
            self.points.add_column(name)
            self.curve = name
        return name

    def rename_curve(
        self, name: str, old: str | None = None, merge: Hashable | None = None
    ) -> None:
        """Rename curve *old* (default: the current one); the current curve follows.

        *merge*: renames with the same key in a row are one undo step (typing a name).
        """
        old = self.curve if old is None else old
        name = name.strip()
        if problem := self.curve_name_problem(name, old):
            raise panel_error(problem, "points")
        if name == old:
            return
        with self.point_edit(f"Rename curve {old}", merge):
            table = self.points
            if table is not None and old in table.names:
                columns = {(name if n == old else n): table.column(n) for n in table.names}
                self.points = PointTable(table.field, columns)
            if self.curve == old:
                self.curve = name
        logger.debug("Curve %r renamed to %r.", old, name)

    def set_new_table(self, new: bool) -> None:
        if bool(new) != self.new_table:
            self.new_table = bool(new)
            self.pointsChanged.emit()

    def _init_points_if_requested(self, field_values: np.ndarray) -> None:
        """A new (empty) table on the result's field, keeping the curve names, if asked;
        else the table gets rows for the result's fields it lacks (:meth:`PointTable.with_fields`),
        so that every point keeps its own field and a pick never lands on another one."""
        if not self.new_table and self.points is not None:
            table = self.points.with_fields(field_values)
            if table is not self.points:
                added = table.field.size - self.points.field.size
                self.points = table  # no undo step: the points are the same
                logger.info("Point table: %d field rows added for the new map.", added)
                self.pointsChanged.emit()
            return
        names = self.curve_names() or ["LL 1"]
        table = PointTable(
            field_values, {name: np.full(field_values.shape, np.nan) for name in names}
        )
        curve = self.curve if self.curve in names else names[0]
        if self.points is None:  # the first table: nothing to undo before it
            self.points, self.curve = table, curve
            self.points_undo.clear()
            self.pointsChanged.emit()
        else:
            with self.point_edit("New point table"):
                self.points, self.curve = table, curve
        self.set_new_table(False)
        logger.info("New point extraction table initialized.")

    def map_field(self, b: float) -> float | None:
        """Field of the map column drawn at *b*: its nearest field (see
        :func:`~mag_opt_detective.core.spectra.sample_at`), or None without a map.
        ValueError beyond the map's first and last columns."""
        if self.result is None:
            return None
        try:
            field_values = self.result.base(self._selection.kind).field
        except ValueError:  # no field-step ratio: the field of the other maps
            field_values = self.result.ratio.field
        column = sample_at(field_values, b)
        if column is None:
            lo, hi = field_values.min(), field_values.max()
            raise ValueError(f"B = {b:.4g} T is outside the map ({lo:g} – {hi:g} T)")
        return float(field_values[column])

    def record_point(self, b: float, energy: float) -> int:
        """Record a point clicked at (*b*, *energy*), energy in the display unit (kept in
        cm^-1). It goes to the field of the map column under the click, whose row is added
        if the table lacks it (without a map: the table row of *b*)."""
        if self.points is None:
            raise panel_error("no point table - process data first", "points")
        name = self.curve_name()
        column = self.map_field(b)
        with self.point_edit(f"Record point on {name}"):
            if column is not None:
                b = column
                self.points = self.points.with_fields(np.array([b]))
            row = self.points.set_nearest(name, b, float(to_cm1(energy, self._unit)))
        logger.info("%s: B = %g T -> E = %.4g %s", name, self.points.field[row], energy, self._unit)
        return row

    def record_points(
        self,
        field: np.ndarray,
        energy: np.ndarray,
        curve: str | None = None,
        unit: Unit | str | None = None,
        text: str | None = None,
    ) -> int:
        """Record many points as one undo step (e.g. an auto-pick result); returns how many
        it stored (a field given twice is stored once, with its last energy).

        Each point goes to the row of its own field in *curve* (default: the current one),
        which is added if the table lacks it; *energy* is in *unit* (default: the display
        unit). NaN energies are skipped.
        """
        if self.points is None:
            raise panel_error("no point table - process data first", "points")
        name = self.curve_name() if curve is None else curve.strip()
        if name not in self.curve_names() and (problem := self.curve_name_problem(name)):
            raise panel_error(problem, "points")
        field, energy = np.broadcast_arrays(np.asarray(field, float), np.asarray(energy, float))
        keep = np.isfinite(field) & np.isfinite(energy)
        energy_cm1 = to_cm1(energy[keep], self._unit if unit is None else Unit(unit))
        rows = set()
        with self.point_edit(text or f"Record {int(keep.sum())} points on {name}"):
            self.points = self.points.with_fields(field[keep])
            for b, e in zip(field[keep], energy_cm1, strict=True):
                rows.add(self.points.set_nearest(name, float(b), float(e)))
        logger.info("%s: %d points recorded", name, len(rows))
        return len(rows)

    def remove_point(self, b: float) -> int:
        """Remove the current curve's point in the field row nearest to *b* (if there is one)."""
        if self.points is None:
            raise panel_error("no point table - process data first", "points")
        name = self.curve_name()
        row = self.points.nearest_row(b)
        if name not in self.points.names or np.isnan(self.points.column(name)[row]):
            return row
        with self.point_edit(f"Remove point from {name}"):
            self.points.clear_nearest(name, b)
        logger.info("%s: point at B = %g T removed", name, self.points.field[row])
        return row

    def drop_curve(self, name: str | None = None) -> None:
        """Delete curve *name* (default: the current one); its neighbour becomes current.

        The table always keeps one curve: deleting the last leaves an empty "LL 1".
        """
        if self.points is None:
            return
        name = self.curve_name() if name is None else name
        names = self.points.names
        with self.point_edit(f"Delete curve {name}"):
            self.points.drop(name)
            rest = self.points.names
            if not rest:
                rest = ["LL 1"]
                self.points.add_column(rest[0])
            if self.curve == name or self.curve not in rest:
                index = names.index(name) if name in names else 0
                self.curve = rest[min(index, len(rest) - 1)]
        logger.info("Curve %r dropped.", name)

    def load_points(self, path: str | Path) -> PointTable:
        """Read a point table; one without a unit in its header is in the display unit."""
        with in_panel("points"):
            table = PointTable.load_tsv(path, default_unit=self._unit)
            for name in table.names:
                if problem := _curve_name_rule(name):
                    raise ValueError(f"{Path(path).name}: curve {name!r}: {problem}")
        with self.point_edit(f"Import {Path(path).name}"):
            self.points = table
            if table.names:
                self.curve = table.names[0]
        self.set_new_table(False)
        logger.info("Loaded points %s (%s)", Path(path).name, ", ".join(table.names))
        return table

    def save_points(self, path: str | Path) -> None:
        if self.points is None:
            raise panel_error("no points to export", "points")
        self.points.save_tsv(path, unit=self._unit)
        logger.info("Exported points to %s", path)

    # --- library -----------------------------------------------------------------------
    # Library maps are kept in cm^-1. A ticked map is on the plot: the ticked maps are drawn
    # together, the one ticked or brought up last on top (:meth:`plotted_entries`). The top
    # map is the result shown (picking, the stacked plot, exports); the others are drawn below
    # it (:meth:`overlay_maps`). Combining ticked maps makes a product, shown until it is saved
    # into the library or discarded. Process shows the processed sweep again and keeps the
    # ticks (:meth:`showing`); the sweep a library map hides is kept (:meth:`show_processed`).
    plottedChanged = Signal()  # what the library puts on the plot changed (ticks, order, source)
    overlayMapsChanged = Signal()  # the maps below the top one changed, the result did not
    opacityChanged = Signal(float)
    productChanged = Signal()  # a product was made, renamed, saved or discarded

    def entry(self, key: int) -> LibraryEntry:
        for entry in self.library:
            if entry.key == key:
                return entry
        raise panel_error("that map is no longer in the library", "library")

    def _unique_name(self, name: str) -> str:
        names = {entry.name for entry in self.library}
        if name not in names:
            return name
        n = 2
        while f"{name} ({n})" in names:
            n += 1
        return f"{name} ({n})"

    def add_map(
        self,
        fmap: FieldMap,
        name: str,
        kind: str = KIND_LABELS[PlotKind.RATIO],
        source: str = "",
        provenance: Provenance | None = None,
    ) -> LibraryEntry:
        """Add a processed map (kept in cm^-1) to the library, not ticked."""
        self._next_key += 1
        entry = LibraryEntry(
            name=self._unique_name(name.strip() or "Map"),
            fmap=fmap.to_unit(Unit.CM1),
            kind=kind,
            source=source,
            key=self._next_key,
            provenance=provenance,
        )
        self.library.append(entry)
        self.libraryChanged.emit()
        return entry

    def load_table(self, path: str | Path, field_values: np.ndarray | None = None) -> LibraryEntry:
        """Add an exported table to the library (kept in cm^-1).

        A table without a unit in its header is in the display unit. *field_values* replace
        the field read from the header. The kind comes from the name an export gives
        (``…_Data.csv``; R(B)/R(0) without one).
        """
        with in_panel("library"):
            fmap = load_tsv(path, default_unit=self._unit)
            file_unit = fmap.unit
            if field_values is not None:
                if field_values.size != fmap.field.size:
                    raise ValueError(
                        f"custom field range has {field_values.size} values, "
                        f"the table has {fmap.field.size}"
                    )
                fmap = fmap.replace(field=field_values)
        provenance = Provenance("table", datetime.now(), source=str(path))
        name = Path(path).name
        entry = self.add_map(fmap, name, kind_from_name(name), str(path), provenance)
        logger.info("Library: loaded %s (energy in %s)", Path(path).name, file_unit)
        return entry

    def sweep_name(self) -> str:
        """The name of the processed sweep: its files' common prefix."""
        used = self._processed_with
        files = used.sample_files.field if used is not None else ()
        return common_prefix([Path(f).name for f in files]).strip(SEPARATORS) or "Processed map"

    def result_name(self) -> str:
        """A name for the map shown: the sweep's common file prefix, or the library map's."""
        if self.result_source == "library":
            return self._library_name or "Library map"
        return self.sweep_name()

    def processed_result(self) -> ProcessResult | None:
        """The result of the last Process, shown or hidden by a library map (None before)."""
        return self.result if self.result_source == "process" else self._kept_result

    def processed_in_library(self) -> bool:
        """The processed sweep's R(B)/R(0) map, as it is, is in the library already."""
        result = self.processed_result()
        return result is not None and any(e.fmap is result.ratio for e in self.library)

    def add_processed(self, name: str | None = None) -> LibraryEntry:
        """Add the processed sweep's R(B)/R(0) map to the library, with how it was made."""
        if self.result_source == "process":
            self.settle_baseline()  # with the region it counts as having
        result = self.processed_result()
        if result is None:
            raise panel_error("process a sweep first: there is no processed map to add", "library")
        p = self._processed_with or self._processing
        mode = p.reference_used()
        reference = {ReferenceMode.NONE: "none", ReferenceMode.SELF: "the sample itself"}.get(
            mode, f"a separate sweep of {len(p.reference_files.field)} files"
        )
        if mode is not ReferenceMode.NONE and p.smooth:
            reference += f", smoothed (window {p.sg_window}, order {p.sg_poly})"
        provenance = Provenance(
            "process",
            datetime.now(),
            source=self.sweep_name(),
            files=(len(p.sample_files.field), len(p.sample_files.zero)),
            reference=reference,
            window=p.energy_cut,
            baseline=result.baseline_region,
        )
        entry = self.add_map(result.ratio, name or self.sweep_name(), provenance=provenance)
        logger.info("Library: added the processed R(B)/R(0) as %r", entry.name)
        return entry

    def remove_entry(self, entry: LibraryEntry) -> None:
        if entry not in self.library:
            return
        plotted = entry.key in self._plotted
        self.library.remove(entry)
        logger.info("Library: removed %r", entry.name)
        if plotted:
            self._plotted.remove(entry.key)
        self.libraryChanged.emit()
        if plotted and self._showing_maps():
            self._show_next()  # the next map comes on top

    def update_entry(self, entry: LibraryEntry, **changes) -> None:
        """Change an entry's ``used`` tick, ``energy_cut`` (cm^-1) or ``field_cut`` (T).

        Ticking shows the map on top of the ticked ones, unticking takes it off the plot; a
        map that cannot be shown or left (e.g. the baseline region misses the next one) keeps
        its tick and the error is raised. Limits are kept as given: maps plotted cut to their
        limits are drawn with new ones by :meth:`redraw_ticked` (the panel calls it once typing
        pauses), so a limit typed half-way never fails.
        """
        unknown = set(changes) - {"used", "energy_cut", "field_cut"}
        if unknown:
            raise TypeError(f"cannot change {sorted(unknown)} of a library entry")
        old_used, before = entry.used, list(self._plotted)
        changed = set()
        for name, value in changes.items():
            if name != "used":
                value = tuple(None if v is None else float(v) for v in value)
            else:
                value = bool(value)
            if getattr(entry, name) != value:
                setattr(entry, name, value)
                changed.add(name)
        if not changed:
            return
        if "used" in changed:
            if entry.key in self._plotted:
                self._plotted.remove(entry.key)
            if entry.used:
                self._plotted.append(entry.key)
        self.entryChanged.emit(entry.key)
        if "used" in changed:
            try:
                self._show_ticked()
            except Exception:
                entry.used, self._plotted = old_used, before  # the limits stay
                self.entryChanged.emit(entry.key)
                raise

    def redraw_ticked(self) -> None:
        """Draw the ticked maps on the plot again with their limits, when they are plotted
        cut to them (after limits changed)."""
        if self._showing_maps() and not self._full_range:
            self._show_ticked()

    def ticked(self) -> list[LibraryEntry]:
        return [entry for entry in self.library if entry.used]

    def plotted_entries(self) -> list[LibraryEntry]:
        """The ticked maps in the order they are drawn: bottom first, the top one last."""
        by_key = {entry.key: entry for entry in self.library}
        return [by_key[key] for key in self._plotted if key in by_key]

    def showing(self) -> str:
        """What the plot shows: the processed sweep (:data:`SHOWS_PROCESS`), the ticked library
        maps (:data:`SHOWS_MAPS`), the product (:data:`SHOWS_PRODUCT`), or nothing ("")."""
        if self.result is None:
            return ""
        if self.result_source != "library":
            return SHOWS_PROCESS
        return self._library_shows

    def _showing_maps(self) -> bool:
        return self.showing() == SHOWS_MAPS

    def plot_title(self, overlays: bool = True) -> str:
        """The name of the map shown; with *overlays*, also the maps drawn below it."""
        shows = self.showing()
        if shows != SHOWS_MAPS:
            return self.result_name() if shows else ""
        entries = self.plotted_entries()
        title = self.result_name()
        if overlays and len(entries) == 2:
            title += f" over {entries[0].name}"
        elif overlays and len(entries) > 2:
            title += f" over {len(entries) - 1} maps"
        return title

    def plot_entry(self, entry: LibraryEntry) -> ProcessResult:
        """Show *entry* on top of the ticked maps (ticking it); the map on top is plotted
        again (with the baseline region of the processing options)."""
        if entry.used and self._plotted and self._plotted[-1] == entry.key:
            self._shown_top = None
            self._show_ticked()
            return self.result
        logger.info("-" * 40)
        logger.info("Plotting library map %r", entry.name)
        before = list(self._plotted)
        if entry.used:
            self._plotted.remove(entry.key)
            self._plotted.append(entry.key)
            try:
                self._show_ticked()
            except Exception:
                self._plotted = before
                raise
        else:
            self.update_entry(entry, used=True)
        return self.result

    def show_ticked(self) -> None:
        """Show the ticked maps again (after Process or a product hid them)."""
        if not self._plotted:
            raise panel_error("tick the maps to show first", "library")
        self._show_ticked()

    def _show_ticked(self) -> None:
        """Show the ticked maps; without any, the processed sweep (or nothing)."""
        entries = self.plotted_entries()
        if not entries:
            if self._showing_maps():
                self._shown_top = None
                if self._kept_result is not None:
                    self.show_processed()
                else:
                    self._clear_result()
            self.plottedChanged.emit()
            return
        top = entries[-1]
        shown = (top.key, None if self._full_range else (top.energy_cut, top.field_cut))
        if self._showing_maps() and shown == self._shown_top:
            self.overlayMapsChanged.emit()
        else:
            self._show_library(self.entry_map(top), top.name, SHOWS_MAPS)
            self._shown_top = shown
        self.plottedChanged.emit()

    def _show_next(self) -> None:
        """Show the ticked maps after the top one left; when the next one cannot be shown, the
        processed sweep (or nothing) instead, and raise why."""
        try:
            self._show_ticked()
        except Exception:
            self._leave_library()
            raise

    def _leave_library(self) -> None:
        """Show the processed sweep, or nothing, instead of library maps (the ticks stay)."""
        self._library_shows, self._shown_top = SHOWS_MAPS, None
        if self._kept_result is not None:
            self.show_processed()
        else:
            self._clear_result()
        self.plottedChanged.emit()

    def show_processed(self) -> None:
        """Show the processed sweep a library map hid (Process is not run again)."""
        result = self._kept_result
        if self.result_source == "process":
            return
        if result is None:
            raise panel_error("process a sweep first: there is no processed map", "library")
        self._kept_result = None
        self._library_name = ""
        self.result, self.result_source = result, "process"
        self._init_points_if_requested(result.ratio.field)
        self.resultChanged.emit()
        self._update_changed()
        self.settle_baseline()  # a live region changed while the sweep was hidden
        self.plottedChanged.emit()

    def _clear_result(self) -> None:
        """Show nothing (no map ticked and no processed sweep to go back to)."""
        self.result, self.result_source, self._library_name = None, "", ""
        self._shown_top = None
        self.resultChanged.emit()
        self._update_changed()

    @property
    def full_range(self) -> bool:
        """Library maps are plotted whole (else cut to their E and B limits)."""
        return self._full_range

    def set_full_range(self, full: bool) -> None:
        if bool(full) != self._full_range:
            self._full_range = bool(full)
            if self._showing_maps():
                self._show_ticked()

    @property
    def overlay_opacity(self) -> float:
        """Opacity of the ticked maps drawn above the bottom one (0 - 1)."""
        return self._opacity

    def set_overlay_opacity(self, opacity: float) -> None:
        opacity = min(max(float(opacity), 0.0), 1.0)
        if opacity != self._opacity:
            self._opacity = opacity
            self.opacityChanged.emit(opacity)

    def entry_map(self, entry: LibraryEntry) -> FieldMap:
        """*entry*'s map as plotted (cm^-1): whole, or cut to its limits."""
        return entry.fmap if self._full_range else self.cut_map(entry)

    def cut_map(self, entry: LibraryEntry) -> FieldMap:
        """*entry*'s map cut to its E and B limits (errors name it, in the display unit)."""
        fmap = entry.fmap
        if entry.energy_cut != (None, None):
            rng = entry.energy_cut
            self.check_energy_range(f"{entry.name}: E range", rng, fmap.energy, "library")
            fmap = crop_energy(fmap, *rng)
        lo, hi = entry.field_cut
        if (lo, hi) == (None, None):
            return fmap
        if lo is not None and hi is not None and lo >= hi:
            raise panel_error(
                f"{entry.name}: B range: the first value must be below the second; it is "
                f"{fmt(lo)} – {fmt(hi)} T",
                "library",
            )
        b = fmap.field
        if not ((lo is None or b >= lo - FIELD_TOL) & (hi is None or b <= hi + FIELD_TOL)).any():
            raise panel_error(
                f"{entry.name}: B range {fmt(lo)} – {fmt(hi)} T contains no data "
                f"(the map spans {b.min():g} – {b.max():g} T)",
                "library",
            )
        with in_panel("library"):
            return crop_field(fmap, lo, hi)

    def overlay_layers(self) -> list[OverlayLayer]:
        """The ticked maps below the top one, bottom first, each as drawn (as the top one is
        shown: plot kind, derivative, baseline region, display unit) or why it is not: a map
        of another kind than the top one, or one that cannot be shown so (e.g. its limits or
        the baseline region hold no data)."""
        entries = self.plotted_entries()
        if not (self._showing_maps() and entries):  # (none while the top one is replaced)
            return []
        s = self._selection
        parts = tuple((e.key, e.kind, e.energy_cut, e.field_cut, e.fmap) for e in entries)
        key = (self.result, s, self._unit, self._full_range, parts)
        if self._layers is not None and self._layers[0] == key:
            return self._layers[1]
        top = entries[-1]
        region = self.result.baseline_region  # the top map's: library maps keep theirs
        layers = []
        for entry in entries[:-1]:
            if entry.kind != top.kind:
                problem = f"it is {entry.kind} and the top map {top.kind}"
                layers.append(OverlayLayer(entry, None, problem))
                continue
            try:
                fmap = self.entry_map(entry)
                if region is not None and not _has_values(fmap, region):
                    raise ValueError("the baseline region holds none of its data")
                result = ProcessResult.from_map(fmap, region)
                shown = result.get(s.kind, s.order, s.axis, s.physical, self._unit)
                layers.append(OverlayLayer(entry, shown))
            except ValueError as exc:
                problem = str(exc).removeprefix(f"{entry.name}: ")
                layers.append(OverlayLayer(entry, None, problem))
                logger.debug("Library: %s is not drawn: %s", entry.name, problem)
        self._layers = (key, layers)
        return layers

    def overlay_maps(self) -> list[FieldMap]:
        """The ticked maps drawn below the top one, bottom first (see :meth:`overlay_layers`)."""
        return [layer.fmap for layer in self.overlay_layers() if layer.fmap is not None]

    def _show_library(self, fmap: FieldMap, name: str, shows: str) -> ProcessResult:
        """Show a library map or the product (*shows*), keeping the processed sweep."""
        if self.result_source == "process":
            self.settle_baseline()  # the processed map leaves with the region it counts as having
        kept = self.result if self.result_source == "process" else self._kept_result
        old = self._library_name, self._library_shows, self._kept_result
        self._library_name, self._library_shows = name, shows  # read when the result is shown
        self._kept_result = kept
        try:
            return self.from_map(fmap)
        except Exception:
            self._library_name, self._library_shows, self._kept_result = old  # nothing changed
            raise

    # --- combining ticked maps (the product) --------------------------------------------
    def combine_preview(self, method: str, entries: list[LibraryEntry] | None = None):
        """What :meth:`make_product` would give for *entries* (default: the ticked maps)."""
        parts = tuple(self.ticked() if entries is None else entries)
        try:
            fmap, facts = self._combine(method, parts)
        except EXPECTED_ERRORS as exc:
            return CombinePreview(method, parts, problem=str(exc))
        return CombinePreview(method, parts, fmap, facts)

    def _combine(
        self, method: str, parts: tuple[LibraryEntry, ...]
    ) -> tuple[FieldMap, tuple[tuple[str, float, float], ...]]:
        """Combine *parts*, each cut to its limits; errors say why not, naming the maps."""
        if method not in COMBINE_METHODS:
            raise ValueError(f"unknown way to combine maps: {method!r}")
        if len(parts) < 2:
            raise panel_error("tick at least two maps to merge or average them", "library")
        kinds = list(dict.fromkeys(p.kind for p in parts))
        if len(kinds) > 1:
            what = ", ".join(f"{p.name} is {p.kind}" for p in parts)
            raise panel_error(f"combine maps of one kind: {what}", "library")
        maps = [self.cut_map(p) for p in parts]
        if method in ("energy", "average"):
            first = maps[0].field
            if any(m.field.shape != first.shape or not np.allclose(m.field, first) for m in maps):
                fields = "; ".join(
                    _fields_text(p.name, m) for p, m in zip(parts, maps, strict=True)
                )
                how = "merging by energy" if method == "energy" else "averaging"
                raise panel_error(
                    f"{how} needs the same fields in every map ({fields}): cut them to the "
                    "same fields with B limits, or merge by field",
                    "library",
                )
        if method == "energy":
            return self._merge_energy(parts, maps)
        lo = max(m.energy.min() for m in maps)
        hi = min(m.energy.max() for m in maps)
        common = energy_mask(maps[0].energy, lo, hi).sum() >= 2
        if not common:
            spans = "; ".join(
                f"{p.name} {energy_range_text((m.energy.min(), m.energy.max()), self._unit)}"
                for p, m in zip(parts, maps, strict=True)
            )
            raise panel_error(f"the maps share no energy range ({spans})", "library")
        facts: list[tuple[str, float, float]] = []
        if any(m.energy.min() < lo or m.energy.max() > hi for m in maps):
            facts.append(("common", float(lo), float(hi)))
        with in_panel("library"):
            if method == "field":
                fmap = merge_field([(m, None, None) for m in maps])
                facts += _field_facts(maps)
            else:
                fmap = average_maps(maps)
        return fmap, tuple(facts)

    def _merge_energy(self, parts, maps):
        seams = energy_seams([(m, None, None) for m in maps])
        for seam in seams:
            if seam.kind == "inside":
                inner, outer = parts[seam.upper], parts[seam.lower]
                span = energy_range_text((seam.lo, seam.hi), self._unit)
                raise panel_error(
                    f"{inner.name} ({span}) lies inside the energy range of {outer.name}: "
                    "give them E limits that meet, or average repeated measurements",
                    "library",
                )
        with in_panel("library"):
            fmap = merge_energy([(m, None, None) for m in maps])
        facts = tuple((s.kind, s.lo, s.hi) for s in seams if s.kind in ("overlap", "gap"))
        return fmap, facts

    def make_product(self, method: str, entries: list[LibraryEntry] | None = None) -> Product:
        """Combine *entries* (default: the ticked maps) by *method* (:data:`COMBINE_METHODS`)
        into the product, which replaces the one before and is shown."""
        parts = tuple(self.ticked() if entries is None else entries)
        fmap, facts = self._combine(method, parts)
        records = tuple(
            PartRecord(p.name, p.kind, p.energy_cut, p.field_cut, p.provenance) for p in parts
        )
        provenance = Provenance(method, datetime.now(), parts=records, facts=facts)
        name = f"{PRODUCT_PREFIX[method]}: " + " + ".join(p.name for p in parts)
        product = Product(name, fmap, parts[0].kind, provenance)
        old, self.product = self.product, product
        try:
            self._show_library(fmap, name, SHOWS_PRODUCT)
        except Exception:
            self.product = old
            raise
        logger.info("-" * 40)
        names = [p.name for p in parts]
        b, e = fmap.field, from_cm1(fmap.energy[[0, -1]], self._unit)
        logger.info(
            "%s %s: %d fields, B = %g … %g T, E = %.6g … %.6g %s, %d energies.",
            PRODUCT_PREFIX[method],
            names,
            b.size,
            b.min(),
            b.max(),
            e[0],
            e[1],
            self._unit,
            fmap.energy.size,
        )
        self.productChanged.emit()
        self.plottedChanged.emit()
        return product

    def _product(self) -> Product:
        if self.product is None:
            raise panel_error("there is no product: combine ticked maps first", "library")
        return self.product

    def rename_product(self, name: str) -> None:
        product = self._product()
        name = name.strip()
        if name and name != product.name:
            product.name = name
            if self.showing() == SHOWS_PRODUCT:
                self._library_name = name
            self.productChanged.emit()

    def plot_product(self) -> ProcessResult:
        """Show the product (in place of the ticked maps)."""
        product = self._product()
        if self.showing() != SHOWS_PRODUCT:
            self._show_library(product.fmap, product.name, SHOWS_PRODUCT)
            self.plottedChanged.emit()
        return self.result

    def save_product(self) -> LibraryEntry:
        """Add the product to the library, with how it was made (once)."""
        product = self._product()
        if product.saved is not None:
            with contextlib.suppress(ValueError):
                return self.entry(product.saved)
        entry = self.add_map(
            product.fmap, product.name, kind=product.kind, provenance=product.provenance
        )
        product.saved = entry.key
        logger.info("Library: saved the product as %r", entry.name)
        self.productChanged.emit()
        return entry

    def product_table(self) -> tuple[FieldMap, Range | None]:
        """The product as its table is exported: as plotted, with the baseline region of the
        processing options (the one it is shown with), in the display unit; and that region
        (cm^-1, None without one)."""
        product = self._product()
        if self.showing() == SHOWS_PRODUCT:
            result = self.result
        else:
            region = self._baseline()
            if region is not None:
                self.check_energy_range(
                    "baseline region", region, product.fmap.energy, "processing"
                )
            with in_panel("processing"):
                result = ProcessResult.from_map(product.fmap, region)
        return result.ratio.to_unit(self._unit), result.baseline_region

    def discard_product(self) -> None:
        """Drop the product; the ticked maps (or the processed sweep) come back on the plot.
        When the ticked maps cannot be shown, the processed sweep (or nothing) comes instead
        and the error is raised."""
        if self.product is None:
            return
        shown = self.showing() == SHOWS_PRODUCT
        self.product = None
        self.productChanged.emit()
        if not shown:
            return
        self._library_shows, self._shown_top = SHOWS_MAPS, None
        if self._plotted:
            self._show_next()
        else:
            self._leave_library()


def _has_values(fmap: FieldMap, region: Range) -> bool:
    """*fmap* (cm^-1) has a value (not NaN) in the energy *region* (cm^-1)."""
    return bool(np.isfinite(fmap.values[energy_mask(fmap.energy, *region)]).any())


def _curve_name_rule(name: str) -> str | None:
    """Why *name* (stripped) breaks the rule for curve names, or None."""
    if not name:
        return "enter a curve name"
    if not CURVE_NAME.fullmatch(name):
        return "use letters, digits, spaces and _ . + - (at most 32)"
    return None


def _selection_scale(unit: Unit, selection: PlotSelection) -> float:
    """:func:`derivative_scale` of the map *selection* shows in *unit*."""
    s = selection
    return derivative_scale(unit, s.order, s.axis == Axis.ENERGY, s.physical)


def _mirrored(
    lo: float, hi: float, old: tuple[float, float] | None, centre: float
) -> tuple[float, float] | None:
    """Levels symmetric about *centre* after an edit of symmetric levels *old*, or None when
    the edit moved both ends off the centre (a shift)."""
    tol = 1e-9 * max(abs(hi - lo), abs(centre), 1e-300)
    if abs((lo + hi) / 2 - centre) <= tol:
        return lo, hi
    if old is not None and abs(hi - old[1]) <= tol and lo < centre:
        return lo, 2 * centre - lo
    if old is not None and abs(lo - old[0]) <= tol and hi > centre:
        return 2 * centre - hi, hi
    return None


SEPARATORS = "_- "


def common_prefix(names: list[str]) -> str:
    """The start all *names* share, cut after its last separator (``_``, ``-`` or space)."""
    if not names:
        return ""
    prefix = names[0]
    for name in names[1:]:
        while not name.startswith(prefix):
            prefix = prefix[:-1]
    cut = max(prefix.rfind(s) for s in SEPARATORS)
    return prefix[: cut + 1] if cut >= 0 else ""


def _sorted_files(files: SweepFiles) -> SweepFiles:
    return SweepFiles(zero=tuple(sort_paths(files.zero)), field=tuple(sort_paths(files.field)))


# ---------------------------------------------------------------------- library text
_EXPORT_KIND = re.compile(
    r"_(?P<kind>Ratio_AVR|Ratio_Step|Ratio|Data)(?P<order>_1stDer|_2ndDer)?(?:_perUnit)?$"
)


def kind_from_name(name: str) -> str:
    """The kind of a map from the name an export gives it (``S1_Data.csv`` -> ``"Data"``,
    ``S1_Ratio_1stDer.csv`` -> ``"R(B)/R(0) · 1st derivative"``); R(B)/R(0) for other names."""
    match = _EXPORT_KIND.search(Path(name).stem)
    if match is None:
        return KIND_LABELS[PlotKind.RATIO]
    kind = next(k for k, export in EXPORT_NAMES.items() if export == match["kind"])
    order = {"_1stDer": 1, "_2ndDer": 2}.get(match["order"] or "", 0)
    return KIND_LABELS[kind] + (f" · {ORDINALS[order]} derivative" if order else "")


def energy_range_text(rng: Range, unit: Unit) -> str:
    """An energy range (cm^-1) in *unit*, e.g. ``350 – 3200 cm⁻¹``."""
    lo, hi = convert_range(rng, Unit.CM1, unit)
    return format_range(lo, hi, unit_text(unit))


def cut_text(energy_cut: Range, field_cut: Range, unit: Unit) -> str:
    """Cut limits in words (``E ≤ 450 cm⁻¹, B 0 – 8 T``); "" without any."""
    parts = []
    for name, (lo, hi), shown, u in (
        ("E", energy_cut, convert_range(energy_cut, Unit.CM1, unit), unit_text(unit)),
        ("B", field_cut, field_cut, "T"),
    ):
        if lo is not None and hi is not None:
            parts.append(f"{name} {format_range(shown[0], shown[1], u)}")
        elif lo is not None:
            parts.append(f"{name} ≥ {format_number(shown[0])} {u}")
        elif hi is not None:
            parts.append(f"{name} ≤ {format_number(shown[1])} {u}")
    return ", ".join(parts)


def fact_text(fact: tuple[str, float, float], unit: Unit) -> tuple[str, str]:
    """(text, level) of something combining did (see :meth:`AppController.combine_preview`);
    the level is ``"info"``, or ``"warn"`` for what may need other limits."""
    what, lo, hi = fact
    if what == "overlap":
        middle = f"{format_number(float(from_cm1((lo + hi) / 2, unit)))} {unit_text(unit)}"
        if hi - lo <= 1e-9 * max(abs(hi), 1.0):  # the limits meet in one sample
            return f"Joined at {middle}", "info"
        return f"Overlap {energy_range_text((lo, hi), unit)} joined at {middle}", "info"
    if what == "gap":
        return f"No data in {energy_range_text((lo, hi), unit)}: left empty", "warn"
    if what == "common":
        return f"Cut to the energies all maps share, {energy_range_text((lo, hi), unit)}", "info"
    if what == "averaged":
        n = int(lo)
        return f"{n} field{'s' if n != 1 else ''} in more than one map averaged", "info"
    if what == "interleaved":
        span = format_range(lo, hi, "T")
        return f"Fields of two maps alternate in {span}: set B limits to join them", "warn"
    raise ValueError(f"unknown fact {what!r}")


def _fields_text(name: str, fmap: FieldMap) -> str:
    b = fmap.field
    fields = f"{b.size} field{'s' if b.size != 1 else ''}"
    return f"{name} {fields}, {format_range(b.min(), b.max(), 'T')}"


def _field_facts(maps: list[FieldMap]) -> list[tuple[str, float, float]]:
    """What merging *maps* by field does: fields measured more than once are averaged, and
    the fields of overlapping maps may alternate."""
    facts: list[tuple[str, float, float]] = []
    fields = np.sort(np.concatenate([m.field for m in maps]))
    repeated = np.diff(fields) <= FIELD_TOL
    if repeated.any():
        facts.append(("averaged", float(np.count_nonzero(repeated)), 0.0))
    ordered = sorted((m.field for m in maps), key=lambda b: (b.min(), b.max()))
    for a, b in itertools.pairwise(ordered):
        b_lo, a_hi = float(b.min()), float(a.max())
        a_in, b_in = a[(a >= b_lo) & (a <= a_hi)], b[(b >= b_lo) & (b <= a_hi)]
        if a_in.size + b_in.size < 3:
            continue
        same = a_in.size == b_in.size and np.allclose(np.sort(a_in), np.sort(b_in), atol=FIELD_TOL)
        if not same:
            facts.append(("interleaved", b_lo, a_hi))
    return facts
