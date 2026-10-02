"""Journal figures drawn with matplotlib's object API (never pyplot, never a Qt backend).

:func:`render` turns a :class:`FigureState` into a :class:`~matplotlib.figure.Figure` of
an exact physical size; :func:`save` writes it as PDF, SVG, EPS, PS, PNG or TIFF and
:func:`rasterize` draws it for a preview. Text stays editable in every vector format.
"""

from __future__ import annotations

import logging
import weakref
from collections.abc import Iterator
from contextlib import contextmanager
from functools import lru_cache
from pathlib import Path

import matplotlib as mpl
import numpy as np
from matplotlib import font_manager, patheffects
from matplotlib.axes import Axes
from matplotlib.backend_bases import FigureCanvasBase
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.backends.backend_pdf import FigureCanvasPdf
from matplotlib.backends.backend_ps import FigureCanvasPS
from matplotlib.backends.backend_svg import FigureCanvasSVG
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Colormap, LinearSegmentedColormap, ListedColormap, Normalize
from matplotlib.figure import Figure
from PIL import Image
from PIL.PngImagePlugin import PngInfo

from mag_opt_detective import __version__
from mag_opt_detective.core import colormaps
from mag_opt_detective.core.units import Unit
from mag_opt_detective.export.presets import JournalPreset, get_preset
from mag_opt_detective.export.state import FigureState, Range

logger = logging.getLogger(__name__)

MM_PER_INCH = 25.4
FIELD_LABEL = "Magnetic field (T)"
INTENSITY_LABEL = "Intensity (a.u.)"
# file suffix -> matplotlib format
FORMATS: dict[str, str] = {
    ".pdf": "pdf",
    ".svg": "svg",
    ".eps": "eps",
    ".ps": "ps",
    ".png": "png",
    ".tif": "tiff",
    ".tiff": "tiff",
}
# Vector formats are drawn by these canvases, imported here rather than loaded by name
# through matplotlib's backend registry so that PyInstaller bundles them.
_VECTOR_CANVASES: dict[str, type[FigureCanvasBase]] = {
    "pdf": FigureCanvasPdf,
    "svg": FigureCanvasSVG,
    "eps": FigureCanvasPS,
    "ps": FigureCanvasPS,
}
CREATOR = f"Magneto-Optical Detective {__version__}"
FALLBACK_FONT = "DejaVu Sans"  # ships with matplotlib

_UNIT_TEXT = {Unit.CM1: "cm$^{-1}$", Unit.MEV: "meV", Unit.THZ: "THz"}
# part of the colour map used to colour traces by field (as the stacked plot in the app)
_FIELD_COLOURS = (0.1, 0.82)
_MARKERS = ("o", "s", "^", "D", "v", "p")
_OUTLINE = "#1a1a1a"  # dark edge around light overlays on a map
_DASHES = (0, (3.0, 2.0))  # in pt, not scaled with the line width
_PAD_IN = 1.0 / 72.0  # constrained-layout padding around the axes and labels

_EDITABLE_TEXT = {"pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none"}
# rcParams each figure was made with; save and rasterize draw with the same ones
_FIGURE_RC: weakref.WeakKeyDictionary[Figure, dict] = weakref.WeakKeyDictionary()


# ---------------------------------------------------------------------- styling
def colormap(name: str) -> Colormap:
    """Matplotlib colour map built from the stops in :mod:`core.colormaps`; NaN is white."""
    table = [(pos, tuple(c / 255 for c in rgb)) for pos, rgb in colormaps.stops(name)]
    cmap = LinearSegmentedColormap.from_list(name, table, N=256)
    return cmap.with_extremes(bad="white")


def energy_label(unit: Unit | str) -> str:
    """Axis label of the energy axis, e.g. ``Energy (meV)`` (cm⁻¹ as math text)."""
    return f"Energy ({_UNIT_TEXT[Unit(unit)]})"


@lru_cache
def resolve_font(families: tuple[str, ...]) -> str:
    """First installed font of *families*; logs a warning when it falls back."""
    installed = {entry.name for entry in font_manager.fontManager.ttflist}
    for name in families:
        if name in installed:
            if name != families[0]:
                logger.warning("Font %s is not installed; using %s.", families[0], name)
            return name
    logger.warning("None of %s is installed; using %s.", ", ".join(families), FALLBACK_FONT)
    return FALLBACK_FONT


def figure_rc(preset: JournalPreset, font_size_pt: float, line_width_pt: float) -> dict:
    """rcParams of a figure: fonts, sizes, line widths and editable text in every format."""
    font = resolve_font(tuple(preset.font_family))
    family = [font] if font == FALLBACK_FONT else [font, FALLBACK_FONT]  # glyph fallback
    fs, lw = float(font_size_pt), float(line_width_pt)
    rc: dict = {
        "font.family": family,
        "font.size": fs,
        "axes.labelsize": fs,
        "axes.titlesize": fs,
        "xtick.labelsize": fs,
        "ytick.labelsize": fs,
        "legend.fontsize": fs,
        "figure.titlesize": preset.label_size_pt,
        # math text (cm$^{-1}$, $B$) in the same font as the rest of the figure
        "mathtext.fontset": "custom",
        "mathtext.rm": font,
        "mathtext.sf": font,
        "mathtext.cal": font,
        "mathtext.it": f"{font}:italic",
        "mathtext.bf": f"{font}:bold",
        "mathtext.bfit": f"{font}:italic:bold",
        "mathtext.tt": "DejaVu Sans Mono",
        "mathtext.default": "regular",
        "text.usetex": False,
        "axes.unicode_minus": True,
        "axes.formatter.useoffset": False,
        "axes.formatter.limits": (-3, 4),
        "axes.linewidth": lw,
        "axes.labelpad": 0.4 * fs,
        "axes.titlepad": 0.5 * fs,
        "lines.linewidth": lw,
        "lines.markeredgewidth": lw,
        "lines.scale_dashes": False,
        "patch.linewidth": lw,
        "legend.frameon": False,
        "image.interpolation": "nearest",
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",
        "savefig.edgecolor": "white",
        "savefig.transparent": False,
        "savefig.bbox": "standard",  # keep the exact page size
        "figure.constrained_layout.w_pad": _PAD_IN,
        "figure.constrained_layout.h_pad": _PAD_IN,
        "figure.constrained_layout.wspace": 0.0,
        "figure.constrained_layout.hspace": 0.0,
        **_EDITABLE_TEXT,  # embedded TrueType fonts in PDF/PS, <text> elements in SVG
        "pdf.use14corefonts": False,
        "ps.useafm": False,
        "ps.papersize": "figure",
        "svg.hashsalt": "mag-opt-detective",  # reproducible ids
    }
    for axis in ("xtick", "ytick"):
        rc |= {
            f"{axis}.direction": "out",
            f"{axis}.major.width": lw,
            f"{axis}.minor.width": 0.75 * lw,
            f"{axis}.major.size": 0.45 * fs,
            f"{axis}.minor.size": 0.25 * fs,
            f"{axis}.major.pad": 0.3 * fs,
        }
    return rc


# ---------------------------------------------------------------------- render
def render(
    state: FigureState,
    *,
    preset: JournalPreset | str,
    width_mm: float,
    height_mm: float,
    font_size_pt: float | None = None,
    line_width_pt: float | None = None,
    panel_label: str | None = None,
    dpi: float | None = None,
) -> Figure:
    """Figure of exactly *width_mm* × *height_mm* showing *state* in the *preset* style.

    Font size, line width and dpi default to the preset's. *panel_label* (e.g. ``"a"``)
    is drawn bold in the top-left corner in the preset's style; None leaves it out.
    """
    preset = get_preset(preset)
    if not (width_mm > 0 and height_mm > 0):
        raise ValueError("width and height must be positive")
    if state.fmap.values.size == 0:
        raise ValueError("the map has no data")
    fs = preset.font_size_pt if font_size_pt is None else float(font_size_pt)
    lw = preset.line_width_pt if line_width_pt is None else float(line_width_pt)
    if fs <= 0 or lw <= 0:
        raise ValueError("font size and line width must be positive")
    rc = figure_rc(preset, fs, lw)
    with mpl.rc_context(rc):
        fig = Figure(
            figsize=(width_mm / MM_PER_INCH, height_mm / MM_PER_INCH),
            dpi=preset.raster_dpi if dpi is None else float(dpi),
            layout="constrained",
        )
        FigureCanvasAgg(fig)
        ax = fig.add_subplot()
        if state.kind == "map":
            _draw_map(fig, ax, state, fs, lw)
        else:
            _draw_stacked(fig, ax, state, fs, lw)
        if state.title:
            ax.set_title(state.title)
        if panel_label:
            fig.suptitle(
                preset.panel_label(panel_label),
                x=_PAD_IN / fig.get_figwidth(),
                ha="left",
                fontsize=preset.label_size_pt,
                fontweight="bold",
            )
    _FIGURE_RC[fig] = rc
    return fig


def _sorted_map(state: FigureState) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Field, energy and values with both axes rising."""
    fmap = state.fmap
    i, j = np.argsort(fmap.energy, kind="stable"), np.argsort(fmap.field, kind="stable")
    return fmap.field[j], fmap.energy[i], fmap.values[np.ix_(i, j)]


def _edges(axis: np.ndarray) -> np.ndarray:
    """Pixel edges around rising centres *axis* (a single sample is 1 wide)."""
    if axis.size == 1:
        return np.array([axis[0] - 0.5, axis[0] + 0.5])
    mid = (axis[:-1] + axis[1:]) / 2
    return np.concatenate([[2 * axis[0] - mid[0]], mid, [2 * axis[-1] - mid[-1]]])


def _is_uniform(axis: np.ndarray) -> bool:
    if axis.size < 3:
        return True
    step = np.diff(axis)
    return bool(np.allclose(step, step.mean(), rtol=1e-6, atol=0.0))


def _extent(axis: np.ndarray) -> tuple[float, float]:
    """Default view range: first to last sample (a single sample: its pixel)."""
    if axis.size == 1:
        lo, hi = _edges(axis)
        return float(lo), float(hi)
    return float(axis[0]), float(axis[-1])


def _apply_range(ax: Axes, which: str, rng: Range | None, default: tuple[float, float]) -> None:
    lo, hi = (None, None) if rng is None else rng
    lo = default[0] if lo is None else float(lo)
    hi = default[1] if hi is None else float(hi)
    (ax.set_xlim if which == "x" else ax.set_ylim)(lo, hi)


def robust_levels(values: np.ndarray) -> tuple[float, float]:
    """Colour levels from the 1st/99th percentile (as the app's automatic levels)."""
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return 0.0, 1.0
    lo, hi = np.percentile(finite, [1, 99])
    return float(lo), float(hi)


def _levels(state: FigureState, values: np.ndarray) -> tuple[float, float]:
    lo, hi = robust_levels(values) if state.levels is None else sorted(map(float, state.levels))
    if lo == hi:
        lo, hi = lo - 0.5, hi + 0.5
    return lo, hi


def _style_colorbar(cbar, label: str, lw: float) -> None:
    cbar.outline.set_linewidth(lw)
    cbar.solids.set_rasterized(True)  # no seams between the colour patches in PDF viewers
    cbar.ax.tick_params(which="both", width=lw)
    if label:
        cbar.set_label(label)


def _draw_map(fig: Figure, ax: Axes, state: FigureState, fs: float, lw: float) -> None:
    field, energy, values = _sorted_map(state)
    cmap = colormap(state.cmap)
    norm = Normalize(*_levels(state, values))
    if _is_uniform(field) and _is_uniform(energy):
        x, y = _edges(field), _edges(energy)  # pixel centres on the grid
        image = ax.imshow(
            values,
            cmap=cmap,
            norm=norm,
            origin="lower",
            extent=(x[0], x[-1], y[0], y[-1]),
            aspect="auto",
            interpolation="nearest",
        )
    else:  # uneven field steps: one cell per sample, drawn as an image in vector files
        image = ax.pcolormesh(
            _edges(field), _edges(energy), values, cmap=cmap, norm=norm, rasterized=True
        )
    ax.set_xlabel(FIELD_LABEL if state.x_label is None else state.x_label)
    ax.set_ylabel(energy_label(state.fmap.unit) if state.y_label is None else state.y_label)
    _apply_range(ax, "x", state.x_range, _extent(field))
    _apply_range(ax, "y", state.y_range, _extent(energy))

    stroke = [patheffects.withStroke(linewidth=lw + 1.0, foreground=_OUTLINE)]
    for curve in state.curves:
        ax.plot(
            curve.x,
            curve.y,
            color="white",
            linewidth=lw,
            linestyle=_DASHES if curve.style == "model" else "-",
            dash_capstyle="butt",
            path_effects=stroke,
            label=curve.label or "_nolegend_",
            scalex=False,
            scaley=False,
        )
    size = 0.55 * fs
    for k, points in enumerate(state.points):
        ax.plot(
            points.x,
            points.y,
            linestyle="none",
            marker=_MARKERS[k % len(_MARKERS)],
            markersize=size,
            markeredgewidth=lw,
            markerfacecolor="white" if points.current else "none",
            markeredgecolor=_OUTLINE if points.current else "white",
            path_effects=None if points.current else stroke,
            label=points.label or "_nolegend_",
            scalex=False,
            scaley=False,
        )
    if state.colorbar:
        _style_colorbar(fig.colorbar(image, ax=ax, pad=0.02, aspect=25), state.colorbar_label, lw)


def _field_norm(field: np.ndarray) -> Normalize:
    """Fields of all traces (shown or not) onto 0..1 for the trace colours."""
    lo, hi = float(field.min()), float(field.max())
    return Normalize(lo - 0.5, hi + 0.5) if lo == hi else Normalize(lo, hi)


def _field_cmap(name: str) -> Colormap:
    """The part of colour map *name* used for traces (the ends can vanish on white)."""
    lo, hi = _FIELD_COLOURS
    return ListedColormap(colormap(name)(np.linspace(lo, hi, 256)), name=f"{name}_traces")


def _trace_colours(state: FigureState, shown: np.ndarray) -> list:
    if not state.stacked.color_by_field:
        return ["black"] * shown.size
    field = state.fmap.field
    return list(_field_cmap(state.cmap)(_field_norm(field)(field[shown])))


def _field_tolerance(field: np.ndarray) -> float:
    """Half the smallest field step: a point belongs to the trace whose pixel holds it."""
    steps = np.diff(np.unique(field))
    return 0.5 if steps.size == 0 else 0.5 * float(steps.min())


def _draw_stacked(fig: Figure, ax: Axes, state: FigureState, fs: float, lw: float) -> None:
    fmap, options = state.fmap, state.stacked
    order = np.argsort(fmap.energy, kind="stable")
    energy, values = fmap.energy[order], fmap.values[order]
    shown = np.arange(0, fmap.field.size, options.every)
    position = {int(j): k for k, j in enumerate(shown)}
    for k, (j, colour) in enumerate(zip(shown, _trace_colours(state, shown), strict=True)):
        ax.plot(energy, values[:, j] + k * options.offset, color=colour, linewidth=lw)

    tolerance = _field_tolerance(fmap.field)
    size = 0.55 * fs
    for n, points in enumerate(state.points):
        xs, ys = [], []
        for b, e in zip(points.x, points.y, strict=True):
            if not (np.isfinite(b) and np.isfinite(e)):
                continue
            j = int(np.abs(fmap.field - b).argmin())
            k = position.get(j)
            if k is None or abs(fmap.field[j] - b) > tolerance:
                continue
            y = np.interp(e, energy, values[:, j], left=np.nan, right=np.nan)
            if np.isfinite(y):
                xs.append(e)
                ys.append(y + k * options.offset)
        ax.plot(
            xs,
            ys,
            linestyle="none",
            marker=_MARKERS[n % len(_MARKERS)],
            markersize=size,
            markeredgewidth=lw,
            markerfacecolor="black" if points.current else "white",
            markeredgecolor="black",
            label=points.label or "_nolegend_",
            scalex=False,
            scaley=False,
        )
    ax.set_xlabel(energy_label(fmap.unit) if state.x_label is None else state.x_label)
    ax.set_ylabel(INTENSITY_LABEL if state.y_label is None else state.y_label)
    ax.margins(x=0.0, y=0.03)
    _apply_range(ax, "x", state.x_range, _extent(energy))
    if state.y_range is not None:
        _apply_range(ax, "y", state.y_range, ax.get_ylim())
    if state.colorbar and options.color_by_field:
        mappable = ScalarMappable(_field_norm(fmap.field), _field_cmap(state.cmap))
        cbar = fig.colorbar(mappable, ax=ax, pad=0.02, aspect=25)
        _style_colorbar(cbar, state.colorbar_label or FIELD_LABEL, lw)


# ---------------------------------------------------------------------- output
def _rc(fig: Figure) -> dict:
    """rcParams *fig* was rendered with; text stays editable in other figures too."""
    return {**_EDITABLE_TEXT, **_FIGURE_RC.get(fig, {})}


def rasterize(fig: Figure, dpi: float) -> np.ndarray:
    """The figure drawn at *dpi* as an RGBA ``uint8`` array (for previews and raster files).

    The array is ``round(size_in_inches * dpi)`` pixels in each direction.
    """
    if dpi <= 0:
        raise ValueError("dpi must be positive")
    size, old_dpi, old_canvas = fig.get_size_inches().copy(), fig.dpi, fig.canvas
    pixels = np.round(size * dpi)
    try:
        canvas = FigureCanvasAgg(fig)
        # Agg truncates the pixel size, so ask for a hair more than the rounded size
        fig.set_size_inches((pixels + 1e-3) / dpi, forward=False)
        fig.dpi = dpi
        with mpl.rc_context(_rc(fig)):
            canvas.draw()
        return np.array(canvas.buffer_rgba())
    finally:
        fig.set_size_inches(size, forward=False)
        fig.dpi = old_dpi
        fig.set_canvas(old_canvas)


def save(fig: Figure, path: str | Path, *, dpi: float) -> Path:
    """Write *fig* to *path* in the format of its suffix (.pdf .svg .eps .ps .png .tif .tiff).

    Vector formats keep the exact page size and editable text; *dpi* sets their embedded
    images. PNG and TIFF are written in RGB with *dpi* stored in the file.
    """
    path = Path(path)
    fmt = FORMATS.get(path.suffix.lower())
    if fmt is None:
        raise ValueError(
            f"unknown figure format {path.suffix!r}; use one of {', '.join(sorted(FORMATS))}"
        )
    if dpi <= 0:
        raise ValueError("dpi must be positive")
    if fmt in ("png", "tiff"):
        rgb = Image.fromarray(rasterize(fig, dpi)[..., :3])
        if fmt == "png":
            info = PngInfo()
            info.add_text("Software", CREATOR)
            rgb.save(path, format="PNG", dpi=(dpi, dpi), pnginfo=info)
        else:
            rgb.save(path, format="TIFF", dpi=(dpi, dpi), compression="tiff_lzw")
        return path
    old_canvas = fig.canvas
    try:
        canvas = _VECTOR_CANVASES[fmt](fig)
        with mpl.rc_context(_rc(fig)), _quiet_fonttools():
            canvas.print_figure(path, format=fmt, dpi=dpi, metadata={"Creator": CREATOR})
    finally:
        fig.set_canvas(old_canvas)
    return path


@contextmanager
def _quiet_fonttools() -> Iterator[None]:
    """Hide fontTools' INFO line per font subset of a PDF/PS save (it floods the app log)."""
    log = logging.getLogger("fontTools")
    level = log.level
    log.setLevel(max(level, logging.WARNING))
    try:
        yield
    finally:
        log.setLevel(level)
