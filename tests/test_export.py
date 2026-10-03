import dataclasses
import logging
import re
import subprocess
import sys
import textwrap
import xml.etree.ElementTree as ET
import zlib

import matplotlib as mpl
import numpy as np
import pytest
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.collections import QuadMesh
from matplotlib.image import AxesImage
from matplotlib.text import Text
from PIL import Image

from mag_opt_detective.core import colormaps
from mag_opt_detective.core.spectra import FieldMap
from mag_opt_detective.export import (
    APS,
    CUSTOM,
    FORMATS,
    NATURE,
    PRESETS,
    Curve,
    FigureState,
    PointSet,
    StackedOptions,
    colormap,
    energy_label,
    get_preset,
    rasterize,
    render,
    resolve_font,
    save,
)
from mag_opt_detective.export.figure import FIELD_LABEL, INTENSITY_LABEL, figure_rc

MM = 25.4


@pytest.fixture
def fmap() -> FieldMap:
    """A small Landau-fan-like map in meV on a uniform grid."""
    energy = np.linspace(10.0, 120.0, 221)
    field = np.linspace(0.0, 16.0, 17)
    values = np.ones((energy.size, field.size))
    for j, b in enumerate(field):
        for n in range(3):
            e0 = 15.0 * (np.sqrt(n) + np.sqrt(n + 1)) * np.sqrt(b)
            values[:, j] -= 0.1 / (1 + ((energy - e0) / 3.0) ** 2)
    return FieldMap(energy, field, values, unit="meV")


@pytest.fixture
def map_state(fmap) -> FigureState:
    b = np.linspace(0.0, 16.0, 50)
    return FigureState(
        kind="map",
        fmap=fmap,
        levels=(0.9, 1.0),
        colorbar_label="Relative transmission",
        curves=[Curve(b, 15.0 * np.sqrt(b), "model", "n=0"), Curve(b, 30 + b, "plain")],
        points=[
            PointSet([4.0, 9.0], [30.0, 45.0], "L0", current=True),
            PointSet([4.0], [72.0], "L1"),
        ],
    )


def nature_single(state: FigureState, height_mm: float = 70.0, **kwargs):
    return render(state, preset=NATURE, width_mm=89.0, height_mm=height_mm, **kwargs)


# ---------------------------------------------------------------------- presets
def test_nature_preset():
    assert NATURE.widths_mm == {"single": 89, "1.5 narrow": 120, "1.5 wide": 136, "double": 183}
    assert NATURE.default_width_mm == 89
    assert NATURE.max_height_mm == 247
    assert NATURE.font_family == ("Arial", "Helvetica", "DejaVu Sans")
    assert (NATURE.font_size_pt, NATURE.font_size_range_pt) == (7, (5, 7))
    assert (NATURE.label_size_pt, NATURE.panel_label_style) == (8, "a")
    assert (NATURE.line_width_pt, NATURE.min_line_width_pt, NATURE.max_line_width_pt) == (
        0.5,
        0.25,
        1.0,
    )
    assert (NATURE.raster_dpi, NATURE.raster_dpi_range) == (450, (300, 600))
    assert NATURE.formats == ("pdf", "eps", "tif", "png")
    assert NATURE.colour_mode == "RGB"
    assert NATURE.sources == ("https://www.nature.com/nature/for-authors/final-submission",)
    assert any("editable" in note for note in NATURE.notes)


def test_preset_widths_are_read_only():
    with pytest.raises(TypeError):
        NATURE.widths_mm["single"] = 100.0  # type: ignore[index]
    custom = dataclasses.replace(CUSTOM, widths_mm={"single": 80.0}, default_width="single")
    assert custom.default_width_mm == 80.0
    assert CUSTOM.widths_mm == {}


def test_aps_preset():
    assert APS.widths_mm == {"single": 86, "double": 178}
    assert APS.font_size_pt == 8
    assert (APS.line_width_pt, APS.min_line_width_pt) == (0.75, 0.5)
    assert APS.raster_dpi == 600
    assert APS.formats[:2] == ("eps", "ps")
    assert {"pdf", "png"} <= set(APS.formats)
    assert APS.panel_label_style == "(a)"
    assert APS.colour_mode == "RGB"
    assert "https://journals.aps.org/authors/style-basics" in APS.sources
    assert any(url.endswith("aps-journals-style-guide_tnoyln.pdf") for url in APS.sources)
    notes = " ".join(APS.notes)
    assert "8.5 cm" in notes
    assert "not confirmed" in notes
    assert "grayscale" in notes


def test_custom_preset():
    assert CUSTOM.free_size
    assert (CUSTOM.default_width_mm, CUSTOM.default_height_mm) == (120, 90)
    assert (CUSTOM.font_size_pt, CUSTOM.raster_dpi) == (8, 600)
    assert CUSTOM.check(300.0, 250.0) == []


def test_every_journal_preset_has_sources():
    assert set(PRESETS) == {"nature", "aps", "custom"}
    for preset in PRESETS.values():
        assert all(url.startswith("https://") for url in preset.sources)
        assert preset.notes
        if not preset.free_size:
            assert preset.sources, preset.name


def test_get_preset():
    assert get_preset("nature") is NATURE
    assert get_preset("APS (Physical Review)") is APS
    assert get_preset(" Custom ") is CUSTOM
    assert get_preset(APS) is APS
    with pytest.raises(ValueError, match="unknown preset"):
        get_preset("Science")


def test_panel_label_styles():
    assert NATURE.panel_label("A") == "a"
    assert NATURE.panel_label("(b)") == "b"
    assert APS.panel_label("c") == "(c)"
    assert APS.panel_label("(d)") == "(d)"
    assert APS.panel_label("  ") == NATURE.panel_label("()") == ""


def test_preset_check():
    assert NATURE.check(89, 70, font_size_pt=7, line_width_pt=0.5, dpi=450) == []
    problems = NATURE.check(200, 300, font_size_pt=4, line_width_pt=0.1, dpi=100)
    assert len(problems) == 5
    assert NATURE.check(89, 70, line_width_pt=1.5) == [
        "Nature: lines should be at most 1 pt, not 1.5 pt."
    ]
    assert APS.check(86, 300) == []  # APS gives no height limit
    assert CUSTOM.check(0, 10) == ["Width and height must be positive."]


# ---------------------------------------------------------------------- styling
@pytest.mark.parametrize("name", colormaps.names())
def test_colormap_matches_core_lut(name):
    table = colormaps.lut(name, 256)
    cmap = colormap(name)
    index = np.array([0, 1, 37, 100, 128, 200, 254, 255])
    rgb = np.rint(np.asarray(cmap(index / 255))[:, :3] * 255).astype(int)
    assert np.abs(rgb - table[index].astype(int)).max() <= 1
    assert cmap(np.nan) == (1.0, 1.0, 1.0, 1.0)  # NaN is white


def test_energy_labels():
    assert energy_label("meV") == "Energy (meV)"
    assert energy_label("THz") == "Energy (THz)"
    assert energy_label("cm-1") == "Energy (cm$^{-1}$)"


def test_font_fallback_warns(caplog):
    with caplog.at_level(logging.WARNING, logger="mag_opt_detective.export.figure"):
        assert resolve_font(("NoSuchFont-1f3a", "DejaVu Sans")) == "DejaVu Sans"
    assert "NoSuchFont-1f3a is not installed" in caplog.text
    caplog.clear()
    with caplog.at_level(logging.WARNING, logger="mag_opt_detective.export.figure"):
        assert resolve_font(("NoSuchFont-77c2",)) == "DejaVu Sans"
    assert "using DejaVu Sans" in caplog.text


def test_render_uses_the_fallback_font(map_state):
    preset = dataclasses.replace(CUSTOM, font_family=("NoSuchFont-9b0e", "DejaVu Sans"))
    fig = render(map_state, preset=preset, width_mm=100, height_mm=80)
    ax = fig.axes[0]
    assert ax.xaxis.label.get_fontfamily() == ["DejaVu Sans"]


def test_figure_rc_keeps_text_editable():
    rc = figure_rc(NATURE, 7.0, 0.5)
    assert rc["pdf.fonttype"] == rc["ps.fonttype"] == 42
    assert rc["svg.fonttype"] == "none"
    assert rc["axes.linewidth"] == rc["xtick.major.width"] == rc["ytick.major.width"] == 0.5
    assert rc["xtick.labelsize"] == rc["axes.labelsize"] == 7.0
    assert rc["font.family"][-1] == "DejaVu Sans"


# ---------------------------------------------------------------------- map
def test_render_has_the_exact_size_and_preset_style(map_state):
    fig = nature_single(map_state)
    np.testing.assert_allclose(fig.get_size_inches() * MM, [89.0, 70.0])
    assert fig.dpi == NATURE.raster_dpi
    ax = fig.axes[0]
    assert ax.spines["left"].get_linewidth() == 0.5
    assert ax.xaxis.label.get_fontsize() == 7.0
    assert ax.xaxis.get_ticklabels()[0].get_fontsize() == 7.0
    assert ax.xaxis.get_major_ticks()[0].tick1line.get_markeredgewidth() == 0.5
    fig = nature_single(map_state, font_size_pt=6.0, line_width_pt=0.25, dpi=300)
    assert fig.axes[0].yaxis.label.get_fontsize() == 6.0
    assert fig.axes[0].spines["bottom"].get_linewidth() == 0.25
    assert fig.dpi == 300


def test_labels_stay_inside_the_figure(map_state):
    for preset, width, height in [(NATURE, 89, 40), (APS, 86, 60), (APS, 178, 60)]:
        fig = render(map_state, preset=preset, width_mm=width, height_mm=height, panel_label="a")
        with mpl.rc_context(figure_rc(preset, preset.font_size_pt, preset.line_width_pt)):
            fig.canvas.draw()
            bbox = fig.get_tightbbox(fig.canvas.get_renderer())
        w, h = fig.get_size_inches()
        assert bbox.x0 >= 0 and bbox.y0 >= 0, preset.name
        assert bbox.x1 <= w + 1e-9 and bbox.y1 <= h + 1e-9, preset.name


def test_map_image_puts_pixel_centres_on_the_grid(map_state, fmap):
    fig = nature_single(map_state)
    ax = fig.axes[0]
    (image,) = ax.images
    assert isinstance(image, AxesImage)
    de, db = fmap.energy[1] - fmap.energy[0], fmap.field[1] - fmap.field[0]
    np.testing.assert_allclose(
        image.get_extent(),
        [-db / 2, 16 + db / 2, 10 - de / 2, 120 + de / 2],
    )
    np.testing.assert_array_equal(image.get_array(), fmap.values)
    assert (image.norm.vmin, image.norm.vmax) == (0.9, 1.0)
    # default view: first to last sample
    assert ax.get_xlim() == (0.0, 16.0)
    assert ax.get_ylim() == (10.0, 120.0)
    assert ax.get_xlabel() == FIELD_LABEL
    assert ax.get_ylabel() == "Energy (meV)"


def test_map_descending_axes_are_sorted(fmap):
    flipped = FieldMap(fmap.energy[::-1], fmap.field[::-1], fmap.values[::-1, ::-1], "meV")
    fig = nature_single(FigureState("map", flipped))
    np.testing.assert_array_equal(fig.axes[0].images[0].get_array(), fmap.values)


def test_map_uneven_field_steps_use_cells_around_each_sample():
    field = np.array([0.0, 1.0, 2.0, 4.0])
    energy = np.linspace(100.0, 200.0, 11)
    state = FigureState("map", FieldMap(energy, field, np.ones((11, 4))), colorbar=False)
    ax = nature_single(state).axes[0]
    assert not ax.images
    (mesh,) = (c for c in ax.collections if isinstance(c, QuadMesh))
    assert mesh.get_rasterized()
    xs = np.unique(mesh.get_coordinates()[..., 0])
    np.testing.assert_allclose(xs, [-0.5, 0.5, 1.5, 3.0, 5.0])
    assert ax.get_ylabel() == "Energy (cm$^{-1}$)"


def test_map_levels_ranges_and_labels(map_state, fmap):
    finite = fmap.values.ravel()
    state = dataclasses.replace(
        map_state,
        levels=None,
        x_range=(2.0, None),
        y_range=(None, 80.0),
        x_label="$B$ (T)",
        y_label="",
        colorbar=False,
        title="Sample A",
    )
    fig = nature_single(state)
    ax = fig.axes[0]
    image = ax.images[0]
    np.testing.assert_allclose((image.norm.vmin, image.norm.vmax), np.percentile(finite, [1, 99]))
    assert ax.get_xlim() == (2.0, 16.0)
    assert ax.get_ylim() == (10.0, 80.0)
    assert (ax.get_xlabel(), ax.get_ylabel()) == ("$B$ (T)", "")
    assert ax.get_title() == "Sample A"
    assert len(fig.axes) == 1  # no colour bar
    flat = dataclasses.replace(map_state, levels=(2.0, 2.0))
    image = nature_single(flat).axes[0].images[0]
    assert (image.norm.vmin, image.norm.vmax) == (1.5, 2.5)


def test_large_wavenumbers_have_no_exponent():
    energy = np.linspace(8000.0, 12000.0, 41)
    fmap = FieldMap(energy, np.linspace(0.0, 4.0, 5), np.ones((41, 5)), unit="cm-1")
    fig = nature_single(FigureState("map", fmap, colorbar=False))
    FigureCanvasAgg(fig).draw()
    ax = fig.axes[0]
    assert ax.yaxis.get_offset_text().get_text() == ""
    assert "12000" in [t.get_text() for t in ax.get_yticklabels()]


def test_map_colorbar(map_state):
    fig = nature_single(map_state)
    assert len(fig.axes) == 2
    cax = fig.axes[1]
    assert cax.get_ylabel() == "Relative transmission"
    assert cax.spines["outline"].get_linewidth() == 0.5


def test_map_overlays(map_state):
    ax = nature_single(map_state).axes[0]
    model, plain, current, other = ax.lines
    assert model.get_color() == "white"
    assert model.get_linestyle() == "--"
    assert model.get_path_effects()
    assert model.get_label() == "n=0"
    assert plain.get_linestyle() == "-"
    assert current.get_marker() != other.get_marker()
    assert current.get_markerfacecolor() == "white"  # current curve: filled
    assert other.get_markerfacecolor() == "none"  # other curves: open
    for line in ax.lines:
        assert line.get_linewidth() == 0.5 or line.get_linestyle() == "None"
        assert line.get_clip_on() and line.get_clip_box() is not None  # never outside the axes
    # overlays do not widen the view
    assert ax.get_xlim() == (0.0, 16.0)


def test_markers_are_clipped_at_the_axes(map_state):
    state = dataclasses.replace(
        map_state, curves=[], points=[PointSet([8.0], [120.0], current=True)], colorbar=False
    )
    fig = nature_single(state, dpi=300)
    ax = fig.axes[0]
    rgba = rasterize(fig, 300)
    x, top = ax.transData.transform((8.0, 120.0))
    row = rgba.shape[0] - int(top)  # the top spine
    assert (rgba[row - 12 : row - 3, int(x), :3] == 255).all()  # the marker is cut at the edge
    assert (rgba[row + 3, int(x), :3] == 255).all()  # its white face shows inside


def test_nan_is_white(fmap):
    values = np.full(fmap.values.shape, np.nan)
    state = FigureState("map", fmap.with_values(values), levels=(0, 1), colorbar=False)
    dark = FigureState("map", fmap.with_values(np.zeros_like(values)), levels=(0, 1))
    for st, expected in [(state, [255, 255, 255]), (dark, list(colormaps.lut("magma")[0]))]:
        fig = nature_single(st, dpi=100)
        ax = fig.axes[0]
        rgba = rasterize(fig, 100)
        x, y = ax.transAxes.transform((0.5, 0.5)) * 100 / fig.dpi
        assert list(rgba[rgba.shape[0] - int(y), int(x), :3]) == expected


def test_panel_label(map_state):
    fig = nature_single(map_state, panel_label="A")
    label = fig.get_suptitle()
    assert label == "a"
    (text,) = [t for t in fig.findobj(Text) if t.get_text() == label]
    assert text.get_fontweight() == "bold"
    assert text.get_fontsize() == 8.0
    assert text.get_horizontalalignment() == "left"
    fig = render(map_state, preset=APS, width_mm=86, height_mm=60, panel_label="b")
    assert fig.get_suptitle() == "(b)"
    assert nature_single(map_state).get_suptitle() == ""
    blank = render(map_state, preset=APS, width_mm=86, height_mm=60, panel_label=" ")
    assert blank.get_suptitle() == ""


def test_render_rejects_bad_input(map_state):
    with pytest.raises(ValueError, match="positive"):
        render(map_state, preset=NATURE, width_mm=0, height_mm=50)
    with pytest.raises(ValueError, match="kind"):
        FigureState("contour", map_state.fmap)
    with pytest.raises(ValueError, match="style"):
        Curve([1.0], [2.0], style="dotted")
    with pytest.raises(ValueError, match="shape"):
        PointSet([1.0, 2.0], [3.0])
    with pytest.raises(ValueError, match="every"):
        StackedOptions(0.1, every=0)


# ---------------------------------------------------------------------- stacked
def test_render_stacked(fmap):
    fmap = FieldMap(fmap.energy, fmap.field[:7], fmap.values[:, :7], "meV")
    e = 40.0
    state = FigureState(
        kind="stacked",
        fmap=fmap,
        cmap="viridis",
        points=[
            PointSet(fmap.field[[2, 3, 4]], [e, e, e], "L0", current=True),
            PointSet([fmap.field[6] + 0.01], [e], "L1"),
            PointSet([fmap.field[0]], [500.0], "outside the spectrum"),
        ],
        stacked=StackedOptions(offset=0.2, every=2, color_by_field=True),
    )
    fig = nature_single(state)
    ax = fig.axes[0]
    traces, points = ax.lines[:4], ax.lines[4:]
    assert len(traces) == 4  # fields 0, 2, 4, 6
    cmap = colormap("viridis")
    for k, (line, j) in enumerate(zip(traces, [0, 2, 4, 6], strict=True)):
        np.testing.assert_array_equal(line.get_xdata(), fmap.energy)
        np.testing.assert_allclose(line.get_ydata(), fmap.values[:, j] + 0.2 * k)
        t = (fmap.field[j] - fmap.field[0]) / (fmap.field[6] - fmap.field[0])
        colour = mpl.colors.to_rgba(line.get_color())
        np.testing.assert_allclose(colour, cmap(0.1 + 0.72 * t), atol=0.01)
    # field 3 is not shown, so its point is skipped; the others sit on their traces
    np.testing.assert_allclose(points[0].get_xdata(), [e, e])
    expected = [np.interp(e, fmap.energy, fmap.values[:, j]) + 0.2 * k for j, k in [(2, 1), (4, 2)]]
    np.testing.assert_allclose(points[0].get_ydata(), expected)
    assert points[0].get_markerfacecolor() == "black"
    np.testing.assert_allclose(
        points[1].get_ydata(), [np.interp(e, fmap.energy, fmap.values[:, 6]) + 0.6]
    )
    assert points[1].get_markerfacecolor() == "white"
    assert len(points[2].get_xdata()) == 0
    assert ax.get_xlim() == (10.0, 120.0)
    assert ax.get_xlabel() == "Energy (meV)"
    assert ax.get_ylabel() == INTENSITY_LABEL
    # colour bar of the field
    assert len(fig.axes) == 2
    assert fig.axes[1].get_ylabel() == FIELD_LABEL
    np.testing.assert_allclose(fig.axes[1].get_ylim(), (0.0, 6.0))


def test_stacked_field_colours_span_all_fields(fmap):
    fmap = FieldMap(fmap.energy, fmap.field[:6], fmap.values[:, :6], "meV")  # 0 .. 5 T
    state = FigureState("stacked", fmap, stacked=StackedOptions(0.1, every=2))
    fig = nature_single(state)
    np.testing.assert_allclose(fig.axes[1].get_ylim(), (0.0, 5.0))
    colour = mpl.colors.to_rgba(fig.axes[0].lines[2].get_color())  # 4 T
    np.testing.assert_allclose(colour, colormap("magma")(0.1 + 0.72 * 0.8), atol=0.01)


def test_render_stacked_without_field_colours(fmap):
    state = FigureState(
        "stacked",
        fmap,
        y_range=(0.5, 3.0),
        x_range=(20.0, None),
        curves=[Curve([1.0], [20.0])],  # not drawn on stacked spectra
        stacked=StackedOptions(0.1, color_by_field=False),
    )
    fig = nature_single(state)
    ax = fig.axes[0]
    assert len(ax.lines) == fmap.field.size
    assert {line.get_color() for line in ax.lines} == {"black"}
    assert ax.get_ylim() == (0.5, 3.0)
    assert ax.get_xlim() == (20.0, 120.0)
    assert len(fig.axes) == 1


# ---------------------------------------------------------------------- output
@pytest.mark.parametrize("dpi", [300, 600])
def test_png_has_the_exact_pixel_size(map_state, tmp_path, dpi):
    fig = nature_single(map_state, height_mm=70.0)
    path = save(fig, tmp_path / "figure.png", dpi=dpi)
    with Image.open(path) as image:
        assert image.size == (round(89 / MM * dpi), round(70 / MM * dpi))
        assert image.mode == "RGB"
        np.testing.assert_allclose(image.info["dpi"], (dpi, dpi), atol=0.01)
    # the figure itself is unchanged
    np.testing.assert_allclose(fig.get_size_inches() * MM, [89.0, 70.0])
    assert fig.dpi == NATURE.raster_dpi


def test_rasterize_size(map_state):
    fig = nature_single(map_state, height_mm=50.0)
    rgba = rasterize(fig, 150)
    assert rgba.shape == (round(50 / MM * 150), round(89 / MM * 150), 4)
    assert rgba.dtype == np.uint8


def test_tiff(map_state, tmp_path):
    path = save(nature_single(map_state), tmp_path / "figure.tif", dpi=300)
    with Image.open(path) as image:
        assert image.format == "TIFF"
        assert image.mode == "RGB"
        assert image.size == (round(89 / MM * 300), round(70 / MM * 300))


def _pdf_text(data: bytes) -> list[str]:
    """Text of every BT ... ET block in the (compressed) PDF content streams."""

    def unescape(s: bytes) -> bytes:
        def repl(m: re.Match) -> bytes:
            esc = m.group(1)
            if esc[:1].isdigit():
                return bytes([int(esc, 8) & 0xFF])
            return {b"n": b"\n", b"r": b"\r", b"t": b"\t", b"b": b"\b", b"f": b"\f"}.get(esc, esc)

        return re.sub(rb"\\([0-7]{1,3}|.)", repl, s, flags=re.S)

    texts = []
    for stream in re.findall(rb"stream\r?\n(.*?)endstream", data, re.S):
        try:
            content = zlib.decompress(stream)
        except zlib.error:
            continue
        for block in re.findall(rb"BT(.*?)ET", content, re.S):
            pieces = re.findall(rb"\(((?:\\.|[^\\)])*)\)", block, re.S)
            texts.append(b"".join(unescape(p) for p in pieces).decode("utf-16-be"))
    return texts


def test_pdf_page_size_and_editable_text(map_state, tmp_path):
    fig = nature_single(map_state, panel_label="a")
    data = save(fig, tmp_path / "figure.pdf", dpi=300).read_bytes()
    (box,) = re.findall(rb"/MediaBox\s*\[([^\]]*)\]", data)
    np.testing.assert_allclose([float(v) for v in box.split()], [0, 0, 89 / MM * 72, 70 / MM * 72])
    # fonttype 42: an embedded TrueType font, no Type 3 glyph procedures or outlines
    assert b"/FontFile2" in data
    assert b"/Type3" not in data
    text = _pdf_text(data)
    for label in (FIELD_LABEL, "Energy (meV)", "Relative transmission", "a"):
        assert label in text


def test_svg_keeps_text(map_state, tmp_path):
    data = save(nature_single(map_state), tmp_path / "figure.svg", dpi=300).read_bytes()
    root = ET.fromstring(data)
    assert root.get("width") == f"{89 / MM * 72:.6f}pt"
    texts = root.findall(".//{http://www.w3.org/2000/svg}text")
    assert FIELD_LABEL in ["".join(t.itertext()) for t in texts]
    assert all("font-family" in t.get("style", "") for t in texts)


def test_save_keeps_the_figure_canvas(map_state, tmp_path):
    fig = nature_single(map_state)
    canvas = fig.canvas
    for suffix in (".pdf", ".svg", ".eps", ".ps", ".png"):
        save(fig, tmp_path / f"figure{suffix}", dpi=100)
        assert fig.canvas is canvas


def test_save_hides_fonttools_info(map_state, tmp_path, caplog):
    fonttools = logging.getLogger("fontTools")
    level = fonttools.level
    with caplog.at_level(logging.INFO):
        save(nature_single(map_state), tmp_path / "figure.pdf", dpi=100)
    assert not [r for r in caplog.records if r.name.startswith("fontTools")]
    assert fonttools.level == level  # no lasting change to the library's logger


def test_eps_and_ps_save(map_state, tmp_path):
    fig = nature_single(map_state)
    eps = save(fig, tmp_path / "figure.eps", dpi=150).read_bytes()
    assert eps.startswith(b"%!PS-Adobe-3.0 EPSF-3.0")
    assert b"/FontType 42 def" in eps
    (box,) = re.findall(rb"%%HiResBoundingBox: ([^\n]*)", eps)
    np.testing.assert_allclose([float(v) for v in box.split()], [0, 0, 89 / MM * 72, 70 / MM * 72])
    ps = save(fig, tmp_path / "figure.ps", dpi=150).read_bytes()
    assert ps.startswith(b"%!PS-Adobe-3.0")
    assert b"/FontType 42 def" in ps


def test_save_rejects_unknown_formats(map_state, tmp_path):
    assert set(FORMATS) == {".pdf", ".svg", ".eps", ".ps", ".png", ".tif", ".tiff"}
    with pytest.raises(ValueError, match="unknown figure format"):
        save(nature_single(map_state), tmp_path / "figure.jpg", dpi=300)


def test_export_does_not_import_qt_or_pyplot(tmp_path):
    code = textwrap.dedent(
        """
        import sys
        from pathlib import Path

        import numpy as np

        from mag_opt_detective.core.spectra import FieldMap
        from mag_opt_detective.export import Curve, FigureState, PointSet, render, save

        fmap = FieldMap(np.linspace(1.0, 2.0, 5), np.linspace(0.0, 1.0, 3), np.ones((5, 3)))
        for kind in ("map", "stacked"):
            state = FigureState(
                kind, fmap, curves=[Curve([0, 1], [1, 2])], points=[PointSet([0.5], [1.5])]
            )
            fig = render(state, preset="nature", width_mm=89, height_mm=60, panel_label="a")
            for suffix in (".png", ".pdf", ".svg", ".eps"):
                save(fig, Path(sys.argv[1]) / (kind + suffix), dpi=100)
        bad = sorted(
            m
            for m in sys.modules
            if m.split(".")[0] in ("PySide6", "pyqtgraph", "PyQt5", "PyQt6", "tkinter")
            or m == "matplotlib.pyplot"
            or m.startswith(("matplotlib.backends.backend_qt", "matplotlib.backends.backend_tk"))
        )
        print(",".join(bad))
        """
    )
    out = subprocess.run(
        [sys.executable, "-c", code, str(tmp_path)], capture_output=True, text=True, check=True
    ).stdout
    assert out.strip() == ""
    assert len(list(tmp_path.iterdir())) == 8


def test_export_imports_its_vector_canvases():
    """matplotlib loads them by name; a plain import lets PyInstaller see and bundle them."""
    code = textwrap.dedent(
        """
        import sys

        import mag_opt_detective.export

        wanted = [f"matplotlib.backends.backend_{name}" for name in ("agg", "pdf", "svg", "ps")]
        print(",".join(m for m in wanted if m not in sys.modules))
        """
    )
    out = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=True
    ).stdout
    assert out.strip() == ""
