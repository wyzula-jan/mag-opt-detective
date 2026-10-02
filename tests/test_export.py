import numpy as np
import pytest

from mag_opt_detective.core.spectra import FieldMap
from mag_opt_detective.export import (
    APS,
    CUSTOM,
    NATURE,
    PRESETS,
    Curve,
    FigureState,
    PointSet,
    StackedOptions,
    get_preset,
)


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


def test_preset_check():
    assert NATURE.check(89, 70, font_size_pt=7, line_width_pt=0.5, dpi=450) == []
    problems = NATURE.check(200, 300, font_size_pt=4, line_width_pt=0.1, dpi=100)
    assert len(problems) == 5
    assert NATURE.check(89, 70, line_width_pt=1.5) == [
        "Nature: lines should be at most 1 pt, not 1.5 pt."
    ]
    assert APS.check(86, 300) == []  # APS gives no height limit
    assert CUSTOM.check(0, 10) == ["Width and height must be positive."]


# ---------------------------------------------------------------------- state
def test_figure_state():
    fmap = FieldMap(np.linspace(1.0, 2.0, 5), np.linspace(0.0, 1.0, 3), np.ones((5, 3)))
    state = FigureState("map", fmap)
    assert (state.x_range, state.levels, state.x_label, state.curves) == (None, None, None, [])
    assert (state.stacked.offset, state.stacked.every, state.stacked.color_by_field) == (0, 1, True)
    curve = Curve([0, 1], [2, 3])
    assert curve.x.dtype == float and curve.style == "model"
    with pytest.raises(ValueError, match="kind"):
        FigureState("contour", fmap)
    with pytest.raises(ValueError, match="style"):
        Curve([1.0], [2.0], style="dotted")
    with pytest.raises(ValueError, match="shape"):
        PointSet([1.0, 2.0], [3.0])
    with pytest.raises(ValueError, match="every"):
        StackedOptions(0.1, every=0)
