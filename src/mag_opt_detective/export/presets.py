"""Figure rules of the journals we submit to, kept in one table with their sources."""

from __future__ import annotations

from dataclasses import dataclass

PANEL_LABEL_STYLES: tuple[str, ...] = ("a", "(a)")


@dataclass(frozen=True, eq=False)
class JournalPreset:
    """Sizes in mm, fonts and lines in pt, resolution in dpi.

    *widths_mm* names the column widths; *font_family* lists the fonts in fallback
    order. *panel_label_style* is "a" (bold lowercase letter) or "(a)". Limits that a
    journal does not give are None. *notes* are shown to the user next to the preset.
    """

    key: str
    name: str
    widths_mm: dict[str, float]
    default_width: str | None
    default_height_mm: float
    max_height_mm: float | None
    font_family: tuple[str, ...]
    font_size_pt: float
    font_size_range_pt: tuple[float, float]
    label_size_pt: float
    panel_label_style: str
    line_width_pt: float
    min_line_width_pt: float
    max_line_width_pt: float | None
    raster_dpi: int
    raster_dpi_range: tuple[int, int]
    formats: tuple[str, ...]
    colour_mode: str
    sources: tuple[str, ...]
    notes: tuple[str, ...]
    free_size: bool = False

    def __post_init__(self) -> None:
        if self.panel_label_style not in PANEL_LABEL_STYLES:
            raise ValueError(f"unknown panel label style {self.panel_label_style!r}")
        if self.default_width is not None and self.default_width not in self.widths_mm:
            raise ValueError(f"default width {self.default_width!r} is not in widths_mm")

    @property
    def default_width_mm(self) -> float:
        """Width of the default column (the free default for presets without columns)."""
        if self.default_width is None:
            return CUSTOM_DEFAULT_SIZE_MM[0]
        return self.widths_mm[self.default_width]

    def panel_label(self, label: str) -> str:
        """*label* (``"b"``, ``"B"`` or ``"(b)"``) in this preset's style."""
        letter = label.strip().strip("()").strip().lower()
        return letter if self.panel_label_style == "a" else f"({letter})"

    def check(
        self,
        width_mm: float,
        height_mm: float,
        font_size_pt: float | None = None,
        line_width_pt: float | None = None,
        dpi: float | None = None,
    ) -> list[str]:
        """Plain-language warnings for values outside this preset's rules (empty if none)."""
        problems = []
        if width_mm <= 0 or height_mm <= 0:
            problems.append("Width and height must be positive.")
        if not self.free_size and self.widths_mm:
            widest = max(self.widths_mm.values())
            if width_mm > widest + 1e-9:
                problems.append(f"{self.name}: width {width_mm:g} mm is wider than {widest:g} mm.")
        if self.max_height_mm is not None and height_mm > self.max_height_mm + 1e-9:
            problems.append(
                f"{self.name}: height {height_mm:g} mm is more than {self.max_height_mm:g} mm."
            )
        if font_size_pt is not None:
            lo, hi = self.font_size_range_pt
            if not lo <= font_size_pt <= hi:
                problems.append(
                    f"{self.name}: text should be {lo:g}–{hi:g} pt, not {font_size_pt:g} pt."
                )
        if line_width_pt is not None:
            if line_width_pt < self.min_line_width_pt:
                problems.append(
                    f"{self.name}: lines should be at least {self.min_line_width_pt:g} pt, "
                    f"not {line_width_pt:g} pt."
                )
            elif self.max_line_width_pt is not None and line_width_pt > self.max_line_width_pt:
                problems.append(
                    f"{self.name}: lines should be at most {self.max_line_width_pt:g} pt, "
                    f"not {line_width_pt:g} pt."
                )
        if dpi is not None:
            lo, hi = self.raster_dpi_range
            if not lo <= dpi <= hi:
                problems.append(f"{self.name}: raster images should be {lo}–{hi} dpi, not {dpi:g}.")
        return problems


CUSTOM_DEFAULT_SIZE_MM = (120.0, 90.0)
FONTS = ("Arial", "Helvetica", "DejaVu Sans")

NATURE = JournalPreset(
    key="nature",
    name="Nature",
    widths_mm={"single": 89.0, "1.5 narrow": 120.0, "1.5 wide": 136.0, "double": 183.0},
    default_width="single",
    default_height_mm=70.0,
    max_height_mm=247.0,
    font_family=FONTS,
    font_size_pt=7.0,
    font_size_range_pt=(5.0, 7.0),
    label_size_pt=8.0,
    panel_label_style="a",
    line_width_pt=0.5,
    min_line_width_pt=0.25,
    max_line_width_pt=1.0,
    raster_dpi=450,
    raster_dpi_range=(300, 600),
    formats=("pdf", "eps", "tif", "png"),
    colour_mode="RGB",
    sources=("https://www.nature.com/nature/for-authors/final-submission",),
    notes=(
        "Widths: 89 mm (single column), 120–136 mm (1.5 columns), 183 mm (double column); "
        "height up to the full page depth of 247 mm.",
        "Text 5–7 pt in Helvetica or Arial; panel labels 8 pt bold lowercase (a, b, c).",
        "Lines 0.25–1 pt.",
        "Text must stay editable: fonts are embedded as TrueType, never as outlines.",
        "Vector PDF or EPS preferred; raster TIFF or PNG at 300 dpi or more, in RGB.",
    ),
)

APS = JournalPreset(
    key="aps",
    name="APS (Physical Review)",
    widths_mm={"single": 86.0, "double": 178.0},
    default_width="single",
    default_height_mm=65.0,
    max_height_mm=None,
    font_family=FONTS,
    font_size_pt=8.0,
    font_size_range_pt=(8.0, 12.0),
    label_size_pt=8.0,
    panel_label_style="(a)",
    line_width_pt=0.75,
    min_line_width_pt=0.5,
    max_line_width_pt=None,
    raster_dpi=600,
    raster_dpi_range=(600, 1200),
    formats=("eps", "ps", "pdf", "png"),
    colour_mode="RGB",
    sources=(
        "https://journals.aps.org/authors/style-basics",
        "https://res.cloudinary.com/apsphysics/image/upload/v1715884920/"
        "aps-journals-style-guide_tnoyln.pdf",
    ),
    notes=(
        "Single column 8.6 cm (APS style basics); the APS style guide says 8.5 cm.",
        "The two-column width of 17.8 cm is not confirmed by an official APS page.",
        "Lettering at least 2 mm high (about 8 pt); lines at least 0.5 pt (0.18 mm).",
        "Colour is shown online; the figure must stay readable in grayscale.",
        "EPS or PS preferred, PDF and PNG accepted; raster images at 600 dpi.",
    ),
)

CUSTOM = JournalPreset(
    key="custom",
    name="Custom",
    widths_mm={},
    default_width=None,
    default_height_mm=CUSTOM_DEFAULT_SIZE_MM[1],
    max_height_mm=None,
    font_family=FONTS,
    font_size_pt=8.0,
    font_size_range_pt=(4.0, 24.0),
    label_size_pt=8.0,
    panel_label_style="(a)",
    line_width_pt=0.75,
    min_line_width_pt=0.25,
    max_line_width_pt=None,
    raster_dpi=600,
    raster_dpi_range=(72, 1200),
    formats=("pdf", "svg", "eps", "ps", "png", "tif"),
    colour_mode="RGB",
    sources=(),
    notes=("No journal rules: width and height are free.",),
    free_size=True,
)

PRESETS: dict[str, JournalPreset] = {p.key: p for p in (NATURE, APS, CUSTOM)}


def get_preset(name: str | JournalPreset) -> JournalPreset:
    """The preset with key or display name *name* (case-insensitive); presets pass through."""
    if isinstance(name, JournalPreset):
        return name
    wanted = name.strip().lower()
    for preset in PRESETS.values():
        if wanted in (preset.key, preset.name.lower()):
            return preset
    raise ValueError(f"unknown preset {name!r}; choose from {', '.join(PRESETS)}")
