"""The rail's data states (Sample, Reference), its labels, and Process without reference files."""

from mag_opt_detective.gui import icons


def test_filled_icons_draw_the_glyph_on_a_tile(qapp):
    icons.clear_cache()
    filled = icons.icon("activity", "#ffffff", fill="#7a2a8c")
    assert filled.cacheKey() == icons.icon("activity", "#ffffff", fill="#7a2a8c").cacheKey()
    assert filled.cacheKey() != icons.icon("activity", "#ffffff").cacheKey()
    image = filled.pixmap(20, 20).toImage()
    assert image.pixelColor(10, 0).name() == "#7a2a8c"  # the tile
    colours = {image.pixelColor(x, y).name() for x in range(20) for y in range(20)}
    assert "#ffffff" in colours  # the glyph on it
    for name in ("activity", "layers"):  # the SVGs stay Lucide's outlines
        assert b'fill="none"' in icons.svg_data(name)
