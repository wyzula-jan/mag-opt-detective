"""Draw the app icon: a Landau fan (E ∝ √(nB)) in the magma colour map on a rounded tile.

    python packaging/make_icon.py

writes ``src/mag_opt_detective/gui/app_icon.png`` (1024 px). The app uses it as its window
icon and PyInstaller converts it to the .icns / .ico of the bundles.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

from mag_opt_detective.core.colormaps import lut

SIZE = 1024
OUT = Path(__file__).resolve().parents[1] / "src" / "mag_opt_detective" / "gui" / "app_icon.png"


def rounded_tile(size: int, margin: float = 0.06, radius: float = 0.2) -> np.ndarray:
    """Alpha (0-1) of a rounded square with anti-aliased edges."""
    c = (np.arange(size) + 0.5) / size
    x, y = np.meshgrid(c, c)
    half = 0.5 - margin
    qx = np.maximum(np.abs(x - 0.5) - (half - radius), 0.0)
    qy = np.maximum(np.abs(y - 0.5) - (half - radius), 0.0)
    distance = np.hypot(qx, qy) - radius  # < 0 inside
    return np.clip(0.5 - distance * size, 0.0, 1.0)


def landau_fan(size: int) -> np.ndarray:
    """Intensity 0-1 of a few Landau-level lines on a faint background, field to the right."""
    c = (np.arange(size) + 0.5) / size
    b, e = np.meshgrid(c, 1.0 - c)  # energy up
    b = (b - 0.2) / 0.74  # the lines start inside the tile
    width = 0.012 + 0.006 * np.clip(b, 0, 1)
    value = 0.08 + 0.12 * e
    for n in range(1, 6):
        line = 0.2 + 0.28 * np.sqrt(n * np.clip(b, 0, None))
        value += (0.95 / n**0.35) * np.exp(-(((e - line) / width) ** 2)) * np.clip(40 * b, 0, 1)
    return np.clip(value, 0.0, 1.0)


def icon(size: int = SIZE) -> Image.Image:
    colours = lut("magma")
    index = np.round(landau_fan(size) * (len(colours) - 1)).astype(int)
    rgb = colours[index][..., :3].astype(np.uint8)
    alpha = (rounded_tile(size) * 255).round().astype(np.uint8)
    return Image.fromarray(np.dstack([rgb, alpha]), "RGBA")


if __name__ == "__main__":
    icon().save(OUT, optimize=True)
    print(OUT)
