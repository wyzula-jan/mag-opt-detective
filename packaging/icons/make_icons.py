"""Build the app icon files from the full-bleed artwork (macOS only, needs iconutil).

    uv run --with pillow python packaging/icons/make_icons.py

artwork-1024.png -> mag-opt-detective.icns (macOS icon grid: 824 px rounded square with
continuous corners on a 1024 px canvas, soft shadow), mag-opt-detective.ico (rounded square
filling the canvas, no shadow) and the 512 px window icons in mag_opt_detective/resources.
"""

import subprocess
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image, ImageChops, ImageFilter

HERE = Path(__file__).parent
ARTWORK = HERE / "artwork-1024.png"
RESOURCES = HERE.parents[1] / "src" / "mag_opt_detective" / "resources"
CANVAS = 1024
SUPERSAMPLE = 4


def rounded_square(size: int, radius: float = 0.225, n: float = 3.26) -> Image.Image:
    """Mask of Apple's rounded square: corner radius 22.5 % of the side, continuous corners.

    Each corner is a superellipse quadrant spanning 1.528 r whose 45-degree point matches a
    circular corner of radius r.
    """
    s = size * SUPERSAMPLE
    c = 1.528 * radius * s
    t = np.arange(s) + 0.5
    d = np.maximum(np.maximum(c - t, t - (s - c)), 0) / c
    dx, dy = np.meshgrid(d, d)
    inside = (dx**n + dy**n) <= 1.0
    mask = Image.fromarray((inside * 255).astype(np.uint8), "L")
    return mask.resize((size, size), Image.LANCZOS)


def tile(art: Image.Image, body: int, rim: bool) -> tuple[Image.Image, Image.Image]:
    mask = rounded_square(body)
    out = Image.new("RGBA", (body, body))
    out.paste(art.convert("RGBA").resize((body, body), Image.LANCZOS), (0, 0), mask)
    if rim:  # faint inner edge highlight, as in Apple's templates
        edge = ImageChops.subtract(mask, mask.filter(ImageFilter.MinFilter(5)))
        light = Image.new("RGBA", (body, body), (255, 255, 255, 0))
        light.putalpha(edge.point(lambda v: round(v * 0.18)))
        out = Image.alpha_composite(out, light)
    return out, mask


def macos_icon(art: Image.Image) -> Image.Image:
    body = 824
    margin = (CANVAS - body) // 2
    front, mask = tile(art, body, rim=True)
    alpha = Image.new("L", (CANVAS, CANVAS), 0)
    alpha.paste(mask.point(lambda v: round(v * 0.30)), (margin, margin + 10))
    out = Image.new("RGBA", (CANVAS, CANVAS), (0, 0, 0, 0))
    out.putalpha(alpha.filter(ImageFilter.GaussianBlur(10)))
    out.alpha_composite(front, (margin, margin))
    return out


def plain_icon(art: Image.Image, fill: float = 0.94) -> Image.Image:
    body = round(CANVAS * fill)
    front, _ = tile(art, body, rim=False)
    out = Image.new("RGBA", (CANVAS, CANVAS))
    out.alpha_composite(front, ((CANVAS - body) // 2, (CANVAS - body) // 2))
    return out


def main() -> None:
    art = Image.open(ARTWORK)
    mac = macos_icon(art)
    plain = plain_icon(art)

    with tempfile.TemporaryDirectory() as tmp:
        iconset = Path(tmp) / "mag-opt-detective.iconset"
        iconset.mkdir()
        for s in (16, 32, 128, 256, 512):
            mac.resize((s, s), Image.LANCZOS).save(iconset / f"icon_{s}x{s}.png")
            mac.resize((2 * s, 2 * s), Image.LANCZOS).save(iconset / f"icon_{s}x{s}@2x.png")
        icns = HERE / "mag-opt-detective.icns"
        subprocess.run(["iconutil", "-c", "icns", str(iconset), "-o", str(icns)], check=True)

    sizes = [(s, s) for s in (16, 24, 32, 48, 64, 128, 256)]
    plain.save(HERE / "mag-opt-detective.ico", sizes=sizes)
    RESOURCES.mkdir(exist_ok=True)
    mac.resize((512, 512), Image.LANCZOS).save(RESOURCES / "icon-macos.png", optimize=True)
    plain.resize((512, 512), Image.LANCZOS).save(RESOURCES / "icon.png", optimize=True)


if __name__ == "__main__":
    main()
