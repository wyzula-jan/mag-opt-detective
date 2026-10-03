"""The screenshots of the docs site and the README, which docs/make_screenshots.py draws.

A module of its own, without Qt, so that tests/test_site.py can read the list.
"""

from __future__ import annotations

from collections.abc import Iterable

SCHEMES = ("light", "dark")
SUFFIX = ".webp"
# the scenes of the docs site (docs/site/images/), in the order they are drawn
SITE = (
    "main-window",
    "points",
    "export-window",
    "processing",
    "stacked",
    "derivative",
    "autopick",
    "models",
    "fit",
    "library",
)
EXTRA = ("reference",)  # drawn only with --all
README = ("main-window", "points", "export-window")  # docs/images/: copies of the site's


def file_names(scenes: Iterable[str]) -> list[str]:
    """The image files of *scenes*: ``<scene>-light.webp`` and ``<scene>-dark.webp`` each."""
    return [f"{name}-{scheme}{SUFFIX}" for name in scenes for scheme in SCHEMES]
