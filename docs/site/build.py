"""Build the documentation and download site into docs/site/_build/ (standard library only).

    python docs/site/build.py              # into docs/site/_build/ (replaced)
    python docs/site/build.py --out DIR    # into DIR, which must be new or empty

Each file in pages/ is a fragment of plain HTML that starts with a comment holding its
title and description::

    <!--
    title: Picking points
    description: Pick transition energies by hand, or let auto-pick follow a line.
    -->
    <h1>Picking points</h1>
    ...

layout.html wraps every fragment in the page head, the header with the main navigation and
the footer; the documentation pages also get the sidebar and links to the previous and
next page. The style sheet, the icon and images/ are copied next to the pages. Every link is
relative, so the site works from a local server and from a project page such as
https://wyzula-jan.github.io/mag-opt-detective/. The build fails on a broken internal link,
image or anchor.
"""

from __future__ import annotations

import argparse
import html
import re
import shutil
import sys
from html.parser import HTMLParser
from pathlib import Path
from typing import NamedTuple
from urllib.parse import unquote, urlsplit

HERE = Path(__file__).resolve().parent
PAGES = HERE / "pages"
IMAGES = HERE / "images"
LAYOUT = HERE / "layout.html"
STATIC = ("site.css", "favicon.svg")
OUT = HERE / "_build"

SITE_NAME = "Magneto-Optical Detective"
SEPARATOR = " \N{MIDDLE DOT} "

# the header's navigation: (file, label)
MAIN_NAV = (
    ("index.html", "Home"),
    ("download.html", "Download"),
    ("getting-started.html", "Docs"),
    ("release-notes.html", "Release notes"),
    ("about.html", "About"),
)
DOCS_ENTRY = "getting-started.html"  # where the header's "Docs" leads
# the documentation in reading order: the sidebar and the previous / next links
DOCS = (
    ("getting-started.html", "Getting started"),
    ("processing.html", "Processing"),
    ("viewing.html", "Viewing"),
    ("picking.html", "Picking points"),
    ("models.html", "Models and fitting"),
    ("library.html", "Library"),
    ("figures.html", "Journal figures"),
    ("shortcuts.html", "Keyboard shortcuts"),
    ("settings.html", "Settings and files"),
)
DOC_NAMES = tuple(name for name, _label in DOCS)

META = re.compile(r"\A\s*<!--(?P<meta>.*?)-->\s*", re.DOTALL)
PLACEHOLDER = re.compile(r"\{\{\s*(\w+)\s*\}\}")
H2 = re.compile(r'<h2 id="(?P<id>[^"]+)"[^>]*>(?P<text>.*?)</h2>', re.DOTALL)
TAG = re.compile(r"<[^>]+>")
TOC_MARK = "<!-- toc -->"
MEASUREMENT_DATA = "data_to_test"  # the lab data folder: never part of the site


class Page(NamedTuple):
    """One page: its file name, title, description and body (HTML)."""

    name: str
    title: str
    description: str
    body: str


def read_page(path: Path) -> Page:
    """The fragment *path*; ValueError if its metadata comment is missing or incomplete."""
    text = path.read_text(encoding="utf-8")
    match = META.match(text)
    if match is None:
        raise ValueError(f"{path.name}: no metadata comment at the top")
    meta = {}
    for line in match["meta"].strip().splitlines():
        key, sep, value = line.partition(":")
        if not sep:
            raise ValueError(f"{path.name}: metadata line without a colon: {line.strip()!r}")
        meta[key.strip()] = value.strip()
    missing = [key for key in ("title", "description") if not meta.get(key)]
    if missing:
        raise ValueError(f"{path.name}: metadata needs {', '.join(missing)}")
    return Page(path.name, meta["title"], meta["description"], text[match.end() :])


def read_pages(folder: Path = PAGES) -> dict[str, Page]:
    """Every fragment in *folder*, checked against the navigation."""
    pages = {path.name: read_page(path) for path in sorted(folder.glob("*.html"))}
    listed = [name for name, _label in MAIN_NAV] + list(DOC_NAMES)
    missing = sorted(set(listed) - set(pages))
    if missing:
        raise ValueError(f"no page fragment for {', '.join(missing)}")
    unlisted = sorted(set(pages) - set(listed))
    if unlisted:
        raise ValueError(f"pages not in the navigation: {', '.join(unlisted)}")
    return pages


# ---------------------------------------------------------------------- rendering
def nav_link(href: str, label: str, current: str | None) -> str:
    attr = f' aria-current="{current}"' if current else ""
    return f'<li><a href="{href}"{attr}>{html.escape(label)}</a></li>'


def main_nav(page: Page) -> str:
    items = []
    for name, label in MAIN_NAV:
        if name == page.name:
            current = "page"
        elif name == DOCS_ENTRY and page.name in DOC_NAMES:
            current = "true"  # the section the page is in
        else:
            current = None
        items.append("        " + nav_link(name, label, current))
    return "      <ul>\n" + "\n".join(items) + "\n      </ul>"


def table_of_contents(body: str) -> str:
    """A list of the page's h2 headings with ids, put where the body has ``<!-- toc -->``."""
    entries = [
        f'<li><a href="#{m["id"]}">{TAG.sub("", m["text"]).strip()}</a></li>'
        for m in H2.finditer(body)
    ]
    if not entries:
        return ""
    return (
        '<nav class="toc" aria-labelledby="toc-title">\n'
        '<p id="toc-title">On this page</p>\n'
        "<ul>\n" + "\n".join(entries) + "\n</ul>\n</nav>"
    )


def docs_main(page: Page) -> str:
    """The documentation layout: sidebar, the page, and links to its neighbours."""
    index = DOC_NAMES.index(page.name)
    side = "\n".join(
        "    " + nav_link(name, label, "page" if name == page.name else None)
        for name, label in DOCS
    )
    pager = []
    if index > 0:
        name, label = DOCS[index - 1]
        pager.append(
            f'<a class="prev" href="{name}" rel="prev"><span>Previous</span>'
            f"{html.escape(label)}</a>"
        )
    if index + 1 < len(DOCS):
        name, label = DOCS[index + 1]
        pager.append(
            f'<a class="next" href="{name}" rel="next"><span>Next</span>{html.escape(label)}</a>'
        )
    body = page.body.replace(TOC_MARK, table_of_contents(page.body))
    return (
        '<div class="wrap docs">\n'
        '  <nav class="side-nav" aria-labelledby="docs-nav-title">\n'
        '    <p class="side-title" id="docs-nav-title">Documentation</p>\n'
        f"    <ul>\n{side}\n    </ul>\n"
        "  </nav>\n"
        f'  <main id="content" class="doc">\n{body.rstrip()}\n'
        '<nav class="pager" aria-label="Previous and next page">\n'
        + "\n".join(pager)
        + "\n</nav>\n  </main>\n</div>"
    )


def render(page: Page, layout: str) -> str:
    """The complete HTML of *page*."""
    if page.name in DOC_NAMES:
        main = docs_main(page)
    else:
        body = page.body.replace(TOC_MARK, table_of_contents(page.body))
        main = f'<main id="content" class="page-{page.name.removesuffix(".html")}">\n'
        main += f"{body.rstrip()}\n</main>"
    title = SITE_NAME if page.name == "index.html" else page.title + SEPARATOR + SITE_NAME
    values = {
        "title": html.escape(title),
        "description": html.escape(page.description),
        "nav": main_nav(page),
        "main": main,
    }

    def fill(match: re.Match) -> str:
        key = match[1]
        if key not in values:
            raise ValueError(f"layout.html: unknown placeholder {{{{ {key} }}}}")
        return values[key]

    return PLACEHOLDER.sub(fill, layout)


def build(out: Path = OUT) -> list[Path]:
    """Write the site into *out*; returns the pages written.

    The default folder is replaced; any other must be new or empty, so nothing else is lost.
    ValueError for a page problem or a broken internal link.
    """
    out = Path(out)
    if out.exists():
        if out.resolve() == OUT.resolve():
            shutil.rmtree(out)
        elif any(out.iterdir()):
            raise ValueError(f"{out} is not empty")
    out.mkdir(parents=True, exist_ok=True)
    layout = LAYOUT.read_text(encoding="utf-8")
    written = []
    for page in read_pages().values():
        path = out / page.name
        path.write_text(render(page, layout), encoding="utf-8", newline="\n")
        written.append(path)
    for name in STATIC:
        shutil.copy2(HERE / name, out / name)
    shutil.copytree(IMAGES, out / "images")
    found = problems(out)
    if found:
        raise ValueError("broken site:\n  " + "\n  ".join(found))
    return written


# ---------------------------------------------------------------------- checks
class _PageRefs(HTMLParser):
    """The ids, the local references (href, src, srcset) and the images without alt text."""

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.ids: set[str] = set()
        self.refs: list[str] = []
        self.no_alt: list[str] = []

    def handle_starttag(self, tag, attrs):
        values = dict(attrs)
        if values.get("id"):
            self.ids.add(values["id"])
        for key in ("href", "src"):
            if values.get(key):
                self.refs.append(values[key])
        if values.get("srcset"):
            self.refs += [part.split()[0] for part in values["srcset"].split(",") if part.strip()]
        if tag == "img" and values.get("alt") is None:
            self.no_alt.append(values.get("src", "?"))


def _is_external(url: str) -> bool:
    parts = urlsplit(url)
    return bool(parts.scheme or parts.netloc)


def problems(out: Path) -> list[str]:
    """Broken internal links, images and anchors, images without alt text, absolute local
    paths (they would break under the project page's /mag-opt-detective/ prefix) and
    mentions of the measurement data in the built site at *out*."""
    out = Path(out)
    pages = {path: _PageRefs() for path in sorted(out.rglob("*.html"))}
    for path, refs in pages.items():
        refs.feed(path.read_text(encoding="utf-8"))
    ids = {path.resolve(): refs.ids for path, refs in pages.items()}
    found = []
    for path, refs in pages.items():
        name = path.relative_to(out).as_posix()
        found += [f"{name}: image without alt text: {src}" for src in refs.no_alt]
        for url in refs.refs:
            if _is_external(url) or url.startswith(("mailto:", "data:")):
                continue
            parts = urlsplit(url)
            if parts.path.startswith("/"):
                found.append(f"{name}: absolute link {url}")
                continue
            target = (path.parent / unquote(parts.path)).resolve() if parts.path else path.resolve()
            anchors = ids.get(target, set())
            if not target.is_file():
                found.append(f"{name}: broken link {url}")
            elif parts.fragment and target.suffix == ".html" and parts.fragment not in anchors:
                found.append(f"{name}: no anchor #{parts.fragment} in {url}")
    for path in sorted(out.rglob("*")):
        text = path.read_text(encoding="utf-8") if path.suffix in (".html", ".css", ".svg") else ""
        if MEASUREMENT_DATA in text.lower():
            found.append(f"{path.relative_to(out).as_posix()}: mentions the measurement data")
    return found


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--out", type=Path, default=OUT, help="output folder (default: %(default)s)"
    )
    args = parser.parse_args(argv)
    try:
        written = build(args.out)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print(f"Built {len(written)} pages into {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
