"""The docs site (docs/site): it builds, and its pages, links and images hold together."""

import importlib.util
import re
from html.parser import HTMLParser
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SITE = ROOT / "docs" / "site"
README_IMAGES = ROOT / "docs" / "images"
RELEASES = "https://github.com/wyzula-jan/mag-opt-detective/releases/latest/download/"
MAX_IMAGE_BYTES = 200_000  # lossless WebP of a 1400 x 900 window: about 140 kB
MAX_IMAGES = 24  # 10 scenes, each light and dark, and room for two more
DARK = "(prefers-color-scheme: dark)"


def load_build():
    return load_script("site_build", SITE / "build.py")


def load_script(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def build():
    return load_build()


@pytest.fixture(scope="module")
def site(build, tmp_path_factory):
    """The site built into a temporary folder."""
    out = tmp_path_factory.mktemp("site") / "out"
    build.build(out)
    return out


class Outline(HTMLParser):
    """The parts of a page the tests look at."""

    def __init__(self):
        super().__init__()
        self.h1 = 0
        self.title = ""
        self.images: list[dict] = []
        self.current: list[str] = []  # aria-current values in the header's navigation
        self._in = {"title": False, "header": False}

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if tag in self._in:
            self._in[tag] = True
        if tag == "h1":
            self.h1 += 1
        if tag == "img":
            self.images.append(attrs)
        if tag == "a" and self._in["header"] and "aria-current" in attrs:
            self.current.append(attrs["aria-current"])

    def handle_endtag(self, tag):
        if tag in self._in:
            self._in[tag] = False

    def handle_data(self, data):
        if self._in["title"]:
            self.title += data


def outline(path: Path) -> Outline:
    parser = Outline()
    parser.feed(path.read_text(encoding="utf-8"))
    return parser


class Pictures(HTMLParser):
    """Each image of a page with the sources of its <picture> (none outside one)."""

    def __init__(self):
        super().__init__()
        self.images: list[tuple[dict, list[dict]]] = []
        self._sources: list[dict] | None = None

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if tag == "picture":
            self._sources = []
        elif tag == "source" and self._sources is not None:
            self._sources.append(attrs)
        elif tag == "img":
            self.images.append((attrs, self._sources or []))

    def handle_endtag(self, tag):
        if tag == "picture":
            self._sources = None


def images_in(folder: Path) -> set[str]:
    """The names of the files in *folder*, without hidden ones such as .DS_Store."""
    return {path.name for path in folder.iterdir() if not path.name.startswith(".")}


def webp_size(path: Path) -> tuple[int, int]:
    """Width and height of the lossless WebP file at *path* (AssertionError for another)."""
    head = path.read_bytes()[:25]
    assert head[:4] == b"RIFF" and head[8:16] == b"WEBPVP8L", f"{path.name}: no lossless WebP"
    assert head[20] == 0x2F, path.name  # the VP8L signature
    bits = int.from_bytes(head[21:25], "little")
    return (bits & 0x3FFF) + 1, (bits >> 14 & 0x3FFF) + 1


def themed_images(text: str, root: Path, where: str) -> set[str]:
    """The file names of the screenshots in the HTML *text*, whose paths are relative to
    *root*, after checking each: the light image in a <picture> whose only source is its dark
    twin for a dark appearance, both of the same size (and of the size the <img> gives)."""
    parser = Pictures()
    parser.feed(text)
    names = set()
    for img, sources in parser.images:
        light = img.get("src", "")
        assert light.endswith("-light.webp"), f"{where}: {light} is not a light screenshot"
        dark = light.removesuffix("-light.webp") + "-dark.webp"
        found = [(source.get("srcset"), source.get("media")) for source in sources]
        assert found == [(dark, DARK)], f"{where}: {light} has no dark twin: {found}"
        size = webp_size(root / light)
        assert webp_size(root / dark) == size, f"{where}: {light} and {dark} differ in size"
        if img.get("width"):
            assert (int(img["width"]), int(img["height"])) == size, f"{where}: {light}"
        names |= {Path(light).name, Path(dark).name}
    return names


def test_every_page_is_built_with_a_title_and_one_heading(build, site):
    names = {name for name, _label in build.MAIN_NAV} | set(build.DOC_NAMES)
    assert {path.name for path in SITE.joinpath("pages").glob("*.html")} == names
    for name in names:
        page = outline(site / name)
        assert page.h1 == 1, name
        assert page.title.strip().endswith("Magneto-Optical Detective"), name
        assert page.current, f"{name}: no current item in the header navigation"
    assert (site / "site.css").is_file() and (site / "favicon.svg").is_file()


def test_internal_links_images_and_anchors_resolve(build, site):
    assert build.problems(site) == []


def test_the_link_check_finds_broken_links(build, tmp_path):
    (tmp_path / "images").mkdir()
    (tmp_path / "images" / "ok.png").write_bytes(b"")
    (tmp_path / "other.html").write_text('<h2 id="there">x</h2>', encoding="utf-8")
    (tmp_path / "page.html").write_text(
        '<a href="other.html#there">fine</a> <a href="https://example.org">external</a>'
        '<img src="images/ok.png" alt="fine">'
        '<a href="missing.html">1</a> <a href="other.html#nowhere">2</a> <a href="#gone">3</a>'
        '<img src="images/missing.png" alt="4"> <img src="images/ok.png">'
        '<source srcset="images/dark.png"> <a href="/root.html">7</a> Data_to_test',
        encoding="utf-8",
    )
    found = build.problems(tmp_path)
    assert len(found) == 8, found
    assert all(problem.startswith("page.html: ") for problem in found)


def test_no_page_mentions_the_measurement_data(site):
    for folder in (SITE, site):
        for path in folder.rglob("*"):
            if (
                path.suffix in (".html", ".css", ".svg", ".md", ".py")
                and "_build" not in path.parts
            ):
                text = path.read_text(encoding="utf-8").lower()
                # the build script names it only in its own check
                allowed = text.count('"data_to_test"') if path.name == "build.py" else 0
                assert text.count("data_to_test") == allowed, path


def test_images_are_used_have_alt_text_and_stay_small(site):
    pages = list(site.glob("*.html"))
    text = "\n".join(path.read_text(encoding="utf-8") for path in pages)
    images = [SITE / "images" / name for name in sorted(images_in(SITE / "images"))]
    assert 0 < len(images) <= MAX_IMAGES
    for image in images:
        assert image.suffix == ".webp", image.name
        assert image.stat().st_size <= MAX_IMAGE_BYTES, image.name
        assert f"images/{image.name}" in text, f"{image.name} is not used"
    for path in pages:
        for img in outline(path).images:
            assert img.get("alt", "").strip(), f"{path.name}: {img.get('src')} has no alt text"
            assert img.get("width") and img.get("height"), f"{path.name}: {img.get('src')}"


def test_every_screenshot_follows_the_appearance(site):
    shown = set()
    for path in sorted(site.glob("*.html")):
        shown |= themed_images(path.read_text(encoding="utf-8"), site, path.name)
    assert shown == images_in(SITE / "images")


def test_the_screenshot_check_needs_a_dark_twin_of_the_same_size(tmp_path):
    def webp(name: str, size: tuple[int, int], chunk: bytes = b"VP8L") -> None:
        bits = (size[0] - 1) | (size[1] - 1) << 14
        head = b"RIFF\0\0\0\0WEBP" + chunk + b"\0\0\0\0\x2f" + bits.to_bytes(4, "little")
        (tmp_path / name).write_bytes(head)

    webp("a-light.webp", (1400, 900))
    webp("a-dark.webp", (1400, 900))
    webp("b-light.webp", (1400, 900))
    webp("b-dark.webp", (1400, 899))
    webp("c-light.webp", (1400, 900), chunk=b"VP8 ")  # lossy
    webp("c-dark.webp", (1400, 900), chunk=b"VP8 ")
    pair = '<picture><source srcset="{}-dark.webp" media="{}"><img src="{}-light.webp"></picture>'
    found = themed_images(pair.format("a", DARK, "a"), tmp_path, "")
    assert found == {"a-light.webp", "a-dark.webp"}
    assert webp_size(tmp_path / "a-light.webp") == (1400, 900)
    for text in (
        '<img src="a-light.webp">',
        '<img src="a-dark.webp">',
        pair.format("a", "(prefers-color-scheme: light)", "a"),
        pair.format("b", DARK, "a"),
        pair.format("b", DARK, "b"),  # not the same size
        pair.format("c", DARK, "c"),  # not lossless
    ):
        with pytest.raises(AssertionError):
            themed_images(text, tmp_path, "")


def test_the_readme_screenshots_follow_the_appearance():
    text = (ROOT / "README.md").read_text(encoding="utf-8")
    assert not re.search(r"!\[[^\]]*\]\(docs/", text), "a screenshot without its dark twin"
    shown = themed_images(text, ROOT, "README.md")
    assert shown == images_in(README_IMAGES)
    for name in shown:  # copies of the site's, so git keeps each picture once
        site = (SITE / "images" / name).read_bytes()
        assert (README_IMAGES / name).read_bytes() == site, name


def test_the_screenshot_script_draws_every_image():
    shots = load_script("screenshot_list", ROOT / "docs" / "screenshot_list.py")
    assert set(shots.README) <= set(shots.SITE)
    assert set(shots.file_names(shots.SITE)) == images_in(SITE / "images")
    assert set(shots.file_names(shots.README)) == images_in(README_IMAGES)


def test_the_downloads_are_the_assets_the_bundles_workflow_publishes(site):
    workflow = (ROOT / ".github" / "workflows" / "bundles.yml").read_text(encoding="utf-8")
    assets = re.findall(r"archive:\s*(\S+)", workflow)
    assert len(assets) == 3
    for name in ("download.html", "index.html"):
        text = (site / name).read_text(encoding="utf-8")
        for asset in assets:
            assert RELEASES + asset in text, f"{name}: no link to {asset}"
        linked = set(re.findall(re.escape(RELEASES) + r'([^"]+)"', text))
        assert linked == set(assets), name


def test_the_pages_workflow_runs_only_by_hand():
    yaml = pytest.importorskip("yaml")
    path = ROOT / ".github" / "workflows" / "pages.yml"
    workflow = yaml.safe_load(path.read_text(encoding="utf-8"))
    triggers = workflow.get("on", workflow.get(True))  # YAML 1.1 reads a bare "on" as True
    assert set(triggers) == {"workflow_dispatch"}
    steps = [step.get("run", "") for job in workflow["jobs"].values() for step in job["steps"]]
    assert any("docs/site/build.py" in run for run in steps)


# ---------------------------------------------------------------------- release notes
CHANGELOG = """\
# Changelog

Intro text that the page does not show.

## 0.2.0 - 2026-10-05

### Added

- **gui:** add a ruler
- **core:** read `.dpt` files <b>now</b>
  in two lines

### Changed

- **io:** write units. **Breaking:** tables name the unit.

## 0.1.0 - 2026-10-03

The first release.
It plots sweeps.

### Known limitations

- Not signed.
"""


def test_the_release_notes_are_built_from_the_changelog(build, tmp_path):
    changelog = tmp_path / "CHANGELOG.md"
    changelog.write_text(CHANGELOG, encoding="utf-8")
    out = tmp_path / "out"
    build.build(out, changelog=changelog)
    assert build.problems(out) == []
    text = (out / "release-notes.html").read_text(encoding="utf-8")
    assert "No release yet" not in text and "Intro text" not in text
    assert text.index('id="v0.2.0"') < text.index('id="v0.1.0"')  # newest first
    assert '<h2 id="v0.2.0">Version 0.2.0</h2>' in text
    assert '<time datetime="2026-10-05">5 October 2026</time>' in text
    assert text.count('<span class="badge">Pre-release</span>') == 2
    assert "<h3>Added</h3>" in text and "<h3>Known limitations</h3>" in text
    assert (
        "<li><strong>core:</strong> read <code>.dpt</code> files &lt;b&gt;now&lt;/b&gt; "
        "in two lines</li>"
    ) in text
    assert "<strong>Breaking:</strong> tables name the unit." in text
    assert "<p>The first release. It plots sweeps.</p>" in text
    assert outline(out / "release-notes.html").h1 == 1


def test_the_release_notes_before_the_first_release(build, tmp_path):
    header = "# Changelog\n\nNothing released.\n"
    assert build.changelog_html(header) == build.NO_RELEASE
    assert build.read_changelog(tmp_path / "missing.md") == build.NO_RELEASE
    assert "No release yet" in build.NO_RELEASE
    assert "Pre-release" not in build.changelog_html("## 1.0.0 - 2027-01-04\n\n- Public.\n")
    with pytest.raises(ValueError, match="not a release heading"):
        build.changelog_html(header + "\n## Unreleased\n\n- a\n")


def test_the_release_notes_page_shows_the_repository_changelog(build, site):
    text = (site / "release-notes.html").read_text(encoding="utf-8")
    changelog = (ROOT / "CHANGELOG.md").read_text(encoding="utf-8")
    versions = re.findall(r"^## (\S+) - ", changelog, re.MULTILINE)
    if versions:
        assert f'id="v{versions[0]}"' in text
    else:
        assert "No release yet" in text
    assert "<!-- changelog -->" not in text
