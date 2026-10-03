"""The docs site (docs/site): it builds, and its pages, links and images hold together."""

import importlib.util
import re
from html.parser import HTMLParser
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SITE = ROOT / "docs" / "site"
RELEASES = "https://github.com/wyzula-jan/mag-opt-detective/releases/latest/download/"
MAX_IMAGE_BYTES = 400_000
MAX_IMAGES = 15


def load_build():
    spec = importlib.util.spec_from_file_location("site_build", SITE / "build.py")
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
    images = sorted(SITE.joinpath("images").glob("*"))
    assert 0 < len(images) <= MAX_IMAGES
    for image in images:
        assert image.suffix == ".png", image.name
        assert image.stat().st_size <= MAX_IMAGE_BYTES, image.name
        assert f"images/{image.name}" in text, f"{image.name} is not used"
    for path in pages:
        for img in outline(path).images:
            assert img.get("alt", "").strip(), f"{path.name}: {img.get('src')} has no alt text"
            assert img.get("width") and img.get("height"), f"{path.name}: {img.get('src')}"


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
