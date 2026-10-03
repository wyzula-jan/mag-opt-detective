"""Help menu: the documentation, the feature request and the bug report (gui/links.py), and
the GitHub issue forms they open (.github/ISSUE_TEMPLATE)."""

import getpass
import logging
import platform
import re
import sys
from pathlib import Path
from urllib.parse import parse_qsl, urlsplit

import numpy as np
import pyqtgraph as pg
import pytest
import scipy
from PySide6 import __version__ as pyside_version
from PySide6.QtCore import Qt, QUrl, qVersion
from PySide6.QtGui import QAccessible, QDesktopServices, QKeySequence
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication

import gui_helpers
from mag_opt_detective import __version__
from mag_opt_detective.gui import licences, links
from mag_opt_detective.gui.main_window import SHORTCUTS

window, errors = gui_helpers.window, gui_helpers.errors  # shared fixtures

ROOT = Path(__file__).resolve().parents[1]
FORMS = ROOT / ".github" / "ISSUE_TEMPLATE"
FEATURE_URL = (
    "https://github.com/wyzula-jan/mag-opt-detective/issues/new?template=feature_request.yml"
)
LINKS = ("documentation", "request_feature", "report_bug")


@pytest.fixture
def opened(monkeypatch):
    """The addresses the app asks the web browser to open (none is really opened)."""
    urls: list[str] = []

    def open_url(url: QUrl) -> bool:
        urls.append(url.toString(QUrl.ComponentFormattingOption.FullyEncoded))
        return True

    monkeypatch.setattr(QDesktopServices, "openUrl", open_url)
    return urls


def query(url: str) -> dict[str, str]:
    parts = urlsplit(url)
    assert f"{parts.scheme}://{parts.netloc}{parts.path}" == links.NEW_ISSUE
    return dict(parse_qsl(parts.query, strict_parsing=True))


# ---------------------------------------------------------------------- menu
def test_the_help_menu(window):
    menu = window.help_menu
    assert menu.title() == "&Help"
    assert [a.text() for a in menu.actions()] == [
        "&Documentation",
        "Request a &feature…",
        "Report a &bug…",
        "",
        "&Shortcuts",
        "",
        "&About",
    ]
    names = [window.commands[n] for n in (*LINKS, "shortcuts", "about")]
    assert [a for a in menu.actions() if not a.isSeparator()] == names
    assert all(a.statusTip() for a in names)
    assert all(not window.commands[n].icon().isNull() for n in LINKS)
    # the system's help keys first (the menu shows Cmd+? on macOS), and F1 everywhere
    keys = window.commands["documentation"].shortcuts()
    help_keys = QKeySequence.keyBindings(QKeySequence.StandardKey.HelpContents)
    assert keys[: len(help_keys)] == help_keys and QKeySequence("F1") in keys
    assert len(set(k.toString() for k in keys)) == len(keys)
    assert ("F1", "Open the documentation (on macOS also ⌘?)") in SHORTCUTS  # Help > Shortcuts
    menu.adjustSize()
    items = QAccessible.queryAccessibleInterface(menu)
    spoken = [items.child(i).text(QAccessible.Text.Name) for i in range(items.childCount())]
    assert [t for t in spoken if t] == [
        "Documentation",
        "Request a feature…",
        "Report a bug…",
        "Shortcuts",
        "About",
    ]


def test_the_links_open_in_the_web_browser(window, opened):
    window.commands["documentation"].trigger()
    window.commands["request_feature"].trigger()
    assert opened == [links.DOCS_URL, FEATURE_URL]
    assert links.DOCS_URL == "https://wyzula-jan.github.io/mag-opt-detective/"


def test_f1_opens_the_documentation(window, opened, qtbot):
    with qtbot.waitExposed(window):
        window.show()
    qtbot.waitUntil(window.isActiveWindow)  # shortcuts work in the active window
    window.plots.map.view.setFocus()
    QTest.keyClick(window.plots.map.view, Qt.Key.Key_F1)
    assert opened == [links.DOCS_URL]


def test_the_bug_report_is_filled_in_with_the_environment(window, opened, sweep):
    window.commands["report_bug"].trigger()
    fields = query(opened[-1])
    assert fields == {
        "template": "bug_report.yml",
        "version": __version__,
        "install": "From source",
        "os": links.system(),
        "python": platform.python_version(),
        "qt": f"Qt {qVersion()}, PySide6 {pyside_version}",
        "libraries": f"numpy {np.__version__}, scipy {scipy.__version__}, "
        f"pyqtgraph {pg.__version__}",
        "plot": "Nothing processed",
    }
    assert platform.machine() in fields["os"]

    gui_helpers.load_sweep(window, sweep)
    gui_helpers.process(window)
    gui_helpers.select(window, order=1)
    gui_helpers.set_unit(window, "meV")
    window.commands["report_bug"].trigger()
    assert query(opened[-1])["plot"] == "Map: R(B)/R(0) · 1st derivative d/dE per point, meV"
    window.plot_area.set_current_view("reference")
    assert window.plot_summary() == "Reference: R(B)/R(0), meV"


def test_the_bug_report_sends_no_personal_data(window, opened, sweep, monkeypatch):
    monkeypatch.setattr(links, "system", lambda: "Some OS 1.0 (x86_64)")  # its own tests below
    gui_helpers.load_sweep(window, sweep)  # files in a temporary folder of the user's
    gui_helpers.process(window)
    window.commands["report_bug"].trigger()
    url = opened[-1]
    assert len(url) < 2000  # GitHub and browsers take 8000 characters; stay well under
    fields = query(url)
    text = " ".join(fields.values())  # the values only: the address names "wyzula-jan"
    home = Path.home()
    private = {str(home), home.name, getpass.getuser(), platform.node()}  # user and host
    private |= {Path(f).name for f in sweep["field"]} | {Path(sweep["field"][0]).parent.name}
    for value in private:
        if len(value) >= 3:
            assert value not in text, value
    rest = " ".join(v for k, v in fields.items() if k != "os").replace("R(B)/R(0)", "")
    assert "/" not in rest and "\\" not in rest  # no paths at all


SYSTEMS = [  # sys.platform, the platform functions' results, the system named
    ("darwin", {"mac_ver": ("15.6", ("", "", ""), "arm64"), "machine": "arm64"},
     "macOS 15.6 (arm64)"),
    ("darwin", {"mac_ver": ("", ("", "", ""), ""), "release": "24.6.0", "machine": "arm64"},
     "Darwin 24.6.0 (arm64)"),
    ("win32", {"release": "11", "version": "10.0.26100", "machine": "AMD64"},
     "Windows 11 10.0.26100 (AMD64)"),
    ("linux", {"freedesktop_os_release": {"PRETTY_NAME": "Debian GNU/Linux 12 (bookworm)"},
               "release": "6.1.0-18-amd64", "machine": "x86_64"},
     "Debian GNU/Linux 12 (bookworm), kernel 6.1.0-18-amd64 (x86_64)"),
    ("linux", {"freedesktop_os_release": OSError, "system": "Linux", "release": "6.1.0",
               "machine": "aarch64"},
     "Linux, kernel 6.1.0 (aarch64)"),
]  # fmt: skip


@pytest.mark.parametrize(("name", "results", "expected"), SYSTEMS)
def test_the_system_name(monkeypatch, name, results, expected):
    monkeypatch.setattr(sys, "platform", name)
    for function, result in results.items():

        def fake(result=result):
            if result is OSError:
                raise OSError("no /etc/os-release")
            return result

        monkeypatch.setattr(platform, function, fake)
    assert links.system() == expected


def test_a_debian_system_name_reaches_the_form_whole(window, opened, monkeypatch):
    debian = "Debian GNU/Linux 12 (bookworm), kernel 6.1.0-18-amd64 (x86_64)"
    monkeypatch.setattr(links, "system", lambda: debian)
    window.commands["report_bug"].trigger()
    assert query(opened[-1])["os"] == debian
    assert "os=Debian%20GNU%2FLinux%2012%20%28bookworm%29" in opened[-1]


def test_the_about_box_shows_the_versions_of_the_bug_form(window, monkeypatch):
    texts = []

    class Box:
        def exec(self) -> int:
            return 0

    monkeypatch.setattr(licences, "about_box", lambda _parent, text: texts.append(text) or Box())
    window.commands["about"].trigger()
    v = links.versions()
    assert f"Python {v['python']}, {v['qt']}<br>{v['libraries']}<br>" in texts[0]
    assert {k: links.environment()[k] for k in v} == v


def test_a_standalone_app_says_so(monkeypatch):
    monkeypatch.setattr(sys, "frozen", True, raising=False)
    assert links.environment()["install"] == "Standalone app"
    assert "plot" not in links.environment()


def test_long_values_are_cut_and_unknown_fields_refused():
    url = links.bug_report_url({"os": "x" * 10_000, "plot": "·" * 10_000})
    assert len(url) < 8000
    assert query(url)["os"] == "x" * links.MAX_VALUE
    with pytest.raises(KeyError, match="home"):
        links.bug_report_url({"home": "/Users/someone"})


def test_no_web_browser_goes_to_the_info_bar(window, errors, monkeypatch, caplog):
    caplog.set_level(logging.INFO, logger="mag_opt_detective")
    monkeypatch.setattr(QDesktopServices, "openUrl", lambda _url: False)
    bar = window.infobar
    bug_url = links.bug_report_url(links.environment(window.plot_summary()))
    for name, url in (("documentation", links.DOCS_URL), ("report_bug", bug_url)):
        QApplication.clipboard().clear()
        window.commands[name].trigger()
        assert errors[-1] == links.NO_BROWSER
        assert f"Address: {url}" in caplog.messages  # the log keeps it too
        assert not bar.action_button.isHidden() and bar.action_button.text() == "Copy address"
        bar.action_button.click()
        assert QApplication.clipboard().text() == url and bar.isHidden()
    assert gui_helpers.infobar_text(window) == ""
    window.commands["request_feature"].trigger()
    assert gui_helpers.infobar_text(window) == (
        "Can't request a feature: No web browser opened the page: copy its address, or find "
        "it in the log"
    )


# ---------------------------------------------------------------------- issue forms
# GitHub's issue form syntax (docs: "Syntax for issue forms" and "for GitHub's form schema")
FORM_KEYS = {"name", "description", "title", "labels", "assignees", "projects", "type", "body"}
ELEMENT_KEYS = {"type", "id", "attributes", "validations"}
ATTRIBUTES = {  # type: (required, allowed)
    "markdown": ({"value"}, {"value"}),
    "textarea": ({"label"}, {"label", "description", "placeholder", "value", "render"}),
    "input": ({"label"}, {"label", "description", "placeholder", "value"}),
    "dropdown": ({"label", "options"}, {"label", "description", "multiple", "options", "default"}),
    "checkboxes": ({"label", "options"}, {"label", "description", "options"}),
}
# query parameters GitHub reads itself: a field id must not take one of them
RESERVED = {"title", "body", "labels", "assignees", "milestone", "projects", "template", "type"}
ID = re.compile(r"[A-Za-z0-9_-]+")


def form_problems(form) -> list[str]:
    """What breaks GitHub's issue form rules in *form* (a parsed YAML file)."""
    if not isinstance(form, dict):
        return ["not a mapping"]
    problems = [f"unknown key {k}" for k in set(form) - FORM_KEYS]
    for key in ("name", "description"):
        if not (isinstance(form.get(key), str) and form[key].strip()):
            problems.append(f"{key} missing")
    for key in ("labels", "assignees"):
        if key in form and not all(isinstance(v, str) for v in form[key]):
            problems.append(f"{key} not a list of names")
    body = form.get("body")
    if not (isinstance(body, list) and body):
        return [*problems, "body missing"]
    ids, labels = set(), set()
    for i, element in enumerate(body):
        where = f"body[{i}]"
        kind = element.get("type")
        if kind not in ATTRIBUTES:
            problems.append(f"{where}: unknown type {kind}")
            continue
        problems += [f"{where}: unknown key {k}" for k in set(element) - ELEMENT_KEYS]
        required, allowed = ATTRIBUTES[kind]
        attributes = element.get("attributes") or {}
        problems += [f"{where}: {k} missing" for k in required - set(attributes)]
        problems += [f"{where}: unknown attribute {k}" for k in set(attributes) - allowed]
        if kind == "markdown":
            if "id" in element or "validations" in element:
                problems.append(f"{where}: markdown takes no id or validations")
            continue
        label = attributes.get("label")
        if not (isinstance(label, str) and label.strip()) or label in labels:
            problems.append(f"{where}: label empty or repeated")
        labels.add(label)
        if "id" in element:
            id_ = element["id"]
            if not (isinstance(id_, str) and ID.fullmatch(id_)) or id_ in ids | RESERVED:
                problems.append(f"{where}: id {id_!r} invalid, repeated or reserved")
            ids.add(id_)
        validations = element.get("validations", {})
        if set(validations) - {"required"} or not all(
            isinstance(v, bool) for v in validations.values()
        ):
            problems.append(f"{where}: validations")
        if kind == "checkboxes":
            for option in attributes.get("options") or []:
                if not isinstance(option.get("label"), str) or set(option) - {"label", "required"}:
                    problems.append(f"{where}: option {option}")
    if not labels:
        problems.append("no field to fill in")
    return problems


def load(name: str):
    yaml = pytest.importorskip("yaml")
    return yaml.safe_load((FORMS / name).read_text(encoding="utf-8"))


@pytest.mark.parametrize("name", [links.BUG_FORM, links.FEATURE_FORM])
def test_the_issue_forms_follow_github_rules(name):
    form = load(name)
    assert form_problems(form) == []


def test_the_rule_check_finds_mistakes():
    bad = {
        "name": "x",
        "body": [
            {"type": "markdown", "id": "intro", "attributes": {}},
            {"type": "input", "id": "os", "attributes": {"label": "OS", "size": 3}},
            {"type": "input", "id": "os", "attributes": {"label": "OS"}},
            {"type": "textarea", "id": "title", "attributes": {"label": "T"}},
            {"type": "slider", "attributes": {"label": "S"}},
        ],
    }
    assert form_problems(bad) == [
        "description missing",
        "body[0]: value missing",
        "body[0]: markdown takes no id or validations",
        "body[1]: unknown attribute size",
        "body[2]: label empty or repeated",
        "body[2]: id 'os' invalid, repeated or reserved",
        "body[3]: id 'title' invalid, repeated or reserved",
        "body[4]: unknown type slider",
    ]


def test_the_bug_form_has_a_field_for_each_query_parameter(window, opened):
    form = load(links.BUG_FORM)
    inputs = [e["id"] for e in form["body"] if e["type"] == "input"]
    assert tuple(inputs) == links.ENVIRONMENT_FIELDS
    window.commands["report_bug"].trigger()
    assert set(query(opened[-1])) - {"template"} <= set(inputs)
    texts = [e["id"] for e in form["body"] if e["type"] == "textarea"]
    assert {"description", "steps", "expected", "actual"} <= set(texts)
    (data,) = [e for e in form["body"] if e["type"] == "checkboxes"]
    assert all(option["required"] for option in data["attributes"]["options"])
    assert "bug" in form["labels"]


def test_the_feature_form_and_the_chooser():
    form = load(links.FEATURE_FORM)
    assert [e.get("id") for e in form["body"] if e["type"] != "markdown"] == [
        "problem",
        "solution",
        "alternatives",
    ]
    intro = next(e for e in form["body"] if e["type"] == "markdown")["attributes"]["value"]
    assert re.findall(r"\]\((https?://[^)]+)\)", intro) == [links.DOCS_URL]  # the docs link
    config = load("config.yml")
    assert set(config) <= {"blank_issues_enabled", "contact_links"}
    assert isinstance(config["blank_issues_enabled"], bool)
    for link in config["contact_links"]:
        assert set(link) == {"name", "url", "about"} and link["url"].startswith("https://")
    assert links.DOCS_URL in [link["url"] for link in config["contact_links"]]
    citation = (ROOT / "CITATION.cff").read_text(encoding="utf-8")
    assert f'url: "{links.DOCS_URL}"' in citation  # one address for the docs site
    assert sorted(p.name for p in FORMS.iterdir()) == sorted(
        [links.BUG_FORM, links.FEATURE_FORM, "config.yml"]
    )
