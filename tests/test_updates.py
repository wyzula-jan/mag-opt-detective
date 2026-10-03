"""mag_opt_detective.updates: version numbers, GitHub's release feed and the release to offer.

No test makes a network request: the feed comes from a fake transport with a sample answer of
GitHub's list-releases API (``tests/data/github_releases.json``, written by hand from its
documentation), and conftest's guard fails any test that tries.
"""

import http.client
import io
import json
import socket
import ssl
import subprocess
import sys
import urllib.request
from email.message import Message
from pathlib import Path
from urllib.error import HTTPError, URLError

import pytest

from mag_opt_detective import updates
from mag_opt_detective.updates import (
    PRERELEASE,
    STABLE,
    UpdateCheckError,
    Version,
    default_channel,
    fetch_releases,
    newest,
    parse_releases,
    update_for,
)

ROOT = Path(__file__).resolve().parents[1]
SAMPLE = Path(__file__).parent / "data" / "github_releases.json"
REPO = "https://github.com/wyzula-jan/mag-opt-detective"
DOWNLOAD = f"{REPO}/releases/download"
OPEN_GITHUB = updates.open_github  # the real one: conftest's guard replaces it in every test


class FakeResponse(io.BytesIO):
    """What urlopen returns: a file-like answer that is also a context manager."""

    status = 200


class FakeTransport:
    """An opener for :func:`fetch_releases` that records its requests."""

    def __init__(self, body: bytes = b"", error: Exception | None = None):
        self.body, self.error = body, error
        self.requests: list[tuple[urllib.request.Request, float]] = []

    def __call__(self, request, timeout):
        self.requests.append((request, timeout))
        if self.error is not None:
            raise self.error
        return FakeResponse(self.body)


def sample() -> list:
    return json.loads(SAMPLE.read_text(encoding="utf-8"))


def item(tag: str, prerelease: bool = False, draft: bool = False, **fields) -> dict:
    """One release as GitHub lists it (the fields the app reads)."""
    return {
        "tag_name": tag,
        "html_url": f"https://github.com/wyzula-jan/mag-opt-detective/releases/tag/{tag}",
        "draft": draft,
        "prerelease": prerelease,
        "assets": [],
        **fields,
    }


# --- version numbers -----------------------------------------------------------------------
@pytest.mark.parametrize(
    ("text", "parsed"),
    [
        ("v0.2.0", Version(0, 2, 0)),
        ("0.2.0", Version(0, 2, 0)),
        ("v10.20.30", Version(10, 20, 30)),
        ("v1.2.0-rc.1", Version(1, 2, 0, ("rc", 1))),
        ("1.0.0-alpha.beta", Version(1, 0, 0, ("alpha", "beta"))),
        ("v1.0.0+20261003", Version(1, 0, 0)),  # build metadata is ignored
        ("0.3.0.dev0", Version(0, 3, 0, dev=0)),  # PEP 440: a version in development
        ("0.3.0.dev", Version(0, 3, 0, dev=0)),
        ("0.3.0rc2", Version(0, 3, 0, ("rc", 2))),
        ("1.0.0rc1.dev0", Version(1, 0, 0, ("rc", 1), dev=0)),
        ("0.3.0.post1", Version(0, 3, 0)),  # a post-release counts as its release
        ("0.2.0+g1a2b3c4", Version(0, 2, 0)),  # a local (git) build of 0.2.0
    ],
)
def test_versions_are_parsed(text, parsed):
    assert Version.parse(text) == parsed


@pytest.mark.parametrize(
    "text", ["", "v1.2", "1.2.3.4", "01.2.3", "nightly", "v1.2.3-", "1.\u0662.3", None, 3]
)
def test_other_text_is_no_version(text):
    with pytest.raises(ValueError):
        Version.parse(text)


def test_versions_are_ordered_as_semantic_versioning_orders_them():
    assert Version.parse("0.10.0") > Version.parse("0.9.1")
    assert Version.parse("1.0.0") > Version.parse("0.99.99")
    assert Version.parse("0.3.0-rc.1") < Version.parse("0.3.0")  # a pre-release before
    chain = [  # the example from semver.org
        "1.0.0-alpha",
        "1.0.0-alpha.1",
        "1.0.0-alpha.beta",
        "1.0.0-beta",
        "1.0.0-beta.2",
        "1.0.0-beta.11",
        "1.0.0-rc.1",
        "1.0.0",
    ]
    versions = [Version.parse(text) for text in chain]
    assert sorted(reversed(versions)) == versions
    assert Version.parse("v1.0.0+build.7") == Version.parse("1.0.0")


def test_a_version_in_development_lies_between_releases():
    dev = Version.parse("0.3.0.dev0")
    assert Version.parse("0.2.0") < dev < Version.parse("0.3.0")
    assert dev.is_prerelease and not Version.parse("0.3.0").is_prerelease
    chain = ["1.0.0.dev0", "1.0.0a1", "1.0.0rc1.dev0", "1.0.0rc1", "1.0.0"]  # as PEP 440
    versions = [Version.parse(text) for text in chain]
    assert sorted(reversed(versions)) == versions
    assert Version.parse("1.0.0rc1") == Version.parse("v1.0.0-rc.1")
    assert Version.parse("0.9.9") < Version.parse("1.0.0.dev0")


def test_versions_print_as_semantic_versions():
    assert str(Version.parse("v0.3.0")) == "0.3.0"
    assert str(Version.parse("v1.2.0-rc.1")) == "1.2.0-rc.1"
    assert str(Version.parse("1.2.0rc1.dev3")) == "1.2.0-rc.1.dev3"


def test_pre_releases_count_before_1_0_0_only():
    assert default_channel(Version.parse("0.2.0")) == PRERELEASE
    assert default_channel(Version.parse("0.99.0.dev0")) == PRERELEASE
    assert default_channel(Version.parse("1.0.0")) == STABLE
    assert default_channel(Version.parse("2.1.0-rc.1")) == STABLE


# --- the release feed ----------------------------------------------------------------------
def test_the_sample_answer_is_read_without_drafts():
    releases = parse_releases(sample())
    assert [str(r.version) for r in releases] == ["0.3.0", "0.2.0", "0.1.0"]  # 0.4.0: a draft
    first = releases[0]
    assert first.tag == "v0.3.0"
    assert first.page == "https://github.com/wyzula-jan/mag-opt-detective/releases/tag/v0.3.0"
    assert first.prerelease  # every 0.x release is a pre-release on GitHub
    assert sorted(first.assets) == [
        "mag-opt-detective-linux.tar.gz",
        "mag-opt-detective-macos.zip",
        "mag-opt-detective-windows.zip",
    ]


def test_the_download_is_the_archive_for_the_system():
    release = parse_releases(sample())[0]
    assert release.download_url("darwin") == f"{DOWNLOAD}/v0.3.0/mag-opt-detective-macos.zip"
    assert release.download_url("win32") == f"{DOWNLOAD}/v0.3.0/mag-opt-detective-windows.zip"
    assert release.download_url("linux") == f"{DOWNLOAD}/v0.3.0/mag-opt-detective-linux.tar.gz"
    assert release.download_url("freebsd") == release.page  # no app for it: the page
    bare = parse_releases([item("v0.3.1")])[0]
    assert bare.download_url("darwin") == bare.page  # no files attached (yet)


def test_the_archive_names_are_those_the_bundles_workflow_attaches():
    workflow = (ROOT / ".github" / "workflows" / "bundles.yml").read_text(encoding="utf-8")
    for name in updates.ASSETS.values():
        assert f"archive: {name}" in workflow


def test_odd_releases_are_left_out_or_made_safe():
    foreign = "https://example.com/mag-opt-detective.zip"
    payload = [
        "not a release",
        item("nightly"),  # no version number
        item("v0.9.0", draft=True),
        {**item("v0.8.0"), "draft": None},  # not known to be published
        item(
            "v0.7.0",
            html_url="https://example.com/releases/v0.7.0",
            assets=[
                {"name": "mag-opt-detective-macos.zip", "browser_download_url": foreign},
                {
                    "name": "mag-opt-detective-linux.tar.gz",
                    "browser_download_url": f"{DOWNLOAD}/v0.7.0/mag-opt-detective-linux.tar.gz",
                    "state": "starter",  # an upload that did not finish
                },
                "not an asset",
            ],
        ),
        {**item("v0.6.0"), "prerelease": None},
    ]
    releases = parse_releases(payload)
    assert [r.tag for r in releases] == ["v0.7.0", "v0.6.0"]
    seven, six = releases
    assert seven.page == "https://github.com/wyzula-jan/mag-opt-detective/releases/tag/v0.7.0"
    assert seven.assets == {}  # nothing outside this repository's releases
    assert six.prerelease  # not known to be stable
    with pytest.raises(UpdateCheckError):
        parse_releases({"message": "Not Found"})


@pytest.mark.parametrize(
    "link",
    [
        f"{DOWNLOAD}/../../../../attacker/repo/releases/download/v0.7.0/app.zip",
        f"{DOWNLOAD}/%2e%2e/%2e%2e/%2e%2e/%2e%2e/attacker/repo/app.zip",
        f"{DOWNLOAD}/v0.7.0\\..\\..\\..\\..\\attacker\\app.zip",
        f"{DOWNLOAD}/v0.7.0/./app.zip",
        f"{DOWNLOAD}//example.com/app.zip",
        f"{DOWNLOAD}/v0.7.0/app.zip?next=https://example.com",
        f"{DOWNLOAD}/v0.7.0/app.zip#top",
        f"{DOWNLOAD}/v0.7.0/app zip",
        "http://github.com/wyzula-jan/mag-opt-detective/releases/download/v0.7.0/app.zip",
        "https://github.com.example.com/wyzula-jan/mag-opt-detective/releases/download/v0.7.0/a",
        "https://user@github.com/wyzula-jan/mag-opt-detective/releases/download/v0.7.0/app.zip",
        "https://github.com:8443/wyzula-jan/mag-opt-detective/releases/download/v0.7.0/app.zip",
        "https://github.com/wyzula-jan/other-repo/releases/download/v0.7.0/app.zip",
        "https://github.com/wyzula-jan/mag-opt-detective/releases-download/v0.7.0/app.zip",
    ],
)
def test_links_that_leave_the_releases_are_refused(link):
    page = link.replace("/download/", "/tag/")
    asset = {"name": "mag-opt-detective-macos.zip", "browser_download_url": link}
    [release] = parse_releases([item("v0.7.0", html_url=page, assets=[asset])])
    assert release.assets == {}  # Download opens the release page instead
    assert release.page == f"{REPO}/releases/tag/v0.7.0"


def test_a_tag_is_used_without_its_spaces_and_odd_assets_are_skipped():
    payload = [
        item(" v0.7.0 ", html_url=None, assets=5),
        item("v0.6.0", assets="mag-opt-detective-macos.zip"),
        item("v0.5.0", assets={"name": "mag-opt-detective-macos.zip"}),
    ]
    seven, six, five = parse_releases(payload)
    assert seven.tag == "v0.7.0" and seven.page == f"{REPO}/releases/tag/v0.7.0"
    assert seven.assets == six.assets == five.assets == {}


def test_newer_equal_and_older_versions():
    releases = parse_releases(sample())
    assert update_for(releases, Version.parse("0.2.0"), PRERELEASE).tag == "v0.3.0"
    assert update_for(releases, Version.parse("0.3.0"), PRERELEASE) is None
    assert update_for(releases, Version.parse("0.4.0"), PRERELEASE) is None  # the draft
    assert update_for(releases, Version.parse("0.3.0.dev0"), PRERELEASE).tag == "v0.3.0"
    assert update_for(releases, Version.parse("0.3.1.dev0"), PRERELEASE) is None
    assert update_for([], Version.parse("0.2.0"), PRERELEASE) is None


def test_the_stable_channel_leaves_pre_releases_out():
    releases = parse_releases(
        [item("v1.1.0-rc.1", prerelease=True), item("v1.0.1"), item("v1.0.0"), item("v1.2.0b1")]
    )
    assert newest(releases, STABLE).tag == "v1.0.1"
    assert newest(releases, PRERELEASE).tag == "v1.2.0b1"  # a pre-release version, also untagged
    assert update_for(releases, Version.parse("1.0.0"), STABLE).tag == "v1.0.1"
    assert update_for(releases, Version.parse("1.0.1"), STABLE) is None
    # 0.x releases are all pre-releases on GitHub: the stable channel finds none
    assert newest(parse_releases(sample()), STABLE) is None


# --- the request ---------------------------------------------------------------------------
def test_the_request_is_one_anonymous_get():
    transport = FakeTransport(SAMPLE.read_bytes())
    releases = fetch_releases("0.2.0", opener=transport)
    assert [r.tag for r in releases] == ["v0.3.0", "v0.2.0", "v0.1.0"]
    [(request, timeout)] = transport.requests
    assert request.full_url == updates.RELEASES_API
    assert request.full_url.startswith("https://api.github.com/repos/wyzula-jan/mag-opt-detective/")
    assert request.get_method() == "GET" and request.data is None
    assert dict(request.header_items()) == {
        "User-agent": "mag-opt-detective/0.2.0",
        "Accept": "application/vnd.github+json",
    }  # no account, no cookies
    assert timeout == updates.TIMEOUT == 5


@pytest.mark.parametrize(
    ("error", "message"),
    [
        (URLError(socket.gaierror(8, "nodename nor servname provided")), "no connection"),
        (URLError(ConnectionRefusedError(61, "Connection refused")), "no connection"),
        (http.client.RemoteDisconnected("closed"), "no connection"),
        (URLError(TimeoutError("timed out")), "did not answer within 5 s"),
        (TimeoutError("The read operation timed out"), "did not answer within 5 s"),
        (URLError(ssl.SSLCertVerificationError(1, "certificate verify failed")), "secure"),
        (HTTPError(updates.RELEASES_API, 403, "rate limit exceeded", {}, None), "how often"),
        (HTTPError(updates.RELEASES_API, 429, "Too Many Requests", {}, None), "how often"),
        (HTTPError(updates.RELEASES_API, 502, "Bad Gateway", {}, None), "HTTP 502"),
        (http.client.IncompleteRead(b"[{"), "cannot read"),
    ],
)
def test_failures_raise_a_short_reason(error, message):
    with pytest.raises(UpdateCheckError, match=message) as info:
        fetch_releases("0.2.0", opener=FakeTransport(error=error))
    assert info.value.__cause__ is error  # the details stay for the log


@pytest.mark.parametrize(
    "body", [b"<html>Bad gateway</html>", b'{"message": "Not Found"}', b"[" * 200_000]
)
def test_an_answer_that_is_no_release_list_fails(body):
    with pytest.raises(UpdateCheckError, match="cannot read"):
        fetch_releases("0.2.0", opener=FakeTransport(body))


def test_a_huge_answer_is_not_read_to_the_end(monkeypatch):
    monkeypatch.setattr(updates, "MAX_BYTES", 100)
    with pytest.raises(UpdateCheckError, match="cannot read"):
        fetch_releases("0.2.0", opener=FakeTransport(b"[" + b" " * 200 + b"]"))


def test_redirects_stay_within_githubs_api():
    handler, request = updates.ApiRedirects(), updates.releases_request("0.2.0")
    moved = "https://api.github.com/repositories/1234/releases?per_page=10"  # a renamed repo
    new = handler.redirect_request(request, io.BytesIO(), 301, "Moved", Message(), moved)
    assert new.full_url == moved and new.get_header("User-agent") == "mag-opt-detective/0.2.0"
    for elsewhere in (
        "https://example.com/releases",
        "http://api.github.com/repos/wyzula-jan/mag-opt-detective/releases",
        "https://github.com/wyzula-jan/mag-opt-detective/releases",
    ):
        answer = io.BytesIO()
        with pytest.raises(UpdateCheckError, match="another address"):
            handler.redirect_request(request, answer, 302, "Found", Message(), elsewhere)
        assert answer.closed


def test_the_request_goes_through_urllib_without_cookies(monkeypatch):
    sent = []

    def offline(director, request, timeout=None):
        sent.append(([type(h) for h in director.handlers], request, timeout))
        raise URLError(OSError(51, "Network is unreachable"))

    monkeypatch.setattr(urllib.request.OpenerDirector, "open", offline)
    with pytest.raises(UpdateCheckError, match="no connection"):
        fetch_releases("0.2.0", opener=OPEN_GITHUB)
    [(handlers, request, timeout)] = sent
    assert updates.ApiRedirects in handlers
    assert urllib.request.HTTPRedirectHandler not in handlers
    assert urllib.request.HTTPCookieProcessor not in handlers
    assert request.full_url == updates.RELEASES_API and timeout == 5


def test_no_test_reaches_the_network(network_guard):
    """conftest's guard: the update check, urllib and http.client all refuse, and a test
    that tried fails."""
    with pytest.raises(AssertionError, match="network"):
        fetch_releases("0.2.0")
    with pytest.raises(AssertionError, match="network"):
        urllib.request.urlopen("https://api.github.com/", timeout=1)
    with pytest.raises(AssertionError, match="network"):
        http.client.HTTPSConnection("api.github.com", timeout=1).request("GET", "/")
    assert network_guard.attempts == ["the update check", "urllib", "http.client"]
    with pytest.raises(pytest.fail.Exception, match=r"the update check, urllib, http\.client"):
        network_guard.check()
    network_guard.attempts.clear()  # tried on purpose


def test_the_module_does_not_import_qt():
    code = (
        "import sys, mag_opt_detective.updates; "
        "sys.exit(any(m.startswith('PySide6') for m in sys.modules))"
    )
    assert subprocess.run([sys.executable, "-c", code], timeout=60).returncode == 0
