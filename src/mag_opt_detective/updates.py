"""New versions on GitHub: version numbers, the release feed and which release to offer (no Qt).

The app only says that a new version exists; it never installs one (the app bundles are not
code-signed, so people download the new one and replace the app). :func:`fetch_releases` is one
anonymous GET of GitHub's releases API: no account, no cookies, no data, a short timeout,
redirects only within the API, and Python's own proxy and certificate handling. Releases are
tagged ``vX.Y.Z`` by ``tools/release.py``; the *Release* workflow marks versions before 1.0.0
as GitHub pre-releases, so while the app is 0.x pre-releases count (:func:`default_channel`).
The app opens only links into this repository's releases on github.com.
"""

from __future__ import annotations

import http.client
import json
import posixpath
import re
import ssl
import sys
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from functools import total_ordering
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit
from urllib.request import HTTPRedirectHandler, Request, build_opener

REPOSITORY = "wyzula-jan/mag-opt-detective"
API_HOST, SITE_HOST = "api.github.com", "github.com"
RELEASES_API = f"https://{API_HOST}/repos/{REPOSITORY}/releases?per_page=10"
RELEASES_PAGE = f"https://{SITE_HOST}/{REPOSITORY}/releases"
RELEASES_PATH = f"/{REPOSITORY}/releases/"  # the pages the app opens: releases and their files
DOWNLOADS_PATH = f"{RELEASES_PATH}download/"  # where the files attached to a release are
TIMEOUT = 5.0  # seconds
MAX_BYTES = 4_000_000  # ten releases with their notes are far less
STABLE, PRERELEASE = "stable", "pre-release"
CHANNELS = (STABLE, PRERELEASE)
# the archives the App bundles workflow attaches to every release, by sys.platform
ASSETS = {
    "darwin": "mag-opt-detective-macos.zip",
    "win32": "mag-opt-detective-windows.zip",
    "linux": "mag-opt-detective-linux.tar.gz",
}

_NUMBER = r"(0|[1-9]\d*)"
_VERSION = re.compile(
    rf"v?{_NUMBER}\.{_NUMBER}\.{_NUMBER}"
    r"(?:-(?P<pre>[0-9A-Za-z-]+(?:\.[0-9A-Za-z-]+)*)"  # SemVer: 1.2.0-rc.1
    r"|(?P<pep>a|b|rc)(?P<pep_n>\d+))?"  # PEP 440: 1.2.0rc1
    r"(?P<post>\.post\d+)?"  # PEP 440: a post-release, counted as its release
    r"(?:\.dev(?P<dev>\d*))?"  # PEP 440: 1.2.0.dev0, in development before 1.2.0
    r"(?:\+[0-9A-Za-z.-]+)?",  # build metadata (a local build): ignored
    re.ASCII,
)


@total_ordering
@dataclass(frozen=True)
class Version:
    """A version number ordered as Semantic Versioning orders it: 0.10.0 > 0.9.1, and a
    pre-release before its release (1.2.0-rc.1 < 1.2.0).

    :meth:`parse` also reads the PEP 440 forms of a version in development, ordered as PEP 440
    orders them: 1.2.0.dev0 < 1.2.0a1 < 1.2.0rc1.dev0 < 1.2.0rc1 < 1.2.0 (all newer than
    1.1.x); a post-release (1.2.0.post1) counts as its release.
    """

    major: int
    minor: int
    patch: int
    pre: tuple[int | str, ...] = ()  # pre-release identifiers; () for a release
    dev: int | None = None  # PEP 440 .devN: in development, before the version without it

    @classmethod
    def parse(cls, text: str) -> Version:
        """``vX.Y.Z`` or ``X.Y.Z``, with an optional pre-release; ValueError otherwise."""
        match = _VERSION.fullmatch(text.strip()) if isinstance(text, str) else None
        if match is None:
            raise ValueError(f"not a version number: {text!r}")
        major, minor, patch = (int(n) for n in match.group(1, 2, 3))
        if match["pre"] is not None:
            pre = tuple(int(p) if p.isdigit() else p for p in match["pre"].split("."))
        elif match["pep"] is not None:
            pre = (match["pep"], int(match["pep_n"]))
        else:
            pre = ()
        dev = None if match["post"] or match["dev"] is None else int(match["dev"] or 0)
        return cls(major, minor, patch, pre, dev)

    @property
    def is_prerelease(self) -> bool:
        return bool(self.pre) or self.dev is not None

    def _key(self) -> tuple:
        # SemVer: numeric identifiers sort before alphanumeric ones, a longer list after its
        # prefix (as tuples do) and a release after its pre-releases. PEP 440: X.Y.Z.devN
        # before every pre-release of X.Y.Z, and .devN before the version it is added to.
        pre = tuple((0, p, "") if isinstance(p, int) else (1, 0, p) for p in self.pre)
        stage = (0, pre) if self.pre else (-1, ()) if self.dev is not None else (1, ())
        dev = (1, 0) if self.dev is None else (0, self.dev)
        return self.major, self.minor, self.patch, stage, dev

    def __lt__(self, other: Version) -> bool:
        if not isinstance(other, Version):
            return NotImplemented
        return self._key() < other._key()

    def __str__(self) -> str:
        text = f"{self.major}.{self.minor}.{self.patch}"
        if self.pre:
            text += f"-{'.'.join(str(p) for p in self.pre)}"
        return text if self.dev is None else f"{text}.dev{self.dev}"


@dataclass(frozen=True)
class Release:
    """A published release: its version, tag, page on GitHub and attached files."""

    version: Version
    tag: str
    page: str  # the release page on GitHub (its notes and files)
    prerelease: bool  # marked as one on GitHub, or a pre-release version
    assets: dict[str, str] = field(default_factory=dict, compare=False)  # file name -> URL

    def download_url(self, platform: str = sys.platform) -> str:
        """The archive of the app for *platform*, or the release page if it has none."""
        return self.assets.get(ASSETS.get(platform, ""), self.page)


class UpdateCheckError(Exception):
    """The releases could not be read; the message says why, for people."""


def default_channel(current: Version) -> str:
    """Pre-releases count while the app is 0.x (every 0.x release is one), not from 1.0.0."""
    return PRERELEASE if current.major == 0 else STABLE


def _ours(url: object, path: str) -> bool:
    """A plain https link on github.com under *path* (this repository's releases, the only
    pages the app opens): no other host, user, port, query or fragment, no escapes or ``..``."""
    if not isinstance(url, str) or not url.isascii() or not url.isprintable():
        return False
    link = urlsplit(url).path
    return (
        url == f"https://{SITE_HOST}{link}"
        and link.startswith(path)
        and posixpath.normpath(link) == link  # no "." or ".." segments, no "//"
        and not any(c in link for c in "%\\ ")
    )


def parse_releases(payload: object) -> list[Release]:
    """The releases in an answer of GitHub's list-releases API.

    Drafts, tags that are no version number and files outside this repository's releases are
    left out; a release page outside them is replaced by the tag's page.
    """
    if not isinstance(payload, list):
        raise UpdateCheckError("GitHub sent an answer the app cannot read")
    releases = []
    for item in payload:
        if not isinstance(item, dict) or item.get("draft") is not False:
            continue
        tag = item.get("tag_name")
        try:
            version = Version.parse(tag)
        except ValueError:
            continue
        tag = tag.strip()  # a version number now: safe in a link
        page = item.get("html_url")
        if not _ours(page, RELEASES_PATH):
            page = f"{RELEASES_PAGE}/tag/{tag}"
        assets = {}
        files = item.get("assets")
        for asset in files if isinstance(files, list) else ():
            if not isinstance(asset, dict) or asset.get("state", "uploaded") != "uploaded":
                continue
            name, url = asset.get("name"), asset.get("browser_download_url")
            if isinstance(name, str) and _ours(url, DOWNLOADS_PATH):
                assets[name] = url
        prerelease = item.get("prerelease") is not False or version.is_prerelease
        releases.append(Release(version, tag, page, prerelease, assets))
    return releases


def newest(releases: Iterable[Release], channel: str) -> Release | None:
    """The newest release of *channel*: every release for pre-release, else the others."""
    candidates = [r for r in releases if channel == PRERELEASE or not r.prerelease]
    return max(candidates, key=lambda r: r.version, default=None)


def update_for(releases: Iterable[Release], current: Version, channel: str) -> Release | None:
    """The newest release of *channel* if it is newer than *current*, else None."""
    release = newest(releases, channel)
    return release if release is not None and release.version > current else None


def releases_request(version: str) -> Request:
    """The GET of the newest releases: who asks (the app and its version) and nothing else."""
    return Request(
        RELEASES_API,
        headers={
            "User-Agent": f"mag-opt-detective/{version}",
            "Accept": "application/vnd.github+json",
        },
    )


class ApiRedirects(HTTPRedirectHandler):
    """Follows a redirect only within GitHub's API over https (a renamed repository answers
    with one); any other target fails the check."""

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        target = urlsplit(newurl)
        if target.scheme != "https" or target.netloc != API_HOST:
            fp.close()
            raise UpdateCheckError("GitHub sent the request to another address")
        return super().redirect_request(req, fp, code, msg, headers, newurl)


def open_github(request: Request, timeout: float):
    """Send *request* with urllib's handlers (proxies, certificates; no cookies), following
    redirects only within GitHub's API: the opener of :func:`fetch_releases`."""
    return build_opener(ApiRedirects).open(request, timeout=timeout)


def _error_text(exc: Exception) -> str:
    if isinstance(exc, HTTPError):
        if exc.code in (403, 429):  # the limit for requests without an account
            return "GitHub limits how often it answers; try again in an hour"
        return f"GitHub answered with an error (HTTP {exc.code})"
    reason = exc.reason if isinstance(exc, URLError) else exc
    if isinstance(reason, TimeoutError):
        return f"GitHub did not answer within {TIMEOUT:g} s"
    if isinstance(reason, ssl.SSLError):
        return "the secure connection to GitHub failed"
    if isinstance(reason, OSError):
        return "no connection to GitHub"
    return "GitHub sent an answer the app cannot read"


def fetch_releases(
    version: str, opener: Callable | None = None, timeout: float = TIMEOUT
) -> list[Release]:
    """Ask GitHub for the newest releases (blocks: call it off the GUI thread).

    *version* is the app's, sent in the User-Agent header as GitHub asks. *opener* takes the
    request and the timeout as :func:`open_github` (the default) does; tests pass a fake.
    Every failure (offline, timeout, rate limit, a redirect elsewhere, an answer that is no
    release list) raises :class:`UpdateCheckError`.
    """
    opener = opener or open_github
    try:
        with opener(releases_request(version), timeout=timeout) as response:
            body = response.read(MAX_BYTES + 1)
        if len(body) > MAX_BYTES:
            raise ValueError("the answer is too long")
        payload = json.loads(body)
    except (OSError, ValueError, RecursionError, http.client.HTTPException) as exc:
        raise UpdateCheckError(_error_text(exc)) from exc
    return parse_releases(payload)
