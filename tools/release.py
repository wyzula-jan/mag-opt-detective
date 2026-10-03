"""Release the next semantic version, computed from the Conventional Commits since the last tag.

    python tools/release.py --dry-run                     # the next version and its section
    python tools/release.py                               # release it (nothing is pushed)
    python tools/release.py --version 1.0.0               # a chosen version instead
    python tools/release.py --first 0.1.0 --notes FILE    # the first release (no tag yet)

The commits since the last ``vX.Y.Z`` tag (merges skipped) decide the version: a breaking
change (``!`` after the type, or a ``BREAKING CHANGE:`` footer) bumps the major version,
``feat`` the minor, ``fix``, ``perf`` and ``revert`` the patch. While the major version is 0 a
breaking change bumps the minor version. ``docs``, ``test``, ``build``, ``ci``, ``chore``,
``style`` and ``refactor`` alone make no release.

A release writes the version into pyproject.toml, uv.lock, the package's ``__init__.py`` and
CITATION.cff (with ``date-released``), puts its section at the top of CHANGELOG.md, commits
these files as ``chore(release): vX.Y.Z`` (the git hooks run) and creates the annotated tag
``vX.Y.Z`` with the section as its message. It refuses to run with uncommitted changes, on a
commit that is already tagged, or when nothing needs a release. ``--notes FILE`` replaces the
generated section with your own text (Markdown, ``###`` headings at most).

The release workflow reads the version with ``--print-version`` and a version's section with
``--print-notes X.Y.Z``. Standard library only, Python 3.12 or newer.
"""

from __future__ import annotations

import argparse
import contextlib
import datetime as dt
import re
import subprocess
import sys
import tomllib
from collections import Counter
from pathlib import Path
from typing import NamedTuple

PYPROJECT = "pyproject.toml"
LOCK = "uv.lock"
INIT = "src/mag_opt_detective/__init__.py"
CITATION = "CITATION.cff"
CHANGELOG = "CHANGELOG.md"
VERSION_FILES = (PYPROJECT, LOCK, INIT, CITATION)
PACKAGE = "mag-opt-detective"  # the project's own entry in uv.lock

CHANGELOG_HEADER = """\
# Changelog

All notable changes to Magneto-Optical Detective, newest first. The format follows Keep a
Changelog, and the versions follow Semantic Versioning. `tools/release.py` writes each section
from the commit messages when a version is released (see Releases in CONTRIBUTING.md).
"""
NO_CHANGES = "No changes to the app itself."

SEMVER = re.compile(r"(0|[1-9]\d*)\.(0|[1-9]\d*)\.(0|[1-9]\d*)")
TAG = re.compile(r"v(0|[1-9]\d*)\.(0|[1-9]\d*)\.(0|[1-9]\d*)")
HEADER = re.compile(
    r"(?P<type>[A-Za-z]+)(?:\((?P<scope>[^()\s][^()]*)\))?(?P<bang>!)?: (?P<text>\S.*)"
)
FOOTER = re.compile(r"BREAKING[ -]CHANGE: *(?P<text>.*)")
SECTION = re.compile(r"^## (?P<version>\S+) - (?P<date>\d{4}-\d{2}-\d{2})[ \t]*$", re.MULTILINE)
SECTION_START = re.compile(r"^## ", re.MULTILINE)
PROJECT_TABLE = re.compile(
    r"^\[project\][ \t]*\n(?P<table>.*?)(?=^\[|\Z)", re.MULTILINE | re.DOTALL
)
PROJECT_VERSION = re.compile(r'^(?P<key>version\s*=\s*)"(?P<version>[^"]*)"', re.MULTILINE)
LOCK_PACKAGE = f'[[package]]\nname = "{PACKAGE}"\nversion = '
LOCK_VERSION = re.compile(rf'^(?P<key>{re.escape(LOCK_PACKAGE)})"(?P<version>[^"]*)"', re.MULTILINE)
INIT_VERSION = re.compile(r'^(?P<key>__version__ = )"(?P<version>[^"]*)"', re.MULTILINE)
CITATION_VERSION = re.compile(r"^(?P<key>version: *)(?P<version>.*?)[ \t]*$", re.MULTILINE)
CITATION_DATE = re.compile(r"^date-released: *[\"']?(?P<date>[^\"'\s]*)[\"']?[ \t]*$", re.MULTILINE)

LEVELS = ("patch", "minor", "major")
PATCH_TYPES = ("fix", "perf", "revert")
# the changelog's sections in their order, and the commit types listed in each (breaking
# changes and reverts of any type go under Changed)
SECTIONS = ("Added", "Changed", "Fixed", "Performance")
SECTION_OF_TYPE = {"feat": "Added", "fix": "Fixed", "perf": "Performance", "revert": "Changed"}


class ReleaseError(Exception):
    """A reason not to release, shown to the user."""


class Commit(NamedTuple):
    """One commit message, parsed. ``type`` is None for a message that is no Conventional
    Commit; ``note`` is the text of a ``BREAKING CHANGE:`` footer."""

    sha: str
    type: str | None
    scope: str | None
    text: str
    breaking: bool = False
    note: str = ""


# ---------------------------------------------------------------------- versions
def parse_version(text: str) -> tuple[int, int, int]:
    """``"1.2.3"`` (or ``"v1.2.3"``) as a tuple; ValueError for anything else."""
    match = SEMVER.fullmatch(text.removeprefix("v"))
    if match is None:
        raise ValueError(f"not a version X.Y.Z: {text!r}")
    return int(match[1]), int(match[2]), int(match[3])


def format_version(version: tuple[int, int, int]) -> str:
    return ".".join(map(str, version))


def parse_commit(message: str, sha: str = "") -> Commit:
    """The type, scope, description and breaking change of a commit *message*."""
    subject, _, body = message.strip().partition("\n")
    subject = subject.strip()
    match = HEADER.fullmatch(subject)
    if match is None:
        return Commit(sha, None, None, subject)
    note = breaking_note(body)
    breaking = bool(match["bang"]) or note is not None
    return Commit(
        sha, match["type"].lower(), match["scope"], match["text"].strip(), breaking, note or ""
    )


def breaking_note(body: str) -> str | None:
    """The text of the body's ``BREAKING CHANGE:`` footer up to the next blank line; None
    without one."""
    lines = body.splitlines()
    for index, line in enumerate(lines):
        match = FOOTER.fullmatch(line.strip())
        if match is None:
            continue
        text = [match["text"].strip()]
        for more in lines[index + 1 :]:
            if not more.strip():
                break
            text.append(more.strip())
        return " ".join(part for part in text if part)
    return None


def commit_level(commit: Commit) -> str | None:
    """The bump *commit* asks for: "major", "minor", "patch" or None."""
    if commit.type is None:
        return None
    if commit.breaking:
        return "major"
    if commit.type == "feat":
        return "minor"
    if commit.type in PATCH_TYPES:
        return "patch"
    return None


def release_level(commits: list[Commit]) -> str | None:
    """The largest bump the *commits* ask for, or None when none of them needs a release."""
    levels = [level for commit in commits if (level := commit_level(commit))]
    return max(levels, key=LEVELS.index) if levels else None


def next_version(last: tuple[int, int, int], level: str) -> tuple[int, int, int]:
    """*last* bumped by *level*; while the major version is 0 a major bump is a minor one."""
    major, minor, patch = last
    if level == "major" and major > 0:
        return major + 1, 0, 0
    if level in ("major", "minor"):
        return major, minor + 1, 0
    return major, minor, patch + 1


# ---------------------------------------------------------------------- changelog
def changelog_entry(commit: Commit) -> str:
    """The changelog line of *commit*: scope in bold, breaking changes flagged."""
    text = f"revert {commit.text}" if commit.type == "revert" else commit.text
    line = f"- **{commit.scope}:** {text}" if commit.scope else f"- {text}"
    if commit.breaking:
        line += f". **Breaking:** {commit.note}" if commit.note else ". **Breaking change.**"
    return line


def changelog_section(commits: list[Commit]) -> str:
    """The body of a release's section: Added, Changed, Fixed and Performance, each sorted by
    scope (in commit order within a scope); empty when no commit belongs there."""
    groups: dict[str, list[str]] = {name: [] for name in SECTIONS}
    for commit in sorted(commits, key=lambda commit: commit.scope or ""):
        if commit.type is None:
            continue
        name = "Changed" if commit.breaking else SECTION_OF_TYPE.get(commit.type)
        if name is None:
            continue
        line = changelog_entry(commit)
        if line not in groups[name]:
            groups[name].append(line)
    return "\n\n".join(
        f"### {name}\n\n" + "\n".join(lines) for name, lines in groups.items() if lines
    )


def changelog_sections(text: str) -> list[tuple[str, str, str]]:
    """The releases in a CHANGELOG.md *text*: (version, date, body), newest first."""
    starts = list(SECTION_START.finditer(text))
    sections = []
    for index, start in enumerate(starts):
        end = starts[index + 1].start() if index + 1 < len(starts) else len(text)
        match = SECTION.match(text, start.start())
        if match is None:
            line = text[start.start() :].partition("\n")[0]
            raise ReleaseError(f"{CHANGELOG}: not a release heading: {line!r}")
        sections.append((match["version"], match["date"], text[match.end() : end].strip()))
    return sections


def add_section(text: str, version: str, date: str, body: str) -> str:
    """CHANGELOG.md *text* with the section of *version* above the newest one."""
    section = f"## {version} - {date}\n\n{body.strip()}\n"
    if not text.strip():
        text = CHANGELOG_HEADER
    first = SECTION_START.search(text)
    if first is None:
        return text.rstrip("\n") + "\n\n" + section
    return text[: first.start()] + section + "\n" + text[first.start() :]


# ---------------------------------------------------------------------- version files
def _replace_version(text: str, pattern: re.Pattern, version: str, name: str) -> str:
    new, count = pattern.subn(lambda match: f'{match["key"]}"{version}"', text, count=1)
    if count != 1:
        raise ReleaseError(f"{name}: no version found")
    return new


def set_pyproject_version(text: str, version: str) -> str:
    """pyproject.toml *text* with *version* in its [project] table (and nowhere else)."""
    table = PROJECT_TABLE.search(text)
    if table is None:
        raise ReleaseError(f"{PYPROJECT}: no [project] table")
    new = _replace_version(table["table"], PROJECT_VERSION, version, PYPROJECT)
    return text[: table.start("table")] + new + text[table.end("table") :]


def set_lock_version(text: str, version: str) -> str:
    """uv.lock *text* with *version* for the project's own package."""
    return _replace_version(text, LOCK_VERSION, version, LOCK)


def set_init_version(text: str, version: str) -> str:
    """The package's __init__.py *text* with *version* as ``__version__``."""
    return _replace_version(text, INIT_VERSION, version, INIT)


def set_citation_version(text: str, version: str, date: str) -> str:
    """CITATION.cff *text* with *version* and ``date-released`` (added after the version)."""
    new, count = CITATION_VERSION.subn(lambda match: match["key"] + version, text, count=1)
    if count != 1:
        raise ReleaseError(f"{CITATION}: no version found")
    line = f'date-released: "{date}"'
    new, count = CITATION_DATE.subn(lambda _match: line, new, count=1)
    if count == 0:
        new = CITATION_VERSION.sub(lambda match: f"{match[0]}\n{line}", new, count=1)
    return new


def project_version(root: Path) -> str:
    """The version in pyproject.toml's [project] table."""
    return tomllib.loads(read_text(root / PYPROJECT))["project"]["version"]


def read_versions(root: Path) -> dict[str, str]:
    """The version written in each version file of the repository at *root*."""
    found = {PYPROJECT: project_version(root)}
    patterns = {LOCK: LOCK_VERSION, INIT: INIT_VERSION, CITATION: CITATION_VERSION}
    for name, pattern in patterns.items():
        match = pattern.search(read_text(root / name))
        if match is None:
            raise ReleaseError(f"{name}: no version found")
        found[name] = match["version"].strip("\"'")
    return found


def citation_date(root: Path) -> str | None:
    """The ``date-released`` of CITATION.cff, or None."""
    match = CITATION_DATE.search(read_text(root / CITATION))
    return match["date"] if match else None


def read_text(path: Path) -> str:
    """The text of *path* with "\\n" line ends."""
    return path.read_bytes().decode("utf-8").replace("\r\n", "\n")


def write_text(path: Path, text: str, like: bytes | None = None) -> None:
    """Write *text*, with "\\r\\n" line ends if the old content *like* had them."""
    if like is not None and b"\r\n" in like:
        text = text.replace("\n", "\r\n")
    path.write_bytes(text.encode("utf-8"))


# ---------------------------------------------------------------------- git
def git(root: Path, *args: str, stdin: str | None = None) -> str:
    """The output of ``git -C root args``; ReleaseError when it fails."""
    result = subprocess.run(
        ["git", "-C", str(root), *args],
        input=stdin,
        capture_output=True,
        text=True,
        encoding="utf-8",
        check=False,
    )
    if result.returncode != 0:
        output = (result.stderr.strip() + "\n" + result.stdout.strip()).strip()
        raise ReleaseError(f"git {' '.join(args)} failed:\n{output}")
    return result.stdout


def repo_root(path: Path) -> Path:
    try:
        return Path(git(path, "rev-parse", "--show-toplevel").strip())
    except (ReleaseError, OSError) as exc:
        raise ReleaseError(f"{path} is not in a git repository") from exc


def release_tags(root: Path, *args: str) -> list[str]:
    """The ``vX.Y.Z`` tags among ``git tag args``, oldest version first."""
    tags = [tag for tag in git(root, "tag", *args).split() if TAG.fullmatch(tag)]
    return sorted(tags, key=parse_version)


def last_release(root: Path) -> str | None:
    """The highest ``vX.Y.Z`` tag on HEAD's history, or None before the first release."""
    tags = release_tags(root, "--merged", "HEAD")
    return tags[-1] if tags else None


def commits_since(root: Path, tag: str | None) -> list[Commit]:
    """The commits after *tag* (all without one) up to HEAD, oldest first, merges skipped."""
    log = git(
        root, "log", "--no-merges", "--format=%H%x1f%B%x1e", f"{tag}..HEAD" if tag else "HEAD"
    )
    commits = []
    for record in log.split("\x1e"):
        if record.strip():
            sha, _, message = record.strip().partition("\x1f")
            commits.append(parse_commit(message, sha))
    return commits[::-1]


def uncommitted(root: Path) -> list[str]:
    """Changed tracked files (staged or not); untracked files do not count."""
    return git(root, "status", "--porcelain", "--untracked-files=no").splitlines()


# ---------------------------------------------------------------------- release
class Plan(NamedTuple):
    """What a release does: the version, why, the section and the files' new texts."""

    version: str
    previous: str | None
    reason: str
    commits: list[Commit]
    section: str
    files: dict[str, str]


def describe_types(commits: list[Commit]) -> str:
    """How many commits of each type, e.g. "3 fix, 1 docs"."""
    counts = Counter(commit.type or "other" for commit in commits)
    return ", ".join(f"{count} {name}" for name, count in counts.most_common())


def bump_reason(level: str, last: tuple[int, int, int], commits: list[Commit]) -> str:
    """Why *level* is the bump, e.g. "minor: feat"."""
    if level == "major":
        if last[0] == 0:
            return "minor: a breaking change while the major version is 0"
        return "major: a breaking change"
    causes = sorted({commit.type for commit in commits if commit_level(commit) == level})
    return f"{level}: {', '.join(causes)}"


def read_notes(path: Path) -> str:
    """A hand-written section from *path*; only ``###`` headings and below are allowed."""
    try:
        notes = read_text(path).strip()
    except OSError as exc:
        raise ReleaseError(f"cannot read the notes: {exc}") from exc
    if not notes:
        raise ReleaseError(f"{path}: the notes are empty")
    if re.search(r"^#{1,2} ", notes, re.MULTILINE):
        raise ReleaseError(f"{path}: use ### headings in the notes, not # or ##")
    return notes


def plan_release(root: Path, args: argparse.Namespace) -> Plan | str:
    """The release to make, or the reason why there is none (a str)."""
    tagged = release_tags(root, "--points-at", "HEAD")
    if tagged:
        raise ReleaseError(f"HEAD is already released as {tagged[-1]}: nothing to release")
    previous = last_release(root)
    commits = commits_since(root, previous)
    if previous is None:
        if args.first is None:
            raise ReleaseError(
                "no vX.Y.Z tag yet: give the first version with --first X.Y.Z and its "
                "changelog section with --notes FILE"
            )
        if args.notes is None:
            raise ReleaseError(
                "the first release needs --notes FILE: a summary of the app instead of "
                f"the {len(commits)} commits since the start"
            )
        version, reason = args.first, "the first release"
    else:
        if args.first is not None:
            raise ReleaseError(f"--first is for the first release only; the last is {previous}")
        if not commits:
            return f"Nothing to release: no commits since {previous}."
        level = release_level(commits)
        last = parse_version(previous)
        if args.version is not None:
            if parse_version(args.version) <= last:
                raise ReleaseError(f"--version {args.version} is not after {previous}")
            version, reason = args.version, "set with --version"
        elif level is None:
            return (
                f"No release: the {len(commits)} commits since {previous} are "
                f"{describe_types(commits)}; only feat, fix, perf, revert or a breaking "
                "change makes a release."
            )
        else:
            version = format_version(next_version(last, level))
            reason = bump_reason(level, last, commits)
    section = read_notes(args.notes) if args.notes else changelog_section(commits) or NO_CHANGES
    return Plan(
        version, previous, reason, commits, section, new_files(root, version, args.date, section)
    )


def new_files(root: Path, version: str, date: str, section: str) -> dict[str, str]:
    """The new text of every file a release of *version* changes."""
    for name in VERSION_FILES:
        if not (root / name).is_file():
            raise ReleaseError(f"{name} is missing")
    changelog = root / CHANGELOG
    old_changelog = read_text(changelog) if changelog.is_file() else ""
    if any(found == version for found, _date, _body in changelog_sections(old_changelog)):
        raise ReleaseError(f"{CHANGELOG} has a section for {version} already")
    return {
        PYPROJECT: set_pyproject_version(read_text(root / PYPROJECT), version),
        LOCK: set_lock_version(read_text(root / LOCK), version),
        INIT: set_init_version(read_text(root / INIT), version),
        CITATION: set_citation_version(read_text(root / CITATION), version, date),
        CHANGELOG: add_section(old_changelog, version, date, section),
    }


def apply(root: Path, plan: Plan) -> str:
    """Write the files, commit them and tag the commit; returns the commit's short hash.

    When the commit fails (e.g. a hook rejects it) the files are put back as they were.
    """
    tag = f"v{plan.version}"
    paths = {name: root / name for name in plan.files}
    old = {name: path.read_bytes() if path.exists() else None for name, path in paths.items()}
    try:
        for name, text in plan.files.items():
            write_text(paths[name], text, like=old[name])
        git(root, "add", "--", *plan.files)
        git(root, "commit", "--quiet", "-m", f"chore(release): {tag}")
    except (ReleaseError, OSError) as exc:
        for name, content in old.items():
            if content is None:
                paths[name].unlink(missing_ok=True)
            else:
                paths[name].write_bytes(content)
        with contextlib.suppress(ReleaseError):
            git(root, "reset", "--quiet", "--", *plan.files)
        raise ReleaseError(f"the release commit failed; the files are unchanged.\n{exc}") from exc
    # whitespace cleanup keeps the section's "###" headings, which the default strips
    message = f"{tag}\n\n{plan.section}\n"
    git(root, "tag", "--annotate", "--cleanup=whitespace", "--file=-", tag, stdin=message)
    return git(root, "rev-parse", "--short", "HEAD").strip()


def release(root: Path, args: argparse.Namespace) -> int:
    changes = uncommitted(root)
    if changes and not args.dry_run:
        raise ReleaseError(
            "uncommitted changes: commit or stash them first\n  " + "\n  ".join(changes)
        )
    plan = plan_release(root, args)
    if isinstance(plan, str):
        print(plan)
        return 0
    tag = f"v{plan.version}"
    since = f"since {plan.previous}" if plan.previous else "since the start"
    print(f"Last release: {plan.previous or 'none'}")
    print(f"Commits {since}: {len(plan.commits)} ({describe_types(plan.commits)})")
    print(f"Next version: {plan.version} ({plan.reason})")
    files = ", ".join(plan.files)
    if args.dry_run:
        if changes:
            print("Warning: uncommitted changes; a release would refuse to run.")
        print(
            f'Dry run, nothing written. A release updates {files}, commits "chore(release): '
            f'{tag}" and creates the annotated tag {tag}.'
        )
        print(f"\n## {plan.version} - {args.date}\n\n{plan.section}")
        return 0
    commit = apply(root, plan)
    print(f"Updated {files}")
    print(f"Committed {commit} chore(release): {tag}")
    print(f"Tagged {tag} (annotated, with the changelog section as its message)")
    print("Nothing was pushed. To publish: git push --follow-tags")
    return 0


def print_notes(root: Path, version: str) -> int:
    text = read_text(root / CHANGELOG) if (root / CHANGELOG).is_file() else ""
    for found, _date, body in changelog_sections(text):
        if found == version:
            print(body)
            return 0
    raise ReleaseError(f"{CHANGELOG} has no section for {version}")


# ---------------------------------------------------------------------- command line
def version_arg(text: str) -> str:
    try:
        return format_version(parse_version(text))
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc


def parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Release the next version: update the version files and CHANGELOG.md, "
        "commit them and create the annotated tag. Nothing is pushed."
    )
    parser.add_argument(
        "-C", "--repo", type=Path, default=Path(), help="the repository (default: the current one)"
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="print the next version and its section only"
    )
    choice = parser.add_mutually_exclusive_group()
    choice.add_argument("--version", type=version_arg, metavar="X.Y.Z", help="release this version")
    choice.add_argument(
        "--first", type=version_arg, metavar="X.Y.Z", help="the first version (no tag yet)"
    )
    choice.add_argument(
        "--print-version", action="store_true", help="print the version in pyproject.toml"
    )
    choice.add_argument(
        "--print-notes", type=version_arg, metavar="X.Y.Z", help="print a CHANGELOG.md section"
    )
    parser.add_argument(
        "--notes", type=Path, metavar="FILE", help="the changelog section, written by hand"
    )
    parser.add_argument(
        "--date",
        type=lambda text: dt.date.fromisoformat(text).isoformat(),
        default=dt.date.today().isoformat(),
        metavar="YYYY-MM-DD",
        help="the release date (default: today)",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        root = repo_root(args.repo)
        if args.print_version:
            print(project_version(root))
            return 0
        if args.print_notes:
            return print_notes(root, args.print_notes)
        return release(root, args)
    except ReleaseError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
