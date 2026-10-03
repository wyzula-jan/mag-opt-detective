# Docs site

The documentation and download site, to be published at
<https://wyzula-jan.github.io/mag-opt-detective/>. Static HTML, built with the standard library
only.

- `pages/*.html`: one fragment of plain HTML per page, starting with a comment that holds its
  `title` and `description`; `<!-- toc -->` puts an "On this page" list of the `h2` headings
  there, and `<!-- changelog -->` (in `release-notes.html`) the releases of the repository's
  `CHANGELOG.md`, which the build reads (written by `tools/release.py`; "No release yet"
  before the first).
- `layout.html`, `site.css`, `favicon.svg`: the frame around every page and its style.
- `images/`: screenshots of the app, rendered from a synthetic sweep by
  `python docs/make_screenshots.py --site` (never from measurement data).
- `build.py`: puts it together in `_build/` (git-ignored), with the navigation, the docs
  sidebar and previous / next links, and fails on a broken internal link, image or anchor. A
  new page also needs an entry in `MAIN_NAV` or `DOCS` there.

## Build

```bash
python docs/site/build.py
```

## Preview

```bash
python -m http.server -d docs/site/_build 8000
```

then open <http://localhost:8000>. All links are relative, so the site works the same under
the project page's `/mag-opt-detective/` path.

## Publish

The *Docs site* workflow (`.github/workflows/pages.yml`) builds and deploys the site to GitHub
Pages. It runs only when started by hand, after Pages is enabled with *GitHub Actions* as the
source.
