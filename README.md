# Magneto-Optical Detective

[![CI](https://github.com/wyzula-jan/mag-opt-detective/actions/workflows/ci.yml/badge.svg)](https://github.com/wyzula-jan/mag-opt-detective/actions/workflows/ci.yml)

Desktop tool to plot and analyse magneto-optical FTIR measurements: field sweeps
recorded with Bruker OPUS, either as OPUS binary files (`*.0`, `*.1`, …) or as
two-column text files written by the OPUS export macro.

- drag & drop zero-field and in-field spectra of a sample (and optionally of a reference)
- R(B)/R(0), raw data and R(B)/R(B-average) colour maps and stacked spectra
- zero-field drift correction, reference correction (with Savitzky-Golay smoothing),
  baseline normalization, 1st/2nd derivatives along energy or field, per data point or
  per unit (meV, cm⁻¹, THz or T)
- energy units cm⁻¹, meV, THz
- point picking on the colour map (e.g. Landau-level positions) with export/import
- export of the current map as a table and of any plot as PNG or SVG
- re-loading of exported maps, merging several spectral ranges or field ranges, and
  averaging repeated measurements
- choices (units, ranges, limits, colours, window layout) are remembered between
  sessions; *View → Reset Settings* restores the defaults

## Installation

Requires Python ≥ 3.12 and [uv](https://docs.astral.sh/uv/).

```bash
uv sync
```

## Running

```bash
uv run mag-opt-detective
```

or `uv run python -m mag_opt_detective`.

### Without Python

Standalone apps for Windows, macOS and Linux are attached to each
[release](https://github.com/wyzula-jan/mag-opt-detective/releases) (and to every run of
the *App bundles* workflow). Unzip and start `mag-opt-detective` (`.exe` on Windows,
*Magneto-Optical Detective.app* on macOS). The bundles are not code-signed: on macOS
right-click the app and choose *Open* the first time, on Windows choose *More info → Run
anyway*.

Build one yourself:

```bash
uv sync --group bundle
uv run pyinstaller packaging/mag-opt-detective.spec --noconfirm
```

`mag-opt-detective --smoke-test` processes a small synthetic measurement and exits with 0;
the workflow runs it on every bundle.

## Input files

| File | Content |
| --- | --- |
| OPUS binary (`*.0`, `*.1`, …) | the single-channel sample spectrum (`ScSm`) is read |
| text (`*.txt`, …) | two whitespace-separated columns: wavenumber (cm⁻¹), intensity |

The file type is detected automatically. With field values **From file names** the field
is taken from the name: `..._a01p250T.txt` → 1.25 T. Files are sorted by field, and the
Sample panel reports missing fields. Otherwise choose **Custom range** and enter start /
step / end.

**Zero field.** Load one zero-field spectrum, or two: measured before and after the
sweep, e.g. `..._a00p000T_a00p000T.txt` and `..._a00p000T_a16p000T.txt`. With two
spectra the zero-field reference is interpolated linearly between the first and the
last field point to compensate drift during the sweep. Files dropped on the Sample or
Reference panel (or a whole sweep folder) are sorted: names whose first field is 0 T go
to the zero-field list.

## Exported files

Tab-separated tables, compatible with the files written by the old versions:

```
Energy (meV)	0.25T	0.50T	...
12.5	1.0012	0.9987	...
```

Picked points are stored as field (rows) × curve name (columns), empty cells for
missing points.

## Library

The Library panel keeps processed maps: **Save current map** adds the R(B)/R(0) map
shown, **Load table…** adds exported tables. Tick the maps to combine and open a map's
row to give it an energy (E min / E max, in the energy unit shown) and field (B min /
B max) range; empty limits mean no cut.

- **Merge by energy** joins spectral ranges (e.g. FIR + MIR) and re-grids the energy
  axis to a uniform step.
- **Merge by field** joins field ranges (e.g. 0–8 T and 8–16 T sweeps); fields measured
  twice are averaged.
- **Average** averages repeated measurements with the same field values.

They need at least two ticked maps.

## Development

See [CONTRIBUTING.md](CONTRIBUTING.md). In short:

```bash
uv sync
uv run pre-commit install --hook-type pre-commit --hook-type commit-msg
uv run pytest
```

Tests that need real measurement data look for a `Data_to_test/` folder in the
repository root and are skipped when it is missing. Measurement data is never
committed.

## History

Versions up to 4.9 were PyQt5 scripts versioned by file name. The last of them is
kept in git under the tag `v4.9-legacy`.
