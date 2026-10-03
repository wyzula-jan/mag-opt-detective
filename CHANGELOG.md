# Changelog

All notable changes to Magneto-Optical Detective, newest first. The format is based on Keep a
Changelog, and the versions follow Semantic Versioning. `tools/release.py` writes each section
from the commit messages when a version is released (see Releases in CONTRIBUTING.md).

## 0.7.0 - 2026-10-04

### Added

- **gui:** watch a measurement folder. Switch on "Watch for new files" for the loaded sweep's folder, or choose File › Watch a folder… for one that may still be empty: every new spectrum is read once it is complete, and the sweep is processed and plotted again with the current settings, keeping the picked points, view ranges and models. Only new files are read. A spectrum written again is read again; files that are not spectra, are still being written or have another energy axis are held back and reported once; a folder that is out of reach is looked at until it is back. The status bar shows what is watched, with a button to stop.

## 0.6.0 - 2026-10-04

### Added

- **gui:** the status bar shows the baseline correction of the map on screen: "Baseline 500 – 880 cm⁻¹" in the display unit, with a Live tag while it is applied live, a dot when the Processing panel holds another region than the one applied, and "No baseline" when a region waits for the next Process. Its tooltip explains the state (or why a region cannot be applied); a click opens the baseline settings.

### Changed

- **gui:** the status bar's summary is shortened with an ellipsis when it does not fit.

## 0.5.0 - 2026-10-04

### Added

- **gui:** update notifications. Once a day, a few seconds after the start, the app asks GitHub whether a newer version is released (one anonymous request, nothing about you or your data; silent when offline). A newer version shows a notice with Download, Release notes and Skip this version, which stays until you close it. Help › Check for updates… asks at once, and Help › Check for updates at startup switches the daily check off. Nothing is ever installed by itself.

### Changed

- **docs:** the screenshots on the documentation site and in the README follow the reader's light or dark appearance.

### Fixed

- **gui:** a window whose building failed now closes without an error.

## 0.4.0 - 2026-10-03

### Added

- **gui:** Help › Documentation (F1, or ⌘? on macOS), Request a feature… and Report a bug…, which open the documentation site and GitHub's issue forms. A bug report is filled in with the app version, how it runs, the system, Python, Qt and library versions and the kind of plot shown, never with file names or paths. When no browser opens, the address can be copied from the message.
- Issue forms for bug reports and feature requests.

## 0.3.0 - 2026-10-03

### Added

- **gui:** an **Auto-scale the histograms** button in the plot toolbar. Off by default: the colour histograms stay still while the levels change (dragging the level region, typed levels, Auto, Fixed or Symmetric), and fit again when the data change, on a double-click or with **Fit to data** (A). On: the histograms follow the levels as before.

### Fixed

- **gui:** the colour histograms no longer jump to a new range after the levels are set with their region.

## 0.2.0 - 2026-10-03

### Added

- **gui:** a slider for every model parameter, including g, E₀, the couplings and the parameters of custom expressions. Sliders work in Range mode or in Relative mode, where a drag changes the value by up to ±1, 10 or 50 % and the handle springs back to the centre; one mode applies to all sliders and is switched from the Models section or a slider's context menu. Arrow keys nudge the value.

### Changed

- **gui:** the Models section puts every parameter on one aligned line (symbol, slider, value with its unit), gives the names of custom parameters their own column, and lines up the fit results in columns. g may be negative; sizes such as the half-gap, the velocity and the couplings stop at 0.

## 0.1.1 - 2026-10-03

### Fixed

- **gui:** pace live baseline updates by their full cost, drawing included, so a large map keeps updating during a drag of the baseline region, and a single slow update no longer switches it to updating on release.

## 0.1.0 - 2026-10-03

The first release. Magneto-Optical Detective plots, picks, fits and exports magneto-optical FTIR field sweeps recorded with Bruker OPUS, as OPUS binary files or as two-column text files.

### Added

- **Workbench:** a rail of panels (Sample, Reference, Processing, Library, Points), the Map, Stacked and Reference plots, an inspector (View, Colour, Traces, Models) and a log drawer, each sliding open and closed; light, dark or following the system.
- **Loading:** drop a sweep folder or files; the fields come from the file names or a custom range, the files are sorted by field and missing fields are reported. One zero-field spectrum is used as it is; two, measured before and after the sweep, correct the drift linearly.
- **Processing:** R(B)/R(0), the data, R(B)/R(B-AVR) and R(B)/R(B-ΔB); first or second derivatives along energy or field, per data point or per unit; reference correction with a separate sweep or the sample itself, optionally Savitzky–Golay smoothed; an energy window, and a baseline band to drag on the plot, which can apply live.
- **Viewing:** a live switch between cm⁻¹, meV and THz that converts maps, axes, ranges, points, models and exports without processing again; an energy range shared by all three plots, a field range for the map and the reference plot and an intensity range for the stacked plot, each Auto or Fixed; colour maps with Auto, Fixed or Symmetric levels and a histogram; stacked spectra with an offset, every n-th field and colours by field; a legend of the picked curves and the models drawn.
- **Picking:** on the map a click records the current curve's energy at the nearest field and Alt-click removes the nearest point; on the stacked plot a click near a trace records at its field. Curves with a table of their points, import and export of point tables, and undo and redo for every edit.
- **Auto-pick:** follow a line field by field, or find every line in a box, rotated box, ellipse or freehand region; maxima, minima or rising or falling inflection points, with a preview to accept or discard.
- **Models and fitting:** massive Dirac transitions, Zeeman and magnon branches (linear or hyperbolic, optionally coupled to give avoided crossings) and custom expressions in B, drawn live; fits to the picked points with value ± σ for every fitted parameter and χ², copied or saved as a table.
- **Library:** keep processed maps and exported tables, cut each to an energy and field range, merge spectral or field ranges and average repeated sweeps.
- **Export:** data tables, quick PNG or SVG images of the plot, and journal figures with a live preview: Nature, APS, Custom and your own presets, PDF, SVG and EPS with editable text, PNG and TIFF.
- **Settings:** units, ranges, colours and layout are remembered between sessions; View › Reset settings restores the defaults.
- **Standalone apps** for Windows, macOS on Apple silicon and Linux, which need no Python.

### Known limitations

- The apps are not code-signed: macOS and Windows ask once before the first launch.
- There is no build for Intel Macs; run from source there.
- In the standalone apps, the first journal figure after installing takes about 20 s while matplotlib builds its font cache.
- The two-column width of the APS preset (17.8 cm) is not confirmed by an official APS page.
