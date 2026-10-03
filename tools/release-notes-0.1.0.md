The first release. Magneto-Optical Detective plots, picks, fits and exports magneto-optical FTIR field sweeps recorded with Bruker OPUS, as OPUS binary files or as two-column text files.

### Added

- **Workbench:** a rail of panels (Sample, Reference, Processing, Library, Points), the Map, Stacked and Reference plots, an inspector (View, Colour, Traces, Models) and a log drawer, each sliding open and closed; light, dark or following the system.
- **Loading:** drop a sweep folder or files; the fields come from the file names or a custom range, the files are sorted by field, missing fields are reported, and one or two zero-field spectra correct the drift during the sweep.
- **Processing:** R(B)/R(0), the data, R(B)/R(B-AVR) and R(B)/R(B-ΔB); first or second derivatives along energy or field, per data point or per unit; reference correction with Savitzky–Golay smoothing, an energy window, and a baseline band to drag on the plot that applies live.
- **Viewing:** a live switch between cm⁻¹, meV and THz that converts maps, axes, ranges, points, models and exports without processing again; field, energy and intensity ranges shared by the map and the stacked plot; colour maps with Auto, Fixed or Symmetric levels and a histogram; stacked spectra with an offset; a legend of the curves and models.
- **Picking:** a click records the energy of the current curve at a field, on the map or the stacked plot, and Alt-click removes it; curves with a table of their points, import and export, and undo and redo for every edit.
- **Auto-pick:** follow a line field by field, or find every line in a box, rotated box, ellipse or freehand region; maxima, minima or inflection points, with a preview to accept or discard.
- **Models and fitting:** massive Dirac transitions, Zeeman and magnon branches (linear or hyperbolic, optionally coupled to give avoided crossings) and custom expressions in B, drawn live; fits to the picked points with value ± σ for every parameter and χ², copied or saved as a table.
- **Library:** keep processed maps and exported tables, cut each to an energy and field range, merge spectral or field ranges and average repeated sweeps.
- **Export:** data tables, quick PNG or SVG images of the plot, and journal figures with a live preview: Nature, APS and your own presets, PDF, SVG and EPS with editable text, PNG and TIFF.
- **Settings:** units, ranges, colours and layout are remembered between sessions; View › Reset Settings restores the defaults.
- **Standalone apps** for Windows, macOS on Apple silicon and Linux, which need no Python.

### Known limitations

- The apps are not code-signed: macOS and Windows ask once before the first launch.
- There is no build for Intel Macs; run from source there.
- The first journal figure after installing takes about 20 s while matplotlib builds its font cache.
