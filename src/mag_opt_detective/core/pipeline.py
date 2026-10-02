"""The processing chain behind the "Process" button."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, StrEnum

from mag_opt_detective.core import processing as proc
from mag_opt_detective.core.processing import Axis
from mag_opt_detective.core.readers import Measurement
from mag_opt_detective.core.spectra import FieldMap


class ReferenceMode(Enum):
    NONE = "none"
    SEPARATE = "separate"  # divide by a separately measured reference
    SELF = "self"  # use the (usually smoothed) sample itself as reference


class PlotKind(StrEnum):
    RATIO = "Ratio"  # R(B)/R(0)
    DATA = "Data"  # R(B) (divided by the reference, if any)
    AVERAGE = "Ratio_AVR"  # R(B)/R(B-average)
    STEP = "Ratio_Step"  # R(B)/R(B - dB), the ratio of neighbouring field steps


@dataclass(frozen=True)
class ProcessOptions:
    reference_mode: ReferenceMode = ReferenceMode.NONE
    smooth_reference: bool = False
    sg_window: int = 11
    sg_poly: int = 2
    baseline_region: tuple[float, float] | None = None


@dataclass(frozen=True, eq=False)
class ProcessResult:
    data: FieldMap
    ratio: FieldMap
    average: FieldMap
    step: FieldMap | None = None  # None when there is only one field value
    reference_data: FieldMap | None = None
    reference_ratio: FieldMap | None = None

    def base(self, kind: PlotKind) -> FieldMap:
        kind = PlotKind(kind)
        if kind is PlotKind.STEP:
            if self.step is None:
                raise ValueError("R(B)/R(B-dB) needs at least two field values")
            return self.step
        return {
            PlotKind.RATIO: self.ratio,
            PlotKind.DATA: self.data,
            PlotKind.AVERAGE: self.average,
        }[kind]

    def get(
        self,
        kind: PlotKind,
        order: int = 0,
        axis: Axis = Axis.ENERGY,
        physical: bool = False,
    ) -> FieldMap:
        """Map of *kind*, differentiated *order* times along *axis*.

        *physical* divides by the real axis spacing instead of one data point.
        """
        fmap = self.base(kind)
        for _ in range(order):
            fmap = proc.derivative(fmap, axis, physical=physical)
        return fmap

    @classmethod
    def from_map(
        cls, fmap: FieldMap, baseline_region: tuple[float, float] | None = None
    ) -> ProcessResult:
        """Wrap an already processed map (Processed tab).

        Data, R(B)/R(0) and R(B)/R(B-average) all show the loaded map; the field-step
        ratio is computed from it.
        """
        corrected = _baseline(fmap, baseline_region)
        step = _baseline(_step_or_none(fmap), baseline_region)
        return cls(data=fmap, ratio=corrected, average=corrected, step=step)


def _step_or_none(fmap: FieldMap) -> FieldMap | None:
    return proc.step_ratio(fmap) if fmap.field.size >= 2 else None


def _baseline(fmap: FieldMap | None, region: tuple[float, float] | None) -> FieldMap | None:
    if fmap is None or region is None:
        return fmap
    return proc.baseline_normalize(fmap, region)


def process(
    sample: Measurement,
    reference: Measurement | None = None,
    options: ProcessOptions | None = None,
) -> ProcessResult:
    """Build Data, R(B)/R(0) and R(B)/R(B-average) maps from a measurement."""
    options = options or ProcessOptions()
    data = sample.spectra
    ratio = proc.ratio_to_zero(sample)
    ref_data: FieldMap | None = None
    ref_ratio: FieldMap | None = None

    if options.reference_mode is ReferenceMode.SEPARATE:
        if reference is None:
            raise ValueError("reference correction selected but no reference is loaded")
        ref_data = proc.interpolate_field(reference.spectra, data.field)
        ref_ratio = proc.interpolate_field(proc.ratio_to_zero(reference), data.field)
    elif options.reference_mode is ReferenceMode.SELF:
        ref_data, ref_ratio = data, ratio

    if ref_data is not None and ref_ratio is not None:
        if options.smooth_reference:
            ref_data = proc.savgol(ref_data, options.sg_window, options.sg_poly)
            ref_ratio = proc.savgol(ref_ratio, options.sg_window, options.sg_poly)
        data = proc.divide(data, ref_data)
        ratio = proc.divide(ratio, ref_ratio)

    average = proc.ratio_to_average(data)
    step = _step_or_none(ratio)
    region = options.baseline_region
    ratio, average, step = (_baseline(m, region) for m in (ratio, average, step))

    return ProcessResult(
        data=data,
        ratio=ratio,
        average=average,
        step=step,
        reference_data=ref_data,
        reference_ratio=ref_ratio,
    )
