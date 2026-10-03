"""The processing chain behind the "Process" button.

Everything is processed in cm^-1, the unit of the measured files; :meth:`ProcessResult.get`
gives a map in the unit that is shown or exported.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from enum import Enum, StrEnum

import numpy as np

from mag_opt_detective.core import processing as proc
from mag_opt_detective.core.processing import Axis
from mag_opt_detective.core.readers import Measurement
from mag_opt_detective.core.spectra import FieldMap
from mag_opt_detective.core.units import Unit


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
    baseline_region: tuple[float, float] | None = None  # in cm^-1


@dataclass(frozen=True, eq=False)
class ProcessResult:
    """The processed maps, all with the energy axis in cm^-1.

    The baseline normalisation of *baseline_region* is applied to R(B)/R(0), R(B)/R(B-average)
    and R(B)/R(B-dB), never to Data or the reference maps. *before_baseline* keeps the result
    without it, so :meth:`with_baseline` can apply another region without processing again.
    """

    data: FieldMap
    ratio: FieldMap
    average: FieldMap
    step: FieldMap | None = None  # None when there is only one field value
    reference_data: FieldMap | None = None
    reference_ratio: FieldMap | None = None
    baseline_region: tuple[float, float] | None = None  # in cm^-1; None: not normalised
    before_baseline: ProcessResult | None = None  # the same without the baseline, if applied

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
        unit: Unit | str = Unit.CM1,
    ) -> FieldMap:
        """Map of *kind* in energy *unit*, differentiated *order* times along *axis*.

        *physical* divides by the real axis spacing instead of one data point, so an
        energy derivative is per *unit*: it scales with ``CM1_PER_UNIT[unit] ** order``
        (:func:`~mag_opt_detective.core.units.derivative_scale`). Derivatives along the
        field and per data point do not depend on the unit.
        """
        fmap = self.base(kind).to_unit(unit)  # convert first: d/dE is per *unit*
        for _ in range(order):
            fmap = proc.derivative(fmap, axis, physical=physical)
        return fmap

    def with_baseline(self, region: tuple[float, float] | None) -> ProcessResult:
        """This result with the baseline normalisation of *region* (cm^-1; None: none)
        instead of the one applied. Only the normalisation is computed (a shift per spectrum),
        from the maps before it, so the result is the same as processing again with *region*.
        """
        base = self.before_baseline or self
        if base.baseline_region is not None:
            raise ValueError("this result does not keep its maps before the baseline")
        if region is None:
            return base
        region = (float(region[0]), float(region[1]))
        ratio = proc.baseline_normalize(base.ratio, region)
        average = ratio if base.average is base.ratio else _baseline(base.average, region)
        return dataclasses.replace(
            base,
            ratio=ratio,
            average=average,
            step=_baseline(base.step, region),
            baseline_region=region,
            before_baseline=base,
        )

    @classmethod
    def from_map(
        cls, fmap: FieldMap, baseline_region: tuple[float, float] | None = None
    ) -> ProcessResult:
        """Wrap an already processed map (Processed tab).

        Data, R(B)/R(0) and R(B)/R(B-average) all show the loaded map; the field-step
        ratio is computed from it, with the field columns sorted first. The map is converted
        to cm^-1 (if needed), the unit of *baseline_region*.
        """
        fmap = _field_sorted(fmap.to_unit(Unit.CM1))
        result = cls(data=fmap, ratio=fmap, average=fmap, step=_step_or_none(fmap))
        return result.with_baseline(baseline_region)


def _field_sorted(fmap: FieldMap) -> FieldMap:
    """*fmap* with its field columns in rising order (the map itself if they are)."""
    if np.all(np.diff(fmap.field) >= 0):
        return fmap
    order = np.argsort(fmap.field, kind="stable")
    return fmap.replace(field=fmap.field[order], values=fmap.values[:, order])


def _step_or_none(fmap: FieldMap) -> FieldMap | None:
    return proc.step_ratio(fmap) if fmap.field.size >= 2 else None


def _baseline(fmap: FieldMap | None, region: tuple[float, float] | None) -> FieldMap | None:
    if fmap is None or region is None:
        return fmap
    return proc.baseline_normalize(fmap, region)


def _in_cm1(measurement: Measurement) -> Measurement:
    spectra = measurement.spectra
    if spectra.unit is Unit.CM1:
        return measurement
    return Measurement(spectra=spectra.to_unit(Unit.CM1), zero=measurement.zero)


def process(
    sample: Measurement,
    reference: Measurement | None = None,
    options: ProcessOptions | None = None,
) -> ProcessResult:
    """Build Data, R(B)/R(0) and R(B)/R(B-average) maps from a measurement.

    The maps are in cm^-1 (measurements in another unit are converted first), the unit
    of ``options.baseline_region``.
    """
    options = options or ProcessOptions()
    sample = _in_cm1(sample)
    reference = None if reference is None else _in_cm1(reference)
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

    result = ProcessResult(
        data=data,
        ratio=ratio,
        average=proc.ratio_to_average(data),
        step=_step_or_none(ratio),
        reference_data=ref_data,
        reference_ratio=ref_ratio,
    )
    return result.with_baseline(options.baseline_region)
