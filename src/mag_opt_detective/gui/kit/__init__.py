"""Reusable widgets for the redesigned window (painted from the palette, light and dark).

Widgets that hold state implement the settings protocol used by ``gui.settings.Persistence``:
``settings_value()`` returns a str, int, float or bool (JSON text for compound state) and
``set_settings_value(value)`` restores it, returning False for invalid values.
"""

from mag_opt_detective.gui.kit.collapsible import CollapsibleSection
from mag_opt_detective.gui.kit.range_control import RangeControl
from mag_opt_detective.gui.kit.range_slider import RangeSlider
from mag_opt_detective.gui.kit.segmented import SegmentedControl
from mag_opt_detective.gui.kit.slide_panel import SlidePanel

__all__ = [
    "CollapsibleSection",
    "RangeControl",
    "RangeSlider",
    "SegmentedControl",
    "SlidePanel",
]
