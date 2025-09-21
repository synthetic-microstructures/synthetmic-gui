from enum import StrEnum, auto
from typing import Any


class DocSize(StrEnum):
    BEAMER = auto()
    THESIS = auto()


class Tab(StrEnum):
    GEN_MIC = "Generated microstructure"
    METRICS_AND_PLOTS = "Full diagram metrics and plots"


class Dimension(StrEnum):
    TWO_D = "2D"
    THREE_D = "3D"


class Distribution(StrEnum):
    CONSTANT = auto()
    UNIFORM = auto()
    LOGNORMAL = auto()


class Phase(StrEnum):
    SINGLE = auto()
    DUAL = auto()
    UPLOAD = "upload target volumes"


class FigureExtension(StrEnum):
    PDF = auto()
    SVG = auto()
    EPS = auto()
    HTML = auto()
    VTK = auto()


class PropertyExtension(StrEnum):
    CSV = auto()
    TXT = auto()


class DiagramView(StrEnum):
    FULL = "Full"
    SLICE = "Slice"
    CLIP = "Clip"


class Colorby(StrEnum):
    TARGET_VOLUMES = "target volumes"
    FITTED_VOLUMES = "fitted volumes"
    VOLUME_ERRORS = "volume errors"
    RANDOM = auto()


class SeedInitializer(StrEnum):
    RANDOM = auto()
    UPLOAD = auto()


PLOT_DEFAULTS: dict[str, Any] = {
    "view": DiagramView.FULL,
    "colorby": Colorby.FITTED_VOLUMES,
    "colormap": "plasma",
    "opacity": 1.0,
    "fig_extension": FigureExtension.HTML,
    "prop_extension": PropertyExtension.CSV,
    "slice_value": 0.0,
    "slice_normal": "x",
    "clip_value": 0.0,
    "clip_normal": "x",
}
FILL_COLOUR: str = "#0073CF"
