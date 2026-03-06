import tomllib
from enum import StrEnum, auto
from typing import Any

with open("pyproject.toml", "rb") as f:
    pyproject_data = tomllib.load(f)


APP_VERSION: str = f"v{pyproject_data['project']['version']}"
APP_NAME: str = "SynthetMic-GUI"
APP_AUTHORS: str = (
    "Synthetic Microstructures Group, "
    "School of Mathematical and Computer Sciences at Heriot-Watt University "
    "and School of Mathematics and Statistics at the University of Glasgow"
)
APP_LICENSE: str = "The MIT License"
APP_LINK: str = "https://david-bourne.shinyapps.io/synthetmic-gui/"

MDICE_LINK: str = "https://mdice.site.hw.ac.uk/"
MACS_LINK: str = (
    "https://www.hw.ac.uk/about/our-schools/mathematical-and-computer-sciences"
)
MS_LINK: str = "https://www.gla.ac.uk/schools/mathematicsstatistics/"
HW_LINK: str = "https://www.hw.ac.uk/"
UOG_LINK: str = "https://www.gla.ac.uk/"

FILL_COLOUR: str = "#0073CF"


class DiagramType(StrEnum):
    LAGUERRE = "Laguerre"
    VORONOI = "Voronoi"


class ExampleDataName(StrEnum):
    BASIC = auto()
    RANDOM = auto()
    BANDED = auto()
    CLUSTERED = auto()
    MIXED = auto()
    INCREASING = auto()
    MIDDLE = auto()
    DP = auto()
    LOGNORMAL = auto()
    EBSD = auto()


class Colorby(StrEnum):
    TARGET_VOLUMES = "target volumes"
    FITTED_VOLUMES = "fitted volumes"
    VOLUME_ERRORS = "volume errors"
    RANDOM = auto()


class DocSize(StrEnum):
    BEAMER = auto()
    THESIS = auto()


class Tab(StrEnum):
    MICRO = "Microstructure"
    METRICS = "Statistics"


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
    FULL = auto()
    SLICE = auto()
    CLIP = auto()


class SeedInitializer(StrEnum):
    RANDOM = auto()
    UPLOAD = auto()


PLOT_DEFAULTS: dict[str, Any] = {
    "view": DiagramView.FULL,
    "colorby": Colorby.FITTED_VOLUMES,
    "colormap": "plasma",
    "opacity": 1.0,
    "slice_value": 0.0,
    "slice_normal": "x",
    "clip_normal": "x",
}
