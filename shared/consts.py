import tomllib
from enum import StrEnum, auto
from types import MappingProxyType

with open("pyproject.toml", "rb") as f:
    pyproject_data = tomllib.load(f)


APP_VERSION: str = f"v{pyproject_data['project']['version']}"
APP_NAME: str = "SynthetMic-GUI"
APP_AUTHORS: str = "R.O. Ibraheem, D.P. Bourne and S.M. Roper"
APP_LICENSE: str = "The MIT License"
APP_LINK: str = "https://david-bourne.shinyapps.io/synthetmic-gui/"

MDICE_LINK: str = "https://mdice.site.hw.ac.uk/"
MACS_LINK: str = (
    "https://www.hw.ac.uk/about/our-schools/mathematical-and-computer-sciences"
)
MS_LINK: str = "https://www.gla.ac.uk/schools/mathematicsstatistics/"
HW_LINK: str = "https://www.hw.ac.uk/"
UOG_LINK: str = "https://www.gla.ac.uk/"

ALGO_PAPER_LINK: str = "https://doi.org/10.1080/14786435.2020.1790053"

FILL_COLOUR: str = "#0073CF"


class DiagramType(StrEnum):
    LAGUERRE = "Laguerre"
    VORONOI = "Voronoi"


class ExampleDataName(StrEnum):
    EBSD = "EBSD"
    LOGNORMAL = "Lognormal"
    BANDED_PERIODIC = "Banded periodic"
    DUAL_PHASE = "Dual phase"
    BASIC = "Basic"
    RANDOM = "Random"
    BANDED = "Banded"
    MIXED = "Mixed"
    CLUSTERED = "Clustered"
    INCREASING = "Increasing"
    MIDDLE = "Middle"


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


PLOT_DEFAULTS: MappingProxyType = MappingProxyType(
    {
        "view": DiagramView.FULL,
        "colorby": Colorby.FITTED_VOLUMES,
        "colormap": "plasma",
        "opacity": 1.0,
        "slice_value": 0.0,
        "slice_normal": "x",
        "clip_normal": "x",
        "add_final_seed_positions": False,
    }
)
