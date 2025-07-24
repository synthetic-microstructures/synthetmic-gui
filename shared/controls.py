from enum import StrEnum, auto


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
    UPLOAD = "upload volumes"


class FigureExtension(StrEnum):
    PDF = auto()
    SVG = auto()
    EPS = auto()
    HTML = auto()
    VTK = auto()


class PropertyExtension(StrEnum):
    CSV = auto()
    TXT = auto()


class Slice(StrEnum):
    FULL = "full diagram"
    ORTHOGONAL = "othorgonal slice"
    X = "slice along x-axis"
    Y = "slice along y-axis"
    Z = "slice along z-axis"


class Colorby(StrEnum):
    ACTUAL_VOLUMES = "actual volumes"
    FITTED_VOLUMES = "fitted volumes"
    VOLUME_ERRORS = "volume erros"
    RANDOM = auto()


class SeedInitializer(StrEnum):
    RANDOM = auto()
    UPLOAD = auto()
