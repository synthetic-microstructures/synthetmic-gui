from enum import StrEnum, auto


class Dimension(StrEnum):
    TWO_D = "2D"
    THREE_D = "3D"


# class Distribution(StrEnum):
#     UNIFORM = auto()
#     NORMAL = auto()
#     LOGNORMAL = auto()


# class Phase(StrEnum):
#     SINGLE = auto()
#     DUAL = auto()


class FigureExtension(StrEnum):
    PDF = auto()
    PNG = auto()
    HTML = auto()


class PropertyExtension(StrEnum):
    CSV = auto()
    TXT = auto()
