import tempfile
from dataclasses import asdict, dataclass
from enum import StrEnum
from typing import Any, Callable, Iterable

import numpy as np
import pyvista as pv
from pysdot import OptimalTransport
from synthetmic import LaguerreDiagramGenerator


class Slice(StrEnum):
    FULL = "full diagram"
    ORTHOGONAL = "othorgonal slice"
    X = "slice along x-axis"
    Y = "slice along y-axis"
    Z = "slice along z-axis"


@dataclass(frozen=True)
class InputData:
    seeds: np.ndarray
    volumes: np.ndarray
    domain: np.ndarray
    periodic: list[bool] | None
    init_weights: np.ndarray | None


@dataclass(frozen=True)
class OutputData:
    diagram_htmls: dict[str, str]
    max_percentage_error: float
    mean_percentage_error: float
    n_grains: int
    volume_ratio: int
    grain_ratio: int
    centroids: np.ndarray
    vertices: np.ndarray
    fitted_volumes: np.ndarray
    actual_volumes: np.ndarray


def calulate_rel_vols(n1: int, n2: int, r: int) -> np.ndarray:
    """
    Function to compute the (relative) volumes of the grains in an idealised
    microstructure with n1 grains of volume v and n2 grains of volume r*v,
    where v is chosen so that the total volume of the grains equals 1.
    """
    vols = np.concatenate((np.ones(n1), r * np.ones(n2)))

    return vols / np.sum(vols)


def create_input_data(
    n_grains: int,
    grain_ratio: float,
    volume_ratio: float,
    periodic: bool,
    *args: Iterable[float],
) -> InputData:
    # TODO: add periodic to the args?

    dim = len(args)
    domain = np.array([[0, s] for s in args])
    domain_vol = np.prod(args)

    seeds = np.random.uniform(
        low=domain.min(axis=1), high=domain.max(axis=1), size=(n_grains, dim)
    )

    # match dist.lower():
    #     case Distribution.UNIFORM:
    #         assert {"low", "high"}.issubset(kwargs), (
    #             "'low' and 'high' must be provided for uniform distribution"
    #         )
    #         seeds = np.random.uniform(
    #             low=kwargs.get("low"), high=kwargs.get("high"), size=(n_grains, dim)
    #         )

    #     case Distribution.NORMAL:
    #         assert {"mean", "std"}.issubset(kwargs), (
    #             "'mean' and 'std' must be provided for normal distribution"
    #         )
    #         seeds = np.random.normal(
    #             loc=kwargs.get("mean"), scale=kwargs.get("std"), size=(n_grains, dim)
    #         )

    #     case Distribution.LOGNORMAL:
    #         assert {"mean", "std"}.issubset(kwargs), (
    #             "'mean' and 'std' must be provided for lognormal distribution"
    #         )
    #         seeds = np.random.lognormal(
    #             mean=kwargs.get("mean"), sigma=kwargs.get("std"), size=(n_grains, dim)
    #         )

    #     case _:
    #         raise ValueError(
    #             f"invalid value for dist: value must be one of {', '.join(Distribution)}"
    #         )

    n1 = int(n_grains / (1 + grain_ratio))
    n2 = n_grains - n1
    volumes = domain_vol * calulate_rel_vols(n1, n2, volume_ratio)

    return InputData(
        seeds=seeds,
        volumes=volumes,
        domain=domain,
        periodic=[True] * dim if periodic else None,
        init_weights=None,
    )


def plot_diagram(
    optimal_transport: OptimalTransport,
    window_size: tuple[int, int] = (400, 400),
    colorby: np.ndarray | list[float] | None = None,
) -> tuple[pv.ImageData, dict[str, str]]:
    with tempfile.NamedTemporaryFile(mode="w", suffix=".vtk", delete=True) as tmp_file:
        filename = tmp_file.name

        optimal_transport.pd.display_vtk(filename)

        otgrid = pv.read(filename)

    if colorby is None:
        colorby = optimal_transport.pd.integrals()

    otcell_col = colorby[otgrid.cell_data["num"].astype(int)]
    otgrid.cell_data["vols"] = otcell_col

    diagram_htmls = dict()

    meshes = (
        otgrid,
        otgrid.slice_orthogonal(),
        otgrid.slice_along_axis(axis="x"),
        otgrid.slice_along_axis(axis="y"),
        otgrid.slice_along_axis(axis="z"),
    )

    for slice, mesh in zip(Slice, meshes):
        otplotter = pv.Plotter(
            off_screen=True,
            window_size=list(window_size),
        )

        otplotter.add_mesh(
            mesh,
            show_edges=True,
            show_scalar_bar=False,
            lighting=False,
        )
        # otplotter.add_scalar_bar(
        #     title="Volumes",
        #     vertical=False,
        #     fmt="%.2f",
        # )

        otplotter.show_axes()

        diagram_htmls[slice] = otplotter.export_html(filename=None).read()

    return otgrid, diagram_htmls


def generate_diagram(
    n_grains: int,
    grain_ratio: float,
    volume_ratio: float,
    tol: float,
    n_iter: int,
    damp_param: float,
    periodic: bool,
    *args: Iterable[float],
) -> OutputData:
    data = create_input_data(
        n_grains,
        grain_ratio,
        volume_ratio,
        periodic,
        *args,
    )

    ldg = LaguerreDiagramGenerator(
        tol=tol, n_iter=n_iter, damp_param=damp_param, verbose=False
    )
    ldg.fit(**asdict(data))

    mesh, diagram_htmls = plot_diagram(
        optimal_transport=ldg.optimal_transport_,
    )

    return OutputData(
        diagram_htmls=diagram_htmls,
        n_grains=n_grains,
        grain_ratio=grain_ratio,
        volume_ratio=volume_ratio,
        max_percentage_error=ldg.max_percentage_error_,
        mean_percentage_error=ldg.mean_percentage_error_,
        # centroids=mesh.cell_centers().points,
        centroids=ldg.get_centroids(),  # FIXME: for now; this is not the same as the pyvista mesh
        vertices=mesh.points,  # FIXME: needs checking; should be replaced with ldg.get_vertices()
        actual_volumes=data.volumes,
        fitted_volumes=ldg.get_fitted_volumes(),
    )


def gt(rhs: float) -> Callable[[float | None], str | None]:
    return lambda x: f"Must be greater than {rhs}" if (x < rhs or x is None) else None


def required() -> Callable[[Any], str | None]:
    return lambda x: "Required" if x is None else None


def integer() -> Callable[[Any], str | None]:
    return lambda x: None if isinstance(x, int) else "An integer is required"


def between(
    left: float,
    right: float,
    left_open: bool = False,
    right_open: bool = False,
    both_open: bool = False,
) -> Callable[[float | None], str | None]:
    def rule(x: float) -> bool:
        if left_open:
            return left < x <= right

        if right_open:
            return left <= x < right

        if both_open:
            return left < x < right

        return left <= x <= right

    return (
        lambda x: None
        if (rule(x) or x is None)
        else f"Must be between {left} and {right}"
    )
