from dataclasses import dataclass
from typing import Any, Callable, Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyvista as pv
import seaborn as sns
from matplotlib.figure import Figure
from synthetmic import LaguerreDiagramGenerator

from shared.controls import Colorby, Distribution, Slice


@dataclass(frozen=True)
class Diagram:
    mesh: pv.PolyData | pv.UnstructuredGrid
    plotters: dict[str, pv.Plotter]
    max_percentage_error: float
    mean_percentage_error: float
    centroids: np.ndarray
    vertices: np.ndarray
    fitted_volumes: np.ndarray
    actual_volumes: np.ndarray
    weights: np.ndarray
    seeds: np.ndarray
    positions: np.ndarray
    domain: np.ndarray


COORDINATES = ("x", "y", "z")
VOLUMES = "volumes"


def sample_single_phase_vols(
    dist: str,
    n_grains: int,
    domain_vol: float,
    **kwargs: dict[str, float],
) -> np.ndarray:
    match dist:
        case Distribution.CONSTANT:
            rel_vol = np.ones(n_grains) / n_grains

            return rel_vol * domain_vol

        case Distribution.UNIFORM:
            if not {"low", "high"}.issubset(kwargs):
                raise ValueError(
                    "'low' and 'high' must be provided for uniform distribution"
                )

            samples = np.random.uniform(
                low=kwargs.get("low"), high=kwargs.get("high"), size=n_grains
            )
            scaling_factor = domain_vol / samples.sum()

            return scaling_factor * samples

        case Distribution.LOGNORMAL:
            if not {"mean", "std"}.issubset(kwargs):
                raise ValueError(
                    "'mean' and 'std' must be provided for lognormal distribution"
                )

            sigma = np.sqrt(np.log(1 + (kwargs.get("std") / kwargs.get("mean")) ** 2))
            mu = -0.5 * sigma**2 + np.log(kwargs.get("mean"))

            samples = np.random.lognormal(mean=mu, sigma=sigma, size=n_grains)
            scaling_factor = domain_vol / samples.sum()

            return scaling_factor * samples

        case _:
            raise ValueError(
                f"invalid value for dist: value must be one of {', '.join(Distribution)}"
            )


def sample_dual_phase_vols(
    dist: tuple[str, str],
    n_grains: tuple[int, int],
    vol_ratio: tuple[float, float],
    domain_vol: float,
    dist_kwargs: tuple[dict[str, float], dict[str, float]],
) -> np.ndarray:
    phase1_vol = (vol_ratio[0] / sum(vol_ratio)) * domain_vol
    phase2_vol = domain_vol - phase1_vol

    domain_vols = (phase1_vol, phase2_vol)

    return np.concatenate(
        tuple(
            [
                sample_single_phase_vols(
                    dist=dist[i],
                    n_grains=n_grains[i],
                    domain_vol=domain_vols[i],
                    **dist_kwargs[i],
                )
                for i in (0, 1)
            ]
        ),
        axis=None,
    )


def sample_seeds(
    n_grains: int, random_state: int | None, *args: Iterable[float]
) -> tuple[np.ndarray, np.ndarray]:
    np.random.seed(random_state)

    dim = len(args)
    domain = np.array([[0, s] for s in args])

    seeds = np.random.uniform(
        low=domain.min(axis=1), high=domain.max(axis=1), size=(n_grains, dim)
    )

    return domain, seeds


def plot_diagram(
    generator: LaguerreDiagramGenerator,
    actual_volumes: np.ndarray,
    colorby: str,
    colormap: str = "plasma",
    window_size: tuple[int, int] = (400, 400),
) -> tuple[pv.PolyData | pv.UnstructuredGrid, dict[str, pv.Plotter]]:
    mesh = generator.get_mesh()

    match colorby:
        case Colorby.ACTUAL_VOLUMES:
            colorby_values = actual_volumes

        case Colorby.FITTED_VOLUMES:
            colorby_values = generator.get_fitted_volumes()

        case Colorby.VOLUME_ERRORS:
            colorby_values = (
                np.abs(generator.get_fitted_volumes() - actual_volumes)
                * 100
                / actual_volumes
            )

        case Colorby.RANDOM:
            colorby_values = np.random.rand(actual_volumes.shape[0])

        case _:
            raise ValueError(
                f"Invalid colorby: {colorby}. Value must be one of {', '.join(Colorby)}"
            )

    mesh.cell_data["vols"] = colorby_values[mesh.cell_data["num"].astype(int)]

    plotters = dict()

    for s, m in zip(
        Slice,
        (
            mesh,
            mesh.slice_orthogonal(),
            mesh.slice_along_axis(axis="x"),
            mesh.slice_along_axis(axis="y"),
            mesh.slice_along_axis(axis="z"),
        ),
    ):
        pl = pv.Plotter(
            off_screen=True,
            window_size=list(window_size),
        )

        pl.add_mesh(
            m,
            show_edges=True,
            show_scalar_bar=False,
            lighting=False,
            cmap=colormap,
        )

        pl.show_axes()

        plotters[s] = pl

    return mesh, plotters


def generate_diagram(
    domain: np.ndarray,
    seeds: np.ndarray,
    volumes: np.ndarray,
    periodic: bool,
    tol: float,
    n_iter: int,
    damp_param: float,
    colorby: str,
    colormap: str,
) -> Diagram:
    generator = LaguerreDiagramGenerator(
        tol=tol, n_iter=n_iter, damp_param=damp_param, verbose=False
    )
    generator.fit(
        seeds=seeds,
        volumes=volumes,
        domain=domain,
        periodic=[True] * domain.shape[0] if periodic else None,
        init_weights=None,
    )

    mesh, plotters = plot_diagram(
        generator=generator,
        actual_volumes=volumes,
        colorby=colorby,
        colormap=colormap,
    )

    return Diagram(
        mesh=mesh,
        plotters=plotters,
        max_percentage_error=generator.max_percentage_error_,
        mean_percentage_error=generator.mean_percentage_error_,
        centroids=generator.get_centroids(),  # FIXME: for now; this is not the same as the pyvista mesh
        vertices=generator.get_vertices(),
        actual_volumes=volumes,
        fitted_volumes=generator.get_fitted_volumes(),
        weights=generator.get_weights(),
        domain=domain,
        seeds=seeds,
        positions=generator.get_positions(),
    )


def gt(rhs: float) -> Callable[[float | None], str | None]:
    return lambda x: f"Must be greater than {rhs}" if (x < rhs or x is None) else None


def required() -> Callable[[Any], str | None]:
    return lambda x: "Required" if x is None else None


def integer(allow_none: bool = False) -> Callable[[Any], str | None]:
    def rule(x: Any) -> bool:
        return isinstance(x, int) or (x is None) if allow_none else isinstance(x, int)

    return lambda x: None if rule(x) else "An integer is required"


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


def to_superscript(n: int) -> str:
    superscripts = str.maketrans("-0123456789", "⁻⁰¹²³⁴⁵⁶⁷⁸⁹")
    return str(n).translate(superscripts)


def format_to_standard_form(x: float, precision: int = 2) -> str:
    s = f"{x:.{precision}e}"
    base, exponent = s.split("e")
    exponent = int(exponent)
    return f"{base} × 10{to_superscript(exponent)}"


def extract_property_as_df(diagram: Diagram) -> dict[str, pd.DataFrame]:
    dim = diagram.seeds.shape[1]
    property_dict = {}

    property_dict["domain"] = pd.DataFrame(
        data=diagram.domain,
        columns=["a", "b"],
    )

    for p, d in zip(
        (
            "seeds",
            "positions",
            "centroids",
            "vertices",
        ),
        (diagram.seeds, diagram.positions, diagram.centroids, diagram.vertices),
    ):
        property_dict[p] = pd.DataFrame(data=d[:, :dim], columns=COORDINATES[:dim])

    for p, d in zip(
        (
            "weights",
            "actual_volumes",
            "fitted_volumes",
        ),
        (diagram.weights, diagram.actual_volumes, diagram.fitted_volumes),
    ):
        property_dict[p] = pd.DataFrame(
            data=d, columns=["weights"] if p == "weights" else ["volumes"]
        )

    return property_dict


def plot_volume_dist(diagram: Diagram) -> Figure:
    fig, ax = plt.subplots(1, 3)

    for i in range(3):
        if i in (0, 2):
            sns.histplot(
                data=diagram.fitted_volumes
                if i == 0
                else np.abs(diagram.actual_volumes - diagram.fitted_volumes),
                kde=True,
                fill=True,
                ax=ax[i],
            )
            ax[i].set_title("Histogram plot")
            ax[i].set_xlabel("Fitted volumes" if i == 0 else "Volume errors")

        else:
            sns.histplot(
                data=diagram.fitted_volumes,
                cumulative=True,
                stat="density",
                element="step",
                fill=False,
                ax=ax[i],
            )
            ax[i].set_title("Cummulative distribution")
            ax[i].set_xlabel("Fitted volumes")

    return fig


def validate_df(
    df: pd.DataFrame,
    expected_colnames: list[str],
    expected_type: str,
    expected_dim: int | None = None,
    bound: tuple[float, float] | None = None,
) -> str | None:
    df_colnames = df.columns.to_list()

    if df_colnames != expected_colnames:
        return f"Column mismatch error: expected {expected_colnames} but got {df_colnames}."

    if expected_dim is not None:
        if len(df_colnames) != expected_dim:
            return f"Dimension mismatch error: expected dimension {expected_dim} but got {len(df_colnames)}."

    df_dtypes = df.dtypes.values.tolist()
    if not all(t == expected_type for t in df_dtypes):
        return f"Data type mismatch error: expected all values to be of {expected_type} but got {df_dtypes}."

    if bound is not None:
        df_min = df.values.min()
        df_max = df.values.max()

        if not all(bound[0] <= val <= bound[1] for val in [df_min, df_max]):
            return f"""Values bound error: expected minimum of all values to be in [{bound[0]}, {bound[1]}];
            but values have min and max as {df_min} and {df_max} respectively."""

    return None


def summarize_df(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()

    row_count = df.shape[0]
    column_count = df.shape[1]
    names = df.columns.tolist()
    column_names = ", ".join(str(name) for name in names)

    info_df = pd.DataFrame(
        {
            "Row Count": [row_count],
            "Column Count": [column_count],
            "Column Names": [column_names],
        }
    )

    return info_df
