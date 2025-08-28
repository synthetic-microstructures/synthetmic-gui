from dataclasses import dataclass
from typing import Any, Callable, Iterable

import matplotlib.pyplot as plt
import matplotlib.ticker as tck
import numpy as np
import pandas as pd
import pyvista as pv
from matplotlib.figure import Figure
from synthetmic import LaguerreDiagramGenerator

from shared.controls import FILL_COLOUR, Colorby, Distribution, Slice


@dataclass(frozen=True)
class Diagram:
    generator: LaguerreDiagramGenerator
    max_percentage_error: float
    mean_percentage_error: float
    centroids: np.ndarray
    vertices: dict[int, list]
    fitted_volumes: np.ndarray
    target_volumes: np.ndarray
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

    domain = np.array([[0, s] for s in args])

    seeds = np.column_stack(
        [np.random.uniform(low=0, high=h, size=n_grains) for h in args]
    )

    return domain, seeds


def plot_diagram(
    generator: LaguerreDiagramGenerator,
    target_volumes: np.ndarray,
    colorby: str,
    colormap: str = "plasma",
    window_size: tuple[int, int] = (400, 400),
    add_final_seed_positions: bool = False,
    opacity: float = 1.0,
) -> tuple[pv.PolyData | pv.UnstructuredGrid, dict[str, pv.Plotter]]:
    mesh = generator.get_mesh()

    match colorby:
        case Colorby.TARGET_VOLUMES:
            colorby_values = target_volumes

        case Colorby.FITTED_VOLUMES:
            colorby_values = generator.get_fitted_volumes()

        case Colorby.VOLUME_ERRORS:
            colorby_values = (
                np.abs(generator.get_fitted_volumes() - target_volumes)
                * 100
                / target_volumes
            )

        case Colorby.RANDOM:
            colorby_values = np.random.rand(target_volumes.shape[0])

        case _:
            raise ValueError(
                f"Invalid colorby: {colorby}. Value must be one of {', '.join(Colorby)}"
            )

    mesh.cell_data["vols"] = colorby_values[mesh.cell_data["num"].astype(int)]

    final_seed_positions = generator.get_positions()
    n_samples, space_dim = final_seed_positions.shape

    NUM_SLICES = 3

    plotters = dict()

    for s, m in zip(
        Slice,
        (
            mesh,
            mesh.slice_orthogonal(),
            mesh.slice_along_axis(n=NUM_SLICES, axis="x"),
            mesh.slice_along_axis(n=NUM_SLICES, axis="y"),
            mesh.slice_along_axis(n=NUM_SLICES, axis="z"),
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
            opacity=opacity,
        )

        if space_dim == 2:
            pl.camera_position = "xy"

        elif space_dim == 3:
            pl.camera_position = "yz"
            pl.camera.azimuth = 45

        if s == Slice.FULL and add_final_seed_positions:
            if space_dim == 2:
                final_seed_positions = np.column_stack(
                    (final_seed_positions, np.zeros(n_samples))
                )

            pl.add_points(
                points=final_seed_positions,
                render_points_as_spheres=True,
                color="black",
                point_size=5,
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

    return Diagram(
        generator=generator,
        max_percentage_error=generator.max_percentage_error_,
        mean_percentage_error=generator.mean_percentage_error_,
        centroids=generator.get_centroids(),
        vertices=generator.get_vertices(),
        target_volumes=volumes,
        fitted_volumes=generator.get_fitted_volumes(),
        weights=generator.get_weights(),
        domain=domain,
        seeds=seeds,
        positions=generator.get_positions(),
    )


def gt(rhs: float) -> Callable[[float | None], str | None]:
    return lambda x: f"Must be greater than {rhs}" if (x <= rhs or x is None) else None


def gte(rhs: float) -> Callable[[float | None], str | None]:
    return (
        lambda x: f"Must be greater than or equal to {rhs}"
        if (x < rhs or x is None)
        else None
    )


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

    return lambda x: (
        None if (rule(x) or x is None) else f"Must be between {left} and {right}"
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
            "seeds_initial",
            "seeds_final",
            "centroids",
        ),
        (
            diagram.seeds,
            diagram.positions,
            diagram.centroids,
        ),
    ):
        property_dict[p] = pd.DataFrame(data=d[:, :dim], columns=COORDINATES[:dim])

    for p, d in zip(
        (
            "weights",
            "target_volumes",
            "fitted_volumes",
        ),
        (diagram.weights, diagram.target_volumes, diagram.fitted_volumes),
    ):
        property_dict[p] = pd.DataFrame(
            data=d, columns=["weights"] if p == "weights" else ["volumes"]
        )

    return property_dict


def plot_volume_dist(diagram: Diagram) -> Figure:
    fig = plt.figure()

    errors = (
        np.abs(diagram.target_volumes - diagram.fitted_volumes)
        * 100
        / diagram.target_volumes
    )

    ALPHA = 0.75
    EC = "black"
    BINS = "auto"

    for i in range(4):
        ax = fig.add_subplot(2, 2, i + 1)

        if i in (0, 2):
            ax.hist(
                x=(diagram.fitted_volumes if i == 0 else errors),
                color=FILL_COLOUR,
                bins=BINS,
                alpha=ALPHA,
                ec=EC,
            )
            ax.set_title("Volume distribution" if i == 0 else "Error distribution")
            ax.set_xlabel("Fitted volumes" if i == 0 else "Volume errors (%)")
            ax.set_ylabel("Frequency")

        elif i == 1:
            ax.hist(
                diagram.fitted_volumes,
                weights=diagram.fitted_volumes,
                density=False,
                color=FILL_COLOUR,
                alpha=ALPHA,
                ec=EC,
            )
            ax.set_title("Volume-weighted volume distribution")
            ax.set_xlabel("Fitted volumes")
            ax.set_ylabel("Normalized frequency")

        else:
            num_vertices_list = []

            space_dim = diagram.seeds.shape[1]

            if space_dim == 2:
                for vertices in diagram.vertices.values():
                    num_vertices_list.append(len(vertices))

            elif space_dim == 3:
                for faces in diagram.vertices.values():
                    for vertices in faces:
                        num_vertices_list.append(len(vertices))

            else:
                raise ValueError(
                    f"invalid space_dim: {space_dim}; value must be 2 or 3."
                )

            min_n, max_n = min(num_vertices_list), max(num_vertices_list)
            bins = np.linspace(
                min_n - 0.5,
                max_n + 0.5,
                num=max_n - min_n + 2,
            )
            ax.hist(
                num_vertices_list,
                bins=bins,
                color=FILL_COLOUR,
                alpha=ALPHA,
                ec=EC,
            )

            title_suffix = "per face" if space_dim == 3 else "per grain"
            ax.set_title(f"Distribution of the number of vertices {title_suffix}")
            ax.set_xlabel("Number of vertices")
            ax.set_ylabel("Frequency")
            ax.xaxis.set_major_locator(tck.MaxNLocator(integer=True))

    return fig


def validate_df(
    df: pd.DataFrame,
    expected_colnames: list[str],
    expected_type: str,
    file: str,
    expected_dim: tuple[int, int] | None = None,
    bounds: dict[str, Iterable[float]] | None = None,
) -> str | None:
    df_colnames = df.columns.to_list()

    if df_colnames != expected_colnames:
        return f"Column mismatch error in the uploaded {file} file: expected {expected_colnames} but got {df_colnames}. Please try again."

    if expected_dim is not None:
        if df.shape != expected_dim:
            return f"Dimension mismatch error in the uploaded {file} file: expected dimension {expected_dim} but got {df.shape}. Please try again."

    df_dtypes = df.dtypes.values.tolist()
    if not all(t == expected_type for t in df_dtypes):
        return f"Data type mismatch error in the uploaded {file} file: expected all values to be of {expected_type} but got {df_dtypes}. Please try again."

    if bounds is not None:
        msg = []
        for c, b in bounds.items():
            c_min = df[c].values.min()
            c_max = df[c].values.max()

            if not all(b[0] <= val <= b[1] for val in [c_min, c_max]):
                msg.append(
                    f"""expected {c}-coordinate values to be in [{b[0]:.2f}, {b[1]:.2f}]
                but values are in [{c_min:.2f}, {c_max:.2f}]"""
                )

        if msg:
            return (
                f"Value bound error in the uploaded {file} file: "
                + "; ".join(msg)
                + ". Please try again."
            )

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
