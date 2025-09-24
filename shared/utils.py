import io
import itertools
import json
import tempfile
import tomllib
import zipfile
from dataclasses import asdict, dataclass
from typing import Any, Callable

import matplotlib.pyplot as plt
import matplotlib.ticker as tck
import numpy as np
import pandas as pd
import pyvista as pv
from matplotlib.figure import Figure
from pysdot import ConvexPolyhedraAssembly, PowerDiagram
from synthetmic import LaguerreDiagramGenerator
from synthetmic.data.utils import SynthetMicData

from shared.controls import (
    FILL_COLOUR,
    Colorby,
    Distribution,
    FigureExtension,
    PropertyExtension,
)


@dataclass(frozen=True)
class Diagram:
    centroids: np.ndarray
    vertices: dict[int, list]
    fitted_volumes: np.ndarray
    target_volumes: np.ndarray
    weights: np.ndarray
    seeds: np.ndarray
    positions: np.ndarray
    domain: np.ndarray
    mesh: pv.PolyData | pv.UnstructuredGrid
    plotter: pv.Plotter
    clips: tuple[pv.UnstructuredGrid, pv.UnstructuredGrid] | None = None


COORDINATES = ("x", "y", "z")
VOLUMES = "volumes"


def sample_single_phase_vols(
    dist: str,
    n_grains: int,
    domain_vol: float,
    **kwargs,
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
    n_grains: int, random_state: int | None, *args
) -> tuple[np.ndarray, np.ndarray]:
    np.random.seed(random_state)

    domain = np.array([[0, s] for s in args])

    seeds = np.column_stack(
        [np.random.uniform(low=0, high=h, size=n_grains) for h in args]
    )

    return domain, seeds


def prepare_mesh(
    mesh: pv.PolyData | pv.UnstructuredGrid,
    target_volumes: np.ndarray,
    fitted_volumes: np.ndarray,
    colorby: str,
) -> pv.PolyData | pv.UnstructuredGrid:
    match colorby:
        case Colorby.TARGET_VOLUMES:
            colorby_values = target_volumes

        case Colorby.FITTED_VOLUMES:
            colorby_values = fitted_volumes

        case Colorby.VOLUME_ERRORS:
            colorby_values = (
                np.abs(fitted_volumes - target_volumes) * 100 / target_volumes
            )

        case Colorby.RANDOM:
            np.random.seed(42)  # for reproducibility in a given app session
            colorby_values = np.random.rand(target_volumes.shape[0])

        case _:
            raise ValueError(
                f"Invalid colorby: {colorby}. Value must be one of {', '.join(Colorby)}"
            )

    mesh.cell_data["vols"] = colorby_values[mesh.cell_data["num"].astype(int)]

    return mesh


def generate_clip_diagram(
    data: SynthetMicData,
    generator: LaguerreDiagramGenerator,
    colorby: str,
    clip_normal: str,
    clip_value: float,
    invert: bool,
    add_remains_as_wireframe: bool,
    colormap: str = "plasma",
    window_size: tuple[int, int] = (400, 400),
    opacity: float = 1.0,
) -> Diagram:
    if clip_normal not in COORDINATES:
        raise ValueError(
            f"Invalid for slice_normal: {clip_normal}. Value must be one of {COORDINATES}."
        )

    domain_map = dict(zip(COORDINATES, data.domain))
    a, b = domain_map[clip_normal]
    if not (a < clip_value < b):
        raise ValueError(
            f"clip_center: {clip_value} is out of domain along the specified normal. Value must be in ({a}, {b})."
        )

    fitted_volumes = generator.get_fitted_volumes()
    mesh = prepare_mesh(
        mesh=generator.get_mesh(),
        target_volumes=data.volumes,
        fitted_volumes=fitted_volumes,
        colorby=colorby,
    )
    clips = mesh.clip(
        normal=clip_normal,
        origin=(
            0,
            0,
            0,
        ),  # ensure that origin is taken as (0,0, 0) as all domain originates from 0.
        value=clip_value,
        return_clipped=True,
        crinkle=True,
        invert=not invert,  # note: invert is negated to have a more intuitive behaviour;
        # so if clip_value is 0.1 and invert is false, the bits up to 0.1 will be stored as the first element in the returned tuple.
    )

    pl = pv.Plotter(
        off_screen=True,
        window_size=list(window_size),
    )
    pl.add_mesh(
        clips[0],  # type: ignore
        label="Clipped",
        show_edges=True,
        show_scalar_bar=False,
        lighting=False,
        cmap=colormap,
        opacity=opacity,
        interpolate_before_map=True,
        scalars="vols",
    )

    if add_remains_as_wireframe:
        pl.add_mesh(
            clips[1],  # type: ignore
            style="wireframe",
            label="Remains",
            show_scalar_bar=False,
            lighting=False,
            opacity=opacity,
            cmap=colormap,
            scalars="vols",
        )

    if len(data.domain) == 2:
        pl.camera_position = "xy"

    pl.show_axes()  # type: ignore

    return Diagram(
        centroids=generator.get_centroids(),
        vertices=generator.get_vertices(),
        target_volumes=data.volumes,
        fitted_volumes=fitted_volumes,
        weights=generator.get_weights(),
        domain=data.domain,
        seeds=data.seeds,
        positions=generator.get_positions(),
        plotter=pl,
        mesh=mesh,
        clips=clips,  # type: ignore
    )


def generate_full_diagram(
    data: SynthetMicData,
    generator: LaguerreDiagramGenerator,
    colorby: str,
    colormap: str = "plasma",
    window_size: tuple[int, int] = (400, 400),
    add_final_seed_positions: bool = False,
    opacity: float = 1.0,
) -> Diagram:
    fitted_volumes = generator.get_fitted_volumes()
    mesh = prepare_mesh(
        mesh=generator.get_mesh(),
        target_volumes=data.volumes,
        fitted_volumes=fitted_volumes,
        colorby=colorby,
    )

    final_seed_positions = generator.get_positions()
    n_samples, space_dim = final_seed_positions.shape

    pl = pv.Plotter(
        off_screen=True,
        window_size=list(window_size),
    )

    if space_dim == 2:
        pl.camera_position = "xy"

    pl.add_mesh(
        mesh,
        show_edges=True,
        show_scalar_bar=False,
        lighting=False,
        cmap=colormap,
        opacity=opacity,
        scalars="vols",
    )

    if add_final_seed_positions:
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

    pl.show_axes()  # type: ignore

    return Diagram(
        centroids=generator.get_centroids(),
        vertices=generator.get_vertices(),
        target_volumes=data.volumes,
        fitted_volumes=fitted_volumes,
        weights=generator.get_weights(),
        domain=data.domain,
        seeds=data.seeds,
        positions=generator.get_positions(),
        plotter=pl,
        mesh=mesh,
    )


def generate_slice_diagram(
    data: SynthetMicData,
    generator: LaguerreDiagramGenerator,
    slice_normal: str,
    slice_value: float,
    colorby: str,
    colormap: str = "plasma",
    window_size: tuple[int, int] = (400, 400),
    opacity: float = 1.0,
) -> Diagram:
    def _sort_normals(normal: str) -> list[str]:
        normals = [c for c in COORDINATES if c != normal]

        return sorted(normals)

    def _map_normal_to_domain(
        normal: str, domain_map: dict[str, np.ndarray]
    ) -> np.ndarray:
        normals = _sort_normals(normal)

        return np.array([domain_map[n] for n in normals])

    def _map_normal_to_periodic(
        normal: str, periodic_map: dict[str, bool]
    ) -> list[bool]:
        normals = _sort_normals(normal)
        return [periodic_map[n] for n in normals]

    def _map_normal_to_seeds(
        normal: str, seeds_map: dict[str, np.ndarray]
    ) -> np.ndarray:
        normals = _sort_normals(normal)

        return np.column_stack([seeds_map[n] for n in normals])

    if slice_normal not in COORDINATES:
        raise ValueError(
            f"Invalid for slice_normal: {slice_normal}. Value must be one of {COORDINATES}."
        )

    if len(data.domain) != 3:
        raise ValueError(
            "Invalid space dimension. Slice can only be generated for a space dimension of 3"
        )

    domain_map = dict(zip(COORDINATES, data.domain))
    domain = _map_normal_to_domain(slice_normal, domain_map)  # type: ignore

    a, b = domain_map[slice_normal]
    if not (a <= slice_value <= b):
        raise ValueError(
            f"slice_center: {slice_value} is out of domain along the specified normal. Value must be in [{a}, {b}]."
        )

    periodic = None
    if data.periodic is not None:
        periodic_map = dict(zip(COORDINATES, data.periodic))
        periodic = _map_normal_to_periodic(slice_normal, periodic_map)  # type: ignore

    seeds_map = dict(zip(COORDINATES, generator.get_positions().T))
    seeds = _map_normal_to_seeds(slice_normal, seeds_map)  # type: ignore

    omega = ConvexPolyhedraAssembly()
    mins = domain[:, 0].copy()
    maxs = domain[:, 1].copy()
    lens = domain[:, 1] - domain[:, 0]
    if periodic is not None:
        for k, p in enumerate(periodic):
            if p:
                mins[k] = mins[k] - lens[k]
                maxs[k] = maxs[k] + lens[k]

    omega.add_box(mins, maxs)
    weights = generator.get_weights() - (slice_value - seeds_map[slice_normal]) ** 2

    pd = PowerDiagram(
        positions=seeds,
        weights=weights,
        domain=omega,
    )

    # If there is periodicity, then add the replicants
    if periodic is not None:
        periodic_dict = {True: [-1, 0, 1], False: [0]}
        periodic_list = [periodic_dict[p] for p in periodic]

        cartesian_periodic = list(itertools.product(*periodic_list))

        for rep in cartesian_periodic:
            if rep != (0, 0, 0):
                pd.add_replication(rep * lens)

    with tempfile.NamedTemporaryFile(mode="w", suffix=".vtk", delete=True) as tmp_file:
        filename = tmp_file.name

        pd.display_vtk(filename)
        mesh = pv.read(filename)

    # note: base target and fitted volumes are used
    mesh = prepare_mesh(
        mesh=mesh,  # type: ignore
        target_volumes=data.volumes,
        fitted_volumes=generator.get_fitted_volumes(),
        colorby=colorby,
    )

    pl = pv.Plotter(
        off_screen=True,
        window_size=list(window_size),
    )
    pl.add_mesh(
        mesh,  # type: ignore
        show_edges=True,
        show_scalar_bar=False,
        lighting=False,
        cmap=colormap,
        opacity=opacity,
        scalars="vols",
    )
    pl.camera_position = "xy"
    pl.show_axes()  # type: ignore

    vertices = {}
    offsets, coords = pd.cell_polyhedra()  # type: ignore
    for i in range(len(offsets) - 1):
        s, e = offsets[i : i + 2]
        vertices[i] = coords[s:e].tolist()

    return Diagram(
        centroids=pd.centroids(),
        vertices=vertices,
        target_volumes=np.array([]),  # no target volumes for slice
        fitted_volumes=pd.integrals(),
        weights=weights,
        domain=domain,
        seeds=seeds,
        positions=pd.get_positions(),  # type: ignore
        plotter=pl,
        mesh=mesh,  # type: ignore
    )


def fit_data(
    domain: np.ndarray,
    seeds: np.ndarray,
    volumes: np.ndarray,
    periodic: list[bool],
    tol: float,
    n_iter: int,
    damp_param: float,
) -> tuple[SynthetMicData, LaguerreDiagramGenerator]:
    generator = LaguerreDiagramGenerator(
        tol=tol, n_iter=n_iter, damp_param=damp_param, verbose=False
    )
    data = SynthetMicData(
        seeds=seeds,
        volumes=volumes,
        domain=domain,
        periodic=periodic,
        init_weights=None,
    )
    generator.fit(**asdict(data))

    return data, generator


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
        columns=["a", "b"],  # type: ignore
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
        property_dict[p] = pd.DataFrame(data=d[:, :dim], columns=COORDINATES[:dim])  # type: ignore

    props = ("weights", "fitted_volumes")
    data = (diagram.weights, diagram.fitted_volumes)

    if len(diagram.target_volumes) != 0:
        props += ("target_volumems",)
        data += (diagram.target_volumes,)

    for p, d in zip(props, data):
        property_dict[p] = pd.DataFrame(
            data=d,
            columns=["weights"] if p == "weights" else ["volumes"],  # type: ignore
        )

    return property_dict


def calculate_num_vertices_3d(
    grain_face_vertices: dict[int, list], precision: int
) -> list[int]:
    res = []

    for grain_faces in grain_face_vertices.values():
        grain_vertices = np.array([vertex for face in grain_faces for vertex in face])
        unique_vertices = np.unique(np.round(grain_vertices, precision), axis=0)

        res.append(len(unique_vertices))

    return res


def plot_volume_dist(
    seeds: np.ndarray,
    target_volumes: np.ndarray,
    fitted_volumes: np.ndarray,
    vertices: dict[int, list],
) -> Figure:
    fig = plt.figure()

    errors = np.abs(target_volumes - fitted_volumes) * 100 / target_volumes

    ALPHA = 0.75
    EC = "black"
    PRECISION = 8

    for i in range(4):
        ax = fig.add_subplot(2, 2, i + 1)

        if i in (0, 2):
            ax.hist(
                x=(fitted_volumes if i == 0 else errors),
                color=FILL_COLOUR,
                alpha=ALPHA,
                ec=EC,
            )
            ax.set_title("Volume distribution" if i == 0 else "Error distribution")
            ax.set_xlabel("Fitted volumes" if i == 0 else "Volume errors (%)")
            ax.set_ylabel("Frequency")

        elif i == 1:
            ax.hist(
                fitted_volumes,
                weights=fitted_volumes,
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

            space_dim = seeds.shape[1]

            if space_dim == 2:
                for v in vertices.values():
                    num_vertices_list.append(len(v))

            elif space_dim == 3:
                num_vertices_list = calculate_num_vertices_3d(
                    vertices, precision=PRECISION
                )
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
                bins=bins,  # type: ignore
                color=FILL_COLOUR,
                alpha=ALPHA,
                ec=EC,
            )

            ax.set_title("Distribution of the number of vertices per grain")
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
    bounds: dict[str, tuple[float, ...]] | None = None,
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
            c_min = df[c].min()
            c_max = df[c].max()

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


def create_full_download_bytes(diagram: Diagram) -> bytes:
    diagram_prop = extract_property_as_df(diagram)

    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zipf:
        for fname, df in diagram_prop.items():
            buffer = io.BytesIO()
            df.to_csv(buffer, index=False)
            buffer.seek(0)

            for ext in PropertyExtension:
                zipf.writestr(f"{fname}.{ext}", buffer.getvalue())

            buffer.close()

        # write the vertices to json
        buffer = io.StringIO()
        json.dump(
            diagram.vertices,
            buffer,
            indent=4,
        )
        buffer.seek(0)
        zipf.writestr("vertices.json", buffer.getvalue())
        buffer.close()

        for ext in FigureExtension:
            with tempfile.NamedTemporaryFile(
                mode="w", suffix=f".{ext}", delete=True
            ) as tmp_file:
                filename = tmp_file.name

                match ext:
                    case (
                        FigureExtension.PDF | FigureExtension.EPS | FigureExtension.SVG
                    ):
                        diagram.plotter.save_graphic(filename)

                    case FigureExtension.HTML:
                        diagram.plotter.export_html(filename)

                    case FigureExtension.VTK:
                        diagram.mesh.save(filename, binary=False)

                with open(filename, "rb") as f:
                    content = f.read()

            zipf.writestr(f"diagram.{ext}", content)

        if diagram.clips is not None:
            for i, clip in enumerate(diagram.clips, start=1):
                with tempfile.NamedTemporaryFile(
                    mode="w", suffix=".vtk", delete=True
                ) as temp_file:
                    clip.save(temp_file.name, binary=False)
                    with open(temp_file.name, "rb") as f:
                        content = f.read()

                zipf.writestr(f"clip_{i}.vtk", content)

    zip_buffer.seek(0)

    return zip_buffer.getvalue()


def get_app_version() -> str:
    with open("./pyproject.toml", "rb") as f:
        pyproject_data = tomllib.load(f)

    version = pyproject_data["project"]["version"]

    return f"v{version}"


def compute_cut_interval(
    normal: str, coordinates: tuple[str, ...], domain: np.ndarray
) -> tuple[float, float]:
    if len(coordinates) != len(domain):
        raise ValueError("coordinates and domain don't match.")

    if normal not in coordinates:
        raise ValueError(f"'{normal}' is not in the given coordinates '{coordinates}'.")

    domain_map = dict(zip(coordinates, domain))
    a, b = domain_map[normal]

    return float(a), float(b)
