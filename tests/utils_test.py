from itertools import product

import numpy as np
import pytest
from synthetmic import LaguerreDiagramGenerator
from synthetmic.data import toy
from synthetmic.data.utils import create_constant_volumes

from shared.utils import calculate_num_vertices, fit_data

TEST_DATA = (
    (
        np.array(
            [
                [0.37454012, 0.15599452],
                [0.95071431, 0.05808361],
                [0.73199394, 0.86617615],
                [0.59865848, 0.60111501],
                [0.15601864, 0.70807258],
            ]
        ),
        24,
        12,
    ),
    (
        np.array(list(product(*[[0.25, 0.75]] * 3))),
        64,
        27,
    ),
)


@pytest.mark.parametrize(
    "seeds, exp_num_vertices_per_grain, exp_num_unique_vertices", TEST_DATA
)
def test_total_num_vertices(
    seeds: np.ndarray, exp_num_vertices_per_grain: int, exp_num_unique_vertices: int
) -> None:
    tol = 1.0
    n_iter = 0
    damp_param = 1.0

    n_grains, space_dim = seeds.shape
    domain, domain_volume = toy._create_unit_domain(space_dim=space_dim)
    volumes = create_constant_volumes(n_grains=n_grains, domain_volume=domain_volume)

    periodic = [False] * space_dim

    _, gen = fit_data(
        domain=domain,
        seeds=seeds,
        volumes=volumes,
        periodic=periodic,
        tol=tol,
        n_iter=n_iter,
        damp_param=damp_param,
    )

    grain_face_vertices = [v for v in gen.get_vertices().values()]

    num_vertices_per_grain, num_unique_vertices = calculate_num_vertices(
        grain_face_vertices=grain_face_vertices,
        space_dim=space_dim,
        precision=2,
    )

    assert sum(num_vertices_per_grain) == exp_num_vertices_per_grain
    assert num_unique_vertices == exp_num_unique_vertices

    return None


@pytest.mark.parametrize("seeds", [td[0] for td in TEST_DATA])
def test_period_consistency(seeds: np.ndarray) -> None:
    tol = 1.0
    n_iter = 0
    damp_param = 1.0

    n_grains, space_dim = seeds.shape
    domain, domain_volume = toy._create_unit_domain(space_dim=space_dim)
    volumes = create_constant_volumes(n_grains=n_grains, domain_volume=domain_volume)
    periodic = [False] * space_dim

    pkg_generator = LaguerreDiagramGenerator(
        tol=tol, n_iter=n_iter, damp_param=damp_param, verbose=True
    )
    pkg_generator.fit(seeds=seeds, volumes=volumes, domain=domain, periodic=periodic)
    pkg_count = [len(v) for v in pkg_generator.get_vertices().values()]

    _, gui_generator = fit_data(
        domain=domain,
        seeds=seeds,
        volumes=volumes,
        periodic=periodic,
        tol=tol,
        n_iter=n_iter,
        damp_param=damp_param,
    )
    gui_count = [len(v) for v in gui_generator.get_vertices().values()]

    assert len(pkg_count) == len(gui_count)
    assert pkg_count == gui_count

    return None
