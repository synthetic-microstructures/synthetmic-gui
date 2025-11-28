from pathlib import Path

import numpy as np
import pytest
from synthetmic import LaguerreDiagramGenerator

from shared.utils import fit_data


@pytest.fixture
def _seeds() -> np.ndarray:
    return np.loadtxt(Path("assets", "data", "five_2d_seeds.csv"), delimiter=",")


def test_verts_count(_seeds) -> None:
    tol = 1.0
    n_iter = 0
    damp_param = 1.0

    pkg_generator = LaguerreDiagramGenerator(
        tol=tol,
        n_iter=n_iter,
        damp_param=damp_param,
        verbose=True,
    )

    n_grains, space_dim = _seeds.shape

    domain = np.array([[0, 1] for _ in range(space_dim)])
    domain_vol = np.prod(domain[:, 1] - domain[:, 0])

    volumes = (np.ones(n_grains) / n_grains) * domain_vol

    periodic = [False] * space_dim

    pkg_generator.fit(seeds=_seeds, volumes=volumes, domain=domain, periodic=periodic)
    pkg_count = [len(v) for v in pkg_generator.get_vertices().values()]
    # pkg_count.sort()

    _, gui_generator = fit_data(
        domain=domain,
        seeds=_seeds,
        volumes=volumes,
        periodic=periodic,
        tol=tol,
        n_iter=n_iter,
        damp_param=damp_param,
    )
    gui_count = [len(v) for v in gui_generator.get_vertices().values()]
    # gui_count.sort()

    print(pkg_count)
    print(gui_count)

    assert len(pkg_count) == len(gui_count)
    assert pkg_count == gui_count

    return None
