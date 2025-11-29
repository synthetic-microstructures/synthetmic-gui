import numpy as np
import pytest
from synthetmic import LaguerreDiagramGenerator
from synthetmic.data import toy
from synthetmic.data.utils import create_constant_volumes

from shared.utils import fit_data


@pytest.fixture
def _seeds() -> np.ndarray:
    return np.array(
        [
            [0.37454012, 0.15599452],
            [0.95071431, 0.05808361],
            [0.73199394, 0.86617615],
            [0.59865848, 0.60111501],
            [0.15601864, 0.70807258],
        ]
    )


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
    domain, domain_volume = toy._create_unit_domain(space_dim=space_dim)
    volumes = create_constant_volumes(n_grains=n_grains, domain_volume=domain_volume)

    periodic = [False] * space_dim

    pkg_generator.fit(seeds=_seeds, volumes=volumes, domain=domain, periodic=periodic)
    pkg_count = [len(v) for v in pkg_generator.get_vertices().values()]

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

    print(pkg_count)
    print(gui_count)

    assert len(pkg_count) == len(gui_count)
    assert pkg_count == gui_count

    return None
