"""Tests unitarios de configuración y malla (config.py, grid_mod.py)."""
import numpy as np

from rrwpt import config
from rrwpt.grid_mod import ind2sub, init_fem, init_grid, sub2ind


def test_make_ctrl_defaults_and_overrides():
    c = config.make_ctrl(**{"n_pts_x": 60, "tim.tend": 30, "crit.t_crit": 20})
    assert c.n_pts_x == 60
    assert c.tim.tend == 30
    assert c.crit.t_crit == 20
    # cantidades derivadas (finalize_ctrl)
    assert c.tim.tstep == int(np.ceil(30 / c.tim.deltQS))
    assert c.Tstatvalid[0] == 0.0
    assert c.Tstatvalid[-1] == c.tim.tstep * c.tim.deltQS


def test_sub2ind_ind2sub_roundtrip_matlab_order():
    """La numeración global es column-major (orden 'F'), como MATLAB."""
    shape = (5, 4, 3)
    rng = np.random.default_rng(0)
    ys = rng.integers(0, 5, 20)
    xs = rng.integers(0, 4, 20)
    zs = rng.integers(0, 3, 20)
    lin = sub2ind(shape, ys, xs, zs)
    # equivalencia con np.ravel_multi_index en orden F
    expected = np.ravel_multi_index((ys, xs, zs), shape, order="F")
    np.testing.assert_array_equal(lin, expected)
    back = ind2sub(shape, lin)
    np.testing.assert_array_equal(back[0], ys)
    np.testing.assert_array_equal(back[1], xs)
    np.testing.assert_array_equal(back[2], zs)


def test_grid_2d_dimensions():
    c = config.make_ctrl(**{"Two_D": 1, "n_pts_x": 30, "n_pts_y": 20,
                            "d_pts_x": 10.0, "d_pts_y": 5.0})
    grid = init_fem(init_grid(c))
    assert grid.nd == 2
    np.testing.assert_array_equal(grid.data.n_el, [20, 30])   # (ny, nx)
    np.testing.assert_allclose(grid.d_tot, [20 * 5.0, 30 * 10.0])
