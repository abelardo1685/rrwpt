"""Flujo humo end-to-end: FEM + superposición + RWPT inverso + delineación
en una malla 2-D pequeña (60x60). Presupuesto: < 60 s en CPU.
"""
import numpy as np

from rrwpt import config, pipeline


def _small_ctrl(**extra):
    over = {
        "Two_D": 1,
        "n_pts_x": 60, "n_pts_y": 60,
        "d_pts_x": 15.0, "d_pts_y": 15.0,
        "Ex_pump": 0,                 # un solo pozo
        "het": 1,
        "tim.tend": 30, "tim.deltQS": 10,   # 3 pasos cuasi-estacionarios
        "crit.t_crit": 20,
        "trns.npartic_QS": 150,
        "stru.intexA": 200.0, "stru.intexE": 200.0,
        "stru.inteyA": 120.0, "stru.inteyE": 120.0,
    }
    over.update(extra)
    return config.make_ctrl(**over)


def test_end_to_end_small_mesh_heterogeneous():
    ctrl = _small_ctrl()
    rng = np.random.default_rng(7)
    grid, flowpar, model, ctrl = pipeline.setup(ctrl, rng)
    res = pipeline.run_realization(ctrl, grid, model, flowpar, rng=rng,
                                   verbose=False)

    # mapa probabilístico: vector nodal (ny+1)*(nx+1), escala 0-100 %
    n2d = (grid.n_pts[0] + 1) * (grid.n_pts[1] + 1)
    assert res.A50temporal.shape == (n2d,)
    assert np.isfinite(res.A50temporal).all()
    assert res.A50temporal.min() >= 0.0
    assert res.A50temporal.max() == 100.0        # normalizado al máximo
    assert (res.A50temporal > 0).sum() > 10      # delineó una zona no trivial

    pm = pipeline.prob_map_2d(res, grid)
    assert pm.shape == (grid.n_pts[0] + 1, grid.n_pts[1] + 1)

    # el campo log-K usado es heterogéneo y del tamaño de la malla de elementos
    assert res.yk.shape == (grid.data.nel,)
    assert res.yk.std() > 0.0


def test_end_to_end_reproducible_with_same_seed():
    maps = []
    for _ in range(2):
        ctrl = _small_ctrl(**{"trns.npartic_QS": 60})
        rng = np.random.default_rng(11)
        grid, flowpar, model, ctrl = pipeline.setup(ctrl, rng)
        res = pipeline.run_realization(ctrl, grid, model, flowpar, rng=rng,
                                       verbose=False)
        maps.append(res.A50temporal)
    np.testing.assert_array_equal(maps[0], maps[1])


def test_end_to_end_homogeneous_runs():
    ctrl = _small_ctrl(**{"het": 0, "trns.npartic_QS": 60})
    rng = np.random.default_rng(3)
    grid, flowpar, model, ctrl = pipeline.setup(ctrl, rng)
    res = pipeline.run_realization(ctrl, grid, model, flowpar, rng=rng,
                                   verbose=False)
    assert np.isfinite(res.A50temporal).all()
    assert res.A50temporal.max() == 100.0
    assert res.yk.std() == 0.0                   # log-K constante
