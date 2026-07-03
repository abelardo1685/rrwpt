"""Paridad MATLAB <-> Python de los mapas probabilísticos de WHPA.

Datos (tests/data/, ver README.md ahí):
  matlab_prob_map.npy         referencia MATLAB, escenario homogéneo (500 real.)
  matlab_prob_map_het.npy     referencia MATLAB, heterogéneo (Y_original)
  python_prob_map_*_quick.npy mapas Python archivados de la corrida "quick"
                              (malla completa 450x450x4, 6 pozos, tend=60 d,
                              t_crit=30 d, 150 part/QS; semillas 11 y 2026)
  Y_original_450x450.mat      campo log-K idéntico al usado por MATLAB

Dos niveles:
 1. Tests rápidos sobre los mapas ARCHIVADOS: consistencia de formato y el
    criterio físico de contención — la corrida quick usa un horizonte menor
    (60 d vs 360 d), por lo que su envolvente debe ser (casi) un subconjunto
    de la envolvente MATLAB. Medido en los datos: contención ~0.998-0.999.
 2. Test `slow` (opt-in: pytest -m slow, ~horas): regenera el mapa Python con
    los parámetros EXACTOS de MATLAB y exige solape (Jaccard) alto.
"""
import numpy as np
import pytest

from conftest import require_data

CELL_KM2 = 15 * 15 / 1e6


def _containment(inner, outer, thr=0.0):
    """Fracción de la envolvente `inner` contenida en la de `outer`."""
    a, b = outer > thr, inner > thr
    return (a & b).sum() / max(b.sum(), 1)


def _load(name):
    return np.load(str(require_data(name)[0]))


# ------------------------------------------------ nivel 1: mapas archivados
@pytest.mark.parametrize("mat_name,py_name", [
    ("matlab_prob_map.npy", "python_prob_map_homo_quick.npy"),
    ("matlab_prob_map_het.npy", "python_prob_map_het_quick.npy"),
])
def test_archived_maps_format(mat_name, py_name):
    require_data(mat_name, py_name)
    pm_mat, pm_py = _load(mat_name), _load(py_name)
    assert pm_mat.shape == pm_py.shape == (451, 451)   # nodos de 450x450
    for pm in (pm_mat, pm_py):
        assert np.isfinite(pm).all()
        assert pm.min() >= 0.0 and pm.max() == pytest.approx(100.0)


@pytest.mark.parametrize("mat_name,py_name", [
    ("matlab_prob_map.npy", "python_prob_map_homo_quick.npy"),
    ("matlab_prob_map_het.npy", "python_prob_map_het_quick.npy"),
])
def test_quick_python_envelope_contained_in_matlab(mat_name, py_name):
    """La corrida quick (tend=60 d) debe delinear DENTRO de la envolvente
    MATLAB (tend=360 d): mismo modelo, horizonte menor => subconjunto.
    Valores medidos al archivar: 0.999 (homo) y 0.998 (het)."""
    require_data(mat_name, py_name)
    pm_mat, pm_py = _load(mat_name), _load(py_name)
    assert _containment(pm_py, pm_mat) >= 0.95
    # el núcleo de alta probabilidad Python también cae en la envolvente MATLAB
    core_in = (pm_mat > 0)[pm_py > 50].mean()
    assert core_in >= 0.95


def test_y_original_field_loads_and_matches_matlab_stats():
    """El campo K prescrito (idéntico al de MATLAB) carga y es plausible."""
    scipy_io = pytest.importorskip("scipy.io")
    (path,) = require_data("Y_original_450x450.mat")
    Y = scipy_io.loadmat(str(path), squeeze_me=True)["Y"]
    assert Y.shape == (450, 450, 4)
    yk = Y.reshape(-1, order="F")           # MATLAB: reshape(Y,1,810000)'
    assert np.isfinite(yk).all()
    assert -10.0 < yk.mean() < -3.0         # log-K en el rango físico del modelo
    assert yk.std() > 0.1                   # heterogéneo de verdad


# ------------------------------------- nivel 2: regeneración a malla completa
@pytest.mark.slow
def test_regenerate_full_parity_heterogeneous():
    """Contraste "manzana con manzana": mismo campo K que MATLAB, parámetros
    exactos (450x450x4, 6 pozos, tend=360, t_crit=180, 1000 part/QS).
    ~2 h de CPU. Correr con: pytest -m slow tests/test_matlab_parity.py
    """
    from scipy.io import loadmat

    from rrwpt import config, pipeline

    mat_path, y_path = require_data("matlab_prob_map_het.npy",
                                    "Y_original_450x450.mat")
    pm_mat = np.load(str(mat_path))
    Y = loadmat(str(y_path), squeeze_me=True)["Y"]
    yk = Y.reshape(-1, order="F")

    ctrl = config.make_ctrl(**{
        "Two_D": 0,
        "n_pts_x": 450, "n_pts_y": 450, "n_pts_z": 4,
        "d_pts_x": 15.0, "d_pts_y": 15.0, "d_pts_z": 15.0,
        "het": 1, "Ex_pump": 1, "well_radius": 15.0,
        "tim.tend": 360, "tim.deltQS": 10, "crit.t_crit": 180,
        "trns.npartic_QS": 1000,
    })
    rng = np.random.default_rng(2026)
    grid, flowpar, model, ctrl = pipeline.setup(ctrl, rng)
    res = pipeline.run_realization(ctrl, grid, model, flowpar, rng=rng,
                                   verbose=False, yk=yk)
    pm_py = pipeline.prob_map_2d(res, grid)

    A, B = pm_mat > 0, pm_py > 0
    jac0 = (A & B).sum() / max((A | B).sum(), 1)
    area_rel = abs(B.sum() - A.sum()) / A.sum()
    # tolerancias: mismo K, solo difiere el random-walk estocástico
    assert jac0 >= 0.7
    assert area_rel <= 0.25
