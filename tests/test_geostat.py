"""Tests unitarios de geoestadística (geostat.py).

Cubren: covarianza Matérn (identidad analítica con el modelo exponencial en
kappa=0.5, valor en el origen, decaimiento), nicer_primes y el generador
espectral por circulant embedding (Dietrich & Newsam): reproducibilidad por
semilla, espectro no negativo y estadísticos empíricos del campo.
"""
from types import SimpleNamespace as NS

import numpy as np
import pytest

from rrwpt import config, geostat, grid_mod


def _model(name="matern", variance=1.0, kappa=1.5, lam=(80.0, 80.0)):
    m = NS()
    m.name = name
    m.variance = variance
    m.kappa = kappa
    m.lam = np.asarray(lam, dtype=float)
    m.micro = 0.0
    m.nugget = 0.0
    m.maxprime = 7
    return m


# ------------------------------------------------------------- covarianza
def test_matern_at_origin_equals_variance():
    m = _model(variance=0.7, kappa=1.2)
    assert geostat.evaluate_covariance(m, 0.0) == pytest.approx(0.7)


def test_matern_kappa_half_is_exponential():
    """Matérn con kappa=1/2 == exponencial: C(h) = v * exp(-2h*sqrt(1/2)).

    Con la parametrización del código (arg = 2h*sqrt(kappa)) y la identidad
    K_{1/2}(x) = sqrt(pi/(2x)) e^{-x}.
    """
    m = _model(variance=1.3, kappa=0.5)
    h = np.linspace(0.01, 5.0, 200)
    expected = 1.3 * np.exp(-2.0 * h * np.sqrt(0.5))
    np.testing.assert_allclose(geostat.evaluate_covariance(m, h), expected,
                               rtol=1e-10)


def test_matern_monotone_decay_and_positivity():
    m = _model(variance=1.0, kappa=1.5)
    h = np.linspace(0.0, 10.0, 500)
    q = geostat.evaluate_covariance(m, h)
    assert np.all(np.diff(q) <= 1e-12)      # no creciente
    assert np.all(q >= 0.0)
    assert q[0] == pytest.approx(1.0)
    assert q[-1] < 1e-3                     # decae a ~0


# ------------------------------------------------------------ nicer_primes
def test_nicer_primes_properties():
    n = np.array([97, 128, 251, 450])
    out = geostat.nicer_primes(n, pmax=7)
    assert np.all(out >= n)                 # nunca reduce el tamaño
    for v in out:
        rest = int(v)
        for p in (2, 3, 5, 7):
            while rest % p == 0:
                rest //= p
        assert rest == 1, f"{v} tiene factores primos > 7"


# ------------------------------------------- circulant embedding / campos
def _grid_and_model(n=64, d=15.0, lam=120.0, variance=0.6, kappa=1.0):
    ctrl = config.make_ctrl(**{
        "Two_D": 1, "n_pts_x": n, "n_pts_y": n, "d_pts_x": d, "d_pts_y": d,
        "stru.variA": variance, "stru.variE": variance,
        "stru.intexA": lam, "stru.intexE": lam,
        "stru.inteyA": lam, "stru.inteyE": lam,
        "stru.kapA": kappa, "stru.kapE": kappa,
        "stru.muA": 0.0, "stru.muE": 0.0,
    })
    grid = grid_mod.init_fem(grid_mod.init_grid(ctrl))
    ctrl, model = geostat.init_model(grid, ctrl, np.random.default_rng(0))
    # media determinista para poder verificar estadísticos del campo
    model.beta = np.zeros(model.nbeta)
    model.Qbb = np.zeros((model.nbeta, model.nbeta))
    return grid, model


def test_embedding_spectrum_nonnegative_and_shape():
    grid, model = _grid_and_model()
    fftqe = geostat.initialize_fft_cov(model, grid)
    assert np.all(fftqe >= 0.0)             # embedding válido (PSD numérica)
    assert np.all(np.asarray(fftqe.shape) >= grid.n_pts)  # dominio embebido >= original


def test_randomfield_seed_reproducibility():
    grid, model = _grid_and_model()
    Y1, _, _ = geostat.generate_randomfield(model, grid, rng=np.random.default_rng(123))
    Y2, _, _ = geostat.generate_randomfield(model, grid, rng=np.random.default_rng(123))
    Y3, _, _ = geostat.generate_randomfield(model, grid, rng=np.random.default_rng(124))
    np.testing.assert_array_equal(Y1, Y2)   # misma semilla => campo idéntico
    assert not np.array_equal(Y1, Y3)       # semilla distinta => campo distinto


def test_randomfield_empirical_statistics():
    """Media y varianza empíricas del ensamble ~ parámetros del modelo."""
    grid, model = _grid_and_model(n=48, variance=0.6)
    rng = np.random.default_rng(2026)
    fields, fftqe = [], None
    for _ in range(40):
        Y, _, fftqe = geostat.generate_randomfield(model, grid, rng=rng, fftqe=fftqe)
        fields.append(Y)
    stack = np.stack(fields)
    assert np.isfinite(stack).all()
    assert abs(stack.mean()) < 0.15                      # media ~ 0 (beta=0)
    assert 0.6 * 0.6 < stack.var() < 0.6 * 1.5           # varianza ~ 0.6 (tolerancia MC)


def test_homogeneous_field_is_constant():
    ctrl = config.make_ctrl(**{"Two_D": 1, "n_pts_x": 20, "n_pts_y": 20, "het": 0})
    grid = grid_mod.init_fem(grid_mod.init_grid(ctrl))
    yk, mui = geostat.uncond_homo_ksim(ctrl, grid, rng=np.random.default_rng(1))
    assert yk.shape == (grid.npts,)
    assert np.all(yk == mui)
    assert mui == pytest.approx((ctrl.stru.muA + ctrl.stru.muE) / 2)
