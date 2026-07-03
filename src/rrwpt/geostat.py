"""Modelo geoestadístico y generación de campos aleatorios (2D y 3D).

Espejo de:
  INIT/init_modelRai_ARP.m
  SPECTRAL_GEOSTATS/{generate_randomfield, initialize_FFT_cov, find_embedding,
                     evaluate_separation, evaluate_covariance, extraction,
                     nicer_primes}.m
  Abelardo/uncond_homo_Ksim.m

El método espectral (Dietrich & Newsam) es independiente de la dimensión:
todas las operaciones usan fftn/ifftn y mallas n-d, por lo que el caso 3D
queda cubierto con el mismo código que el 2D.
"""
from types import SimpleNamespace as NS

import numpy as np
from numpy.fft import fftn, ifftn
from scipy.special import gamma as gamma_fn, kv as besselk

from .grid_mod import ndgrid_setup


# ---------------------------------------------------------------- model init
def init_model(grid, ctrl, rng=None):
    """init_modelRai_ARP.m — parámetros del variograma y tendencias."""
    rng = rng or np.random.default_rng()
    model = NS()
    model.name = "matern" if ctrl.use_var_matern_kappa == 1 else "gaussian"

    if ctrl.use_variable_logK == 1:
        muki = ctrl.stru.muA + (ctrl.stru.muE - ctrl.stru.muA) * rng.random()
    else:
        muki = (ctrl.stru.muA + ctrl.stru.muE) / 2

    if ctrl.use_variable_logK_variance == 1:
        model.variance = ctrl.stru.variA + (ctrl.stru.variE - ctrl.stru.variA) * rng.random()
    else:
        model.variance = (ctrl.stru.variA + ctrl.stru.variE) / 2

    if ctrl.use_var_corr_length == 1:
        intex = ctrl.stru.intexA + (ctrl.stru.intexE - ctrl.stru.intexA) * rng.random()
        intey = ctrl.stru.inteyA + (ctrl.stru.inteyE - ctrl.stru.inteyA) * rng.random()
        model.lam = np.array([intey, intex])
        ctrl.lam = intey
    else:
        if ctrl.Two_D == 1:
            model.lam = np.array([(ctrl.stru.inteyA + ctrl.stru.inteyE) / 2,
                                  (ctrl.stru.intexA + ctrl.stru.intexE) / 2])
        else:
            model.lam = np.array([(ctrl.stru.inteyA + ctrl.stru.inteyE) / 2,
                                  (ctrl.stru.intexA + ctrl.stru.intexE) / 2,
                                  (ctrl.stru.intezA + ctrl.stru.intezE) / 2])
        ctrl.lam = model.lam[0]
    # alias para compatibilidad con dispersión RWPT (ctrl.lambda en MATLAB)
    ctrl.lambda_ = ctrl.lam

    if ctrl.use_var_shapefactor == 1:
        model.kappa = ctrl.stru.kapA + (ctrl.stru.kapE - ctrl.stru.kapA) * rng.random()
    else:
        model.kappa = (ctrl.stru.kapA + ctrl.stru.kapE) / 2

    model.micro = 0.0
    model.nugget = 0.0
    model.nbeta = 3
    model.beta = np.array([muki, 0.0, 0.0])
    model.Qbb = np.eye(model.nbeta) * 0.0625
    model.flag_kit = 0
    model.flag_zh = 0
    model.zh_smoother = 0.0
    model.maxprime = 7
    model.periodicity = np.zeros(grid.nd, dtype=int)

    # funciones de tendencia: constante + tendencias lineales x e y
    trend1 = np.ones(grid.npts)
    tx = grid.x_pts[1].reshape(-1, order="F")
    tx = tx / tx.max(); tx = tx - tx.mean()
    ty = grid.x_pts[0].reshape(-1, order="F")
    ty = ty / ty.max(); ty = ty - ty.mean()
    model.trends = np.column_stack([trend1, tx, ty])
    return ctrl, model


# ------------------------------------------------------------- covariance
def evaluate_covariance(model, h_eff):
    """evaluate_covariance.m"""
    h = np.asarray(h_eff, dtype=float)
    v = model.variance
    name = model.name
    if name == "nugget":
        Q = v * (h == 0)
    elif name == "exponential":
        Q = v * np.exp(-h)
    elif name == "gaussian":
        Q = v * np.exp(-h ** 2)
    elif name == "exppower":
        Q = v * np.exp(-h ** model.kappa)
    elif name == "spherical":
        Q = v * (1 - 1.5 * h + 0.5 * h ** 3)
        Q = np.where(h > 1, 0.0, Q)
    elif name == "cubic":
        Q = v * (1 - 7 * h ** 2 + 8.75 * h ** 3 - 3.5 * h ** 5 + 0.75 * h ** 7)
        Q = np.where(h > 1, 0.0, Q)
    elif name == "hole":
        Q = v * (1 - h) * np.exp(-h)
    elif name == "power":
        Q = v * (1 - np.maximum(h, 0)) ** model.kappa
    elif name == "cauchy":
        Q = v * (1 + h ** 2) ** (-model.kappa)
    elif name == "whittle":
        with np.errstate(invalid="ignore"):
            Q = v * h * besselk(1, h)
        Q = np.where(h == 0, v, Q)
    elif name == "matern":
        k = model.kappa
        arg = h * np.sqrt(k) * 2
        with np.errstate(invalid="ignore", divide="ignore", over="ignore"):
            Q = v / (2 ** (k - 1) * gamma_fn(k)) * arg ** k * besselk(k, arg)
        Q = np.where(h == 0, v, Q)
        Q = np.nan_to_num(Q, nan=0.0)
    else:
        raise ValueError(f"modelo geoestadístico no reconocido: {name}")

    if model.nugget != 0:
        Q = np.where(h == 0, Q + model.nugget, Q)
    return Q


def evaluate_separation(model, grid):
    """evaluate_separation.m — distancia efectiva anisótropa."""
    h2 = np.zeros(grid.x_pts[0].shape)
    lambda2 = np.maximum(model.lam, np.finfo(float).eps * np.asarray(grid.d_pts))
    for i in range(grid.nd):
        h2 += ((grid.x_pts[i] - grid.x_pts[i].min()) / lambda2[i]) ** 2
    return np.sqrt(h2 + model.micro ** 2) - model.micro


# ------------------------------------------------------------- embedding
def _factor(n):
    f, i = [], 2
    while i * i <= n:
        while n % i == 0:
            f.append(i); n //= i
        i += 1
    if n > 1:
        f.append(n)
    return f


def nicer_primes(numbers, pmax=7):
    """nicer_primes.m — números >= n con factores primos <= pmax."""
    numbers = np.atleast_1d(np.asarray(numbers, dtype=int))
    nicer = np.ones_like(numbers)
    for idx, n in np.ndenumerate(numbers):
        val, rest = 1, int(n)
        factors = _factor(rest)
        while True:
            nice = [f for f in factors if f <= pmax]
            bad = [f for f in factors if f > pmax]
            val *= int(np.prod(nice)) if nice else 1
            if not bad:
                break
            rest = int(np.prod(bad)) + 1
            factors = _factor(rest)
        nicer[idx] = val
    if pmax > 2:
        smaller = [p for p in (2, 3, 5, 7, 11) if p < pmax]
        if smaller:
            nicer = np.minimum(nicer, nicer_primes(numbers, smaller[-1]))
    return nicer


def find_embedding(model, grid):
    """find_embedding.m — tamaño mínimo del dominio embebido periódico."""
    minsize = 0.0
    minh_effe = float(np.min(np.asarray(grid.d_pts) / np.maximum(model.lam, np.finfo(float).eps)))
    noise = abs(1 - evaluate_covariance(model, minh_effe) / (model.variance + model.nugget))
    while True:
        mincorr = evaluate_covariance(model, minsize) / (model.variance + model.nugget)
        if mincorr < 1e-2 * noise + 1e-3:
            break
        minsize += minh_effe

    embed = NS()
    d_pts = np.asarray(grid.d_pts, dtype=float)
    embed.n_pts = np.ceil(np.maximum(grid.d_tot, minsize * model.lam) / d_pts
                          + minsize * model.lam / d_pts).astype(int)
    embed.n_pts = nicer_primes(embed.n_pts, model.maxprime)
    embed.d_pts = d_pts.copy()
    periodic = np.asarray(model.periodicity) == 1
    embed.n_pts = np.where(periodic, np.asarray(grid.n_pts), embed.n_pts)
    embed = ndgrid_setup(embed)

    # periodicidad del dominio embebido + suavizado microescala en el reflejo
    for i in range(grid.nd):
        embed.x_pts[i] = np.minimum(embed.x_pts[i], embed.d_tot[i] - embed.x_pts[i])
    for i in range(grid.nd):
        half = embed.d_tot[i] / 2
        embed.x_pts[i] = (np.sqrt(half ** 2 + model.lam[i] ** 2)
                          - np.sqrt((half - embed.x_pts[i]) ** 2 + model.lam[i] ** 2))
    return embed


def initialize_fft_cov(model, grid):
    """initialize_FFT_cov.m — FFTn de la primera fila embebida de la covarianza."""
    gride = find_embedding(model, grid)
    h_effe = evaluate_separation(model, gride)
    Qe = evaluate_covariance(model, h_effe)
    return np.abs(fftn(Qe.reshape(tuple(gride.n_pts), order="F")))


# ------------------------------------------------------------- generation
def generate_randomfield(model, grid, rng=None, fftqe=None):
    """generate_randomfield.m (método Dietrich & Newsam, flag_kit=0).

    Devuelve (Y, beta, FFTQe); Y con forma grid.n_pts (orden y,x[,z]).
    Funciona idéntico en 2D y 3D.
    """
    rng = rng or np.random.default_rng()
    if fftqe is None:
        fftqe = initialize_fft_cov(model, grid)
    npts_e = fftqe.size
    sqrt_qlambda = np.sqrt(fftqe / npts_e)

    for _attempt in range(100):   # tope de reintentos (evita cuelgue si un sorteo degenera)
        if model.flag_kit == 0:  # Dietrich & Newsam
            eps_c = (rng.standard_normal(fftqe.shape)
                     + 1j * rng.standard_normal(fftqe.shape))
            Ye = np.real(ifftn(eps_c * sqrt_qlambda)) * npts_e
        else:  # Kitanidis
            eps_c = np.exp(1j * np.angle(fftn(rng.standard_normal(fftqe.shape))))
            eps_c.flat[0] = 0
            Ye = np.real(ifftn(eps_c * sqrt_qlambda)) * npts_e

        if model.flag_zh > 0:
            from scipy.special import erf, erfinv
            Ye = Ye / np.sqrt(model.variance)
            sgn = -1.0 if model.flag_zh == 1 else 1.0
            Ye = sgn * erfinv(2 * (1 - model.zh_smoother) * erf(np.abs(Ye) * np.sqrt(0.5))
                              - (1 - model.zh_smoother)) * np.sqrt(2)
            Ye = Ye * np.sqrt(model.variance)

        # extracción del campo original del campo embebido
        slicer = tuple(slice(0, n) for n in grid.n_pts)
        Y = Ye[slicer]

        # media (incierta) + tendencias
        if np.linalg.det(np.atleast_2d(model.Qbb)) == 0:
            beta = np.asarray(model.beta, dtype=float)
        else:
            L = np.linalg.cholesky(np.atleast_2d(model.Qbb))
            beta = np.asarray(model.beta, dtype=float) + L @ rng.standard_normal(model.nbeta)
        Y = Y + (model.trends @ beta).reshape(tuple(grid.n_pts), order="F")

        if not np.isnan(Y).any() and not np.isinf(Y).any():
            return Y, beta, fftqe
    raise RuntimeError("generate_randomfield: campo con NaN/Inf tras 100 intentos "
                       "(revisa los parámetros geoestadísticos del PANEL)")


def uncond_homo_ksim(ctrl, grid, rng=None):
    """uncond_homo_Ksim.m — campo homogéneo (log-K constante)."""
    rng = rng or np.random.default_rng()
    if ctrl.use_variable_logK == 1:
        mui = ctrl.stru.muA + (ctrl.stru.muE - ctrl.stru.muA) * rng.random()
    else:
        mui = (ctrl.stru.muA + ctrl.stru.muE) / 2
    return np.full(int(np.prod(grid.n_pts)), mui), mui


# ------------------------------------------------- conditional generation
def _cov_obs(model, grid, obs_idx):
    """Covarianza C(x_obs, ·) sobre toda la malla y C(x_obs, x_obs),
    evaluada con el mismo modelo anisótropo de evaluate_covariance.

    obs_idx: índices lineales (order='F') sobre la malla de elementos.
    Para <=100 observaciones la evaluación directa es más simple y barata
    que la superposición FFT del kriging MATLAB (options.superpos='fft');
    el resultado es el mismo (misma función de covarianza).
    """
    coords = np.stack([x.reshape(-1, order="F") for x in grid.x_pts], axis=1)
    lam2 = np.maximum(model.lam, np.finfo(float).eps * np.asarray(grid.d_pts))
    obs_xy = coords[obs_idx]                       # (n_obs, nd)
    nobs, nel, nd = obs_xy.shape[0], coords.shape[0], coords.shape[1]
    # distancias efectivas anisótropas acumulando por dimensión: evita el array
    # intermedio (n_obs, nel, nd) que a 450x450x4 pesa varios GB (causa de cuelgues).
    d_og2 = np.zeros((nobs, nel))
    d_oo2 = np.zeros((nobs, nobs))
    for k in range(nd):
        og = (obs_xy[:, k][:, None] - coords[None, :, k]) / lam2[k]
        d_og2 += og * og
        oo = (obs_xy[:, k][:, None] - obs_xy[None, :, k]) / lam2[k]
        d_oo2 += oo * oo
    h_og = np.sqrt(d_og2 + model.micro ** 2) - model.micro
    h_oo = np.sqrt(d_oo2 + model.micro ** 2) - model.micro
    return evaluate_covariance(model, h_og), evaluate_covariance(model, h_oo)


def generate_conditional_field(model, grid, obs_idx, obs_values, obs_error,
                               rng=None, fftqe=None, cov_cache=None):
    """Realización condicionada a observaciones de log-K — puerto del método
    de Generar_K_Field_condicionados.m (condicionamiento por kriging de
    residuos aleatorizados; Journel):

        1. Y_uc  = realización incondicional (espectral, como siempre)
        2. d_j   = y_j − Y_uc(x_j) + ε_j,   ε_j ~ N(0, obs_error)
        3. ξ     = (C_oo + obs_error·I)⁻¹ d        (kriging simple, media 0)
        4. Y_c   = Y_uc + C_·o ξ

    obs_idx    : índices lineales (order='F') de los elementos observados
    obs_values : valores de log-K medidos (n_obs,)
    obs_error  : varianza del error de medición (escalar; y.error del .m)
    cov_cache  : (C_og, C_oo) precalculadas para reusar entre realizaciones
                 del mismo modelo/diseño (la reposición genera cientos).

    Devuelve (Y_cond (n_pts), fftqe, cov_cache).
    """
    rng = rng or np.random.default_rng()
    obs_idx = np.asarray(obs_idx, dtype=int)
    obs_values = np.asarray(obs_values, dtype=float)

    Y_uc, _, fftqe = generate_randomfield(model, grid, rng=rng, fftqe=fftqe)
    if cov_cache is None:
        cov_cache = _cov_obs(model, grid, obs_idx)
    C_og, C_oo = cov_cache

    d = obs_values - Y_uc.reshape(-1, order="F")[obs_idx] \
        + rng.standard_normal(obs_idx.size) * np.sqrt(obs_error)
    A = C_oo + obs_error * np.eye(obs_idx.size)
    ksi = np.linalg.solve(A, d)
    estimate = (ksi @ C_og).reshape(tuple(grid.n_pts), order="F")
    return Y_uc + estimate, fftqe, cov_cache


def conditional_ensemble(model, grid, obs_idx, obs_values, obs_error,
                         n_real, rng=None, randomize_structure=None):
    """Reposición del ensemble (arquitectura B del lazo RL+PreDIA): genera
    n_real campos condicionados a las muestras acumuladas.

    randomize_structure: callable opcional model->model que aleatoriza los
    parámetros estructurales por realización (como el .m, que sorteaba
    lambda y kappa); None usa el modelo fijo y cachea covarianzas/FFT.
    """
    rng = rng or np.random.default_rng()
    out = np.empty((n_real, int(np.prod(grid.n_pts))), dtype=np.float32)
    fftqe = cov_cache = None
    for i in range(n_real):
        m_i = randomize_structure(model) if randomize_structure else model
        if randomize_structure:           # estructura nueva => caches inválidas
            fftqe = cov_cache = None
        Y_c, fftqe, cov_cache = generate_conditional_field(
            m_i, grid, obs_idx, obs_values, obs_error, rng, fftqe, cov_cache)
        out[i] = Y_c.reshape(-1, order="F")
    return out


def uncsim_kfield(ctrl, model, grid, rng=None):
    """UncSim_Kfield_calculation.m (rutas het=0/1, sin het_geo ni map_1).

    Si ctrl.cond.Kflag == 1 (y het == 1), el campo se condiciona a las muestras
    ctrl.cond.{obs_idx, yobs} por kriging (cond_hetero_simu.m) en vez de generar
    una realización incondicional.
    """
    if ctrl.het == 1 and ctrl.het_geo == 0:
        cond = getattr(ctrl, "cond", None)
        if (cond is not None and int(getattr(cond, "Kflag", 0)) == 1
                and getattr(cond, "obs_idx", None) is not None):
            # reusa C_og/C_oo y la FFT de covarianza entre realizaciones del
            # ensamble (se calculan una sola vez; clave para no saturar RAM a 450²)
            Y_c, fftqe, cache = generate_conditional_field(
                model, grid, cond.obs_idx, cond.yobs, cond.r_K, rng=rng,
                fftqe=getattr(cond, "fftqe", None),
                cov_cache=getattr(cond, "cov_cache", None))
            cond.fftqe = fftqe
            cond.cov_cache = cache
            return Y_c.reshape(-1, order="F")
        Y, _, _ = generate_randomfield(model, grid, rng=rng)
        return Y.reshape(-1, order="F")          # vector log-K por elemento
    if ctrl.het == 0 and ctrl.het_geo == 0:
        yk, _ = uncond_homo_ksim(ctrl, grid, rng=rng)
        return yk
    raise NotImplementedError(
        "het_geo=1 (campos geológicos) y map_1=1 (cargar .mat) no migrados")
