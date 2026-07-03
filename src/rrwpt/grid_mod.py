"""Malla regular y estructuras FEM — espejo de:
  INIT/init_gridrainer.m, SPECTRAL_GEOSTATS/ndgrid_setup.m, INIT/init_FEM.m.

Orden de numeración global de nodos: secuencia (y, x, z), column-major,
idéntico a MATLAB reshape(1:npts, n_pts). Internamente 0-based.
"""
from types import SimpleNamespace as NS

import numpy as np


def ndgrid_setup(grid):
    """ndgrid_setup.m: completa la malla con coordenadas y totales."""
    grid.n_pts = np.asarray(grid.n_pts, dtype=int)
    grid.d_pts = np.asarray(grid.d_pts, dtype=float)
    grid.npts = int(np.prod(grid.n_pts))
    grid.nd = len(grid.n_pts)
    grid.d_tot = grid.d_pts * grid.n_pts
    grid.x_vec = [np.arange(n) * d for n, d in zip(grid.n_pts, grid.d_pts)]
    grid.x_pts = list(np.meshgrid(*grid.x_vec, indexing="ij"))
    return grid


def init_grid(ctrl):
    """init_gridrainer.m (sin malla extendida: ctrl.use_ext_grid=0)."""
    grid = NS()
    if ctrl.Two_D == 0:
        grid.n_pts = [ctrl.n_pts_y, ctrl.n_pts_x, ctrl.n_pts_z]
        grid.d_pts = [ctrl.d_pts_y, ctrl.d_pts_x, ctrl.d_pts_z]
    else:
        grid.n_pts = [ctrl.n_pts_y, ctrl.n_pts_x]
        grid.d_pts = [ctrl.d_pts_y, ctrl.d_pts_x]
    grid = ndgrid_setup(grid)
    grid.extend = 0
    grid.data = NS()
    grid.data.domain_len = grid.d_tot.copy()
    grid.data.n_el = grid.n_pts.copy()
    return grid


def init_fem(grid):
    """init_FEM.m: indexación de nodos/elementos e incidencias.

    el2pts guarda, para cada elemento, el índice global (0-based) del nodo
    inferior-izquierdo-frontal; incidence_el son los offsets relativos a los
    2^nd nodos del elemento, numerados en secuencia y,x,z.
    """
    d = grid.data
    n_el = np.asarray(d.n_el, dtype=int)
    n_pts = n_el + 1
    npts = int(np.prod(n_pts))
    nel = int(np.prod(n_el))
    el_len = np.asarray(d.domain_len, dtype=float) / n_el

    d.n_pts = n_pts
    d.npts = npts
    d.nel = nel
    d.el_len = el_len
    d.d_pts = el_len
    d.V_el = float(np.prod(el_len))
    d.elpts = 2 ** len(n_el)
    d.elpt2 = d.elpts ** 2

    if len(n_el) == 2:
        ny, nx = n_pts
        # offsets de los 4 nodos: [0 1 0 1] + ny*[0 0 1 1]
        incidence_el = np.array([0, 1, 0, 1]) + ny * np.array([0, 0, 1, 1])
        all_pts = np.arange(npts).reshape(n_pts, order="F")
        el2pts = all_pts[: ny - 1, : nx - 1].reshape(nel, order="F")
        incidence_pts = np.concatenate([
            -ny + np.array([-1, 0, 1]),
            np.array([-1, 0, 1]),
            ny + np.array([-1, 0, 1]),
        ])
    elif len(n_el) == 3:
        ny, nx, nz = n_pts
        incidence_el = (np.array([0, 1, 0, 1, 0, 1, 0, 1])
                        + ny * np.array([0, 0, 1, 1, 0, 0, 1, 1])
                        + ny * nx * np.array([0, 0, 0, 0, 1, 1, 1, 1]))
        all_pts = np.arange(npts).reshape(n_pts, order="F")
        el2pts = all_pts[: ny - 1, : nx - 1, : nz - 1].reshape(nel, order="F")
        star2d = np.concatenate([
            -ny + np.array([-1, 0, 1]),
            np.array([-1, 0, 1]),
            ny + np.array([-1, 0, 1]),
        ])
        incidence_pts = np.concatenate([
            -nx * ny + star2d, star2d, nx * ny + star2d
        ])
    else:
        raise ValueError("init_fem: número de dimensiones no permitido")

    d.incidence_el = incidence_el
    d.all_pts = all_pts
    d.el2pts = el2pts
    d.incidence_pts = incidence_pts
    return grid


def sub2ind(shape, *subs):
    """Equivalente a sub2ind de MATLAB con índices 0-based, orden 'F'."""
    return np.ravel_multi_index(tuple(np.asarray(s) for s in subs),
                                tuple(shape), order="F")


def ind2sub(shape, ind):
    """Equivalente a ind2sub de MATLAB con índices 0-based, orden 'F'."""
    return np.unravel_index(np.asarray(ind), tuple(shape), order="F")
