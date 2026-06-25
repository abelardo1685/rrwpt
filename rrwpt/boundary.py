"""Condiciones de frontera — espejo de:
  SHARED/select_plane.m, SHARED/select_plane_nocorner.m,
  INIT/init_BC_RRWPT.m, Auxiliars/Change_BC.m, FEM_Dtensor/FEM_BC_2d.m.

Los índices devueltos son 0-based sobre la numeración global de nodos
(column-major, secuencia y,x,z), equivalentes a los de MATLAB menos 1.
"""
import numpy as np

from .grid_mod import sub2ind


def select_plane(n_pts, y=None, x=None, z=None):
    """select_plane.m — nodos de un plano de frontera (y/x/z = 0 ó 1)."""
    n_ = list(int(v) for v in n_pts)
    if len(n_) == 2:
        n_ = n_ + [1]
    ny, nx, nz = n_
    shape = (ny, nx, nz)

    if y == 0:
        ys = np.zeros((nz, nx), dtype=int)
        xs = np.tile(np.arange(nx), (nz, 1))
        zs = np.tile(np.arange(nz)[:, None], (1, nx))
    elif y == 1:
        ys = np.full((nz, nx), ny - 1, dtype=int)
        xs = np.tile(np.arange(nx), (nz, 1))
        zs = np.tile(np.arange(nz)[:, None], (1, nx))
    elif x == 0:
        if len(n_pts) == 2:
            ys = np.tile(np.arange(ny), (nz, 1))
            xs = np.zeros((nz, ny), dtype=int)
            zs = np.tile(np.arange(nz)[:, None], (1, ny))
        else:  # rama 3D de select_plane.m (orden de recorrido distinto)
            ys = np.tile(np.arange(ny)[:, None], (1, nz))
            xs = np.zeros((ny, nz), dtype=int)
            zs = np.tile(np.arange(nz), (ny, 1))
    elif x == 1:
        if len(n_pts) == 2:
            ys = np.tile(np.arange(ny), (nz, 1))
            xs = np.full((nz, ny), nx - 1, dtype=int)
            zs = np.tile(np.arange(nz)[:, None], (1, ny))
        else:
            ys = np.tile(np.arange(ny)[:, None], (1, nz))
            xs = np.full((ny, nz), nx - 1, dtype=int)
            zs = np.tile(np.arange(nz), (ny, 1))
    elif z == 0:
        ys = np.tile(np.arange(ny), (nx, 1))
        xs = np.tile(np.arange(nx)[:, None], (1, ny))
        zs = np.zeros((nx, ny), dtype=int)
    elif z == 1:
        ys = np.tile(np.arange(ny), (nx, 1))
        xs = np.tile(np.arange(nx)[:, None], (1, ny))
        zs = np.full((nx, ny), nz - 1, dtype=int)
    else:
        raise ValueError("select_plane: especificar y, x o z en {0,1}")

    return sub2ind(shape, ys, xs, zs).reshape(-1)


def select_plane_nocorner(n_pts, y=None, x=None, z=None):
    """select_plane_nocorner.m — en 2D coincide con select_plane (la rama 2D
    del archivo MATLAB es idéntica); en 3D excluye el doble conteo usando el
    recorrido alternativo del archivo original."""
    return select_plane(n_pts, y, x, z)


def init_bc(grid, ctrl, flowpar):
    """init_BC_RRWPT.m — pozos + Dirichlet para h (solo cantidades usadas
    por el núcleo migrado: well_pts, fix_pts_h, fix_value_h, in_*_h)."""
    n_pts = grid.data.n_pts
    n_el = grid.data.n_el

    welly_el = np.array([round(n_el[0] * ctrl.inP[1])])
    wellx_el = np.array([round(n_el[1] * ctrl.inP[0])])
    if ctrl.Ex_pump == 1:
        welly_el = np.concatenate([welly_el, np.round(n_el[0] * ctrl.Ex_inP[:, 1]).astype(int)])
        wellx_el = np.concatenate([wellx_el, np.round(n_el[1] * ctrl.Ex_inP[:, 0]).astype(int)])

    if len(n_el) == 2:
        # MATLAB: sub2ind(n_el+1, welly_el, wellx_el); índices aquí 0-based
        well_pts = sub2ind(tuple(n_pts), welly_el - 1, wellx_el - 1)[:, None]
    else:
        wellz_el = np.arange(1, n_el[2] + 2)  # pozo totalmente penetrante
        well_pts = np.zeros((len(welly_el), len(wellz_el)), dtype=int)
        for i in range(len(welly_el)):
            well_pts[i, :] = sub2ind(tuple(n_pts),
                                     np.full(len(wellz_el), welly_el[i] - 1),
                                     np.full(len(wellz_el), wellx_el[i] - 1),
                                     wellz_el - 1)

    left = select_plane(n_pts, x=0)
    right = select_plane(n_pts, x=1)
    fix_pts_h = np.concatenate([left, right])
    # init_BC_RRWPT.m: dh*Lx/100 en x=0, 0 en x=1
    fix_value_h = np.concatenate([
        np.full(left.size, flowpar.dh * grid.n_pts[1] * grid.d_pts[1] / 100),
        np.zeros(right.size),
    ])

    ctrl.bc.well_pts = well_pts
    ctrl.bc.fix_pts_h = fix_pts_h
    ctrl.bc.fix_value_h = fix_value_h
    ctrl.bc.in_pts_h = np.array([], dtype=int)
    ctrl.bc.in_el_h = np.array([], dtype=int)
    ctrl.bc.in_value_h = np.array([])
    return ctrl


def change_bc(varpar, grid, ctrl, n_pts):
    """Change_BC.m -> FEM_BC_2d.m — Dirichlet según dirección de flujo.

    HeadDir==0:   gradiente x (izquierda alta, derecha 0), valor dh*Lx (sin /100).
    HeadDir==90:  gradiente y (arriba alto), valor dh*Ly.
    HeadDir==270: gradiente y invertido.
    """
    dh = varpar.dh
    if varpar.HeadDir == 0:
        pts = np.concatenate([select_plane(n_pts, x=0), select_plane(n_pts, x=1)])
        nl = select_plane(n_pts, x=0).size
        vals = np.concatenate([
            np.full(nl, dh * grid.n_pts[1] * grid.d_pts[1]),
            np.zeros(select_plane(n_pts, x=1).size),
        ])
    elif varpar.HeadDir == 90:
        top = select_plane_nocorner(n_pts, y=0)
        bot = select_plane_nocorner(n_pts, y=1)
        pts = np.concatenate([top, bot])
        vals = np.concatenate([
            np.full(top.size, dh * grid.n_pts[0] * grid.d_pts[1]),
            np.zeros(bot.size),
        ])
    elif varpar.HeadDir == 270:
        top = select_plane_nocorner(n_pts, y=0)
        bot = select_plane_nocorner(n_pts, y=1)
        pts = np.concatenate([top, bot])
        vals = np.concatenate([
            np.zeros(top.size),
            np.full(bot.size, dh * grid.n_pts[0] * grid.d_pts[1]),
        ])
    else:
        raise ValueError("change_bc: HeadDir de referencia debe ser 0, 90 o 270")
    return pts, vals
