"""Núcleo FEM para flujo — espejo de:
  FEM_Dtensor/{FEM_matrix_stiffness, FEM_matrix_mass, FEM_stiff_h,
               FEM_source_infl, FEM_q, FEM_Vector_qp_3d,
               FEM_preparation_recharge_dimension}.m
  y la rama href0 de FEM_kernel_Dtensor_quasiStat.m.

Decisión de diseño (documentada en docs/MIGRACION.md): el solver multigrid
bicgmg.m de MATLAB se sustituye por scipy.sparse.linalg.spsolve (equivalente
al modo 'umf'/UMFPACK que el propio init_SOLVERS.m ofrecía como alternativa).
Se resuelve exactamente el mismo sistema lineal; la diferencia es solo el
algoritmo de solución, no la física.
"""
import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import spsolve

from .grid_mod import sub2ind

try:
    import pyamg
    HAS_PYAMG = True
except ImportError:
    HAS_PYAMG = False

# Tolerancia equivalente a FEMresh de init_SOLVERS.m (bicgmg iteraba a 1e-10)
AMG_TOL = 1e-10

# Caché de sistemas ensamblados/factorizados: la fase de referencia resuelve
# 12+ lados derechos con la MISMA matriz (un par por pozo); reutilizar la
# jerarquía AMG (o la LU) reduce las 17 resolvelas a ~5 setups.
_SYS_CACHE = {}
_SYS_CACHE_MAX = 6


def clear_solver_cache():
    _SYS_CACHE.clear()


# --------------------------------------------------------- element matrices
def fem_matrix_stiffness(el_len, nd):
    """FEM_matrix_stiffness.m — matriz de rigidez del elemento (2D/3D)."""
    if nd == 2:
        M = (np.array([[+2, +1, -2, -1],
                       [+1, +2, -1, -2],
                       [-2, -1, +2, +1],
                       [-1, -2, +1, +2]], dtype=float) * el_len[0] / el_len[1] / 6)
        M += (np.array([[+2, -2, +1, -1],
                        [-2, +2, -1, +1],
                        [+1, -1, +2, -2],
                        [-1, +1, -2, +2]], dtype=float) * el_len[1] / el_len[0] / 6)
        return M
    A = np.array([[4, 2, -4, -2, 2, 1, -2, -1],
                  [2, 4, -2, -4, 1, 2, -1, -2],
                  [-4, -2, 4, 2, -2, -1, 2, 1],
                  [-2, -4, 2, 4, -1, -2, 1, 2],
                  [2, 1, -2, -1, 4, 2, -4, -2],
                  [1, 2, -1, -2, 2, 4, -2, -4],
                  [-2, -1, 2, 1, -4, -2, 4, 2],
                  [-1, -2, 1, 2, -2, -4, 2, 4]], dtype=float)
    B = np.array([[4, -4, 2, -2, 2, -2, 1, -1],
                  [-4, 4, -2, 2, -2, 2, -1, 1],
                  [2, -2, 4, -4, 1, -1, 2, -2],
                  [-2, 2, -4, 4, -1, 1, -2, 2],
                  [2, -2, 1, -1, 4, -4, 2, -2],
                  [-2, 2, -1, 1, -4, 4, -2, 2],
                  [1, -1, 2, -2, 2, -2, 4, -4],
                  [-1, 1, -2, 2, -2, 2, -4, 4]], dtype=float)
    C = np.array([[4, 2, 2, 1, -4, -2, -2, -1],
                  [2, 4, 1, 2, -2, -4, -1, -2],
                  [2, 1, 4, 2, -2, -1, -4, -2],
                  [1, 2, 2, 4, -1, -2, -2, -4],
                  [-4, -2, -2, -1, 4, 2, 2, 1],
                  [-2, -4, -1, -2, 2, 4, 1, 2],
                  [-2, -1, -4, -2, 2, 1, 4, 2],
                  [-1, -2, -2, -4, 1, 2, 2, 4]], dtype=float)
    return (A * el_len[0] * el_len[2] / el_len[1] / 36
            + B * el_len[1] * el_len[2] / el_len[0] / 36
            + C * el_len[0] * el_len[1] / el_len[2] / 36)


def fem_matrix_mass(el_len, nd):
    """FEM_matrix_mass.m — matriz de masa (casos transitorios)."""
    if nd == 2:
        S = np.array([[4, 2, 2, 1],
                      [2, 4, 1, 2],
                      [2, 1, 4, 2],
                      [1, 2, 2, 4]], dtype=float) * el_len[0] * el_len[1] / 36
        return S
    S = np.array([[8, 4, 4, 2, 4, 2, 2, 1],
                  [4, 8, 2, 4, 2, 4, 1, 2],
                  [4, 2, 8, 4, 2, 1, 4, 2],
                  [2, 4, 4, 8, 1, 2, 2, 4],
                  [4, 2, 2, 1, 8, 4, 4, 2],
                  [2, 4, 1, 2, 4, 8, 2, 4],
                  [2, 1, 4, 2, 4, 2, 8, 4],
                  [1, 2, 2, 4, 2, 4, 4, 8]], dtype=float)
    return S * el_len[0] * el_len[1] * el_len[2] / 216


# --------------------------------------------------------------- assembly
def fem_stiff_h(K, fix_pts_h, grid):
    """FEM_stiff_h.m — ensamblaje de la matriz global de rigidez.

    Devuelve (M_h_mod, M_h_d): la matriz modificada para Dirichlet
    (filas/columnas de nodos fijos a 0, diagonal 1) y las columnas de los
    nodos Dirichlet antes de modificar (para el término r_h_mod).
    No se ensambla I_h (in_el_h vacío en todo el flujo migrado).
    """
    d = grid.data
    K = np.asarray(K, dtype=float).reshape(-1)
    nd = len(d.n_el)
    M_el = fem_matrix_stiffness(d.el_len, nd)
    elpts = d.elpts
    npts = d.npts

    rows, cols, vals = [], [], []
    for i in range(elpts):
        for j in range(elpts):
            if M_el[i, j] != 0:
                rows.append(d.el2pts + d.incidence_el[i])
                cols.append(d.el2pts + d.incidence_el[j])
                vals.append(M_el[i, j] * K)
    rows = np.concatenate(rows)
    cols = np.concatenate(cols)
    vals = np.concatenate(vals)
    M_h = sp.coo_matrix((vals, (rows, cols)), shape=(npts, npts)).tocsr()

    M_h_d = M_h[:, fix_pts_h].copy()

    # eliminación simétrica de Dirichlet
    mask = np.zeros(npts, dtype=bool)
    mask[fix_pts_h] = True
    M = M_h.tocoo()
    keep = ~(mask[M.row] | mask[M.col])
    M_h_mod = sp.coo_matrix((M.data[keep], (M.row[keep], M.col[keep])),
                            shape=(npts, npts)).tocsr()
    diag = sp.coo_matrix((np.ones(fix_pts_h.size),
                          (fix_pts_h, fix_pts_h)), shape=(npts, npts))
    M_h_mod = (M_h_mod + diag).tocsc()
    return M_h_mod, M_h_d


def fem_q(K, h, grid):
    """FEM_q.m — descarga específica nodal (promedio de contribuciones)."""
    d = grid.data
    K = np.asarray(K, dtype=float).reshape(-1)
    h = np.asarray(h, dtype=float).reshape(-1)
    npts = d.npts
    inc = d.incidence_el
    e0 = d.el2pts
    el_len = d.el_len

    qx = np.zeros(npts)
    qy = np.zeros(npts)
    qz = np.zeros(npts)
    counter = np.zeros(npts)

    def acc(arr, node_off, dh, length):
        np.add.at(arr, e0 + inc[node_off], -dh * K / length)

    if len(d.n_el) == 2:
        dhx13 = h[e0 + inc[2]] - h[e0 + inc[0]]
        dhx24 = h[e0 + inc[3]] - h[e0 + inc[1]]
        for n, dh in ((0, dhx13), (1, dhx24), (2, dhx13), (3, dhx24)):
            acc(qx, n, dh, el_len[1])
        dhy12 = h[e0 + inc[1]] - h[e0 + inc[0]]
        dhy34 = h[e0 + inc[3]] - h[e0 + inc[2]]
        for n, dh in ((0, dhy12), (1, dhy12), (2, dhy34), (3, dhy34)):
            acc(qy, n, dh, el_len[0])
        for n in range(4):
            np.add.at(counter, e0 + inc[n], 1.0)
    else:
        pairs_x = ((0, 2), (1, 3), (2, 2), (3, 3), (4, 6), (5, 7), (6, 6), (7, 7))
        base_x = {2: (2, 0), 3: (3, 1), 6: (6, 4), 7: (7, 5)}
        for n, hi in pairs_x:
            hh, ll = base_x[hi]
            acc(qx, n, h[e0 + inc[hh]] - h[e0 + inc[ll]], el_len[1])
        pairs_y = ((0, 1), (1, 1), (2, 3), (3, 3), (4, 5), (5, 5), (6, 7), (7, 7))
        base_y = {1: (1, 0), 3: (3, 2), 5: (5, 4), 7: (7, 6)}
        for n, hi in pairs_y:
            hh, ll = base_y[hi]
            acc(qy, n, h[e0 + inc[hh]] - h[e0 + inc[ll]], el_len[0])
        pairs_z = ((0, 4), (4, 4), (1, 5), (5, 5), (2, 6), (6, 6), (3, 7), (7, 7))
        base_z = {4: (4, 0), 5: (5, 1), 6: (6, 2), 7: (7, 3)}
        for n, hi in pairs_z:
            hh, ll = base_z[hi]
            acc(qz, n, h[e0 + inc[hh]] - h[e0 + inc[ll]], el_len[2])
        for n in range(8):
            np.add.at(counter, e0 + inc[n], 1.0)

    counter[counter == 0] = 1.0
    return np.column_stack([qx / counter, qy / counter, qz / counter])


def fem_vector_qp_3d(ctrl, Qp_ii):
    """FEM_Vector_qp_3d.m — distribución del caudal en la columna del pozo."""
    n_nodes = ctrl.bc.well_pts.shape[1]
    discharge = Qp_ii / n_nodes
    qp_vector = np.full(n_nodes, 2.0 * discharge)
    qp_vector[0] = discharge
    qp_vector[-1] = discharge
    return qp_vector


def _solve(M_mod, r_h):
    return spsolve(M_mod, r_h)


def _get_system(K, fix_pts_h, grid):
    """Ensambla (o recupera de caché) el sistema y su solver.

    Devuelve (M_h_d, solve) donde solve(r) resuelve M_h_mod·h = r con AMG+CG
    a tolerancia AMG_TOL (equivalente numérico de bicgmg.m) o, sin pyamg,
    con factorización LU reutilizable.
    """
    key = (hash(K.tobytes()), hash(np.ascontiguousarray(fix_pts_h).tobytes()),
           int(grid.data.npts))
    if key in _SYS_CACHE:
        return _SYS_CACHE[key]

    M_h_mod, M_h_d = fem_stiff_h(K, fix_pts_h, grid)
    if HAS_PYAMG:
        ml = pyamg.smoothed_aggregation_solver(M_h_mod.tocsr(), max_coarse=500)

        def solve(r, _ml=ml):
            return _ml.solve(r, tol=AMG_TOL, accel="cg")
    else:
        lu = sp.linalg.splu(M_h_mod.tocsc())

        def solve(r, _lu=lu):
            return _lu.solve(r)

    if len(_SYS_CACHE) >= _SYS_CACHE_MAX:
        _SYS_CACHE.pop(next(iter(_SYS_CACHE)))
    _SYS_CACHE[key] = (M_h_d, solve)
    return M_h_d, solve


# -------------------------------------------------- reference head scenarios
def solve_reference_head(K, varpar, grid, ctrl):
    """Rama href0==1 de FEM_kernel_Dtensor_quasiStat.m.

    Escenarios según (use_well, use_gwn):
      (0,0) flujo base    -> lim=1, Dirichlet de change_bc
      (1,0) bombeo        -> lim=2, Dirichlet=0, fuente puntual Qp(ii)
      (0,1) recarga       -> lim=2, Dirichlet=0, fuente distribuida qr(ii)
    Devuelve dict con h, qx, qy, qz de tamaño (npts, lim).
    """
    from .boundary import change_bc

    d = grid.data
    npts = d.npts
    n_pts = d.n_pts
    K = np.asarray(K, dtype=float).reshape(-1)

    fix_pts_h, fix_value_h = change_bc(varpar, grid, ctrl, n_pts)

    use_well = ctrl.use_well
    use_gwn = ctrl.use_gwn
    if (use_well and not use_gwn) or (not use_well and use_gwn):
        lim = 2
    else:
        lim = 1

    h = np.zeros((npts, lim))
    qx = np.zeros((npts, lim))
    qy = np.zeros((npts, lim))
    qz = np.zeros((npts, lim))

    # caso especial: recarga 3D con slice extra (FEM_preparation_recharge_dimension.m)
    if (not use_well) and use_gwn and ctrl.Two_D == 0:
        for ii in range(lim):
            h[:, ii] = _recharge_3d_head(ctrl, K, np.atleast_1d(varpar.qr)[ii], grid)
            q = fem_q(K, h[:, ii], grid)
            qx[:, ii], qy[:, ii], qz[:, ii] = q[:, 0], q[:, 1], q[:, 2]
        return {"h": h, "qx": qx, "qy": qy, "qz": qz}

    M_h_d, solve = _get_system(K, fix_pts_h, grid)
    hinfl = np.zeros(npts)  # in_el_h vacío => FEM_source_infl devuelve 0

    for ii in range(lim):
        if use_well and not use_gwn:
            # bombeo: drawdown cero en fronteras
            fv = np.zeros_like(fix_value_h)
            r_h_mod = M_h_d @ fv
            ptsrc = np.zeros(npts)
            wp = np.atleast_2d(ctrl.bc.well_pts)[ctrl.num_well]
            if ctrl.Two_D == 0:
                ptsrc[wp] = fem_vector_qp_3d(ctrl, np.atleast_1d(varpar.Qp)[ii])
                wellsource = ptsrc
            else:
                ptsrc[wp] = 1.0
                wellsource = ptsrc * np.atleast_1d(varpar.Qp)[ii] / wp.size
            r_h = hinfl - r_h_mod - wellsource
            r_h[fix_pts_h] = fv
        elif (not use_well) and use_gwn:
            # recarga 2D homogénea
            fv = np.zeros_like(fix_value_h)
            r_h_mod = M_h_d @ fv
            qr_ii = np.atleast_1d(varpar.qr)[ii]
            gwn = (np.ones(npts) * qr_ii * np.prod(grid.d_pts) * 0.25 / 1000.0) / (86400.0 * 365.0)
            r_h = hinfl - r_h_mod + gwn
            r_h[fix_pts_h] = fv
        else:
            # flujo base de fondo
            r_h_mod = M_h_d @ fix_value_h
            r_h = hinfl - r_h_mod
            r_h[fix_pts_h] = fix_value_h

        h_sim = solve(r_h)
        q = fem_q(K, h_sim, grid)
        h[:, ii] = h_sim
        qx[:, ii], qy[:, ii], qz[:, ii] = q[:, 0], q[:, 1], q[:, 2]

    return {"h": h, "qx": qx, "qy": qy, "qz": qz}


def _recharge_3d_head(ctrl, K, qr_val, grid):
    """FEM_preparation_recharge_dimension.m — recarga 3D vía slice adicional."""
    from types import SimpleNamespace as NS
    from .grid_mod import init_grid, init_fem

    ny, nx, nz = ctrl.n_pts_y, ctrl.n_pts_x, ctrl.n_pts_z
    ctrl2 = NS(**vars(ctrl))
    ctrl2.n_pts_z = nz + 1
    ctrl2.Two_D = 0
    g2 = init_fem(init_grid(ctrl2))
    n_pts2 = g2.data.n_pts

    K3 = K.reshape((ny, nx, nz), order="F")
    K_new = np.concatenate([K3, K3[:, :, :1]], axis=2)
    K2 = K_new.reshape(-1, order="F")

    from .boundary import select_plane
    fix_pts = np.concatenate([select_plane(n_pts2, x=0), select_plane(n_pts2, x=1)])
    fix_val = np.zeros(fix_pts.size)

    M_d, solve2 = _get_system(K2, fix_pts, g2)

    yy = np.tile(np.arange(n_pts2[0])[:, None], (1, n_pts2[1]))
    xx = np.tile(np.arange(n_pts2[1]), (n_pts2[0], 1))
    z0 = np.zeros_like(yy)
    list1 = sub2ind(tuple(n_pts2), yy, xx, z0).reshape(-1)
    list2 = sub2ind(tuple(n_pts2), yy, xx, z0 + 1).reshape(-1)
    list_rech = np.concatenate([list1, list2])

    gwn = np.zeros(g2.data.npts)
    volume = float(np.prod(g2.d_pts))
    gwn[list_rech] = (volume * qr_val * 0.125 / 1000.0) / (86400.0 * 365.0)
    gwn[fix_pts] = 0.0

    r_h = gwn - (M_d @ fix_val)
    r_h[fix_pts] = fix_val
    h_sim = solve2(r_h)

    cube = h_sim.reshape(tuple(n_pts2), order="F")
    return cube[:, :, 1:].reshape(-1, order="F")
