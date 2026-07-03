"""(Reverse) Random Walk Particle Tracking y delineación — espejo de:
  FEM_Dtensor/RRWPT_tempcont.m  y  RW/{RW_Vel_direction_Calculation,
  RW_Dispersion, RW_dDD_Calculation, RW_B_Calculation, RW_Linear_Int_vel,
  RW_X_Calculation, RW_mirror_particles, RW_mirror_particles_Dispersion,
  RW_Bounce_particles_well_location, RW_Dispersion_index,
  RW_particle_3d_dim_size, RW_3D_cilinder_distribution}.m
  FEM_Dtensor/{particlepos, radxy, radflux, fluxconv}.m

Equivalencias documentadas (ver docs/MIGRACION.md):
 * El truco MATLAB de "translation in time" (inyectar todo el lote al inicio
   del QS y desplazar trayectorias en el tiempo) es matemáticamente idéntico,
   bajo cuasi-estacionariedad, a inyectar cada sub-lote en su TTI y rastrear
   desde ahí; aquí se usa la inyección escalonada directa.
 * Las trayectorias no se almacenan (YPt/Tt y los .mat de Data_RWPT): la
   huella binaria de cada lote se acumula en línea, que es lo único que el
   código MATLAB utiliza para la delineación (variable `plot` binaria).
 * MATLAB deja de inyectar partículas cuando t > t_crit/deltQS
   (X0W(:,:)=0 en RW_Particle_injection.m); esos lotes nunca contribuyen a
   los mapas, por lo que aquí simplemente no se inyectan.
"""
from types import SimpleNamespace as NS

import numpy as np
from scipy.interpolate import RegularGridInterpolator

from .grid_mod import ind2sub

# Réplica exacta del .m: la componente Bxx (2D) usa `at`; la teoría sugiere
# `al` (el propio archivo conserva una versión anterior comentada). Cambiar a
# True usa `al` — requiere validación del autor del modelo.
B_LONGITUDINAL_FIX = False


# ------------------------------------------------------------- velocidades
def vel_direction(ctrl, grid, qx, qy, qz):
    """RW_Vel_direction_Calculation.m — inversión (reverse) y matrices."""
    n1 = tuple(np.asarray(grid.n_pts) + 1)
    poro = ctrl.trns.n
    qx_m = (-qx).reshape(n1, order="F") / poro
    qy_m = (-qy).reshape(n1, order="F") / poro
    if ctrl.Two_D == 0:
        qz_m = (-qz).reshape(n1, order="F") / poro
    else:
        qz_m = np.zeros_like(qy_m)
    return qx_m, qy_m, qz_m


def make_interp(grid, field_m, ctrl):
    """Construye el interpolador bilineal/trilineal sobre la malla 0:λ:Nλ
    (interp2/interp3 de RW_Linear_Int_vel.m). Se crea una vez por QS."""
    axes = [np.arange(grid.n_pts[0] + 1) * grid.d_pts[0],
            np.arange(grid.n_pts[1] + 1) * grid.d_pts[1]]
    if ctrl.Two_D == 0:
        axes.append(np.arange(grid.n_pts[2] + 1) * grid.d_pts[2])
    return RegularGridInterpolator(tuple(axes), field_m, method="linear",
                                   bounds_error=False, fill_value=np.nan)


def _interp(grid, field_m, X, ctrl):
    """Versión de conveniencia (construye y evalúa)."""
    rgi = make_interp(grid, field_m, ctrl)
    return rgi(X[:, :3] if ctrl.Two_D == 0 else X[:, :2])


# ------------------------------------------------------------- dispersión
def dispersion_grids(ctrl, grid, qx_m, qy_m, qz_m):
    """RW_Dispersion.m (parte 1) — gradientes del tensor de dispersión."""
    al, at, Dm, poro = ctrl.trns.al, ctrl.trns.at, ctrl.trns.Dm, ctrl.trns.n
    lam = (ctrl.d_pts_y, ctrl.d_pts_x, ctrl.d_pts_z)
    with np.errstate(invalid="ignore", divide="ignore"):
        if ctrl.Two_D == 1:
            absv = np.sqrt(qx_m ** 2 + qy_m ** 2)
            D_diag = at * absv + Dm / poro
            D_1 = (qy_m * qy_m * (al - at)) / absv + D_diag
            D_2 = (qy_m * qx_m * (al - at)) / absv
            D_3 = (qx_m * qy_m * (al - at)) / absv
            D_4 = (qx_m * qx_m * (al - at)) / absv + D_diag
            for D in (D_1, D_2, D_3, D_4):
                np.nan_to_num(D, copy=False)
            dDxx = np.zeros_like(qx_m); dDxy = np.zeros_like(qx_m)
            dDyy = np.zeros_like(qx_m); dDyx = np.zeros_like(qx_m)
            dDxx[:, 1:-1] = (D_3[:, 2:] - D_3[:, :-2]) / (2 * lam[1])
            dDxy[:, 1:-1] = (D_4[:, 2:] - D_4[:, :-2]) / (2 * lam[1])
            dDyy[1:-1, :] = (D_1[2:, :] - D_1[:-2, :]) / (2 * lam[0])
            dDyx[1:-1, :] = (D_2[2:, :] - D_2[:-2, :]) / (2 * lam[0])
            return NS(dDyy=dDyy, dDyx=dDyx, dDxy=dDxy, dDxx=dDxx)

        absv = np.sqrt(qx_m ** 2 + qy_m ** 2 + qz_m ** 2)
        D_diag = at * absv + Dm / poro
        D = {}
        pairs = {1: (qy_m, qy_m, True), 2: (qy_m, qx_m, False), 3: (qy_m, qz_m, False),
                 4: (qx_m, qy_m, False), 5: (qx_m, qx_m, True), 6: (qx_m, qz_m, False),
                 7: (qz_m, qy_m, False), 8: (qz_m, qx_m, False), 9: (qz_m, qz_m, True)}
        for k, (a, b, diag) in pairs.items():
            Dk = (a * b * (al - at)) / absv
            if diag:
                Dk = Dk + D_diag
            D[k] = np.nan_to_num(Dk)
        z = np.zeros_like(qx_m)
        g = NS(dDxx=z.copy(), dDxy=z.copy(), dDxz=z.copy(),
               dDyy=z.copy(), dDyx=z.copy(), dDyz=z.copy(),
               dDzy=z.copy(), dDzx=z.copy(), dDzz=z.copy())
        g.dDxx[:, 1:-1, :] = (D[5][:, 2:, :] - D[5][:, :-2, :]) / (2 * lam[1])
        g.dDxy[:, 1:-1, :] = (D[2][:, 2:, :] - D[2][:, :-2, :]) / (2 * lam[1])
        g.dDxz[:, 1:-1, :] = (D[8][:, 2:, :] - D[8][:, :-2, :]) / (2 * lam[1])
        g.dDyy[1:-1, :, :] = (D[1][2:, :, :] - D[1][:-2, :, :]) / (2 * lam[0])
        g.dDyx[1:-1, :, :] = (D[4][2:, :, :] - D[4][:-2, :, :]) / (2 * lam[0])
        g.dDyz[1:-1, :, :] = (D[7][2:, :, :] - D[7][:-2, :, :]) / (2 * lam[0])
        g.dDzy[:, :, 1:-1] = (D[3][:, :, 2:] - D[3][:, :, :-2]) / (2 * lam[2])
        g.dDzx[:, :, 1:-1] = (D[6][:, :, 2:] - D[6][:, :, :-2]) / (2 * lam[2])
        g.dDzz[:, :, 1:-1] = (D[9][:, :, 2:] - D[9][:, :, :-2]) / (2 * lam[2])
        return g


def make_ddd_interp(ctrl, grid, grads):
    """Pre-suma los gradientes del tensor por componente de desplazamiento y
    construye un interpolador por componente (RW_dDD_Calculation.m suma
    yy+xy(+zy) tras interpolar; por linealidad de la interpolación es
    idéntico interpolar la suma)."""
    if ctrl.Two_D == 1:
        comp_y = grads.dDyy + grads.dDxy
        comp_x = grads.dDyx + grads.dDxx
        return [make_interp(grid, comp_y, ctrl), make_interp(grid, comp_x, ctrl)]
    comp_y = grads.dDyy + grads.dDxy + grads.dDzy
    comp_x = grads.dDyx + grads.dDxx + grads.dDzx
    comp_z = grads.dDyz + grads.dDxz + grads.dDzz
    return [make_interp(grid, comp_y, ctrl), make_interp(grid, comp_x, ctrl),
            make_interp(grid, comp_z, ctrl)]


def ddd_at_particles(ctrl, ddd_interps, X, alive):
    """RW_dDD_Calculation.m — gradiente del tensor interpolado en partículas."""
    pts = X[:, :3] if ctrl.Two_D == 0 else X[:, :2]
    dDD = np.zeros_like(X)
    for k, rgi in enumerate(ddd_interps):
        vals = rgi(pts)
        dDD[alive, k] = vals[alive]
    return np.nan_to_num(dDD)


def b_displacement(ctrl, qx_l, qy_l, qz_l, vabs, rand_n, dt, alive):
    """RW_B_Calculation.m — matriz de desplazamiento aleatorio B."""
    al, at = ctrl.trns.al, ctrl.trns.at
    Dme = ctrl.trns.Dm / ctrl.trns.n
    n = qx_l.shape[0]
    with np.errstate(invalid="ignore", divide="ignore"):
        if ctrl.Two_D == 1:
            B = np.zeros((n, 2))
            Wat = np.sqrt(2 * (at * vabs + Dme))
            Wal = np.sqrt(2 * (al * vabs + Dme))
            Byy = qx_l * Wat / vabs
            Byx = qy_l * Wal / vabs
            Bxy = -qy_l * Wat / vabs
            Bxx = qx_l * (Wal if B_LONGITUDINAL_FIX else Wat) / vabs
            for arr in (Byy, Byx, Bxy, Bxx):
                np.nan_to_num(arr, copy=False)
            B[alive, 0] = (Byy * rand_n[:, 0] + Byx * rand_n[:, 1])[alive] * np.sqrt(dt[alive])
            B[alive, 1] = (Bxy * rand_n[:, 0] + Bxx * rand_n[:, 1])[alive] * np.sqrt(dt[alive])
            return B

        B = np.zeros((n, 3))
        vyxz = vabs
        vyx = np.sqrt(qy_l ** 2 + qx_l ** 2)
        Wal = np.sqrt(2 * (al * vyxz + Dme))
        Wat = np.sqrt(2 * (at * vyxz + Dme))
        Bxx = Wal * qx_l / vyxz
        Bxy = -Wat * qx_l * qz_l / (vyxz * vyx)
        Bxz = -Wat * qy_l / vyx
        Byx = Wal * qy_l / vyxz
        Byy = -Wat * qy_l * qz_l / (vyxz * vyx)
        Byz = Wat * qx_l / vyx
        Bzx = Wal * qz_l / vyxz
        Bzy = Wat * vyx / vyxz
        Bzz = np.zeros(n)
        null = vyx == 0
        s2D = np.sqrt(2 * Dme)
        Bxx = np.where(null, s2D, Bxx); Bxy = np.where(null, 0, Bxy)
        Bxz = np.where(null, 0, Bxz); Byx = np.where(null, 0, Byx)
        Byy = np.where(null, s2D, Byy); Byz = np.where(null, 0, Byz)
        Bzx = np.where(null, 0, Bzx); Bzy = np.where(null, 0, Bzy)
        Bzz = np.where(null, s2D, Bzz)
        for arr in (Bxx, Bxy, Bxz, Byx, Byy, Byz, Bzx, Bzy, Bzz):
            np.nan_to_num(arr, copy=False)
        sq = np.sqrt(dt)
        B[alive, 0] = (Byx * rand_n[:, 0] + Byy * rand_n[:, 1] + Byz * rand_n[:, 2])[alive] * sq[alive]
        B[alive, 1] = (Bxx * rand_n[:, 0] + Bxy * rand_n[:, 1] + Bxz * rand_n[:, 2])[alive] * sq[alive]
        B[alive, 2] = (Bzx * rand_n[:, 0] + Bzy * rand_n[:, 1] + Bzz * rand_n[:, 2])[alive] * sq[alive]
        return B


# ------------------------------------------------------------- fronteras
def mirror_particles(X, lam, N, head_dir, alive, ctrl):
    """RW_mirror_particles.m — rebote (Neumann) o eliminación (Dirichlet)."""
    if ctrl.Two_D == 1:
        X = np.column_stack([X, np.zeros(X.shape[0])])
        lam = np.append(lam, ctrl.d_pts_z)
        N = np.append(N, ctrl.n_pts_z)

    lim = N * lam - lam
    dead = np.zeros(X.shape[0], dtype=bool)

    if head_dir in (0.0, 180.0):
        up = X[:, 0] > lim[0]
        X[up, 0] = lim[0]
        neg = X[:, 0] < 0
        X[neg, 0] = -X[neg, 0]
        for ax in (2, 1):
            out = (X[:, ax] > lim[ax]) | (X[:, ax] < 0)
            X[out, :] = 0.0
            dead |= out
    elif head_dir in (90.0, 270.0):
        up = X[:, 1] > lim[1]
        X[up, 1] = lim[1]
        neg = X[:, 1] < 0
        X[neg, 1] = -X[neg, 1]
        for ax in (2, 0):
            out = (X[:, ax] > lim[ax]) | (X[:, ax] < 0)
            X[out, :] = 0.0
            dead |= out
    else:
        for ax in (2, 1, 0):
            out = (X[:, ax] > lim[ax]) | (X[:, ax] < 0)
            X[out, :] = 0.0
            dead |= out

    alive = alive & ~dead
    if ctrl.Two_D == 1:
        X = X[:, :2]
    return X, alive


def mirror_particles_dispersion(X, lam, N, ctrl):
    """RW_mirror_particles_Dispersion.m — rebote tras el salto dispersivo."""
    if ctrl.Two_D == 1:
        X = np.column_stack([X, np.zeros(X.shape[0])])
        lam = np.append(lam, ctrl.d_pts_z)
        N = np.append(N, ctrl.n_pts_z)
    lim = N * lam - lam
    for ax in range(3):
        over = X[:, ax] > lim[ax]
        X[over, ax] = lim[ax]
        neg = X[:, ax] < 0
        X[neg, ax] = -X[neg, ax]
    if ctrl.Two_D == 1:
        X = X[:, :2]
    return X


def bounce_well(grid, ctrl, X_adv, X, num_well):
    """RW_Bounce_particles_well_location.m — rebote en el radio del pozo."""
    if num_well == 0:
        wx_frac, wy_frac = ctrl.inP[0], ctrl.inP[1]
    else:
        wx_frac, wy_frac = ctrl.Ex_inP[num_well - 1, 0], ctrl.Ex_inP[num_well - 1, 1]
    n_el = grid.data.n_el
    welly = round(n_el[0] * wy_frac) * grid.d_pts[0] - grid.d_pts[0]
    wellx = round(n_el[1] * wx_frac) * grid.d_pts[1] - grid.d_pts[1]

    dist = np.sqrt((wellx - X_adv[:, 1]) ** 2 + (welly - X_adv[:, 0]) ** 2)
    idx = dist < ctrl.well_radius
    if not idx.any():
        return X_adv
    p0 = X[idx, :2]
    pt = X_adv[idx, :2]
    ady = np.abs(p0[:, 1] - pt[:, 1])
    opu = np.abs(p0[:, 0] - pt[:, 0])
    with np.errstate(invalid="ignore", divide="ignore"):
        angle = np.degrees(np.arctan(opu / ady))
    hip = np.abs(ctrl.well_radius - dist[idx])
    y_p = hip * np.sin(np.radians(angle))
    x_p = hip * np.cos(np.radians(angle))
    nuevos = pt + 2 * np.column_stack([y_p, x_p])
    bad_x = np.isnan(nuevos[:, 1]); nuevos[bad_x, 1] = grid.d_pts[1]
    bad_y = np.isnan(nuevos[:, 0]); nuevos[bad_y, 0] = grid.d_pts[0]
    X_adv[idx, :2] = nuevos
    return X_adv


def dispersion_index(X, radius, grid, ctrl, alive):
    """RW_Dispersion_index.m — partículas fuera del radio advectivo."""
    n_el = grid.data.n_el
    welly = round(n_el[0] * ctrl.inP[1]) * grid.d_pts[0]
    wellx = round(n_el[1] * ctrl.inP[0]) * grid.d_pts[1]
    dist = np.sqrt((wellx - X[:, 1]) ** 2 + (welly - X[:, 0]) ** 2)
    return (dist > radius) & alive


# ------------------------------------------------------------- inyección
def particle_3d_dim_size(part):
    """RW_particle_3d_dim_size.m"""
    if part < 1e2:
        raise ValueError("muy pocas partículas para análisis 3D (>=100)")
    if part < 1e5:
        z_tam = 10
    else:
        z_tam = 100
    return int(part // z_tam), z_tam


def particlepos_circular(ctrl, grid, qx_m, qy_m, npartic, num_well, rng):
    """particlepos.m (rama circular) — inyección ponderada por flujo."""
    n1 = np.asarray(grid.n_pts) + 1
    wp_row = np.atleast_2d(ctrl.bc.well_pts)[num_well]
    subs = ind2sub(tuple(n1), wp_row)
    well_my = float(np.mean(subs[0]) + 1)   # subíndices 1-based como MATLAB
    well_mx = float(np.mean(subs[1]) + 1)
    well_z = (subs[2] + 1) if ctrl.Two_D == 0 else None

    if ctrl.Two_D == 1:
        pTot = int(npartic)
    else:
        pTot, ptotz = particle_3d_dim_size(npartic)

    radi = ctrl.use.longinject
    csstep = 1000
    alph = np.linspace(0, 360, csstep)
    IX = well_mx + radi * np.cos(np.radians(alph))
    IY = well_my + radi * np.sin(np.radians(alph))

    # radflux.m: interpolación bilineal de qx,qy en coordenadas de nodo 1-based
    axes_idx = (np.arange(1, n1[0] + 1, dtype=float),
                np.arange(1, n1[1] + 1, dtype=float))
    if ctrl.Two_D == 1:
        qx2, qy2 = qx_m, qy_m
    else:
        kz = int(round(np.mean(well_z))) - 1
        qx2, qy2 = qx_m[:, :, kz], qy_m[:, :, kz]
    rgx = RegularGridInterpolator(axes_idx, qx2, bounds_error=False, fill_value=0.0)
    rgy = RegularGridInterpolator(axes_idx, qy2, bounds_error=False, fill_value=0.0)
    pts = np.column_stack([IY, IX])
    QX = rgx(pts)
    QY = rgy(pts)
    QQ = np.sqrt(QX ** 2 + QY ** 2)             # fluxconv.m
    tot = QQ.sum()
    if tot <= 0:
        alpha = rng.random(pTot) * 360.0
    else:
        cswq = np.cumsum(QQ / tot)
        wp = (1 - cswq.min()) / pTot
        wpi = np.arange(cswq.min(), 1.0, wp)[:pTot]
        wpiR = wpi + wp * rng.random(wpi.size)
        alpha = np.interp(wpiR, cswq, alph)

    IX1 = well_mx + radi * np.cos(np.radians(alpha))
    IY1 = well_my + radi * np.sin(np.radians(alpha))
    if ctrl.Two_D == 1:
        return np.column_stack([IY1 * grid.d_pts[0] - grid.d_pts[0],
                                IX1 * grid.d_pts[1] - grid.d_pts[1]])
    # RW_3D_cilinder_distribution.m: pozo totalmente penetrante
    zz = rng.random(ptotz * IX1.size)
    bottom = ctrl.n_pts_z * ctrl.d_pts_z - ctrl.d_pts_z
    top = ctrl.d_pts_z
    z_axis = zz * (bottom - top) + ctrl.d_pts_z
    IX1s = np.repeat(IX1, ptotz)
    IY1s = np.repeat(IY1, ptotz)
    return np.column_stack([IY1s * grid.d_pts[0] - grid.d_pts[0],
                            IX1s * grid.d_pts[1] - grid.d_pts[1],
                            z_axis])


# ------------------------------------------------------------- tracker
class ParticleTracker:
    """Estado del RWPT inverso para un pozo (sustituye los .mat de Data_RWPT)."""

    def __init__(self, ctrl, grid, num_well, rng):
        self.ctrl = ctrl
        self.grid = grid
        self.num_well = num_well
        self.rng = rng
        dim = 2 if ctrl.Two_D == 1 else 3
        self.X = np.zeros((0, dim))
        self.age = np.zeros(0)            # [s] desde inyección
        self.alive = np.zeros(0, dtype=bool)
        self.batch_qs = np.zeros(0, dtype=int)   # QS de liberación (1-based)
        self.batch_sub = np.zeros(0, dtype=int)  # sub-lote TTI (0..bsteps-1)
        n2d = (grid.n_pts[0] + 1) * (grid.n_pts[1] + 1)
        self.n2d = n2d
        # huella binaria POR SUB-LOTE, como result.m1n{i,col} de MATLAB: el
        # mapa QS emitido es la SUMA de los bsteps sub-lotes (valores 0..10),
        # no la unión — esto pondera el interior de la zona.
        self.footprints = {}              # (QS, sub-lote) -> mapa binario 2D
        self.emitted_maps = []            # mapas QS (uno por TM_step)
        self.recorder = None              # si es lista, graba posiciones por TTI (para video)
        # unión de huellas de lotes ya expirados: en MATLAB los slots m1n no
        # se resetean, por lo que la imagen usada por la optimización
        # (RWPT_drawing_optimization.m) incluye también lotes expirados.
        self.legacy_union = np.zeros(self.n2d, dtype=bool)

    # -- estado (sustituye los X*.mat / X*_moment.mat de Data_RWPT) ---------
    def snapshot(self):
        return {
            "X": self.X.copy(), "age": self.age.copy(),
            "alive": self.alive.copy(), "batch_qs": self.batch_qs.copy(),
            "batch_sub": self.batch_sub.copy(),
            "footprints": {k: v.copy() for k, v in self.footprints.items()},
            "legacy_union": self.legacy_union.copy(),
            "n_emitted": len(self.emitted_maps),
        }

    def restore(self, st):
        self.X = st["X"].copy()
        self.age = st["age"].copy()
        self.alive = st["alive"].copy()
        self.batch_qs = st["batch_qs"].copy()
        self.batch_sub = st["batch_sub"].copy()
        self.footprints = {k: v.copy() for k, v in st["footprints"].items()}
        self.legacy_union = st["legacy_union"].copy()
        del self.emitted_maps[st["n_emitted"]:]

    def current_image(self):
        """Imagen binaria 2D de la zona de captura actual del pozo (unión de
        las huellas de todos los lotes, vivos y expirados) — equivalente a la
        Image_plane_well de RWPT_drawing_optimization.m."""
        img = self.legacy_union.copy()
        for fp in self.footprints.values():
            img |= fp
        return img

    # -- huella binaria ----------------------------------------------------
    def _paint(self, X, alive, age):
        ctrl, grid = self.ctrl, self.grid
        ok = alive & (age <= ctrl.crit.t_crit * 86400.0)
        if not ok.any():
            return
        # round(pos/d): subíndice 1-based con corrimiento de media celda
        iy = np.round(X[ok, 0] / grid.d_pts[0]).astype(int)
        ix = np.round(X[ok, 1] / grid.d_pts[1]).astype(int)
        iy[iy == 0] = 1
        ix[ix == 0] = 1
        ny1 = grid.n_pts[0] + 1
        nx1 = grid.n_pts[1] + 1
        iy = np.clip(iy, 1, ny1)
        ix = np.clip(ix, 1, nx1)
        lin = (ix - 1) * ny1 + (iy - 1)     # column-major 2D (proyección en 3D)
        keys = self.batch_qs[ok] * 1000 + self.batch_sub[ok]
        for key in np.unique(keys):
            sel = keys == key
            fp = self.footprints.setdefault(int(key), np.zeros(self.n2d, dtype=bool))
            fp[lin[sel]] = True

    # -- un paso cuasi-estacionario ----------------------------------------
    def run_qs(self, t, flowpar, fields):
        """Ejecuta el QS t (1-based) con el campo (qx,qy,qz,radius) dado."""
        ctrl, grid, rng = self.ctrl, self.grid, self.rng
        qx_m, qy_m, qz_m, radius = fields
        lam = np.asarray(grid.d_pts, dtype=float)
        N = np.asarray(grid.n_pts, dtype=float)
        bsteps = int(ctrl.tim.deltQS / ctrl.TTI)
        if ctrl.dispersion:
            grads = dispersion_grids(ctrl, grid, qx_m, qy_m, qz_m)
            ddd_interps = make_ddd_interp(ctrl, grid, grads)
        rgi_qx = make_interp(grid, qx_m, ctrl)
        rgi_qy = make_interp(grid, qy_m, ctrl)
        rgi_qz = make_interp(grid, qz_m, ctrl) if ctrl.Two_D == 0 else None

        inject = t <= ctrl.crit.t_crit / ctrl.tim.deltQS  # RW_Particle_injection.m
        if inject:
            X_new = particlepos_circular(ctrl, grid, qx_m, qy_m,
                                         ctrl.trns.npartic_QS, self.num_well, rng)
            # Reparto en sub-lotes ENTRELAZADO como MATLAB (ip(i,:) = i:bsteps:N
            # y RW_ip_3d): cada sub-lote toma 1 de cada bsteps posiciones del
            # círculo de inyección, de modo que TODOS los sub-lotes muestrean
            # el círculo completo (no arcos contiguos). Esto es lo que pondera
            # el interior de la zona (núcleos al 100% en el A50).
            if ctrl.Two_D == 1:
                sub_idx = [np.arange(i, X_new.shape[0], bsteps) for i in range(bsteps)]
            else:
                pTot, ptotz = particle_3d_dim_size(X_new.shape[0])
                ang = np.arange(pTot)
                sub_idx = [np.concatenate(
                    [a * ptotz + np.arange(ptotz) for a in ang[i::bsteps]])
                    for i in range(bsteps)]

        dX = lam.min() / ctrl.space_disc

        for tti in range(bsteps):
            if inject:
                Xi = X_new[sub_idx[tti]]
                self.X = np.vstack([self.X, Xi])
                self.age = np.concatenate([self.age, np.zeros(Xi.shape[0])])
                self.alive = np.concatenate([self.alive, np.ones(Xi.shape[0], dtype=bool)])
                self.batch_qs = np.concatenate(
                    [self.batch_qs, np.full(Xi.shape[0], t, dtype=int)])
                self.batch_sub = np.concatenate(
                    [self.batch_sub, np.full(Xi.shape[0], tti, dtype=int)])
                # MATLAB registra posiciones solo después de cada movimiento
                # (YP se llena dentro del while), no en la inyección.

            if self.X.shape[0] == 0:
                continue

            rem = np.where(self.alive, ctrl.TTI * 86400.0, 0.0)
            _substep = 0
            while (rem > 0).any():
                _substep += 1
                alive = self.alive & (rem > 0)
                X = self.X
                pts = X[:, :3] if ctrl.Two_D == 0 else X[:, :2]
                vx = np.nan_to_num(rgi_qx(pts))
                vy = np.nan_to_num(rgi_qy(pts))
                vz = (np.nan_to_num(rgi_qz(pts))
                      if ctrl.Two_D == 0 else np.zeros_like(vx))
                vabs = np.sqrt(vx ** 2 + vy ** 2 + vz ** 2)
                with np.errstate(divide="ignore"):
                    dt = np.where(vabs > 0, dX / vabs, ctrl.TTI * 86400.0)
                dt = np.minimum(dt, rem)
                dt[~alive] = 0.0

                rand_n = rng.standard_normal(X.shape)

                if ctrl.dispersion:
                    dDD = ddd_at_particles(ctrl, ddd_interps, X, alive)
                else:
                    dDD = np.zeros_like(X)

                # parte advectiva (Euler; RW_X_Calculation.m)
                A = np.zeros_like(X)
                A[alive, 0] = ((vy + dDD[:, 0]) * dt)[alive]
                A[alive, 1] = ((vx + dDD[:, 1]) * dt)[alive]
                if ctrl.Two_D == 0:
                    A[alive, 2] = ((vz + dDD[:, 2]) * dt)[alive]
                X_adv = X.copy()
                X_adv[alive] = X[alive] + A[alive]

                X_adv = bounce_well(grid, ctrl, X_adv, X, self.num_well)
                self.X = X_adv

                self.age[alive] += dt[alive]
                rem[alive] -= dt[alive]

                self.X, self.alive = mirror_particles(
                    self.X, lam.copy(), N.copy(), flowpar.HeadDir, self.alive, ctrl)
                rem[~self.alive] = 0.0

                if ctrl.dispersion:
                    B = b_displacement(ctrl, vx, vy, vz, vabs, rand_n, dt, self.alive)
                    idd = dispersion_index(self.X, radius, grid, ctrl, self.alive)
                    self.X[idd] += B[idd]
                    self.X = mirror_particles_dispersion(self.X, lam.copy(), N.copy(), ctrl)

                self._paint(self.X, self.alive, self.age)

                # salvaguarda anti-cuelgue: el bucle de subpasos DEBE terminar.
                # En operación normal bastan ~cientos de subpasos; si una partícula
                # deja de avanzar (dt≈0 por velocidad extrema, p. ej. un campo
                # condicionado con overshoot), tras este tope se cierra el subpaso
                # para no congelar el kernel de forma ininterrumpible.
                if _substep >= 20000:
                    rem[:] = 0.0
                    break

            # grabación para video: posiciones de TODAS las partículas tras este
            # intervalo de inyección (TTI), con su edad y la dirección del driver
            if self.recorder is not None and self.X.shape[0] > 0:
                self.recorder.append({
                    "t": int(t), "tti": int(tti),
                    "X": self.X.copy(),
                    "age": self.age.copy(),
                    "alive": self.alive.copy(),
                    "HeadDir": float(getattr(flowpar, "HeadDir", np.nan)),
                })

        # emisión del mapa QS (suma de los bsteps sub-lotes binarios, como el
        # bloque "QS = QS + QS_matrix" de RRWPT_tempcont.m) y purga del lote
        n_qs_crit = int(ctrl.crit.t_crit / ctrl.tim.deltQS)
        if t > n_qs_crit:
            tb = t - n_qs_crit
            qs_map = np.zeros(self.n2d, dtype=np.float64)
            for i in range(bsteps):
                fp = self.footprints.pop(tb * 1000 + i, None)
                if fp is not None:
                    qs_map += fp
                    self.legacy_union |= fp
            self.emitted_maps.append(qs_map)
            keep = self.batch_qs != tb
            self.X = self.X[keep]
            self.age = self.age[keep]
            self.alive = self.alive[keep]
            self.batch_qs = self.batch_qs[keep]
            self.batch_sub = self.batch_sub[keep]
