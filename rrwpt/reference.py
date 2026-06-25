"""Campos de referencia y superposición — espejo de:
  TempCond/unc_hfield_reference.m
  TempCond/unc_superpos_hfield.m
  TempCond/rotate_recharge.m
  RW/RW_Advective_radius.m
"""
from types import SimpleNamespace as NS

import numpy as np
from scipy import ndimage

from .fem import solve_reference_head, fem_q
from .flowpar import init_uncert_flowpar


def _clean(slope):
    slope = np.asarray(slope, dtype=float)
    slope[~np.isfinite(slope)] = 0.0
    return slope


def unc_hfield_reference(grid, yk, ctrl, flowpar):
    """unc_hfield_reference.m — campo de referencia h0/q0 y pendientes de
    superposición (mH, mqx, mqy[, mqz]) para Qp, qr y dirh (90/270)."""
    K = np.exp(np.asarray(yk, dtype=float).reshape(-1))
    npts1 = int(np.prod(np.asarray(grid.n_pts) + 1))
    n_wells = np.atleast_2d(ctrl.bc.well_pts).shape[0]

    ref = NS(mH=NS(), mqx=NS(), mqy=NS(), mqz=NS())
    ctrl.href0 = 1
    synth_orig, ctrl.synth = ctrl.synth, 1

    # 1.- escenario de referencia (flujo base, dirh = par0.dirh = 0)
    vp = NS(dh=ctrl.par0.dh, HeadDir=ctrl.par0.dirh, qr=0.0, Qp=0.0)
    ctrl.use_gwn = 0
    ctrl.use_well = 0
    res = solve_reference_head(K, vp, grid, ctrl)
    ref.h0_vec = res["h"][:, 0]
    ref.qx0_vec = res["qx"][:, 0]
    ref.qy0_vec = res["qy"][:, 0]
    if ctrl.Two_D == 0:
        ref.qz0_vec = res["qz"][:, 0]

    # 2.1- pendientes por bombeo (un escenario doble por pozo)
    if ctrl.tim.sup_qpump:
        ref.mH.Qp = np.zeros((npts1, n_wells))
        ref.mqx.Qp = np.zeros((npts1, n_wells))
        ref.mqy.Qp = np.zeros((npts1, n_wells))
        ref.mqz.Qp = np.zeros((npts1, n_wells))
        for w in range(n_wells):
            ctrl.use_uncert_Qp = 0
            ctrl.num_well = w
            fp = init_uncert_flowpar(ctrl, grid)
            vp = NS(dh=ctrl.par0.dh, HeadDir=ctrl.par0.dirh, qr=0.0,
                    Qp=np.array([fp.Qp, ctrl.par0.Qp]))
            ctrl.use_gwn = 0
            ctrl.use_well = 1
            res = solve_reference_head(K, vp, grid, ctrl)
            delx = vp.Qp[0] - ctrl.par0.Qp
            ref.Qp = vp.Qp[0]
            ref.mH.Qp[:, w] = _clean((res["h"][:, 0] - res["h"][:, 1]) / delx)
            ref.mqx.Qp[:, w] = _clean((res["qx"][:, 0] - res["qx"][:, 1]) / delx)
            ref.mqy.Qp[:, w] = _clean((res["qy"][:, 0] - res["qy"][:, 1]) / delx)
            if ctrl.Two_D == 0:
                ref.mqz.Qp[:, w] = _clean((res["qz"][:, 0] - res["qz"][:, 1]) / delx)
    else:
        ref.mH.Qp = np.zeros((npts1, 1))
        ref.mqx.Qp = np.zeros((npts1, 1))
        ref.mqy.Qp = np.zeros((npts1, 1))
        ref.mqz.Qp = np.zeros((npts1, 1))
        ref.Qp = 0.0
    ctrl.use_uncert_Qp = ctrl.tim.sup_qpump

    # 2.2- pendientes por recarga
    if ctrl.tim.sup_recharge:
        ctrl.use_uncert_recharge = 0
        fp = init_uncert_flowpar(ctrl, grid)
        vp = NS(dh=ctrl.par0.dh, HeadDir=0.0, Qp=0.0,
                qr=np.array([fp.qr, ctrl.par0.qr]))
        ctrl.use_gwn = 1
        ctrl.use_well = 0
        res = solve_reference_head(K, vp, grid, ctrl)
        delx = vp.qr[0] - ctrl.par0.qr
        ref.qr = vp.qr[0]
        ref.mH.qr = _clean((res["h"][:, 0] - res["h"][:, 1]) / delx)
        ref.mqx.qr = _clean((res["qx"][:, 0] - res["qx"][:, 1]) / delx)
        ref.mqy.qr = _clean((res["qy"][:, 0] - res["qy"][:, 1]) / delx)
        if ctrl.Two_D == 0:
            ref.mqz.qr = _clean((res["qz"][:, 0] - res["qz"][:, 1]) / delx)
    else:
        ref.mH.qr = np.zeros(npts1)
        ref.mqx.qr = np.zeros(npts1)
        ref.mqy.qr = np.zeros(npts1)
        ref.mqz.qr = np.zeros(npts1)
        ref.qr = 0.0
    ctrl.use_uncert_recharge = ctrl.tim.sup_recharge

    # 2.3- dh: sin pendiente (se escala el campo de referencia con deldh)
    ref.mH.dh = np.zeros(npts1)
    ref.dh = 0.0

    # 2.4- pendientes por dirección (90 y 270 grados)
    for ang, delx in ((90.0, 90.0 - ctrl.par0.dirh), (270.0, 270.0 - 180.0)):
        tag = f"dirh_{int(ang)}"
        if ctrl.tim.sup_dirh:
            vp = NS(dh=ctrl.par0.dh, qr=0.0, Qp=0.0, HeadDir=ang)
            ctrl.use_gwn = 0
            ctrl.use_well = 0
            res = solve_reference_head(K, vp, grid, ctrl)
            ref.dirh = ang
            setattr(ref.mH, tag, _clean((res["h"][:, 0] - ref.h0_vec) / delx))
            setattr(ref.mqx, tag, _clean((res["qx"][:, 0] - ref.qx0_vec) / delx))
            setattr(ref.mqy, tag, _clean((res["qy"][:, 0] - ref.qy0_vec) / delx))
            if ctrl.Two_D == 0:
                setattr(ref.mqz, tag, _clean((res["qz"][:, 0] - ref.qz0_vec) / delx))
        else:
            setattr(ref.mH, tag, np.zeros(npts1))
            setattr(ref.mqx, tag, np.zeros(npts1))
            setattr(ref.mqy, tag, np.zeros(npts1))
            setattr(ref.mqz, tag, np.zeros(npts1))
            ref.dirh = 0.0

    ctrl.href0 = 0
    ctrl.synth = synth_orig
    ref.dh0 = ctrl.par0.dh
    ref.Qp0 = ctrl.par0.Qp
    ref.qr0 = ctrl.par0.qr
    ref.dirh0 = ctrl.par0.dirh
    return ref, ctrl


# ----------------------------------------------------------- superposición
def _fold_heading(head_dir):
    """Plegado del ángulo a [0,90] (unc_superpos_hfield.m)."""
    if 0 <= head_dir <= 90:
        return head_dir
    if 90 < head_dir <= 180:
        return 180 - head_dir
    if 180 < head_dir <= 270:
        return head_dir - 180
    return 360 - head_dir


def unc_superpos_hfield(ref, flowpar, ctrl, grid, K):
    """unc_superpos_hfield.m — campo de flujo del QS actual por superposición.

    Nota (documentada): el .m original asigna `dFelQp` pero usa `delQp`
    (errata); aquí se usa la intención clara: delQp = Qp - Qp0.
    """
    npts1 = int(np.prod(np.asarray(grid.n_pts) + 1))

    delQp = (np.atleast_1d(flowpar.Qp) - ref.Qp0) if ctrl.tim.sup_qpump else np.zeros(1)
    delqr = (flowpar.qr - ref.qr0) if ctrl.tim.sup_recharge else 0.0
    deldirh = _fold_heading(flowpar.HeadDir) if ctrl.tim.sup_dirh else 0.0

    if 90 <= flowpar.HeadDir < 180:
        mH_d, mqx_d, mqy_d = ref.mH.dirh_90, ref.mqx.dirh_90, ref.mqy.dirh_90
        mqz_d = ref.mqz.dirh_90 if ctrl.Two_D == 0 else None
    elif 180 <= flowpar.HeadDir <= 270:
        mH_d, mqx_d, mqy_d = ref.mH.dirh_270, ref.mqx.dirh_270, ref.mqy.dirh_270
        mqz_d = ref.mqz.dirh_270 if ctrl.Two_D == 0 else None
    else:
        # MATLAB no define este caso (dirh fuera de [90,270] no ocurre con la
        # configuración por defecto); usamos pendiente nula y lo registramos.
        mH_d = np.zeros(npts1); mqx_d = mH_d; mqy_d = mH_d; mqz_d = mH_d

    deldh = flowpar.dh / ctrl.par0.dh  # ctrl.tim.riv_scheme == 1

    map_Qp_h = np.zeros(npts1)
    map_Qp_qx = np.zeros(npts1)
    map_Qp_qy = np.zeros(npts1)
    map_Qp_qz = np.zeros(npts1)
    for i in range(min(len(delQp), ref.mH.Qp.shape[1])):
        map_Qp_h += ref.mH.Qp[:, i] * delQp[i]
        map_Qp_qx += ref.mqx.Qp[:, i] * delQp[i]
        map_Qp_qy += ref.mqy.Qp[:, i] * delQp[i]
        if ctrl.Two_D == 0:
            map_Qp_qz += ref.mqz.Qp[:, i] * delQp[i]

    hsim = deldh * (ref.h0_vec + mH_d * deldirh) + map_Qp_h
    qxsim = deldh * (ref.qx0_vec + mqx_d * deldirh) + map_Qp_qx
    qysim = deldh * (ref.qy0_vec + mqy_d * deldirh) + map_Qp_qy
    if ctrl.Two_D == 0:
        qzsim = deldh * (ref.qz0_vec + mqz_d * deldirh) + map_Qp_qz
    else:
        qzsim = np.zeros_like(qysim)

    if ctrl.tim.sup_recharge == 1:
        hsim = deldh * ref.h0_vec
        hsim, qxsim, qysim, qzsim = rotate_recharge(
            ref, delqr, grid, hsim, K, flowpar, map_Qp_h, ctrl)

    radius = rw_advective_radius(qxsim, qysim, qzsim, ctrl, grid) if ctrl.dispersion == 1 else 0.0
    return hsim, qxsim, qysim, qzsim, radius


def rotate_recharge(ref, delqr, grid, hsim, K, flowpar, map_Qp_h, ctrl):
    """rotate_recharge.m — rota el mapa (recarga+flujo base) según HeadDir.

    La maquinaria de padding/imcrop/imrotate_1 de MATLAB se sustituye por
    scipy.ndimage.rotate (bilineal, reshape=False, mode='nearest'), que
    reproduce la rotación bilineal con relleno por replicación de borde.
    """
    n1 = np.asarray(grid.n_pts) + 1
    mapa_recharge = (ref.mH.qr * delqr).reshape(tuple(n1), order="F")
    mapa_baseflow = np.asarray(hsim).reshape(tuple(n1), order="F")
    nz_factor = mapa_recharge.shape[2] if mapa_recharge.ndim == 3 else 1
    map_total = mapa_recharge * nz_factor + mapa_baseflow

    hd = flowpar.HeadDir
    if 90 <= hd <= 180:
        ang = 360 - (180 - hd)
    elif 180 < hd <= 270:
        ang = hd - 180
    else:
        ang = 0.0

    if hd in (0.0, 180.0) or ang == 0.0:
        h_rech = map_total
    else:
        if map_total.ndim == 2:
            h_rech = ndimage.rotate(map_total, ang, reshape=False,
                                    order=1, mode="nearest")
        else:
            h_rech = np.empty_like(map_total)
            for k in range(map_total.shape[2]):
                h_rech[:, :, k] = ndimage.rotate(map_total[:, :, k], ang,
                                                 reshape=False, order=1,
                                                 mode="nearest")

    h = h_rech.reshape(-1, order="F") + map_Qp_h * nz_factor
    q = fem_q(K, h, grid)
    return h, q[:, 0], q[:, 1], q[:, 2]


def rw_advective_radius(qxsim, qysim, qzsim, ctrl, grid):
    """RW_Advective_radius.m — radio del punto de estancamiento.

    imregionalmin se reemplaza por mínimos locales con filtro 3x3
    (misma conectividad 8 del default de MATLAB).
    """
    n1 = np.asarray(grid.n_pts) + 1
    absv = np.sqrt(qxsim ** 2 + qysim ** 2 + qzsim ** 2)
    if ctrl.Two_D == 1:
        va = absv.reshape((n1[0], n1[1]), order="F")
    else:
        va = absv.reshape(tuple(n1), order="F").sum(axis=2)

    mins = (va == ndimage.minimum_filter(va, size=3, mode="nearest"))
    Y, X = np.nonzero(mins)
    # a subíndices 1-based como en MATLAB
    Y = Y + 1
    X = X + 1

    n_el = grid.data.n_el
    welly = round(n_el[0] * ctrl.inP[1])
    wellx = round(n_el[1] * ctrl.inP[0])
    dist = np.sqrt((X - wellx) ** 2 + (Y - welly) ** 2)
    dist = dist[dist != 0]
    if dist.size == 0:
        return float(ctrl.well_radius)
    return float(dist.min() * (grid.d_pts[0] + grid.d_pts[1]) / 2)
