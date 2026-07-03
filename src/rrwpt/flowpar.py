"""Parámetros de flujo y drivers transitorios — espejo de:
  INIT/init_uncertPar_FlowTrans.m
  Abelardo/UncSim_Temporal_parameter_assignation.m (+ _Sobol.m, ruta determinista)
  Auxiliars/Deterministic_transient_iteration.m

Solo se migra la ruta determinista (ctrl.GSA = 0): los valores medios de cada
driver con comportamiento sinusoidal (use_knownTprog = 0). Los muestreos
Sobol/Saltelli y Morris (GSA_sampling.m) quedan pendientes (ver docs).
"""
from types import SimpleNamespace as NS

import numpy as np


def init_uncert_flowpar(ctrl, grid, rng=None):
    """init_uncertPar_FlowTrans.m — ruta 'known' (synth=1, use_uncert_*=0)."""
    rng = rng or np.random.default_rng()
    fp = NS()

    # Bombeo
    if ctrl.use_uncert_Qp == 1 and ctrl.synth != 1:
        fp.Qp = ctrl.tim.QpA + (ctrl.tim.QpE - ctrl.tim.QpA) * rng.random()
    elif ctrl.use_uncert_Qp == 1 and ctrl.href0 == 1:
        fp.Qp = ctrl.par0.Qp
    else:
        fp.Qp_min = ctrl.tim.QpA
        fp.Qp_max = ctrl.tim.QpE
        fp.Qp = fp.Qp_max          # MATLAB: flowpar.Qp = flowpar.Qp_max

    # Gradiente hidráulico
    if ctrl.use_uncert_headgrad == 1 and ctrl.synth != 1:
        fp.dh = ctrl.tim.dhA + (ctrl.tim.dhE - ctrl.tim.dhA) * rng.random()
    elif ctrl.href0 == 1 and ctrl.use_uncert_headgrad == 1:
        fp.dh = ctrl.par0.dh
    else:
        fp.dh = (ctrl.tim.dhA + ctrl.tim.dhE) / 2

    # Dirección del flujo de fondo
    if ctrl.use_uncert_head_BC and ctrl.synth != 1:
        fp.HeadDir = ctrl.tim.dirhA + (ctrl.tim.dirhE - ctrl.tim.dirhA) * rng.random()
    elif ctrl.href0 == 1 and ctrl.use_uncert_head_BC == 1:
        fp.HeadDir = ctrl.par0.dirh
    elif ctrl.use_uncert_head_BC:
        fp.HeadDir = (ctrl.tim.dirhA + ctrl.tim.dirhE) / 2
    else:
        fp.HeadDir = 0.0

    # Recarga
    if ctrl.use_uncert_recharge == 1 and ctrl.synth != 1:
        fp.qr = ctrl.tim.qr0A + (ctrl.tim.qr0E - ctrl.tim.QpA) * rng.random()
    elif ctrl.href0 == 1 and ctrl.use_uncert_recharge == 1:
        fp.qr = ctrl.par0.qr
    else:
        fp.qr = (ctrl.tim.qr0E + ctrl.tim.qr0A) / 2

    fp.varRefPt = grid.n_pts[1] / 2
    return fp


def _sind(deg):
    return np.sin(np.deg2rad(np.asarray(deg, dtype=float)))


def transient_drivers(ctrl):
    """Series sinusoidales deterministas de los 4 drivers (espejo de la ruta
    determinista de UncSim_Temporal_parameter_assignation_Sobol.m, con los
    valores hardcodeados del archivo: amp_i/fase por driver).

    Devuelve un NS con vectores muestreados por QS (longitud tstep):
      dirh_vec [°], dh_vec [-], qr_vec [mm/a], Qp (constante = media, por pozo).
    """
    tt = np.arange(0, ctrl.tim.tend + ctrl.tim.deltQS, ctrl.tim.deltQS, dtype=float)
    freq = (ctrl.tim.T_fr_a + ctrl.tim.T_fr_e) / 2

    drv = NS()

    # dirh: amp_i=50, fase=180 (valores fijados en el .m)
    dirh_mean = (ctrl.tim.dirhA + ctrl.tim.dirhE) / 2
    ampl = ctrl.tim.dirhE - dirh_mean
    vec = dirh_mean + _sind(tt * freq + 180.0) * (ampl * 50.0 / 100.0)
    drv.dirh_vec = np.mod(vec[:-1], 360.0)

    # dh: amp_i=100, fase=180
    dh_mean = (ctrl.tim.dhA + ctrl.tim.dhE) / 2
    ampl = ctrl.tim.dhE - dh_mean
    drv.dh_vec = (dh_mean + _sind(tt * freq + 180.0) * ampl)[:-1]

    # qr: amp_i=100, fase=180
    qr_mean = (ctrl.tim.qr0A + ctrl.tim.qr0E) / 2
    ampl = ctrl.tim.qr0E - qr_mean
    drv.qr_vec = (qr_mean + _sind(tt * freq + 180.0) * ampl)[:-1]

    # Qp: constante en la media para todos los pozos (MATLAB "1) qp mean")
    drv.Qp_mean = (ctrl.tim.QpA + ctrl.tim.QpE) / 2
    return drv


def assign_temporal(flowpar, ctrl, drv, t):
    """UncSim_Temporal_parameter_assignation.m (QS=1) — asigna a flowpar los
    valores del QS t (1-based, como en MATLAB)."""
    n_wells = np.atleast_2d(ctrl.bc.well_pts).shape[0]
    if ctrl.tim.sup_dirh == 1:
        flowpar.HeadDir = float(drv.dirh_vec[t - 1])
    if ctrl.tim.sup_dh == 1:
        flowpar.dh = float(drv.dh_vec[t - 1])
    else:
        flowpar.dh = 0.0
    if ctrl.tim.sup_recharge == 1:
        flowpar.qr = float(drv.qr_vec[t - 1])
    else:
        flowpar.qr = 0.0
    if ctrl.tim.sup_qpump == 1:
        flowpar.Qp = np.full(n_wells, drv.Qp_mean)
    else:
        flowpar.Qp = np.zeros(n_wells)

    if flowpar.HeadDir < 0:
        flowpar.HeadDir += 360.0
    return flowpar
