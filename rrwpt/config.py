"""Configuración de control (ctrl) — espejo de main/MC_config_RRWPT.m + INIT/init_flags.m.

Solo se incluyen las banderas que consume el núcleo físico migrado
(campos K, flujo por superposición, RWPT, delineación). Las banderas de
optimización (PSO/OMOPSO), GSA/Sobol, FPCA y PCE se conservan con sus
valores por defecto "apagado" y están documentadas como pendientes en
docs/MIGRACION.md.
"""
from types import SimpleNamespace as NS

import numpy as np


def make_ctrl(**overrides):
    """Crea la estructura ctrl con los valores de MC_config_RRWPT.m.

    Cualquier campo puede sobreescribirse con keyword arguments usando
    nombres con puntos, p.ej. make_ctrl(**{"tim.tend": 60, "n_pts_x": 100}).
    """
    c = NS(tim=NS(), use=NS(), cond=NS(), crit=NS(), trns=NS(), mov=NS(),
           par0=NS(), stru=NS(), out=NS(), sol=NS(), bc=NS())

    # --- discretización / dominio (MC_config_RRWPT.m líneas 58-97) ---
    c.space_disc = 2          # pasos de espacio por pixel (dX = d_pts/space_disc)
    c.R4 = 0                  # 0 Euler (único esquema migrado), 1 RK4 (pendiente)
    c.Two_D = 1               # 1 => 2D, 0 => 3D
    c.dispersion = 1          # transporte tipo Scheidegger (RWPT con dispersión)
    c.tim.spill = 0           # 0 fuente continua, 1 derrame único
    c.single_trn_batch = 0
    c.well_radius = 15.0      # [m]
    c.n_pts_x = 450
    c.n_pts_y = 450
    c.n_pts_z = 4
    c.d_pts_x = 15.0
    c.d_pts_y = 15.0
    c.d_pts_z = 15.0

    # --- pozos ---
    c.inP = np.array([0.80, 0.50, 0.70])     # [x% y% z%] pozo principal
    c.Ex_pump = 1
    c.Ex_inP = np.array([[0.80, 0.45],
                         [0.80, 0.55],
                         [0.80, 0.60],
                         [0.80, 0.40],
                         [0.80, 0.35]])
    c.use.circular = 1
    c.use.longinject = 4      # [celdas] radio de inyección alrededor del pozo
    c.use.mmtemp = 0

    # --- tiempo ---
    c.n_reali = 1
    c.t_crit = 1
    c.crit.t_crit = 180       # [d] tiempo crítico (delineación t50)
    c.tim.tend = 360          # [d] duración total
    c.trns.npartic_QS = 1000  # partículas por pozo por QS
    c.spec_map = [50]
    c.dti = 1800              # [s] paso de transporte
    c.tim.dtvis = 86400       # [s] paso de visualización
    c.tim.deltQS = 10         # [d] longitud del paso cuasi-estacionario
    c.TTI = 1                 # [d] intervalo de inyección de partículas
    c.tim.periods = 1
    c.RWPTtrans = 1
    c.source_geo = 1
    c.FEM_CODE = 0            # 0 => superposición (única ruta migrada)

    # --- dispersión / difusión ---
    c.trns.n = 0.35
    c.trns.at = 0.10
    c.trns.al = 1.00
    c.trns.Dm = 1e-9

    # --- parámetros espaciales (geostadística) ---
    c.map_1 = 0               # 0: generar campo (1 = cargar .mat precalculado, no migrado)
    c.het = 1
    c.het_geo = 0
    c.use_variable_logK = 0
    c.use_variable_logK_variance = 0
    c.stru.muA = -5.5
    c.stru.muE = -7.5
    c.stru.variA = 0.50
    c.stru.variE = 0.70
    c.use_var_matern_kappa = 1
    c.use_var_corr_length = 0
    c.use_var_shapefactor = 0
    c.stru.nbeta = 0
    c.stru.intexA = 450.0
    c.stru.intexE = 390.0
    c.stru.inteyA = 110.0
    c.stru.inteyE = 160.0
    c.stru.intezA = 25.0
    c.stru.intezE = 25.0
    c.stru.kapA = 0.4
    c.stru.kapE = 0.6
    c.use_big_scale_variaty = 0

    # --- conditioning de campos K a muestras de log-K --------------------
    # (TempCond/cond_hetero_simu.m + Auxiliars/Synthetic_realization.m):
    # condicionamiento por kriging de los campos heterogéneos a n_cond medidas.
    c.cond = NS()
    c.cond.Kflag = 0          # 1 => condicionar campos K a las muestras (kriging)
    c.cond.n_cond = 10        # nº de muestras (medidas) de log-K
    c.cond.r_K = 1.0          # varianza del error de medición (MATLAB ctrl.cond.r_K)
    c.cond.seed_field = None  # semilla del campo "verdadero" de referencia (None = aleatorio cada corrida)
    c.cond.seed_obs = None    # semilla de las ubicaciones de muestreo (None = aleatorio cada corrida)
    c.cond.obs_idx = None     # índices de TODAS las muestras (se llenan en pipeline.setup)
    c.cond.loc_idx = None     # índices de las ubicaciones (x,y), 1 por sondeo (para graficar)
    c.cond.n_layers = None    # nº de capas muestreadas por ubicación (nz en 3D, 1 en 2D)
    c.cond.yobs = None        # valores medidos de log-K (se llenan en pipeline.setup)
    c.cond.Y_ref = None       # campo "verdadero" de referencia (para graficar/diagnóstico)
    c.cond.cov_cache = None    # (C_og, C_oo) del kriging — se calcula 1 vez y se reusa en el ensamble
    c.cond.fftqe = None        # FFT de la covarianza embebida — se reusa entre realizaciones

    # --- escenario de referencia / superposición ---
    c.use_well = 0
    c.use_gwn = 0
    c.par0.dh = 0.01
    c.par0.Qp = 0.001
    c.par0.qr = 10.0
    c.par0.dirh = 0.0
    c.tim.superpos = 1
    c.tim.sup_dirh = 1
    c.tim.sup_dh = 1
    c.tim.sup_qpump = 1
    c.tim.sup_recharge = 1
    c.tim.sup_river_BC = 0
    c.href0 = 0
    c.synth = 1               # 1: usar promedios para drivers temporales

    # --- drivers transitorios ---
    c.use_knownTprog = 0      # 0: comportamiento sinusoidal
    c.Pumping_approach = 3
    c.use_uncert_head_BC = 1
    c.use_uncert_headdir = 0
    c.use_uncert_headgrad = 0
    c.use_uncert_Qp = 0
    c.use_uncert_recharge = 0
    c.use_uncert_river_BC = 0
    c.use_uncert_river_BC_2 = 0
    c.tim.comb = 0
    c.tim.single = 1
    c.fix_sin_par = 0
    c.tim.T_amp_a = 0.0
    c.tim.T_amp_e = 100.0
    c.tim.T_fr_a = 1.0
    c.tim.T_fr_e = 1.0
    c.tim.T_phase_a = 0.0
    c.tim.T_phase_e = 360.0
    c.tim.dhA = 0.0015
    c.tim.dhE = 0.0065
    c.tim.dirhA = 160.0
    c.tim.dirhE = 200.0
    c.tim.QpA = 0.005
    c.tim.QpE = 0.05
    c.tim.qr0A = 50.0
    c.tim.qr0E = 500.0
    c.tim.mhA = 1.0
    c.tim.mhE = 1.5
    c.tim.riv_sing = 2
    c.tim.riv_scheme = 1
    c.tim.riv_uno_a = 1.0
    c.tim.riv_uno_e = 4.0
    c.tim.riv_dos_a = 1.0
    c.tim.riv_dos_e = 4.0

    # --- banderas varias del kernel ---
    c.variable_ref_point = 0
    c.use_ext_grid = 0
    c.geo_window = 0
    c.avarage_to_Ygrid = 0
    c.straty = 0
    c.homo = 0
    c.inclusion = 0

    # --- optimización por enjambre (OMOPSO robusto, MC_config líneas 99-145)
    c.optimization = 0        # activar optimización de bombeos
    c.opt_MO = 1              # 0 mono-objetivo (no migrado), 1 multi-objetivo
    c.opt_RMO = 1             # optimización multi-objetivo ROBUSTA (migrada)
    c.num_k = 1               # nº de escenarios K para robustez
    c.num_d = 0               # nº de escenarios de demanda incierta (0 = sin demanda)
    c.eta = 0.90
    c.time_start_optimization = 18   # QS a partir del cual se permite optimizar
    c.MaxIt = 60              # iteraciones OMOPSO
    c.nPop = 30               # tamaño del enjambre
    c.pos_extr = 2            # multiplicador del bombeo máximo permitido
    c.neg_extr = 0            # multiplicador del bombeo mínimo (0 = sin inyección)
    c.pumping_alpha = 500     # peso de la penalización por déficit de bombeo
    c.Pixel_out = 0           # umbral de pixeles fuera del hull para penalizar
    c.w = (0.1, 0.5)          # rango de peso de inercia
    c.e = 0.001               # epsilon de e-dominancia
    c.c1 = (1.5, 2.0)         # coeficientes de aprendizaje
    c.c2 = (1.5, 2.0)
    c.r1 = (0.0, 1.0)
    c.r2 = (0.0, 1.0)
    c.nGrid = 7               # celdas de la malla adaptativa por objetivo
    c.alpha = 0.1             # inflación de la malla
    c.beta = 2.0              # presión de selección de líderes (ruleta)
    c.gamma = 2.0
    c.mu = 0.1

    # entradas externas y paralelismo de la optimización
    c.opt = NS(
        k_fields_file=None,    # .mat/.npy con la matriz (num_k, nel) de campos log-K
        k_fields_var=None,     # nombre de la variable dentro del .mat (auto si única)
        hull_file=None,        # .mat/.npy con el hull estacionario (prob_map_hull)
        hull_var=None,
        demand_file=None,      # .mat/.csv/.npy de demanda (filas=escenarios)
        demand_var="Mat_MC_Prog",
        demand_scale=0.0275 / 1483.25,  # factor de Objfunc.m (demanda -> m³/s)
        demand_column=1,       # columna usada (MATLAB fija columna=1)
        use_demand=False,      # MATLAB tiene la demanda comentada en
                               # Objective_function.m ("Borrar despues");
                               # True usa demanda_incierta como av_pumping
        n_workers=1,           # procesos paralelos para evaluar el enjambre
        out_dir=None,          # carpeta para rep/e_file por iteración (None = no guardar)
    )

    # --- módulos NO migrados (mantener apagados) ---
    c.GSA = 0
    c.GSA_Sobol = 0
    c.GSA_morris = 0
    c.PCE = 0
    c.OLHS = 0
    c.fpca = 0
    c.part_id = 0
    c.record = 0
    c.swarming = 0
    c.draw_kfield = 0
    c.draw_hfield = 0
    c.mov = NS(dtvis=0)

    # --- flags de salida (init_flags.m + MC_config) ---
    c.out.Y = 1
    c.out.Qt = 0
    c.out.Qy = 0
    c.out.h = 1
    c.out.d = 0
    c.out.sl = 0
    c.out.sl3 = 0
    c.out.c0 = 0
    c.out.ct = 1
    c.out.cmk = 0
    c.out.fx = 0
    c.out.ST = 0
    c.out.fx_y = 0
    c.out.cc = 0
    c.out.cw = 0
    c.out.qx = 1
    c.out.qy = 1
    c.out.qz = 0   # init_flags.m: 1 solo en 3D

    # aplicar overrides "a.b.c" o planos
    for key, val in overrides.items():
        obj = c
        parts = key.split(".")
        for p in parts[:-1]:
            obj = getattr(obj, p)
        setattr(obj, parts[-1], val)

    finalize_ctrl(c)
    return c


def finalize_ctrl(c):
    """Cantidades derivadas (MC_config_RRWPT.m líneas 445-454 + init_flags)."""
    if c.Two_D == 0:
        c.out.qz = 1
    if c.tim.spill == 1 and c.single_trn_batch == 1:
        c.tim.tend = c.crit.t_crit + c.tim.deltQS

    c.tim.tstep = int(np.ceil(c.tim.tend / c.tim.deltQS))
    ttot_vec = np.concatenate(([0.0], np.full(c.tim.tstep, float(c.tim.deltQS))))
    c.Tstatvalid = np.cumsum(ttot_vec)            # [d] tiempo acumulado por QS
    c.trns.npartic = c.trns.npartic_QS * (c.tim.tend // c.tim.deltQS)
    c.part_iter = (c.tim.tend // c.tim.deltQS) * c.trns.npartic_QS

    # solvers (init_SOLVERS.m); el tipo 'bicgmg' se sustituye por scipy
    c.sol.FEMresh = 1e-10
    c.sol.memsave_flag = False
    c.sol.solverh = "scipy-direct"
    return c
