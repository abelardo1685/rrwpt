"""Orquestación de una realización — espejo de TempCond/uncond_control.m
(+ RW_avoiding_Repeated_flow_fields.m y el cierre de RRWPT_tempcont.m),
sin las ramas de optimización (PSO/OMOPSO), GSA ni FPCA/PCE.
"""
from types import SimpleNamespace as NS

import numpy as np

from . import boundary, fem, flowpar as fp_mod, geostat, grid_mod, reference, rwpt


def setup(ctrl, rng=None):
    """Equivalente a la fase de inicialización de Ensemble_generation.m."""
    rng = rng or np.random.default_rng()
    grid = grid_mod.init_fem(grid_mod.init_grid(ctrl))
    flowpar = fp_mod.init_uncert_flowpar(ctrl, grid, rng)
    ctrl = boundary.init_bc(grid, ctrl, flowpar)
    ctrl, model = geostat.init_model(grid, ctrl, rng)
    ctrl = setup_conditioning(ctrl, grid, model)
    return grid, flowpar, model, ctrl


def setup_conditioning(ctrl, grid, model):
    """Genera el campo "verdadero" de referencia y muestrea ctrl.cond.n_cond
    puntos de log-K — espejo de Auxiliars/Synthetic_realization.m.

    Son las medidas a las que se condicionan TODAS las realizaciones del
    ensamble (idénticas dentro de una corrida). El campo de referencia y las
    ubicaciones usan semillas independientes del rng de la simulación:
      - ctrl.cond.seed_field : semilla del campo "verdadero" de referencia
      - ctrl.cond.seed_obs   : semilla de las ubicaciones de muestreo
    Si una semilla es None, esa parte es ALEATORIA en cada corrida (distinta
    cada vez); un entero la hace reproducible. Solo actúa si
    ctrl.cond.Kflag == 1 y el campo es heterogéneo (ctrl.het == 1).
    """
    cond = getattr(ctrl, "cond", None)
    if cond is None or int(getattr(cond, "Kflag", 0)) != 1 or ctrl.het != 1:
        return ctrl
    # campo de referencia INCONDICIONAL (evita recursión con uncsim_kfield)
    rng_field = np.random.default_rng(getattr(cond, "seed_field", None))
    Y_ref_arr, _, _ = geostat.generate_randomfield(model, grid, rng=rng_field)
    Y_ref = Y_ref_arr.reshape(-1, order="F")

    # n_cond UBICACIONES (x,y) aleatorias en el plano horizontal. En 3D cada
    # ubicación aporta TODA su columna vertical (una muestra por capa z), igual
    # que el sondeo de MATLAB; en 2D => n_cond muestras puntuales.
    n_pts = np.asarray(grid.n_pts, dtype=int)
    ny, nx = int(n_pts[0]), int(n_pts[1])
    nz = int(n_pts[2]) if grid.nd == 3 else 1
    n_loc = int(min(max(cond.n_cond, 1), ny * nx))
    rng_obs = np.random.default_rng(getattr(cond, "seed_obs", None))
    flat_xy = rng_obs.choice(ny * nx, size=n_loc, replace=False)  # plano (order F): iy + ny*ix
    iy = flat_xy % ny
    ix = flat_xy // ny
    if grid.nd == 3:
        iz = np.arange(nz)
        # índice lineal (order F) del elemento (iy, ix, iz) = iy + ny*ix + ny*nx*iz
        obs_idx = (iy[:, None] + ny * ix[:, None] + ny * nx * iz[None, :]).reshape(-1)
        loc_idx = iy + ny * ix          # representante z=0 de cada ubicación (para graficar)
    else:
        obs_idx = iy + ny * ix
        loc_idx = obs_idx
    cond.obs_idx = np.asarray(obs_idx, dtype=int)
    cond.loc_idx = np.asarray(loc_idx, dtype=int)   # una entrada por ubicación (x,y)
    cond.n_layers = nz
    cond.yobs = Y_ref[cond.obs_idx]     # medidas (sin ruido; r_K modela la incertidumbre)
    cond.Y_ref = Y_ref
    cond.cov_cache = None               # invalida la caché: las observaciones cambiaron
    cond.fftqe = None
    return ctrl


def run_realization(ctrl, grid, model, flowpar, rng=None, verbose=True,
                    collect_heads=False, yk=None, record_particles=False):
    """uncond_control.m — una realización completa.

    yk : campo log-K prescrito (vector de nel elementos, orden 'F'), espejo
         de la rama map_1=1 de UncSim_Kfield_calculation.m (cargar
         Y_original_450x450.mat en vez de generar). Si es None, se genera
         según ctrl.het.

    Devuelve un NS con:
      A50temporal : mapa probabilístico final (vector (ny+1)*(nx+1), %),
      qs_maps     : lista de mapas binarios por TM_step y pozo,
      yk          : campo log-K usado,
      heads       : (opcional) h por QS para diagnóstico.
    """
    rng = rng or np.random.default_rng()
    n2d = (grid.n_pts[0] + 1) * (grid.n_pts[1] + 1)
    n_wells = np.atleast_2d(ctrl.bc.well_pts).shape[0]

    # 1.- campo de conductividad
    if yk is None:
        yk = geostat.uncsim_kfield(ctrl, model, grid, rng=rng)
    else:
        yk = np.asarray(yk, dtype=float).reshape(-1)
        if yk.size != grid.data.nel:
            raise ValueError(f"yk tiene {yk.size} elementos; la malla requiere "
                             f"{grid.data.nel} (n_el={grid.data.n_el})")
    K = np.exp(yk)

    # 2.- campos de referencia y pendientes para superposición
    if verbose:
        print(" -------------------- Unconditional Head Simulation (referencia)")
    ref, ctrl = reference.unc_hfield_reference(grid, yk, ctrl, flowpar)

    # 2b.- preparación de la optimización robusta (RW_avoiding_Repeated_
    #      flow_fields.m rama opt_RMO: refs por escenario K + hull + demanda)
    opt_on = bool(ctrl.optimization) and bool(ctrl.opt_RMO)
    if opt_on:
        from . import optimization as opt
        hull2 = opt.load_hull(ctrl, grid)
        if ctrl.opt.k_fields_file:
            k_fields = opt.load_k_fields(ctrl, grid)[: max(ctrl.num_k, 1)]
            refs = []
            for yk_k in k_fields:
                if verbose:
                    print(" ---- referencia para escenario K de robustez")
                ref_k, ctrl = reference.unc_hfield_reference(grid, yk_k, ctrl, flowpar)
                refs.append(ref_k)
        else:
            refs = [ref]          # num_k=1 sin archivo: el propio campo
        demand = (opt.load_demand(ctrl)
                  if (ctrl.num_d > 0 and ctrl.opt.demand_file) else None)
        best_pumping_ts = np.zeros((n_wells, ctrl.tim.tstep))
        penalty_ts = [None] * ctrl.tim.tstep
        area_ts = np.zeros((n_wells, ctrl.tim.tstep))

    # 3.- drivers transitorios deterministas
    drv = fp_mod.transient_drivers(ctrl)

    # 4.- bucle QS con RWPT inverso por pozo
    trackers = [rwpt.ParticleTracker(ctrl, grid, w, rng) for w in range(n_wells)]
    if record_particles:
        for tr in trackers:
            tr.recorder = []          # graba posiciones por TTI (para el video)
    heads = [] if collect_heads else None

    for t in range(1, ctrl.tim.tstep + 1):
        fpar = fp_mod.assign_temporal(flowpar, ctrl, drv, t)
        if opt_on:
            snapshots = [tr.snapshot() for tr in trackers]

        hsim, qx, qy, qz, radius = reference.unc_superpos_hfield(
            ref, fpar, ctrl, grid, K)
        if collect_heads:
            heads.append(hsim.copy())
        qx_m, qy_m, qz_m = rwpt.vel_direction(ctrl, grid, qx, qy, qz)
        if verbose:
            print(f"--- QS {t:3d}/{ctrl.tim.tstep}: dirh={fpar.HeadDir:7.2f}  "
                  f"dh={fpar.dh:.5f}  qr={fpar.qr:7.2f}  radio adv={radius:7.1f} m")
        for tr in trackers:
            tr.run_qs(t, fpar, (qx_m, qy_m, qz_m, radius))

        # 4b.- evaluación / optimización (uncond_control.m + RRWPT_tempcont.m)
        if opt_on:
            from . import optimization as opt
            images = [tr.current_image() for tr in trackers]
            cost, area_i, _ = opt.objective_components(
                ctrl, images, hull2, fpar.Qp, best_pumping_ts, t)
            if t < ctrl.time_start_optimization:
                cost[1] = 0.0            # "Saltaste optimización en el tiempo"
            if cost[1] <= 0.0:
                # delineación dentro del hull: se acepta el QS normal
                best_pumping_ts[:, t - 1] = np.atleast_1d(fpar.Qp)
                penalty_ts[t - 1] = cost
                area_ts[:, t - 1] = area_i
            else:
                if verbose:
                    print(f"   >>> {int(cost[1])} pixeles fuera del hull: "
                          f"optimización robusta del bombeo (QS {t})")
                # descartar el QS normal y optimizar (Robust_MO_Optimization.m)
                for tr, st in zip(trackers, snapshots):
                    tr.restore(st)
                ctx = opt.build_opt_context(ctrl, grid, K, refs, flowpar, drv,
                                            trackers, t, hull2, demand,
                                            best_pumping_ts,
                                            seed=int(rng.integers(2**31)))
                best_qp, rep, e_file = opt.robust_mo_optimization(
                    ctx, rng, verbose=verbose)
                # solución final con el bombeo óptimo (ctrl.record=1)
                fpar.Qp = best_qp
                hsim, qx, qy, qz, radius = reference.unc_superpos_hfield(
                    ref, fpar, ctrl, grid, K)
                qx_m, qy_m, qz_m = rwpt.vel_direction(ctrl, grid, qx, qy, qz)
                for tr in trackers:
                    tr.run_qs(t, fpar, (qx_m, qy_m, qz_m, radius))
                images = [tr.current_image() for tr in trackers]
                cost, area_i, _ = opt.objective_components(
                    ctrl, images, hull2, best_qp, best_pumping_ts, t)
                best_pumping_ts[:, t - 1] = best_qp
                penalty_ts[t - 1] = cost
                area_ts[:, t - 1] = area_i
                if verbose:
                    print(f"   >>> Qp óptimo: {np.round(best_qp, 4)}  "
                          f"penalización final: {cost.round(3)}")

    # 5.- mapa probabilístico final (cierre de RRWPT_tempcont.m)
    canvas = np.zeros(n2d)
    qs_maps = []
    for tr in trackers:
        for m in tr.emitted_maps:
            canvas += m
        qs_maps.append(tr.emitted_maps)
    canvas[0] = 0.0
    if canvas.max() > 0:
        canvas = canvas / canvas.max() * 100.0

    out = NS(A50temporal=canvas, qs_maps=qs_maps, yk=yk, heads=heads,
             ref=ref, drivers=drv)
    if record_particles:
        out.particle_frames = [tr.recorder for tr in trackers]   # lista por pozo
    if opt_on:
        out.BestPumping_ts = best_pumping_ts
        out.penalty_ts = penalty_ts
        out.Area_ts = area_ts
    return out


def _run_one(args):
    """Worker para el ensamble paralelo (una realización, semilla propia)."""
    ctrl, grid, model, flowpar, seed = args
    from . import fem
    fem.clear_solver_cache()
    rng = np.random.default_rng(seed)
    res = run_realization(ctrl, grid, model, flowpar, rng=rng, verbose=False)
    return res.A50temporal


def run_ensemble(ctrl, grid, model, flowpar, n_reali=None, n_jobs=None,
                 base_seed=0):
    """Ensamble Monte Carlo paralelo (equivalente al bucle `l` con parfor de
    Ensemble_generation.m). Devuelve la lista de mapas A50 por realización.

    n_jobs: procesos en paralelo (default: min(n_reali, cpu_count())).
    Cada realización usa una semilla independiente (base_seed + l).
    """
    import os
    from concurrent.futures import ProcessPoolExecutor

    n_reali = n_reali or ctrl.n_reali
    n_jobs = n_jobs or min(n_reali, os.cpu_count() or 1)
    tasks = [(ctrl, grid, model, flowpar, base_seed + l) for l in range(n_reali)]
    if n_jobs == 1:
        return [_run_one(t) for t in tasks]
    with ProcessPoolExecutor(max_workers=n_jobs) as ex:
        return list(ex.map(_run_one, tasks))


def prob_map_2d(result, grid):
    """Mapa final como matriz (ny+1, nx+1) para graficar (imagesc MATLAB)."""
    n1 = (grid.n_pts[0] + 1, grid.n_pts[1] + 1)
    return result.A50temporal.reshape(n1, order="F")
