"""rrwpt — Migración Python del modelo MATLAB Transient_RRWPT.

Pipeline (espejo de MC_config_RRWPT.m -> Ensemble_generation.m -> uncond_control.m):

  1. config.make_ctrl()                 <- MC_config_RRWPT.m + init_flags.m
  2. grid_mod.init_grid / init_fem      <- init_gridrainer.m + ndgrid_setup.m + init_FEM.m
  3. geostat.init_model                 <- init_modelRai_ARP.m
     geostat.generate_randomfield       <- SPECTRAL_GEOSTATS/* (2D y 3D)
     geostat.uncond_homo_ksim           <- uncond_homo_Ksim.m
  4. boundary.init_bc                   <- init_BC_RRWPT.m (select_plane*.m)
     boundary.change_bc                 <- Change_BC.m / FEM_BC_2d.m
  5. flowpar.init_uncert_flowpar        <- init_uncertPar_FlowTrans.m
     flowpar.transient_drivers          <- UncSim_Temporal_parameter_assignation*.m
  6. fem.solve_reference_head           <- FEM_kernel_Dtensor_quasiStat.m (rama href0)
     fem.fem_q                          <- FEM_q.m
  7. reference.unc_hfield_reference     <- unc_hfield_reference.m
     reference.unc_superpos_hfield      <- unc_superpos_hfield.m + rotate_recharge.m
  8. rwpt.ParticleTracker               <- RRWPT_tempcont.m + RW_*.m
  9. delineation                        <- bloque de momentos temporales/A50temporal
 10. pipeline.run_realization           <- uncond_control.m (sin optimización/GSA)

Convenciones:
 - Todos los reshape/sub2ind usan orden de columna ('F') para reproducir MATLAB.
 - Los índices de nodo internos son 0-based; las posiciones físicas en metros
   siguen exactamente las fórmulas MATLAB (incluidos los offsets -d_pts).
"""

from . import config, grid_mod, geostat, boundary, flowpar, fem, reference, rwpt, pipeline

__version__ = "0.1.0.dev0"

__all__ = [
    "__version__",
    "config", "grid_mod", "geostat", "boundary", "flowpar", "fem",
    "reference", "rwpt", "pipeline",
]
