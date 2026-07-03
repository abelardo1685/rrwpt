# API — rrwpt

> Esqueleto. Autogenerable con `mkdocstrings` (extra `docs`) una vez que el
> repo esté publicado: cada módulo tiene docstrings con la equivalencia
> archivo-a-archivo con el modelo MATLAB original.

## Módulos

| Módulo | Rol | Espejo MATLAB |
|---|---|---|
| `rrwpt.config` | `make_ctrl(**overrides)` / `finalize_ctrl`: estructura de control completa (banderas, tiempo, transporte, geoestadística, pozos) | `MC_config_RRWPT.m`, `init_flags.m` |
| `rrwpt.grid_mod` | Malla regular 2-D/3-D, estructuras FEM, `sub2ind`/`ind2sub` orden 'F' | `init_gridrainer.m`, `ndgrid_setup.m`, `init_FEM.m` |
| `rrwpt.geostat` | Modelo de covarianza (Matérn y 10 más), circulant embedding espectral (Dietrich & Newsam), campos condicionados por kriging | `SPECTRAL_GEOSTATS/*`, `uncond_homo_Ksim.m` |
| `rrwpt.boundary` | Pozos y condiciones Dirichlet | `init_BC_RRWPT.m`, `select_plane*.m` |
| `rrwpt.flowpar` | Parámetros de flujo inciertos + drivers transitorios sinusoidales | `init_uncertPar_FlowTrans.m`, `UncSim_Temporal_*` |
| `rrwpt.fem` | Ensamblaje y solución FEM (spsolve / pyamg opcional, caché de sistemas) | `FEM_Dtensor/*`, rama href0 de `FEM_kernel_Dtensor_quasiStat.m` |
| `rrwpt.reference` | Campo de referencia + superposición cuasi-estacionaria | `unc_hfield_reference.m`, `unc_superpos_hfield.m` |
| `rrwpt.rwpt` | `ParticleTracker`: RWPT inverso con dispersión (Scheidegger), inyección continua por TTI, delineación | `RRWPT_tempcont.m`, `RW_*.m` |
| `rrwpt.pipeline` | `setup`, `run_realization`, `run_ensemble`, `prob_map_2d` | `uncond_control.m`, `Ensemble_generation.m` |

## Puntos de entrada típicos

```python
ctrl = rrwpt.config.make_ctrl(**overrides)
grid, flowpar, model, ctrl = rrwpt.pipeline.setup(ctrl, rng)
res = rrwpt.pipeline.run_realization(ctrl, grid, model, flowpar, rng=rng)
maps = rrwpt.pipeline.run_ensemble(ctrl, grid, model, flowpar, n_reali, n_jobs, base_seed)
```

## No incluido en v0.1

- `rrwpt.optimization` (OMOPSO robusto): referenciado por import perezoso en
  `pipeline.run_realization` pero **no migrado**; `ctrl.optimization=1` falla
  con `ImportError` (documentado, ADR-004).
- Ramas MATLAB no migradas: `map_1=1` (cargar .mat), `het_geo=1` (campos
  geológicos), `R4=1` (RK4), `FEM_CODE=1`, GSA/Sobol, FPCA, PCE.
