# Quickstart

## Install

```bash
git clone <repo-url> && cd rrwpt        # (repo pendiente de publicación)
pip install -e ".[fast]"                # core + pyamg (recomendado para 3-D)
```

Core dependencies are only `numpy` and `scipy`.

## One realization, one probability map

```python
import numpy as np
from rrwpt import config, pipeline

# 1. Control structure: every physical/numerical parameter with MATLAB-mirror
#    defaults; override any field with dotted keys.
ctrl = config.make_ctrl(**{
    "Two_D": 1,
    "n_pts_x": 90, "n_pts_y": 90, "d_pts_x": 15.0, "d_pts_y": 15.0,
    "het": 1,                        # heterogeneous Matérn log-K field
    "tim.tend": 60, "tim.deltQS": 10,
    "crit.t_crit": 30,               # delineation travel time [d]
    "trns.npartic_QS": 500,
})

# 2. Setup: grid + FEM structures, flow parameters, geostatistical model.
rng = np.random.default_rng(42)      # semilla explícita => reproducible
grid, flowpar, model, ctrl = pipeline.setup(ctrl, rng)

# 3. One realization: reference flow, quasi-steady superposition loop,
#    reverse RWPT per well, time-continuous delineation.
res = pipeline.run_realization(ctrl, grid, model, flowpar, rng=rng, verbose=True)

pmap = pipeline.prob_map_2d(res, grid)   # (ny+1, nx+1), 0-100 %
```

## Monte Carlo ensemble

```python
maps = pipeline.run_ensemble(ctrl, grid, model, flowpar,
                             n_reali=50, n_jobs=8, base_seed=0)
prob = np.mean([m > 0 for m in maps], axis=0) * 100   # WHPA probabilístico
```

Each realization draws an independent conductivity field and transient
drivers with seed `base_seed + l` (process-parallel, checkpoint-free but
embarrassingly parallel — wrap the loop yourself for resumable batches).

## Prescribed conductivity field (MATLAB parity scenario)

```python
from scipy.io import loadmat
Y = loadmat("tests/data/Y_original_450x450.mat", squeeze_me=True)["Y"]
yk = Y.reshape(-1, order="F")            # column-major, como MATLAB
res = pipeline.run_realization(ctrl, grid, model, flowpar, rng=rng, yk=yk)
```

## 3-D

Set `"Two_D": 0` and provide `n_pts_z` / `d_pts_z`. Install the `fast` extra:
without `pyamg` the FEM systems fall back to `scipy.sparse.linalg.spsolve`
(correct but slower on 450x450x4-sized grids).

## Costos orientativos (CPU, 1 núcleo)

| Malla | Config | Tiempo |
|---|---|---|
| 60x60 (2-D, 1 pozo, 3 QS, 150 part/QS) | test humo | < 1 s |
| 450x450x4 (6 pozos, 6 QS, 150 part/QS) | corrida "quick" de validación | ~minutos |
| 450x450x4 (6 pozos, 36 QS, 1000 part/QS) | parámetros exactos MATLAB | ~2 h |
