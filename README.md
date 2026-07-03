# rrwpt

<!-- Badges (placeholders until the repository is published) -->
![CI](https://img.shields.io/badge/CI-pending-lightgrey)
![PyPI](https://img.shields.io/badge/PyPI-not%20released-lightgrey)
![License](https://img.shields.io/badge/license-MIT%20(provisional)-blue)
![Python](https://img.shields.io/badge/python-3.10%2B-blue)

**rrwpt** simulates **transient groundwater flow** (finite elements, quasi-steady
superposition of stress periods) coupled with **reverse random-walk particle
tracking (RWPT)** to delineate **probabilistic wellhead protection areas (WHPA)**
under time-varying pumping, regional gradients and recharge. It includes a
spectral geostatistical generator (circulant embedding, Matérn and 10 other
covariance models, 2-D/3-D, unconditional and kriging-conditioned fields) for
Monte Carlo ensembles of heterogeneous log-conductivity.

It is a validated Python port of the MATLAB model `Transient_RRWPT` used in
Rodríguez-Pretelín & Nowak (2018), *Advances in Water Resources* —
[doi:10.1016/j.advwatres.2018.07.005](https://doi.org/10.1016/j.advwatres.2018.07.005).
MATLAB↔Python parity tests ship with the package (`tests/`).

## Installation

```bash
pip install -e .            # core (numpy + scipy only)
pip install -e ".[fast]"    # + pyamg multigrid solver (recommended for large 3-D grids)
pip install -e ".[dev]"     # + pytest, ruff, build
```

## Quickstart

One realization on a small 2-D aquifer (90 × 90 cells, 6 wells, 60 days), and
its time-continuous WHPA probability map:

```python
import numpy as np
import matplotlib.pyplot as plt
from rrwpt import config, pipeline

ctrl = config.make_ctrl(**{
    "Two_D": 1,                              # 2-D aquifer
    "n_pts_x": 90, "n_pts_y": 90,            # grid cells
    "d_pts_x": 15.0, "d_pts_y": 15.0,        # cell size [m]
    "het": 1,                                # heterogeneous log-K (Matérn field)
    "tim.tend": 60, "tim.deltQS": 10,        # 60 d horizon, 10 d quasi-steady steps
    "crit.t_crit": 30,                       # delineation travel time [d]
    "trns.npartic_QS": 500,                  # particles per well per step
})

rng = np.random.default_rng(42)              # explicit seed => reproducible
grid, flowpar, model, ctrl = pipeline.setup(ctrl, rng)
result = pipeline.run_realization(ctrl, grid, model, flowpar, rng=rng, verbose=False)

pmap = pipeline.prob_map_2d(result, grid)    # (ny+1, nx+1) probability map [%]
plt.imshow(pmap, cmap="viridis"); plt.colorbar(label="WHPA probability [%]")
plt.show()
```

For Monte Carlo ensembles use `pipeline.run_ensemble(...)` (process-parallel,
one independent seed per realization). See `docs/quickstart.md` and
`docs/theory.md` for the method, and `tests/` for validated reference cases.

## Validation

`tests/test_matlab_parity.py` contrasts Python probability maps against the
original MATLAB reference runs (450 × 450 × 4 grid, 6 wells), including a
scenario with the *identical* prescribed conductivity field
(`Y_original_450x450.mat`). Run the fast suite with `pytest`; the full-grid
regeneration (~hours) is opt-in: `pytest -m slow`.

## Citing

If you use rrwpt, please cite the method paper (and this software via
`CITATION.cff`):

> Rodríguez-Pretelín, A., & Nowak, W. (2018). Integrating transient behavior
> as a new dimension to WHPA delineation. *Advances in Water Resources*, 119,
> 178–187. https://doi.org/10.1016/j.advwatres.2018.07.005

## License

MIT (provisional — final license decision pending, see `docs/decisiones.md`).
