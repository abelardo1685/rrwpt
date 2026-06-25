# Transient Probabilistic Delineation

**Probabilistic delineation of Well Head Protection Areas (WHPA) under transient flow
conditions.** This repository contains a Python implementation of a groundwater flow
and **reverse Random‑Walk Particle Tracking (RWPT)** model that delineates capture
zones as **probability maps**, integrating uncertainty from both the **heterogeneous
hydraulic conductivity field** and the **time‑varying (transient) flow drivers**, with
optional **conditioning** of the conductivity fields to measured data.

It reproduces in Python the method of Rodríguez‑Pretelín & Nowak (2018), *Integrating
transient behavior as a new dimension to WHPA delineation* (Advances in Water
Resources, 119, 178–187).

## What the simulator does

Starting from a single set of variables, the notebook:

1. Generates a synthetic **"true" conductivity field** and samples it at a few
   locations (the measured data).
2. Builds **conditioned conductivity fields** that honor those samples (kriging‑based
   conditioning).
3. Runs the transient flow + reverse particle‑tracking simulation, producing the
   time‑of‑travel **protection area** for each realization, and shows a **video** of
   the injected particles aging and being removed over time.
4. Aggregates a Monte Carlo ensemble into the **probabilistic WHPA**, drawn with
   iso‑probability contours (protection levels).

## Repository layout

```
transient-probabilistic-delineation/
├── README.md                  # this file
├── requirements.txt           # Python dependencies (pip install -r requirements.txt)
├── .gitignore
├── Simulador_completo.ipynb   # the simulation notebook (single well: demo + Monte Carlo + particle video)
├── rrwpt/                     # MODEL PACKAGE — all the functions needed to run the notebook
│   ├── __init__.py
│   ├── config.py             # model parameters (make_ctrl)
│   ├── grid_mod.py           # mesh and discretization (FEM)
│   ├── fem.py                # system assembly and flow solver
│   ├── geostat.py            # conductivity fields (geostatistics + conditioning)
│   ├── boundary.py           # boundary conditions
│   ├── flowpar.py            # flow parameters and transient drivers
│   ├── reference.py          # reference flow field + superposition
│   ├── rwpt.py               # reverse particle tracking and area delineation
│   └── pipeline.py           # orchestration: setup() and run_realization()
└── results/                  # generated outputs (git‑ignored)
```

## Installation

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

> `pyamg` is optional (a multigrid solver recommended for 3D); if it is not installed
> the code falls back to `scipy.sparse.linalg.spsolve`.

## Usage

Launch Jupyter **from the repository root** (so that `from rrwpt import ...` finds the
package) and open the notebook:

```bash
jupyter lab        # run from the repository root
```

Open **`Simulador_completo.ipynb`** and run the cells in order. You only edit the
**first cell (variables)** — mesh size, well, conditioning options, number of Monte
Carlo realizations, and the iso‑probability levels for the WHPA contours.

> The default mesh is 450 × 450 × 4; one realization takes a few minutes. Reduce
> `n_pts_x/y` or `trns.npartic_QS` in the first cell for quick experimentation.

## Citation

Rodríguez‑Pretelín, A., & Nowak, W. (2018). *Integrating transient behavior as a new
dimension to WHPA delineation.* Advances in Water Resources, 119, 178–187.
