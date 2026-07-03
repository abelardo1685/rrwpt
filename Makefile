# ============================================================
# Makefile — A2: rrwpt como paquete Python instalable
# (adaptación del Makefile canónico HIDRA a un proyecto de software)
# ============================================================
# Convenciones:
#  - Toda regla es idempotente.
#  - Los tests `slow` (regeneración a malla completa, ~horas) son opt-in.
#  - Semillas de los tests: explícitas en el código de tests (documentadas).

PYTHON  ?= python3
VENV    ?= .venv
PIP     := $(VENV)/bin/pip
PY      := $(VENV)/bin/python

.PHONY: env test test-smoke test-slow lint build check clean-env

## Entorno de desarrollo: venv + instalación editable con extras dev
env:
	$(PYTHON) -m venv $(VENV)
	$(PIP) install --upgrade pip
	$(PIP) install -e ".[dev,fast]"
	$(PY) -c "import rrwpt; print('rrwpt', rrwpt.__version__, 'OK')"

## Suite completa rápida (excluye slow por pyproject addopts; <1 min)
test:
	$(PY) -m pytest tests/ -v

## Solo el humo end-to-end (malla 60x60, <60 s)
test-smoke:
	$(PY) -m pytest tests/test_pipeline_smoke.py -v

## Paridad MATLAB a malla completa 450x450x4 (~2 h CPU; opt-in)
test-slow:
	$(PY) -m pytest tests/ -v -m slow

## Lint (mismo comando que CI)
lint:
	$(PY) -m ruff check src/ tests/

## sdist + wheel en dist/
build:
	$(PIP) install build
	$(PY) -m build

## Lo que corre CI: lint + tests + build
check: lint test build

clean-env:
	rm -rf $(VENV)
