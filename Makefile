# ============================================================
# Makefile canónico del laboratorio HIDRA
# Reproducción completa:  make reproduce
# Verificación rápida:    make reproduce-smoke   (<15 min)
# ============================================================
# Convenciones:
#  - Toda regla es idempotente y reanudable.
#  - Ninguna regla borra salidas previas (append-only por directorio fechado).
#  - Las semillas viven en configs/*.yaml, jamás en el código.

PYTHON  ?= python3
VENV    ?= .venv
PIP     := $(VENV)/bin/pip
PY      := $(VENV)/bin/python

.PHONY: env datos experimentos figuras informe reproduce reproduce-smoke test lint clean-env

## Entorno computacional reproducible
env:
	$(PYTHON) -m venv $(VENV)
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements.lock
	$(PY) -c "import torch, numpy, scipy; print('entorno OK')"

## Descarga/generación de datos (con checksum o semilla; ver data/MANIFIESTO.md)
datos:
	$(PY) -m src.datos --config configs/datos.yaml

## Todos los experimentos pre-registrados (cada eXX es reanudable)
experimentos:
	$(PY) -m experiments.e01 --config configs/e01.yaml
	# añadir e02, e03, ... aquí

## Figuras y tablas de publicación (regenerables siempre desde resultados/)
figuras:
	$(PY) -m figures.build_all

## Informe/manuscrito (LaTeX o docx según proyecto)
informe:
	$(MAKE) -C reports/paper || $(PY) -m reports.build_informe

## Cadena completa
reproduce: env datos experimentos figuras informe
	@echo "== REPRODUCCIÓN COMPLETA TERMINADA =="

## Versión humo: misma cadena, N reducido (config *_smoke.yaml), <15 min
reproduce-smoke: env
	$(PY) -m src.datos --config configs/datos_smoke.yaml
	$(PY) -m experiments.e01 --config configs/e01_smoke.yaml
	$(PY) -m figures.build_all --smoke
	@echo "== HUMO OK =="

test:
	$(PY) -m pytest tests/ -q

lint:
	$(PY) -m ruff check src/ experiments/ || true

clean-env:
	rm -rf $(VENV)
