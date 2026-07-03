# CLAUDE.md — A2: rrwpt, paquete Python + dataset abierto (paper de software)

## Qué es
Empaquetar el simulador rrwpt (fuente de verdad: `0.-Simuladores/1.-Simulador_Transient_RRWPT_python_github_V/` — NO mover ese repo) como paquete pip instalable con tests/CI/docs, y liberar el subset benchmark de 2,000 pares (exportador ya escrito en A1: `experiments/export_subset_zenodo.py`). Destino: Environmental Modelling & Software o JOSS + Zenodo DOI. SIN Nowak (señal de independencia). Envío plan: 2027 T1.

## Protocolo de sesiones (OBLIGATORIO al abrir sesión aquí)
Esta es una **Sesión de Proyecto** del laboratorio HIDRA. Fuente única:
`/home/abelardo/UNAM/3.-Scientific_Project_Manager/LABORATORIO/PROTOCOLO_SESIONES.md`.
ARRANQUE OBLIGATORIO (también ante «inicia protocolo»): (1) leer este CLAUDE.md y el
pre-registro (`experimentos/E01_hipotesis.md`); (2) `git log -5 --oneline` + `git status
--short`; (3) verificar cola activa en `resultados/cola_*.log` (si no dice COLA
COMPLETA/DETENIDA → PROHIBIDO lanzar cómputo); (4) responder con la Confirmación de
Coordinación (formato en PROTOCOLO_SESIONES.md §ARRANQUE).
Territorio: manuscrito/reportes, notebooks, figuras en discusión, exploratorio etiquetado.
NO tocar: experiments/, configs/, Makefile, resultados/ existentes, datos crudos, código en
cuarentena. Corridas largas y agentes pesados = Sesión Coordinadora.

## Estado actual (2026-07-02)
- **Paquete instalable listo (v0.1.0.dev0)**: layout `src/rrwpt/` (copia
  byte-idéntica de la fuente de verdad salvo `__version__`; ADR-002),
  `pyproject.toml` (deps mínimas: numpy+scipy; extras `[fast]`=pyamg,
  `[dev]`, `[docs]`), LICENSE (MIT definitiva, ADR-003+007), CITATION.cff
  (método: Rodríguez-Pretelín & Nowak 2018, AWR, 10.1016/j.advwatres.2018.07.005),
  README en inglés. Verificado: `pip install -e .` + `python -m build` OK.
- **Tests**: 20 tests pytest (19 rápidos, ~1 s, todos pasan; 1 marcado `slow`
  = regeneración de paridad a malla completa, ~2 h, opt-in con `pytest -m slow`).
  Datos de validación MATLAB↔Python en `tests/data/` (12 MB versionados,
  ADR-005; procedencia en `tests/data/README.md`). Estrategia de paridad en
  dos niveles: ADR-006.
- **CI**: `.github/workflows/ci.yml` (ruff + pytest + build en Python 3.11).
  NO publicado aún (ni GitHub ni PyPI).
- **Docs**: `docs/{quickstart,api,theory}.md` (theory = método 2018 con las
  ecuaciones del RWPT inverso). ADRs 001-006 en `docs/decisiones.md`.
- **Makefile**: adaptado a proyecto de software (`env, test, test-smoke,
  test-slow, lint, build, check`).
- **Pendientes que requieren decisión del usuario**: (1) licencia final
  (MIT es placeholder), (2) nombre en PyPI (verificar disponibilidad de
  `rrwpt`), (3) momento de publicar en GitHub/activar CI. Además: exportar el
  subset benchmark 2,000 pares (A1 → Zenodo) sigue pendiente.
