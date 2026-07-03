# decisiones.md — ADRs del proyecto

(contexto → decisión → alternativas → consecuencias; una entrada por decisión no obvia)

---

## ADR-001 (2026-07-02) — Layout `src/` para el paquete

**Contexto.** rrwpt pasa de "carpeta de módulos junto a notebooks" a paquete
pip instalable rumbo a paper de software (EMS/JOSS).

**Decisión.** Layout `src/rrwpt/` con `pyproject.toml` (setuptools,
`packages.find where=["src"]`).

**Alternativas descartadas.** (a) Flat layout (`rrwpt/` en la raíz): permite
importar el paquete sin instalarlo, lo que enmascara errores de empaquetado y
hace que pytest pruebe el árbol de trabajo en vez de la instalación. (b)
Poetry/hatch: sin ventaja para un paquete puro-Python; setuptools es el
estándar que los revisores de JOSS reconocen sin fricción.

**Consecuencias.** Los tests SOLO pasan contra el paquete instalado
(`pip install -e .`), que es exactamente lo que verificará un revisor.

---

## ADR-002 (2026-07-02) — La copia de `src/rrwpt/` se mantiene byte-idéntica a la fuente de verdad

**Contexto.** La fuente de verdad del simulador vive en
`0.-Simuladores/1.-Simulador_Transient_RRWPT_python_github_V/rrwpt/` (regla:
intocable). Ruff marca 22 issues de estilo (E501/E702, F401/F811) en ese
código, que es espejo línea-a-línea del MATLAB original.

**Decisión.** Copiar el código sin reescribirlo (único cambio: añadir
`__version__` en `__init__.py`) y silenciar E501/E702/F401/F811 vía
`per-file-ignores` de ruff en `pyproject.toml`.

**Alternativas descartadas.** Reformatear el espejo MATLAB: rompería la
auditoría por `diff -r` contra la fuente de verdad y contra los `.m`
originales (los `;` y líneas largas replican la estructura MATLAB
deliberadamente).

**Consecuencias.** `diff -r src/rrwpt <fuente>/rrwpt` debe dar SOLO el bloque
`__version__` de `__init__.py`. Cualquier fix del simulador se hace PRIMERO en
la fuente de verdad y se re-copia aquí (unidireccional). Cuando A2 se vuelva
el repo público canónico, esta relación se invertirá (decisión futura).

---

## ADR-003 (2026-07-02) — Licencia MIT como placeholder

**Contexto.** El empaquetado exige un archivo LICENSE; la decisión de licencia
es del autor y afecta la publicación (JOSS exige licencia OSI).

**Decisión.** MIT provisional, con nota interna explícita en el propio
LICENSE y en el README. PENDIENTE DE DECISIÓN DEL USUARIO.

**Alternativas.** GPL-3.0 (copyleft, protege derivados), BSD-3, Apache-2.0
(cláusula de patentes). Todas aceptables para JOSS/EMS.

**Consecuencias.** No publicar a GitHub/PyPI hasta confirmar; quitar la nota
interna del LICENSE al decidir.

---

## ADR-004 (2026-07-02) — Qué queda FUERA del paquete v0.1

**Contexto.** La fuente de verdad contiene más de lo publicable/estable.

**Decisión.** v0.1 incluye solo el núcleo físico validado: config, malla/FEM,
geostat (incl. condicionamiento por kriging), boundary, flowpar, referencia +
superposición, RWPT inverso y pipeline. Quedan fuera:

- `rrwpt.optimization` (OMOPSO robusto): **no existe** ni en la fuente de
  verdad; `pipeline.py` lo importa perezosamente solo si `ctrl.optimization=1`
  (→ ImportError documentado en docs/api.md). No se poda el código que lo
  referencia para no divergir de la fuente (ADR-002).
- Notebooks didácticos y `Simulador_completo.ipynb` (irán al repo público como
  `examples/`, decisión al publicar).
- Ramas MATLAB no migradas: `map_1=1`, `het_geo=1`, `R4=1` (RK4),
  `FEM_CODE=1`, GSA/Sobol, FPCA, PCE (banderas apagadas por defecto).
- El subset benchmark de 2,000 pares (exportador en A1) — es el "Dataset" del
  producto A2 pero se versiona vía Zenodo, no dentro del paquete pip.

**Consecuencias.** El alcance del paper de software es el núcleo
flujo+RWPT+WHPA; la optimización robusta puede ser v0.2/paper aparte.

---

## ADR-005 (2026-07-02) — Datos de validación (12 MB) versionados en git

**Contexto.** La paridad MATLAB↔Python necesita los mapas de referencia
(`matlab_prob_map*.npy`, 1.6 MB c/u), los mapas Python archivados de la
corrida quick y el campo `Y_original_450x450.mat` (5.8 MB). Total ~12 MB
(< 100 MB del umbral acordado). El .gitignore del laboratorio excluye
`*.npy`/`*.mat` globalmente.

**Decisión.** Copiarlos a `tests/data/` con excepción explícita en
`.gitignore` (`!tests/data/*.npy`, `!tests/data/*.mat`) y README de
procedencia. Los tests se saltan con instrucciones si faltan.

**Alternativas descartadas.** (a) Descargarlos de Zenodo en CI: aún no hay
DOI. (b) Git LFS: fricción para revisores, injustificada a 12 MB.

**Consecuencias.** El clon queda autosuficiente para `pytest`; cuando exista
el DOI de Zenodo se podrá migrar a descarga con checksum si el repo engorda.

---

## ADR-006 (2026-07-02) — Estrategia de tests de paridad en dos niveles

**Contexto.** Regenerar el mapa Python con los parámetros exactos de MATLAB
(450x450x4, 36 QS, 1000 part/QS) cuesta ~2 h CPU: inviable en CI.

**Decisión.** Nivel 1 (siempre): tests sobre los mapas ARCHIVADOS con el
criterio físico de contención — la corrida quick (tend=60 d) debe delinear
dentro de la envolvente MATLAB (tend=360 d); medido al archivar: 0.999
(homogéneo) y 0.998 (heterogéneo); umbral del test: ≥0.95. Nivel 2 (opt-in,
`pytest -m slow`): regeneración completa con el MISMO campo K que MATLAB y
umbrales Jaccard≥0.7 / |Δárea|≤25 % (solo difiere el random-walk
estocástico).

**Alternativas descartadas.** Comparar Jaccard directo quick-vs-MATLAB: da
~0.11 por construcción (horizontes distintos), no mide paridad sino subset.

**Consecuencias.** CI corre en ~1 min; la paridad fuerte queda documentada y
ejecutable bajo demanda antes de cada release.
