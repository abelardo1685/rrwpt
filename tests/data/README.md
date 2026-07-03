# Datos de validación (paridad MATLAB <-> Python)

Copiados SIN modificar desde la fuente de verdad del simulador:
`/home/abelardo/UNAM/0.-Simuladores/1.-Simulador_Transient_RRWPT_python_github_V/validation/`
(2026-07-02). Total ~12 MB — se versionan en git (excepción en .gitignore).

| Archivo | Contenido | Origen |
|---|---|---|
| `matlab_prob_map.npy` | Mapa WHPA MATLAB (451x451, %), escenario homogéneo, 500 realizaciones | corrida MATLAB `Transient_RRWPT` |
| `matlab_prob_map_het.npy` | Mapa WHPA MATLAB, heterogéneo con `Y_original` | corrida MATLAB |
| `python_prob_map_homo_quick.npy` | Mapa Python archivado, corrida quick (malla 450x450x4, tend=60 d, t_crit=30 d, 150 part/QS, semilla 11) | notebook `Comparacion_MATLAB_vs_Python_areas.ipynb` |
| `python_prob_map_het_quick.npy` | Ídem, heterogéneo, semilla 2026 | ídem |
| `Y_original_450x450.mat` | Campo log-K (450,450,4) idéntico al usado por MATLAB (variable `Y`) | modelo MATLAB original |

Parámetros MATLAB de referencia: malla 450x450x4 (celdas 15 m), 6 pozos,
`tend=360 d`, `deltQS=10 d`, `t_crit=180 d`, 1000 partículas/pozo/QS.

Métricas medidas al archivar (contención de la envolvente Python-quick en la
MATLAB): homogéneo 0.999, heterogéneo 0.998 — ver `test_matlab_parity.py`.
