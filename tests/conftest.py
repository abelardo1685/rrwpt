"""Configuración compartida de la suite rrwpt.

Los datos de validación (mapas de referencia MATLAB y campo K prescrito)
viven en tests/data/ (~12 MB, versionados; ver tests/data/README.md).
Si faltan, los tests de paridad se saltan con instrucciones.
"""
from pathlib import Path

import pytest

DATA_DIR = Path(__file__).parent / "data"

MISSING_DATA_MSG = (
    "Datos de validación ausentes en tests/data/. Copiarlos desde la fuente "
    "de verdad: 0.-Simuladores/1.-Simulador_Transient_RRWPT_python_github_V/"
    "validation/ (matlab_prob_map*.npy, python_prob_map_*_quick.npy, "
    "Y_original_450x450.mat)."
)


def require_data(*names):
    """Salta el test si falta alguno de los archivos de validación."""
    for name in names:
        if not (DATA_DIR / name).exists():
            pytest.skip(f"{MISSING_DATA_MSG} (falta: {name})")
    return [DATA_DIR / n for n in names]
