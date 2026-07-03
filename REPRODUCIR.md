# Cómo reproducir este proyecto

## Una sola instrucción
```bash
make reproduce          # cadena completa: entorno → datos → experimentos → figuras → informe
make reproduce-smoke    # verificación rápida (<15 min, N reducido)
```

## Con el asistente (Claude Code)
```
/reproducir <este proyecto> --smoke
```
Reconstruye en entorno limpio, compara los números contra los registrados y emite el
certificado de reproducibilidad (`docs/auditoria_reproducibilidad_*.md`).

## Requisitos
- Python <versión> · GPU <modelo, VRAM> (opcional para el humo)
- Espacio en disco: <X GB> · Tiempo completo estimado: <horas>

## Qué produce
| Artefacto | Ruta | Figura/tabla del paper |
|-----------|------|------------------------|
| <métricas e01> | resultados/<...> | Tabla 1 |
| <figura principal> | resultados/figuras/<...> | Fig. 3 |

Los datos crudos se descargan/generan con checksum/semilla declarados en `data/MANIFIESTO.md`.
Semillas y parámetros: `configs/*.yaml`. Cualquier discrepancia: abrir issue o consultar
`docs/decisiones.md`.
