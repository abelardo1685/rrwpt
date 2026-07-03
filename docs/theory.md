# Theory — transient probabilistic WHPA delineation (1 página)

> Esqueleto para el paper de software. Método: Rodríguez-Pretelín & Nowak
> (2018), *Adv. Water Resour.* 119, 178-187,
> [doi:10.1016/j.advwatres.2018.07.005](https://doi.org/10.1016/j.advwatres.2018.07.005).
> Ecuaciones extraídas de los docstrings de `fem.py`, `reference.py` y `rwpt.py`.

## 1. Flujo transitorio por superposición cuasi-estacionaria

Flujo saturado en un acuífero confinado heterogéneo:

$$ S_s \frac{\partial h}{\partial t} = \nabla \cdot \left( K(\mathbf{x}) \nabla h \right) + w(\mathbf{x}, t), $$

con $K = e^{Y}$ y $Y(\mathbf{x})$ un campo aleatorio (Matérn) generado por
circulant embedding espectral (Dietrich & Newsam). El horizonte se divide en
pasos cuasi-estacionarios (QS) de longitud `deltQS`; dentro de cada QS el
campo de velocidad se toma estacionario. En vez de resolver el FEM en cada
QS, `reference.py` resuelve una sola vez los **campos de referencia** (carga
unitaria por gradiente regional, dirección, bombeo por pozo y recarga) y
compone cada QS por **superposición lineal** escalada por los drivers
transitorios $\{dh(t), \theta_h(t), Q_p(t), q_r(t)\}$ (sinusoidales con
amplitud/fase inciertas, `flowpar.py`). La velocidad de poro es

$$ \mathbf{v} = -\frac{K}{\theta} \nabla h = \frac{\mathbf{q}}{\theta}. $$

## 2. RWPT inverso (reverse random walk particle tracking)

Las partículas se liberan EN el pozo y se rastrean **hacia atrás** en el
tiempo (campo de velocidad invertido, $\tilde{\mathbf{v}} = -\mathbf{v}$;
`rwpt.vel_direction`). Cada partícula sigue el esquema de Itô equivalente a
la ecuación de advección–dispersión:

$$ \mathbf{X}_p(t+\Delta t) = \mathbf{X}_p(t) + \left[ \tilde{\mathbf{v}}(\mathbf{X}_p) + \nabla \cdot \mathbf{D}(\mathbf{X}_p) \right] \Delta t + \mathbf{B}(\mathbf{X}_p) \, \boldsymbol{\xi} \sqrt{\Delta t}, \qquad \boldsymbol{\xi} \sim \mathcal{N}(\mathbf{0}, \mathbf{I}), $$

con el tensor de dispersión de Scheidegger

$$ \mathbf{D} = \left( \alpha_T \lVert \mathbf{v} \rVert + D_m \right) \mathbf{I} + \left( \alpha_L - \alpha_T \right) \frac{\mathbf{v} \mathbf{v}^{\mathsf{T}}}{\lVert \mathbf{v} \rVert}, \qquad \mathbf{B}\mathbf{B}^{\mathsf{T}} = 2\,\mathbf{D}, $$

el término de deriva $\nabla \cdot \mathbf{D}$ (corrección de Itô–Fokker–
Planck, `RW_dDD_Calculation`) y reflexión en fronteras/pozo
(`RW_mirror_particles`, `RW_Bounce_particles_well_location`).

## 3. Delineación probabilística continua en el tiempo

La novedad del método 2018: la WHPA deja de ser un objeto estático. En cada
intervalo de inyección `TTI` dentro de cada QS se libera un lote de
partículas por pozo mientras $t \le t_{crit}$; la huella binaria de cada lote
(celdas visitadas hacia atrás dentro del tiempo de viaje crítico
$t_{crit}$, p. ej. la isócrona de 50 días) se acumula en línea. El mapa
final es

$$ P(\mathbf{x}) = \frac{1}{N_{lotes}} \sum_{b} \mathbb{1}\left[ \mathbf{x} \in \Omega_b \right] \times 100\,\%, $$

normalizado a su máximo: la **frecuencia con la que cada celda pertenece a
la zona de captura transitoria**. Sobre un ensamble Monte Carlo de campos
$Y$ y drivers, $P$ se promedia además entre realizaciones, dando la WHPA
probabilística bajo heterogeneidad y transitoriedad conjuntas.

## 4. Verificación

- Paridad MATLAB↔Python: `tests/test_matlab_parity.py` (mismo campo
  $Y$ prescrito, malla 450x450x4, 6 pozos).
- Convenciones MATLAB preservadas: orden column-major ('F') en todos los
  reshape/sub2ind; índices internos 0-based.

<!-- TODO(paper): añadir figura del esquema QS/TTI y la comparación
     homogéneo/heterogéneo de validation/. -->
