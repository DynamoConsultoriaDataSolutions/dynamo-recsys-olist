# Plan de validación

Cómo comprobamos que los datos y el modelo son confiables antes de usarlos en la app.
Los números salen de `python -m src.pipeline` (detalle completo en el README, sección "Evaluación").

**Responsables:** Cesar (evaluación del modelo, secciones 2–5) · Dennys (datos, reproducibilidad y automatización, secciones 1 y 6).

---

## 1. Validación de los datos

Cada corrida del pipeline revisa la tabla de interacciones en el paso 3 (`src/data/data_quality.py`):

| Control | Qué mide | Resultado actual |
|---|---|---|
| Faltantes | Nulos por columna | Solo `review_score` (0,77 %) |
| Nulos disfrazados | Categoría rellenada con `unknown` | 1,42 % de las filas |
| Outliers (IQR) | Valores extremos de `price` y `freight_value` | Se conservan: son ventas reales |
| Cantidades | Unidades por línea | 93 % de las líneas tiene 1 unidad |
| Desbalance | Concentración por categoría y estado | SP = 42 % de los clientes; top 10 categorías = 63 % de las ventas |

**Alarmas:**
- 🔴 **Error, el pipeline se detiene:** menos de 50.000 interacciones, IDs nulos o `price <= 0`.
- 🟡 **Aviso, el pipeline sigue:** más de 5 % de reseñas faltantes o más de 5 % de categoría `unknown`.

Las reglas se prueban con datos de juguete en `tests/test_data_quality.py`, y la sección 4.6 del notebook `02_dd_pipeline_reproducible.ipynb` muestra cómo reaccionan ante datos rotos a propósito.

## 2. Protocolo de evaluación

- **Split temporal:** entrenamiento con compras anteriores al 2018-06-01; test entre 2018-06-01 y 2018-09-01. Simula el uso real: predecir compras futuras con el historial pasado.
- **Qué cuenta como acierto:** productos *nuevos* que el cliente compró en el período de test.
- **Mismo protocolo para todos los modelos:** cada uno implementa `fit()` y `recommend()` y se evalúa con `evaluate()` (`src/evaluation/evaluate.py`), así los resultados son comparables.

## 3. Elección de hiperparámetros sin mirar el test

Los hiperparámetros se eligen en un período de **validación** separado: entrenamiento antes del 2018-03-01 y validación entre 2018-03-01 y 2018-06-01.

| Hiperparámetro | Valor elegido |
|---|---|
| Ventana de popularidad reciente | 30 días |
| Factores del SVD | 4 |

El test se usa una sola vez, al final, para comparar modelos ya configurados.

## 4. Métricas y criterio de decisión

| Métrica | Pregunta que responde |
|---|---|
| HitRate@10 | ¿Al menos una de las 10 recomendaciones fue comprada? |
| NDCG@10 | ¿Los aciertos aparecen arriba en la lista? |
| Precision@K / Recall@K | ¿Qué parte de lo recomendado y de lo comprado coincide? |
| Cobertura | ¿Qué porcentaje del catálogo llega a recomendarse? |

Se reportan para **todos** los clientes y para el segmento **warm** (clientes con compras previas), porque el 98 % de los clientes de test es nuevo.

**Criterio:** se elige el modelo que iguala o supera a los demás en acierto y tiene mayor cobertura → **item-to-item híbrido** (HitRate@10 1,37 % vs 1,19 % del baseline; cobertura 3,29 % vs 0,04 %).

## 5. Incertidumbre

Intervalos de confianza al 95 % por bootstrap sobre usuarios (2.000 remuestreos, `reports/confidence_intervals.csv`).

**Ninguna diferencia de HitRate@10 entre modelos es estadísticamente significativa.** Por eso la mejora en acierto se presenta como tendencia, y el argumento principal para elegir el híbrido es la cobertura.

## 6. Reproducibilidad y automatización

| Mecanismo | Qué garantiza |
|---|---|
| `python -m src.pipeline` | Un solo comando va de los CSV crudos al modelo final (7 pasos) |
| Fechas de corte y `RANDOM_STATE = 42` en `src/config.py` | Mismos resultados en cada corrida |
| Chequeo 7/7 | Al correr desde cero, los resultados clave coinciden con el README |
| GitHub Actions (`.github/workflows/tests.yml`) | `pytest` corre en cada PR y en cada push a `main` |
| MLflow (paso 7) | Cada corrida queda registrada con parámetros y métricas para compararla |

## 7. Limitaciones y próximos pasos

- Segmento warm chico (387 clientes): las diferencias en ese segmento tienen mucha incertidumbre.
- Un solo período de test; validar en más ventanas daría resultados más robustos.
- `n_neighbors` y `max_seeds` del item-to-item están fijos, sin validar.
- El acierto exige el producto exacto; no cuenta productos parecidos.