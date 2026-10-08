# Dynamo Consultoría & Data Solutions — Sistema de recomendación para e-commerce (Olist)

Proyecto Final · Data Science · Henry

Prototipo de un sistema de recomendación de productos para un marketplace, construido sobre el
[Brazilian E-Commerce Public Dataset by Olist](https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce)
siguiendo CRISP-DM y trabajo ágil en sprints.

## Equipo

| Integrante | Rol |
|---|---|
| Hernández Monzalvo, Alicia Jacqueline | Product Owner |
| Palmera, Daniel | Scrum Master |
| Duran, Dennys | Data Scientist |
| Montes de Oca, Julio César | Data Scientist |
| Allende, Nicolas | Data Scientist |
| Jauregui, Gaston Lautaro | Data Scientist |

## Estructura

```
├── app/                    # Demo (Streamlit) y API — Sprint 2
├── docs/                   # plantilla de Pull Request
├── data/
│   ├── raw/                # 9 CSV de Olist (no versionados)
│   └── processed/          # interactions.parquet (no versionado)
├── models/                 # modelos entrenados (no versionados)
├── notebooks/              # EDA, experimentos y explicaciones (NN_tema.ipynb)
├── reports/                # resultados, figuras, Power BI
├── src/
│   ├── config.py           # rutas y parámetros (split temporal, K)
│   ├── data/               # descarga y construcción de interacciones
│   ├── evaluation/         # protocolo de evaluación común
│   └── models/             # recomendadores (popularity, item_based, train_compare)
└── tests/
```

## Cómo correrlo

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate   |   Mac/Linux: source .venv/bin/activate
pip install -r requirements.txt

python -m src.data.download_data      # descarga los CSV (o cópialos a data/raw/)
python -m src.data.make_interactions  # genera data/processed/interactions.parquet
python -m src.models.popularity       # entrena y evalúa los baselines
python -m src.models.train_compare    # valida, entrena y compara todos los modelos
pytest                                # pruebas
```

## Datos: decisiones de limpieza

- Usuario = `customer_unique_id` (`customer_id` cambia en cada pedido).
- Se excluyen pedidos `canceled`, `unavailable` y `created` (98,202 de 99,441 se conservan).
- Pedidos con varias reseñas → se conserva la más reciente (551 casos).
- 610 productos sin categoría → `unknown`.
- Una fila por (pedido, producto); ítems repetidos se suman en `quantity`.

Hallazgos clave para el modelado: sólo **~3%** de los usuarios recompra, **~60%** de los productos
tiene un único comprador y la densidad de la matriz usuario-producto es **0.003%** →
problema dominado por **sparsity, long tail y cold start**.

## Evaluación

Protocolo único para todos los modelos (`src/evaluation/evaluate.py`):

- **Split temporal**: train = compras antes del 2018-06-01; test = 2018-06-01 a 2018-09-01.
- Relevantes = productos nuevos que el usuario compró en test.
- Métricas: Precision@K, Recall@K, HitRate@K, NDCG@K (K = 5, 10, 20) y cobertura de catálogo.
- Se reporta para todos los usuarios y para el segmento **warm** (con historial en train).

Todo modelo nuevo debe implementar `fit(train_df)` y `recommend(user_id, k)` y evaluarse con
`evaluate(...)` para que los resultados sean comparables.

### Validación (elección de hiperparámetros)

Los hiperparámetros se eligen en un periodo de **validación** que no toca el test:
train = compras antes del 2018-03-01, validación = 2018-03-01 a 2018-06-01.

| Hiperparámetro | Cómo se fijó | Valor |
|---|---|---|
| Ventana de popularidad reciente | Validación: 7–180 días y todo el historial | **30 días** (HitRate@10 val = 3.75% vs 3.37% con todo el historial; también la mejor en un segundo periodo, dic-2017 a mar-2018) |
| Filtro por precio en el nivel de categoría | Validación | Desactivado (empeoró el acierto) |
| `n_neighbors` (vecinos por producto) | **Fijo, sin validar** | 50 |
| `max_seeds` (compras usadas como semilla) | **Fijo, sin validar** | 5 |

Nota: con el test, 90 días parecía mejor (+51%), pero elegir así es sobreajustar al test;
por eso la elección se hace sólo en validación.

### Resultados en test (K = 10)

| Modelo | HitRate@10 all (n = 18,533) | HitRate@10 warm (n = 387) | NDCG@10 all | Cobertura |
|---|---|---|---|---|
| Popularidad global (baseline) | 1.19% | 1.03% (4 aciertos) | 0.0054 | 0.04% |
| Popularidad por categoría | 1.21% | 2.07% (8 aciertos) | 0.0050 | 1.67% |
| Popularidad reciente (30 días) | 1.35% | 1.03% (4 aciertos) | 0.0056 | 0.04% |
| Item-to-item sólo co-compra | 1.35% | 1.03% (4 aciertos) | 0.0056 | 1.98% |
| **Item-to-item híbrido** | **1.37%** | **2.07% (8 aciertos)** | **0.0057** | **3.29%** |

all = usuarios de test; warm = usuarios de test con compras previas en train.

### Incertidumbre (IC 95%, bootstrap por usuario)

Diferencias pareadas de HitRate@10, en puntos porcentuales (`reports/confidence_intervals.csv`):

| Comparación | all | warm |
|---|---|---|
| Híbrido − baseline | +0.18 [−0.03, +0.38] | +1.03 [−0.52, +2.58] |
| Popularidad reciente − baseline | +0.16 [−0.04, +0.37] | 0.00 [−1.03, +1.03] |
| Híbrido − popularidad reciente | +0.02 [−0.01, +0.06] | +1.03 [−0.52, +2.84] |
| Híbrido − popularidad por categoría | +0.16 [−0.05, +0.36] | 0.00 [0.00, 0.00] |

**Ninguna diferencia de HitRate@10 es estadísticamente significativa al 95%.** Los resultados se
leen como tendencias, no como mejoras demostradas.

### Interpretación

- **Contra el baseline**, el híbrido sube de 1.19% a 1.37% (+15% relativo), pero **casi toda la
  mejora viene de la recencia**: la popularidad de los últimos 30 días sola llega a 1.35%. Sobre
  ella, el híbrido aporta +0.02 pp (≈ +1.5% relativo).
- **En usuarios con historial**, el híbrido llega a 2.07%, igual que la popularidad por categoría,
  mientras que la co-compra sola se queda en 1.03%. **La ganancia en warm viene de la categoría,
  no de las co-compras** (sólo ~24% de los productos tiene alguna co-compra). Además son 8 aciertos
  contra 4 sobre 387 usuarios: muestra chica, IC amplio.
- **Lo que sí cambia con claridad es la cobertura**: el híbrido recomienda el 3.3% del catálogo
  contra el 0.04% del baseline (~80 veces más productos distintos), con un acierto igual o mejor.

**Modelo elegido: item-to-item híbrido** (`src/models/item_based.py`):

1. Usuario nuevo (98% de los casos) → popularidad de los últimos 30 días.
2. Usuario con historial → (a) productos co-comprados con sus compras (similitud coseno),
   (b) productos populares de la misma categoría, (c) popularidad reciente.

Por qué: iguala o supera a todas las alternativas en acierto y tiene, por mucho, la mayor
cobertura. Con estos datos no se puede afirmar que su acierto sea mejor de forma significativa.

Limitaciones: segmento warm muy chico; el acierto exige el producto exacto; un solo periodo de
test; `n_neighbors` y `max_seeds` sin validar. Reproducir: `python -m src.models.train_compare`.

## Notebooks

| Notebook | Contenido |
|---|---|
| `notebooks/00_explicacion_pr1.ipynb` | Datos, evaluación y baselines, paso a paso con los datos reales |
| `notebooks/01_explicacion_item_to_item.ipynb` | Modelo item-to-item híbrido, validación y resultados |

## Flujo de trabajo en Git

1. Nunca se trabaja directo en `main` (está protegida).
2. Crear rama desde `main` actualizada: `feature/<tarea>`, `fix/<tarea>`, `docs/<tarea>`.
3. Commits pequeños y descriptivos, en español.
4. Abrir Pull Request con la plantilla; **al menos 1 aprobación de otra persona** para hacer merge.
5. Nadie aprueba su propio PR.

```bash
git checkout main && git pull
git checkout -b feature/mi-tarea
# ...cambios...
git add . && git commit -m "Descripción del cambio"
git push -u origin feature/mi-tarea
```
