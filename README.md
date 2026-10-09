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
├── data/
│   ├── raw/                # 9 CSV de Olist (no versionados)
│   └── processed/          # interactions.parquet (no versionado)
├── models/                 # modelos entrenados (no versionados)
├── notebooks/              # EDA y experimentos (NN_iniciales_tema.ipynb)
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
python -m src.data.feature_engineering  # genera features de usuario y producto
python -m src.models.popularity       # entrena y evalúa los baselines
python -m src.models.train_compare    # valida, entrena y compara todos los modelos
pytest                                # pruebas
```
## Feature engineering

El pipeline `src/data/feature_engineering.py` genera variables agregadas a nivel usuario y producto utilizando únicamente información anterior al corte temporal configurado, para evitar data leakage.

Genera:

- `data/processed/user_features.parquet`
- `data/processed/product_features.parquet`

Principales features de usuario:
- cantidad de órdenes
- productos distintos
- gasto total
- precio promedio
- review promedio
- cantidad de categorías
- categoría favorita
- recencia de compra

Principales features de producto:
- compradores distintos
- cantidad de órdenes
- unidades vendidas
- precio promedio
- review promedio
- categoría

Las columnas utilizan prefijos `user_` y `prod_` para facilitar su integración con los modelos.

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

- **Ventana de popularidad reciente:** se probaron 7–180 días y todo el historial. La mejor fue
  **30 días** (HitRate@10 val = 3.75% vs 3.37% con todo el historial), y también fue la mejor en un
  segundo periodo de validación (dic-2017 a mar-2018).
- **Filtro por precio en el nivel de contenido:** empeoró el acierto en validación → se desactivó.

Nota: con el test, 90 días parecía mejor (+51%), pero elegir así es sobreajustar al test;
por eso la elección se hace sólo en validación.

### Resultados en test (K = 10)

| Modelo | HitRate@10 all | HitRate@10 warm | NDCG@10 all | Cobertura | Mejora vs baseline (all) |
|---|---|---|---|---|---|
| Popularidad global (baseline) | 0.0119 | 0.0103 | 0.0054 | 0.04% | — |
| Popularidad por categoría | 0.0121 | 0.0207 | 0.0050 | 1.67% | +2% |
| Popularidad reciente (30 días) | 0.0135 | 0.0103 | 0.0056 | 0.04% | +14% |
| Item-to-item sólo co-compra | 0.0135 | 0.0103 | 0.0056 | 1.98% | +14% |
| **Item-to-item híbrido** | **0.0137** | **0.0207** | **0.0057** | **3.29%** | **+15%** |

all = 18,533 usuarios de test; warm = 387 con historial previo.

**Modelo elegido: item-to-item híbrido** (`src/models/item_based.py`):

1. Usuario nuevo (98% de los casos) → popularidad de los últimos 30 días.
2. Usuario con historial → (a) productos co-comprados con sus compras (similitud coseno),
   (b) productos populares de la misma categoría, (c) popularidad reciente.

Por qué: es el mejor en el total y empata al mejor en usuarios con historial (el doble que el
baseline), y recomienda una parte mucho mayor del catálogo (3.3% vs 0.04%).
La co-compra sola casi no aporta: sólo ~24% de los productos tiene alguna co-compra, por eso
se combina con categoría.

Limitaciones: el segmento warm tiene sólo 387 usuarios (diferencias pequeñas pueden ser ruido) y
el acierto exige el producto exacto. Reproducir: `python -m src.models.train_compare`.

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
