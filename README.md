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
│   └── models/             # recomendadores
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

### Resultados baseline (K = 10)

| Modelo | Segmento | Precision@10 | Recall@10 | HitRate@10 | Cobertura |
|---|---|---|---|---|---|
| Popularidad global | all | 0.0012 | 0.0115 | 0.0119 | 0.04% |
| Popularidad por categoría (pond. rating) | all | 0.0012 | 0.0117 | 0.0120 | 1.67% |
| Popularidad global | warm | 0.0010 | 0.0073 | 0.0103 | 0.04% |
| Popularidad por categoría (pond. rating) | warm | 0.0018 | 0.0168 | 0.0181 | 1.67% |

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
