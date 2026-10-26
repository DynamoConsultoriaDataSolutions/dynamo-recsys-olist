# Dynamo Consultoría & Data Solutions — Sistema de recomendación para e-commerce (Olist)

Proyecto Final · Data Science · Henry

Prototipo de un sistema de recomendación de productos para un marketplace, construido sobre el
[Brazilian E-Commerce Public Dataset by Olist](https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce)
siguiendo CRISP-DM y trabajo ágil en sprints.

## Equipo

| Integrante | Rol |
|---|---|
| Hernández Monzalvo, Alicia Jacqueline | Data Analytics |
| Palmera, Daniel | Data Scientist |
| Duran, Dennys | Data Engineering |
| Montes de Oca, Julio César | Data Scientist |
| Allende, Nicolas | Data Analytics |
| Jauregui, Gaston Lautaro | Data Engineering |

## Estructura

```
├── app/                    # API (api.py), app de demo (app.py) y dashboard (pages/1_Dashboard.py)
├── docs/                   # plan de validación, guía del dashboard y plantilla de Pull Request
├── diagramas/              # diagrama entidad-relación (DER) de las tablas de Olist
├── data/
│   ├── raw/                # 9 CSV de Olist (no versionados)
│   └── processed/          # interactions.parquet (versionado, ~9.5 MB, para el dashboard publicado)
├── models/                 # modelo final item2item_hibrido.joblib (versionado para la API)
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

python -m src.pipeline                # corre TODO de punta a punta (o paso a paso con los comandos de abajo)
python -m src.data.download_data      # descarga los CSV (o cópialos a data/raw/)
python -m src.data.make_interactions  # genera data/processed/interactions.parquet
python -m src.data.data_quality       # reporte de calidad de datos → reports/dq_*.csv
python -m src.data.feature_engineering  # genera features de usuario y producto
python -m src.models.popularity       # entrena y evalúa los baselines
python -m src.models.train_compare    # valida, entrena y compara todos los modelos
python -m src.models.train_final      # entrena el modelo final y lo guarda en models/ para la app
python -m src.models.track_experiments  # registra parámetros y métricas en MLflow (mlflow.db)
mlflow ui --backend-store-uri sqlite:///mlflow.db  # ver los experimentos en http://127.0.0.1:5000
pytest                                # pruebas
```
## Usar el modelo en la app (Streamlit / API)

`python -m src.models.train_final` entrena el **item-to-item híbrido** con todas las compras
(o `--hasta AAAA-MM-DD`) y lo guarda en `models/item2item_hibrido.joblib` (~11 MB, ~10 s).
El archivo está versionado en el repo para poder desplegar la API sin reentrenar. Si se corre de
nuevo el pipeline se regenera (cambia su fecha de creación): conviene no subirlo otra vez salvo que
el modelo cambie a propósito.

```python
from src.models.predict import load_bundle, recommend, user_history, is_known, example_users

bundle = load_bundle()                    # en Streamlit: envolver en @st.cache_resource
recommend(bundle, "<customer_unique_id>", k=10)
# posicion | product_id | category | price | motivo | explicacion
user_history(bundle, "<customer_unique_id>")   # lo que ya compró
example_users(bundle, n=5)                     # clientes con historial para la demo
```

Cada recomendación trae su **motivo**: `co-compra` (otros lo compraron junto), `categoria`
(más vendidos de su categoría) o `tendencia` (más vendido del último mes). Un cliente que no
existe en los datos recibe la tendencia.

### Despliegue de la Demo (Arquitectura Desacoplada)

El sistema cuenta con una arquitectura dividida en un Backend de servicios y una interfaz gráfica interactiva en el Frontend. Ambos componentes se encuentran localizados dentro de la carpeta `app/`.

1. **Backend (API REST con FastAPI):** Se encarga de levantar el modelo `.joblib` en memoria de manera eficiente y exponer los endpoints de consulta de historial, estatus de usuario y generación de recomendaciones. Incorpora resiliencia nativa mediante control de excepciones para conmutar de forma automática al fallback por popularidad ante casos de *User Cold Start*.
2. **Frontend (Interfaz de Usuario con Streamlit):** Una pantalla interactiva diseñada para la presentación con stakeholders. Consume los endpoints del backend mediante solicitudes HTTP y formatea las respuestas utilizando componentes visuales y DataFrames. Incorpora optimización de memoria mediante `@st.cache_resource` para precargar de forma ágil la lista de usuarios de prueba desde el bundle.

#### Instrucciones de Ejecución

Para inicializar el ecosistema completo en tu máquina local, abre dos terminales independientes y asegúrate de tener el entorno virtual activo (`.venv`) en ambas. Ejecuta los siguientes comandos **estrictamente desde la raíz del proyecto**:

* **Terminal 1 - Levantar el Backend (Puerto 8000):**
  ```bash
  uvicorn app.api:app --reload
  ```
* **Terminal 2 - Levantar el Frontend (Puerto 8501):**
  ```bash
  streamlit run app/app.py
  ```

Se puede acceder a la APP desde https://dynamo-recsys-olist-n5ar88xfuj5pp5aubj4uwp.streamlit.app/

*Nota de salud del sistema:* La interfaz gráfica realiza un Health Check rápido apuntando al endpoint base (`GET /`) de la API antes de renderizar componentes pesados. Si el backend está apagado, la aplicación de Streamlit detendrá su ejecución de forma segura exponiendo una alerta técnica descriptiva en la UI en lugar de romper el flujo del sistema.


## Dashboard interactivo

Página de Streamlit con tres pestañas: **Negocio** (ventas, pedidos y clientes con filtros por fecha,
categoría y estado), **Modelos** (comparación de HitRate y cobertura por segmento y K, con intervalos
de confianza) y **Cómo leer los datos** (definiciones, decisiones y limitaciones). No necesita la API ni
el modelo: lee los datos y los reportes de `reports/`.

- Versión publicada: https://dynamo-olist-dashboard.streamlit.app/
- Local: `streamlit run app/app.py` (aparece como página **Dashboard** junto a la app de
  recomendaciones) o `streamlit run app/pages/1_Dashboard.py` (solo el dashboard).
- Detalle y definiciones: [`docs/dashboard.md`](docs/dashboard.md).

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

La tabla resultante, `data/processed/interactions.parquet` (~9.5 MB), está versionada en el repo para
que el dashboard publicado y los notebooks funcionen sin descargar los datos de Kaggle. Los CSV crudos
no se suben (~125 MB). Si se vuelve a correr `make_interactions`, se regenera con el mismo contenido.
Datos derivados del [Brazilian E-Commerce Public Dataset by Olist](https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce)
(licencia CC BY-NC-SA 4.0).

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

El plan de validación completo (datos, protocolo, hiperparámetros, métricas, incertidumbre y
reproducibilidad) está en [`docs/plan_validacion.md`](docs/plan_validacion.md).

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
| `notebooks/EDA.ipynb` | Análisis exploratorio (sparsity, long tail, cold start) y modelo relacional exportado a Excel para Power BI |
| `notebooks/00_explicacion_pr1.ipynb` | Datos, evaluación y baselines, paso a paso con los datos reales |
| `notebooks/01_explicacion_item_to_item.ipynb` | Modelo item-to-item híbrido, validación y resultados |
| `notebooks/02_dd_pipeline_reproducible.ipynb` | Pipeline reproducible: calidad de datos, features, modelos y chequeo de reproducibilidad |

## Modelo colaborativo SVD (Daniel Palmera)

Archivo: `src/models/collaborative.py` | Pruebas: `tests/test_collaborative.py`

Factorización de la matriz usuario-producto (compra = 1) con `TruncatedSVD`. El puntaje de un producto para un usuario es el producto punto entre sus factores latentes, y se recomiendan los mejor puntuados que el usuario no haya comprado. Complementa al item-to-item: aquél usa co-compras directas, éste aprende patrones latentes. Con `rating_weight=True` la matriz usa `review_score/5` en vez de 1.

**Cold start.** Cerca del 98% de los usuarios de test no tiene historial en train y un modelo colaborativo no puede recomendarles. Para ellos, y cuando faltan recomendaciones para completar K, se rellena con popularidad reciente (misma ventana de 30 días que el resto). Así la diferencia frente a los demás modelos se debe sólo a la parte colaborativa y debe leerse en el segmento `warm`.

**Hiperparámetro.** `n_factors` se eligió en el periodo de validación (no en test), igual que la ventana de popularidad. HitRate@10 de validación en todos los usuarios: 3.74% con 4 factores, 3.73% con 8, 16 y 32, y 3.71% con 64. Las diferencias en "todos" son mínimas porque casi todos los usuarios caen en el respaldo de popularidad; en warm (n = 365) gana 4 factores (2.47%) y baja con más factores (0.82% con 64), resultado compatible con la escasez de señal, sin demostrar por sí solo sobreajuste. Se usó `n_factors = 4`.

El barrido se reproduce con `python -m src.models.collaborative` y queda guardado en `reports/collaborative_validation.csv`. La comparación con ratings (`svd_32_rating`) se mantiene como experimento adicional en validación; el modelo comparado en test usa compras, con 4 factores.

**Resultados (test, K = 10)**

| Modelo | HitRate@10 todos | HitRate@10 warm | Cobertura |
|---|---|---|---|
| Popularidad reciente (30 días) | 1.35% | 1.03% | 0.04% |
| Item-to-item híbrido | 1.37% | 2.07% | 3.29% |
| SVD colaborativo (4 factores) | 1.35% | 1.29% | 0.21% |

**Incertidumbre.** SVD obtiene 5 aciertos frente a 8 del híbrido entre los 387 usuarios warm. La diferencia SVD − híbrido es −0.78 puntos porcentuales, con IC 95% [−2.58, +1.03]. El intervalo incluye cero: no demuestra una diferencia de acierto. `reports/confidence_intervals.csv` incluye SVD por segmento y comparaciones pareadas contra baseline, popularidad reciente e híbrido.

**Interpretación.** El SVD no supera al item-to-item: en todos los usuarios empata con popularidad reciente (1.35%) y en warm queda en 1.29%, por debajo del 2.07% del híbrido y de `pop_categoria`. Es el resultado esperable con este dataset: sólo ~3% de los usuarios recompra, ~60% de los productos tiene un único comprador y la densidad de la matriz es 0.003%, así que hay muy poca señal de la que aprender factores latentes. En esta evaluación, el híbrido y la popularidad por categoría obtienen mayor acierto puntual que SVD; la co-compra sola no mejora la popularidad reciente. La diferencia de acierto no implica una superioridad estadísticamente demostrada. El segmento warm tiene pocos usuarios, así que las diferencias entre modelos en ese segmento deben leerse con cautela.

```bash
python -m src.models.collaborative   # validación de 4/8/16/32/64 factores, sin tocar test
python -m src.models.train_compare   # comparación en test e intervalos, incluyendo SVD
pytest tests/test_collaborative.py
```

## Pipeline y MLOps

`python -m src.pipeline` corre 7 pasos en orden y al final muestra un resumen con el estado y los segundos de cada uno.

| Paso | Módulo | Salida |
|---|---|---|
| 1. Datos crudos | `src.data.download_data` | `data/raw/*.csv` |
| 2. Interacciones | `src.data.make_interactions` | `data/processed/interactions.parquet` |
| 3. Calidad de datos | `src.data.data_quality` | `reports/dq_*.csv` y alarmas |
| 4. Features | `src.data.feature_engineering` | features de usuario y producto |
| 5. Modelos | `src.models.train_compare` | `reports/model_comparison.csv`, `confidence_intervals.csv` |
| 6. Modelo final | `src.models.train_final` | `models/item2item_hibrido.joblib` (para la app) |
| 7. Tracking MLflow | `src.models.track_experiments` | `mlflow.db` (un run por modelo) |

**Alarmas de calidad (paso 3).** Después de guardar los reportes, se revisan los datos:
- 🔴 **Error, el pipeline se detiene:** menos de 50.000 interacciones, IDs nulos (`order_id`, `customer_unique_id`, `product_id`) o `price <= 0`.
- 🟡 **Aviso, el pipeline sigue:** más de 5 % de `review_score` faltante o más de 5 % de categoría `unknown`.

Los umbrales están al principio de la sección de alarmas en `src/data/data_quality.py`.

**Tracking de experimentos (paso 7).** Cada corrida registra en MLflow los parámetros (ventana reciente, fechas de corte, semilla) y las métricas de cada modelo (precision, recall, hit_rate, ndcg y coverage, por K y segmento). Para verlas, correr `mlflow ui --backend-store-uri sqlite:///mlflow.db` y abrir http://127.0.0.1:5000 → **Model training → Runs**. `mlflow.db` no se versiona.

**Reproducibilidad.** El split temporal y la semilla (`RANDOM_STATE = 42`) están fijos en `src/config.py`: correr el pipeline desde cero da los mismos números que las tablas de este README.

**Integración continua.** `.github/workflows/tests.yml` corre `pytest` en cada PR y en cada push a `main`. Los tests usan datos de juguete, así que no necesitan los CSV de Kaggle.


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
