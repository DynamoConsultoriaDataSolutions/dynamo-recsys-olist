"""Funciones para consumir el modelo guardado desde la app (Streamlit) o una API.

Antes, generar el modelo una vez:  python -m src.models.train_final

Ejemplo:
    from src.models.predict import load_bundle, recommend, user_history, example_users

    bundle = load_bundle()                       # carga models/item2item_hibrido.joblib
    recommend(bundle, "<customer_unique_id>", k=10)
    # -> DataFrame: posicion, product_id, category, price, motivo

En Streamlit, cachear la carga para no leer el archivo en cada interacción:
    @st.cache_resource
    def get_bundle():
        return load_bundle()
"""
from pathlib import Path

import joblib
import pandas as pd

from src.models.train_final import MODEL_PATH

MOTIVOS = {
    "co-compra": "Otros clientes lo compraron junto con lo que este cliente compró",
    "categoria": "Es de los más vendidos de la categoría que el cliente ya compró",
    "tendencia": "Es de lo más vendido en el último mes",
}


def load_bundle(path=MODEL_PATH):
    """Carga el paquete guardado por train_final.py. Falla con un mensaje claro si no existe."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"No existe {path}. Generarlo primero con:  python -m src.models.train_final")
    return joblib.load(path)


def recommend(bundle, user_id, k=10):
    """Top-k recomendaciones de un cliente, con categoría, precio y motivo de cada una.

    Si el cliente no existe en los datos (cliente nuevo), recibe lo más vendido del último mes.
    """
    pairs = bundle["model"].recommend_with_reasons(user_id, k)
    out = pd.DataFrame(pairs, columns=["product_id", "motivo"])
    out = out.merge(bundle["products"][["product_id", "category", "price"]], on="product_id", how="left")
    out.insert(0, "posicion", range(1, len(out) + 1))
    out["explicacion"] = out["motivo"].map(MOTIVOS)
    return out[["posicion", "product_id", "category", "price", "motivo", "explicacion"]]


def is_known(bundle, user_id):
    """True si el cliente tiene compras en los datos de entrenamiento."""
    return user_id in bundle["model"].history_


def user_history(bundle, user_id):
    """Productos que el cliente ya compró (del más reciente al más antiguo), con categoría y precio."""
    items = bundle["model"].history_.get(user_id, [])
    hist = pd.DataFrame({"product_id": items})
    return hist.merge(bundle["products"][["product_id", "category", "price"]], on="product_id", how="left")


def example_users(bundle, n=5, min_products=2):
    """Algunos clientes con historial, útiles para la demo (orden determinista)."""
    hist = bundle["model"].history_
    users = sorted(u for u, items in hist.items() if len(items) >= min_products)
    return users[:n]
