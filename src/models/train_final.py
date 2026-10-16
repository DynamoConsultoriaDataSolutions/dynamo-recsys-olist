"""Entrena el modelo final (item-to-item híbrido) y lo guarda para que lo use la app.

A diferencia de train_compare.py, que separa train/test para EVALUAR, aquí se entrena con
TODAS las compras disponibles (o hasta la fecha que se indique), porque el objetivo es
recomendar lo más actual posible. Los hiperparámetros son los elegidos en validación.

Salida: models/item2item_hibrido.joblib, un diccionario con:
    model         -> ItemToItemRecommender ya entrenado
    trained_until -> fecha de la última compra usada
    params        -> hiperparámetros
    products      -> catálogo mínimo (categoría, precio mediano, compradores) para mostrar
    created_at, n_users, n_products

Uso:
    python -m src.models.train_final                 # con todos los datos
    python -m src.models.train_final --hasta 2018-06-01
"""
import argparse
from datetime import datetime, timezone

import joblib
import pandas as pd

from src.config import INTERACTIONS_PATH, MODELS_DIR
from src.models.item_based import ItemToItemRecommender

MODEL_PATH = MODELS_DIR / "item2item_hibrido.joblib"
PARAMS = dict(recent_days=30, max_seeds=5, n_neighbors=50, use_content=True, price_tolerance=None)


def build_bundle(df, hasta=None):
    """Entrena con las compras anteriores a `hasta` (o todas) y arma el paquete a guardar."""
    if hasta is not None:
        df = df[df["purchase_ts"] < pd.Timestamp(hasta)]
    model = ItemToItemRecommender(**PARAMS).fit(df)
    products = (df.groupby("product_id")
                .agg(category=("category", "first"), price=("price", "median"),
                     buyers=("customer_unique_id", "nunique"))
                .reset_index())
    return {
        "model": model,
        "params": PARAMS,
        "trained_until": str(df["purchase_ts"].max()),
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "n_users": int(df["customer_unique_id"].nunique()),
        "n_products": int(df["product_id"].nunique()),
        "products": products,
    }


def save_bundle(bundle, path=MODEL_PATH):
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, path, compress=3)
    return path


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--hasta", default=None, help="Usar solo compras anteriores a esta fecha (AAAA-MM-DD)")
    args = parser.parse_args(argv)

    df = pd.read_parquet(INTERACTIONS_PATH)
    bundle = build_bundle(df, args.hasta)
    path = save_bundle(bundle)
    size_mb = path.stat().st_size / 1e6
    print(f"Modelo guardado en {path} ({size_mb:.1f} MB)")
    print(f"Entrenado con compras hasta {bundle['trained_until']} · "
          f"{bundle['n_users']:,} clientes · {bundle['n_products']:,} productos")


if __name__ == "__main__":
    main()
