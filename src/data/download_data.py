"""Descarga el dataset de Olist desde Kaggle a data/raw/.

Requisitos (una sola vez por persona):
  1. pip install kaggle
  2. Kaggle > Settings > API > "Create New Token" -> descarga kaggle.json
  3. Guardarlo en ~/.kaggle/kaggle.json  (Windows: C:\\Users\\<usuario>\\.kaggle\\kaggle.json)

Alternativa manual: descargar el zip desde
https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce y descomprimir los
9 CSV en data/raw/.

Uso:  python -m src.data.download_data
"""
import sys

from src.config import DATA_RAW, KAGGLE_DATASET

EXPECTED_FILES = [
    "olist_customers_dataset.csv",
    "olist_geolocation_dataset.csv",
    "olist_order_items_dataset.csv",
    "olist_order_payments_dataset.csv",
    "olist_order_reviews_dataset.csv",
    "olist_orders_dataset.csv",
    "olist_products_dataset.csv",
    "olist_sellers_dataset.csv",
    "product_category_name_translation.csv",
]


def missing_files():
    return [f for f in EXPECTED_FILES if not (DATA_RAW / f).exists()]


def main():
    DATA_RAW.mkdir(parents=True, exist_ok=True)
    if not missing_files():
        print(f"Los 9 CSV ya están en {DATA_RAW}. Nada que descargar.")
        return
    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
    except ImportError:
        sys.exit("Falta el paquete 'kaggle' (pip install kaggle) o descarga manual; ver docstring.")
    api = KaggleApi()
    api.authenticate()
    print(f"Descargando {KAGGLE_DATASET} ...")
    api.dataset_download_files(KAGGLE_DATASET, path=str(DATA_RAW), unzip=True)
    faltan = missing_files()
    if faltan:
        sys.exit(f"Descarga incompleta, faltan: {faltan}")
    print(f"Listo: {len(EXPECTED_FILES)} archivos en {DATA_RAW}")


if __name__ == "__main__":
    main()
