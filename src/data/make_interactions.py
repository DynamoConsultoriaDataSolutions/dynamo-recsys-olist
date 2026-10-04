"""Construye la tabla de interacciones usuario-producto a partir de los CSV crudos.

Salida: data/processed/interactions.parquet, una fila por (pedido, producto) con:
  order_id, customer_unique_id, product_id, seller_id, purchase_ts,
  price, freight_value, quantity, review_score, category

Decisiones de limpieza (documentadas en el log que imprime el script):
  - Usuario = customer_unique_id (customer_id cambia en cada pedido).
  - Se excluyen pedidos cancelados / no disponibles / sólo "created".
  - Reseñas: si un pedido tiene varias, se conserva la más reciente.
  - Ítems repetidos del mismo producto en un pedido se agregan en quantity.
  - Categorías traducidas al inglés; faltantes -> "unknown".

Uso:  python -m src.data.make_interactions
"""
import pandas as pd

from src.config import DATA_RAW, DATA_PROCESSED, INTERACTIONS_PATH, VALID_ORDER_STATUS


def load_raw():
    read = lambda name, **kw: pd.read_csv(DATA_RAW / f"{name}.csv", **kw)
    return {
        "orders": read("olist_orders_dataset", parse_dates=["order_purchase_timestamp"]),
        "items": read("olist_order_items_dataset"),
        "customers": read("olist_customers_dataset"),
        "products": read("olist_products_dataset"),
        "reviews": read("olist_order_reviews_dataset", parse_dates=["review_answer_timestamp"]),
        "translation": read("product_category_name_translation"),
    }


def build_interactions(raw):
    log = {}
    orders = raw["orders"]
    log["pedidos_totales"] = len(orders)
    orders = orders[orders["order_status"].isin(VALID_ORDER_STATUS)]
    log["pedidos_validos"] = len(orders)

    # Reseña más reciente por pedido
    reviews = raw["reviews"].sort_values("review_answer_timestamp")
    log["pedidos_con_varias_reseñas"] = int(reviews["order_id"].duplicated().sum())
    reviews = reviews.drop_duplicates("order_id", keep="last")[["order_id", "review_score"]]

    # Ítems -> una fila por (pedido, producto)
    items = (
        raw["items"]
        .groupby(["order_id", "product_id", "seller_id"], as_index=False)
        .agg(quantity=("order_item_id", "count"), price=("price", "mean"),
             freight_value=("freight_value", "mean"))
    )

    products = raw["products"].merge(raw["translation"], on="product_category_name", how="left")
    products["category"] = products["product_category_name_english"].fillna(
        products["product_category_name"]).fillna("unknown")
    log["productos_sin_categoria"] = int((products["category"] == "unknown").sum())

    df = (
        items.merge(orders[["order_id", "customer_id", "order_purchase_timestamp"]], on="order_id")
        .merge(raw["customers"][["customer_id", "customer_unique_id"]], on="customer_id")
        .merge(products[["product_id", "category"]], on="product_id", how="left")
        .merge(reviews, on="order_id", how="left")
        .rename(columns={"order_purchase_timestamp": "purchase_ts"})
        .drop(columns="customer_id")
    )
    log["interacciones"] = len(df)
    log["reseña_faltante_%"] = round(df["review_score"].isna().mean() * 100, 2)
    cols = ["order_id", "customer_unique_id", "product_id", "seller_id", "purchase_ts",
            "price", "freight_value", "quantity", "review_score", "category"]
    return df[cols].sort_values("purchase_ts").reset_index(drop=True), log


def summarize(df):
    n_users, n_items = df["customer_unique_id"].nunique(), df["product_id"].nunique()
    orders_per_user = df.groupby("customer_unique_id")["order_id"].nunique()
    buyers_per_item = df.groupby("product_id")["customer_unique_id"].nunique()
    pairs = df[["customer_unique_id", "product_id"]].drop_duplicates().shape[0]
    return {
        "usuarios": n_users,
        "productos": n_items,
        "categorias": df["category"].nunique(),
        "rango_fechas": f"{df['purchase_ts'].min():%Y-%m-%d} a {df['purchase_ts'].max():%Y-%m-%d}",
        "usuarios_que_recompran_%": round((orders_per_user > 1).mean() * 100, 2),
        "productos_con_1_comprador_%": round((buyers_per_item == 1).mean() * 100, 2),
        "densidad_matriz_%": round(pairs / (n_users * n_items) * 100, 5),
    }


def main():
    df, log = build_interactions(load_raw())
    DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
    df.to_parquet(INTERACTIONS_PATH, index=False)
    print("== Limpieza ==")
    for k, v in log.items():
        print(f"  {k}: {v}")
    print("== Resumen ==")
    for k, v in summarize(df).items():
        print(f"  {k}: {v}")
    print(f"Guardado en {INTERACTIONS_PATH}")


if __name__ == "__main__":
    main()
