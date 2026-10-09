"""
Feature engineering de usuarios y productos.

Uso:
    python -m src.data.feature_engineering
"""

import pandas as pd

from src.config import INTERACTIONS_PATH, DATA_PROCESSED, TRAIN_END


def build_user_features(
    df: pd.DataFrame,
    cutoff=TRAIN_END
) -> pd.DataFrame:
    """
    Genera variables agregadas a nivel usuario utilizando
    únicamente información anterior al cutoff.
    """

    df = df.copy()

    # Gasto real considerando cantidad de unidades
    df["item_spend"] = df["price"] * df["quantity"]

    # Promedio global para imputar reviews faltantes
    global_review_mean = df["review_score"].mean()

    user_features = (
        df.groupby("customer_unique_id")
        .agg(
            user_total_orders=("order_id", "nunique"),
            user_total_products=("product_id", "nunique"),
            user_total_spent=("item_spend", "sum"),
            user_avg_price=("price", "mean"),
            user_avg_review=("review_score", "mean"),
            user_total_categories=("category", "nunique"),
            user_last_purchase=("purchase_ts", "max"),
        )
        .reset_index()
    )

    # Categoría favorita por usuario.
    # En caso de empate, se desempata alfabéticamente
    # para mantener un resultado determinista.
    favorite_category = (
        df.groupby(["customer_unique_id", "category"])
        .size()
        .reset_index(name="purchases")
        .sort_values(
            ["customer_unique_id", "purchases", "category"],
            ascending=[True, False, True]
        )
        .drop_duplicates("customer_unique_id")
        [["customer_unique_id", "category"]]
        .rename(columns={"category": "user_favorite_category"})
    )

    user_features = user_features.merge(
        favorite_category,
        on="customer_unique_id",
        how="left"
    )

    # Recencia respecto del cutoff
    user_features["user_recency_days"] = (
        pd.Timestamp(cutoff) - user_features["user_last_purchase"]
    ).dt.days

    user_features = user_features.drop(columns="user_last_purchase")

    # Imputar reviews faltantes con promedio global
    user_features["user_avg_review"] = (
        user_features["user_avg_review"]
        .fillna(global_review_mean)
    )

    return user_features


def build_product_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Genera variables agregadas a nivel producto.
    """

    global_review_mean = df["review_score"].mean()

    product_features = (
        df.groupby("product_id")
        .agg(
            prod_total_buyers=("customer_unique_id", "nunique"),
            prod_total_orders=("order_id", "nunique"),
            prod_total_quantity=("quantity", "sum"),
            prod_avg_price=("price", "mean"),
            prod_avg_review=("review_score", "mean"),
            prod_category=("category", "first"),
        )
        .reset_index()
    )

    # Imputar reviews faltantes con promedio global
    product_features["prod_avg_review"] = (
        product_features["prod_avg_review"]
        .fillna(global_review_mean)
    )

    return product_features


def main(cutoff=TRAIN_END):
    """
    Ejecuta el pipeline de feature engineering.
    """

    df = pd.read_parquet(INTERACTIONS_PATH)

    df["purchase_ts"] = pd.to_datetime(df["purchase_ts"])

    # Usar únicamente información anterior al cutoff
    # para evitar data leakage.
    train_df = df[
        df["purchase_ts"] < pd.Timestamp(cutoff)
    ].copy()

    user_features = build_user_features(
        train_df,
        cutoff=cutoff
    )

    product_features = build_product_features(train_df)

    user_features.to_parquet(
        DATA_PROCESSED / "user_features.parquet",
        index=False
    )

    product_features.to_parquet(
        DATA_PROCESSED / "product_features.parquet",
        index=False
    )

    print("Feature engineering completado")
    print(f"Fecha de corte de entrenamiento: {cutoff}")
    print(f"Interacciones utilizadas: {len(train_df)}")
    print(f"Usuarios: {len(user_features)}")
    print(f"Productos: {len(product_features)}")


if __name__ == "__main__":
    main()