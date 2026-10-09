import pandas as pd

from src.config import INTERACTIONS_PATH, DATA_PROCESSED, TRAIN_END


def build_user_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Genera variables agregadas a nivel usuario utilizando
    únicamente información del período de entrenamiento.
    """

    df = df.copy()

    # Gasto real considerando cantidad de unidades
    df["item_spend"] = df["price"] * df["quantity"]

    user_features = (
        df.groupby("customer_unique_id")
        .agg(
            total_orders=("order_id", "nunique"),
            total_products=("product_id", "nunique"),
            total_spent=("item_spend", "sum"),
            avg_price=("price", "mean"),
            avg_review=("review_score", "mean"),
            total_categories=("category", "nunique"),
            last_purchase=("purchase_ts", "max"),
        )
        .reset_index()
    )

    # Categoría más frecuente por usuario
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
        .rename(columns={"category": "favorite_category"})
    )

    user_features = user_features.merge(
        favorite_category,
        on="customer_unique_id",
        how="left"
    )

    # Días desde última compra hasta el corte de entrenamiento
    user_features["recency_days"] = (
       pd.Timestamp(cutoff) - user_features["last_purchase"]
    ).dt.days

    user_features = user_features.drop(columns="last_purchase")

    return user_features


def build_product_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Genera variables agregadas a nivel producto utilizando
    únicamente información del período de entrenamiento.
    """

    product_features = (
        df.groupby("product_id")
        .agg(
            total_buyers=("customer_unique_id", "nunique"),
            total_orders=("order_id", "nunique"),
            total_quantity=("quantity", "sum"),
            avg_price=("price", "mean"),
            avg_review=("review_score", "mean"),
            category=("category", "first"),
        )
        .reset_index()
    )

    return product_features


def main():
    # Cargar interacciones limpias
    df = pd.read_parquet(INTERACTIONS_PATH)

    # Asegurar formato de fecha
    df["purchase_ts"] = pd.to_datetime(df["purchase_ts"])

    # Trabajar únicamente con datos anteriores al corte
    # para evitar data leakage.
    train_df = df[
        df["purchase_ts"] < pd.Timestamp(TRAIN_END)
    ].copy()

    # Construcción de features
    user_features = build_user_features(train_df)
    product_features = build_product_features(train_df)

    # Guardado
    user_features.to_parquet(
        DATA_PROCESSED / "user_features.parquet",
        index=False
    )

    product_features.to_parquet(
        DATA_PROCESSED / "product_features.parquet",
        index=False
    )

    print("Feature engineering completado")
    print(f"Fecha de corte de entrenamiento: {TRAIN_END}")
    print(f"Interacciones utilizadas: {len(train_df)}")
    print(f"Usuarios: {len(user_features)}")
    print(f"Productos: {len(product_features)}")


if __name__ == "__main__":
    main()