"""Lectura y agregaciones del dashboard, sin depender de la API ni del modelo."""
from pathlib import Path

import pandas as pd


METRICS = ["precision@k", "recall@k", "hit_rate@k", "ndcg@k", "coverage"]


def load_business(root: Path) -> pd.DataFrame:
    """Usa el precio unitario medio del pipeline multiplicado por quantity."""
    df = pd.read_parquet(root / "data/processed/interactions.parquet")
    required = {"order_id", "customer_unique_id", "product_id", "purchase_ts",
                "category", "price", "quantity"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Faltan columnas de interacciones: {', '.join(sorted(missing))}")
    df = df.copy()
    df["purchase_ts"] = pd.to_datetime(df["purchase_ts"], errors="raise")
    if df["purchase_ts"].isna().any():
        raise ValueError("Hay interacciones sin fecha de compra.")
    df["category"] = df["category"].fillna("unknown")
    df["sales_brl"] = df["price"] * df["quantity"]
    df["customer_state"] = "Sin dato"

    raw = root / "data/raw"
    orders_path = raw / "olist_orders_dataset.csv"
    customers_path = raw / "olist_customers_dataset.csv"
    if orders_path.exists() and customers_path.exists():
        orders = pd.read_csv(orders_path, usecols=["order_id", "customer_id"])
        customers = pd.read_csv(customers_path, usecols=["customer_id", "customer_state"])
        geography = orders.merge(customers, on="customer_id", validate="many_to_one")
        df = df.drop(columns="customer_state").merge(
            geography[["order_id", "customer_state"]], on="order_id",
            how="left", validate="many_to_one",
        )
        df["customer_state"] = df["customer_state"].fillna("Sin dato")
    return df


def filter_business(df, start, end, categories=None, states=None):
    """Intervalo inclusivo de días; [] significa ninguna selección, None significa todas."""
    start = pd.Timestamp(start).normalize()
    stop = pd.Timestamp(end).normalize() + pd.Timedelta(days=1)
    if start >= stop:
        raise ValueError("La fecha inicial debe ser anterior o igual a la final.")
    mask = df["purchase_ts"].ge(start) & df["purchase_ts"].lt(stop)
    if categories is not None:
        mask &= df["category"].isin(categories)
    if states is not None:
        mask &= df["customer_state"].isin(states)
    return df.loc[mask].copy()


def business_summary(df):
    orders = int(df["order_id"].nunique())
    customers = int(df["customer_unique_id"].nunique())
    sales = float(df["sales_brl"].sum())
    purchases = df.groupby("customer_unique_id")["order_id"].nunique()
    return {
        "orders": orders, "customers": customers,
        "products": int(df["product_id"].nunique()),
        "units": int(df["quantity"].sum()), "sales_brl": sales,
        "sales_per_order_brl": sales / orders if orders else 0.0,
        "repeat_customers_pct": float(purchases.gt(1).mean() * 100) if customers else 0.0,
    }


def monthly_sales(df):
    return (df.assign(month=df["purchase_ts"].dt.to_period("M").dt.to_timestamp())
            .groupby("month", as_index=False)
            .agg(sales_brl=("sales_brl", "sum"), orders=("order_id", "nunique")))


def category_sales(df):
    return (df.groupby("category", as_index=False)
            .agg(sales_brl=("sales_brl", "sum"), units=("quantity", "sum"),
                 orders=("order_id", "nunique"))
            .sort_values("sales_brl", ascending=False))


def product_sales(df):
    return (df.groupby(["product_id", "category"], as_index=False)
            .agg(sales_brl=("sales_brl", "sum"), units=("quantity", "sum"),
                 buyers=("customer_unique_id", "nunique"))
            .sort_values("sales_brl", ascending=False))


def load_metrics(root):
    df = pd.read_csv(root / "reports/model_comparison.csv")
    required = {"model", "segment", "k", "n_users", *METRICS}
    if not required.issubset(df.columns):
        raise ValueError("model_comparison.csv no contiene las columnas de evaluación esperadas.")
    if df.duplicated(["model", "segment", "k"]).any():
        raise ValueError("Hay resultados repetidos para el mismo modelo, segmento y K.")
    return df


def select_metrics(df, segment, k, models):
    return df.loc[(df["segment"] == segment) & (df["k"] == k)
                  & df["model"].isin(models)].copy()


def load_confidence(root):
    path = root / "reports/confidence_intervals.csv"
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path)
    required = {"comparacion", "segment", "estimado", "ic95_bajo", "ic95_alto"}
    if not required.issubset(df.columns):
        raise ValueError("confidence_intervals.csv no tiene las columnas esperadas.")
    return df


def select_confidence(df, segment, k, models):
    """Los intervalos guardados corresponden exclusivamente a HitRate@10."""
    if df.empty or k != 10:
        return pd.DataFrame()
    return df.loc[(df["segment"] == segment) & df["comparacion"].isin(models)].copy()
