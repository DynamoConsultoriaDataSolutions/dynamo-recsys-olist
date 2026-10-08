"""Reporte de calidad de datos sobre la tabla de interacciones.

Uso:  python -m src.data.data_quality
"""
import pandas as pd

from src.config import DATA_RAW, INTERACTIONS_PATH, REPORTS_DIR


def check_missing(df):
    """Cantidad y porcentaje de nulos por columna."""
    nulos = df.isna().sum()
    pct = (df.isna().mean() * 100).round(2)
    tabla = pd.DataFrame({
        "columna": df.columns,
        "nulos": nulos.values,
        "pct_nulos": pct.values,
    })
    return tabla.sort_values("pct_nulos", ascending=False).reset_index(drop=True)


def check_outliers_iqr(df, col):
    """Outliers de una columna numérica según la regla IQR (1,5 × IQR)."""
    q1 = df[col].quantile(0.25)
    q3 = df[col].quantile(0.75)
    iqr = q3 - q1
    limite_sup = q3 + 1.5 * iqr
    n_outliers = (df[col] > limite_sup).sum()
    return {
        "columna": col,
        "q1": q1,
        "q3": q3,
        "limite_sup": round(limite_sup, 2),
        "n_outliers": int(n_outliers),
        "pct_outliers": round(n_outliers / len(df) * 100, 2),
        "max": df[col].max(),
    }


def check_quantity(df):
    """Frecuencia de cada cantidad de unidades por línea de pedido."""
    conteo = df["quantity"].value_counts().sort_index()
    tabla = conteo.reset_index()
    tabla.columns = ["quantity", "filas"]
    tabla["pct"] = (tabla["filas"] / len(df) * 100).round(2)
    return tabla


def check_imbalance(serie, nombre):
    """Qué tan concentrados están los valores de una variable categórica."""
    pct = serie.value_counts(normalize=True) * 100
    return {
        "variable": nombre,
        "n_valores": serie.nunique(),
        "pct_si_fuera_parejo": round(100 / serie.nunique(), 2),
        "valor_top": pct.index[0],
        "pct_top": round(pct.iloc[0], 2),
        "pct_top10": round(pct.head(10).sum(), 2),
        "n_valores_menos_0.1pct": int((pct < 0.1).sum()),
    }


def estado_por_cliente():
    """Un estado por cliente (customer_unique_id), desde el CSV crudo de clientes."""
    customers = pd.read_csv(DATA_RAW / "olist_customers_dataset.csv")
    return customers.drop_duplicates("customer_unique_id")["customer_state"]


if __name__ == "__main__":
    df = pd.read_parquet(INTERACTIONS_PATH)

    faltantes = check_missing(df)
    outliers = pd.DataFrame([check_outliers_iqr(df, c) for c in ["price", "freight_value"]])
    cantidades = check_quantity(df)
    desbalance = pd.DataFrame([
        check_imbalance(df["category"], "category (ventas)"),
        check_imbalance(estado_por_cliente(), "customer_state (clientes)"),
    ])

    print("== Faltantes ==")
    print(faltantes.to_string(index=False))
    print("\n== Outliers (regla IQR) ==")
    print(outliers.to_string(index=False))
    print("\n== Cantidad de unidades por línea ==")
    print(cantidades.to_string(index=False))
    print("\n== Desbalance ==")
    print(desbalance.to_string(index=False))

    faltantes.to_csv(REPORTS_DIR / "dq_faltantes.csv", index=False)
    outliers.to_csv(REPORTS_DIR / "dq_outliers.csv", index=False)
    cantidades.to_csv(REPORTS_DIR / "dq_cantidades.csv", index=False)
    desbalance.to_csv(REPORTS_DIR / "dq_desbalance.csv", index=False)
    print(f"\nGuardado en {REPORTS_DIR}")