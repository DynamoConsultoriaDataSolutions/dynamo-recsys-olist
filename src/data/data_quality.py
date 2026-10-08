"""Reporte de calidad de datos sobre la tabla de interacciones.

Uso:  python -m src.data.data_quality
"""
import pandas as pd

from src.config import INTERACTIONS_PATH, REPORTS_DIR


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


if __name__ == "__main__":
    df = pd.read_parquet(INTERACTIONS_PATH)

    faltantes = check_missing(df)
    outliers = pd.DataFrame([check_outliers_iqr(df, c) for c in ["price", "freight_value"]])

    print("== Faltantes ==")
    print(faltantes.to_string(index=False))
    print("\n== Outliers (regla IQR) ==")
    print(outliers.to_string(index=False))

    faltantes.to_csv(REPORTS_DIR / "dq_faltantes.csv", index=False)
    outliers.to_csv(REPORTS_DIR / "dq_outliers.csv", index=False)
    print(f"\nGuardado en {REPORTS_DIR}")