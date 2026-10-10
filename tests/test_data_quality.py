import pandas as pd

from src.data.data_quality import check_missing, check_unknown


def test_check_missing():
    df = pd.DataFrame({
        "review_score": [5.0, None, 4.0, 3.0],
        "price": [10.0, 20.0, 30.0, 40.0],
    })
    tabla = check_missing(df).set_index("columna")
    assert tabla.loc["review_score", "nulos"] == 1
    assert tabla.loc["review_score", "pct_nulos"] == 25.0
    assert tabla.loc["price", "nulos"] == 0


def test_check_unknown():
    df = pd.DataFrame({
        "product_id": ["p1", "p2", "p2", "p3"],
        "category": ["toys", "unknown", "unknown", "books"],
    })
    r = check_unknown(df)
    assert r["filas"] == 2
    assert r["pct_filas"] == 50.0
    assert r["productos"] == 1