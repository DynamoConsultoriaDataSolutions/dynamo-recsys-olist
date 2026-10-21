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
    
# ---------------------------------------------------------------- alarmas
from src.data.data_quality import check_alarmas


def datos_sanos():
    return pd.DataFrame({
        "order_id": ["o1", "o2", "o3", "o4"],
        "customer_unique_id": ["c1", "c2", "c3", "c4"],
        "product_id": ["p1", "p2", "p3", "p4"],
        "price": [10.0, 20.0, 30.0, 40.0],
        "review_score": [5.0, 4.0, 3.0, 5.0],
        "category": ["toys", "books", "toys", "books"],
    })
    
def test_alarmas_datos_sanos():
    errores, avisos = check_alarmas(datos_sanos(), min_interacciones =1)
    assert errores == []
    assert avisos == []
    
def test_alarmas_pocas_interacciones():
    errores, _ = check_alarmas(datos_sanos(), min_interacciones=100)
    assert len(errores) == 1
    
def test_alarmas_id_nulo_y_precio_cero():
    df = datos_sanos()
    df.loc[0, "product_id"] = None
    df.loc[1, "price"] = 0.0
    errores,_ = check_alarmas(df, min_interacciones=1)
    assert len(errores) == 2
    
def test_alarmas_aviso_no_es_error():
    df = datos_sanos()
    df.loc[0, "review_score"] = None            # 25% faltante: supera el 5%
    errores, avisos = check_alarmas(df, min_interacciones=1)
    assert errores == []
    assert len(avisos) == 1