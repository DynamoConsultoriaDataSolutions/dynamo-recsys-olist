import pandas as pd
from src.models.track_experiments import metricas_de_modelo, ventana_elegida


def test_ventana_elegida():
    validacion = pd.DataFrame({"recent_days": [None, 7.0, 30.0],
                               "hit_rate@10_val": [0.03, 0.02, 0.04]})
    assert ventana_elegida(validacion) == 30


def test_metricas_de_modelo():
    resultados = pd.DataFrame({
        "model": ["m1", "m2"], "segment": ["all", "all"], "k": [10, 10],
        "precision@k": [0.1, 0.2], "recall@k": [0.3, 0.4], "hit_rate@k": [0.5, 0.6],
        "ndcg@k": [0.7, 0.8], "coverage": [0.9, 1.0],
    })
    metricas = metricas_de_modelo(resultados, "m1")
    assert metricas["hit_rate_10_all"] == 0.5
    assert metricas["coverage_10_all"] == 0.9
    assert len(metricas) == 5