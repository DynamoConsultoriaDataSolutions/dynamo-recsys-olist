"""Registra en MLflow los parámetros y métricas de la última corrida de modelos.

Lee los resultados que dejó train_compare en reports/ y crea un "run" por modelo,
así cada ejecución del pipeline queda guardada y se puede comparar con las anteriores.

Uso:        python -m src.models.track_experiments
Ver runs:   mlflow ui --backend-store-uri sqlite:///mlflow.db   (desde la raíz del repo)
"""
import mlflow
import pandas as pd

from src.config import RANDOM_STATE, REPORTS_DIR, ROOT, TEST_END, TRAIN_END, VAL_START

TRACKING_URI = f"sqlite:///{(ROOT / 'mlflow.db').as_posix()}"
EXPERIMENTO = "recsys-olist"
METRICAS = ["precision", "recall", "hit_rate", "ndcg", "coverage"]


def ventana_elegida(validacion):
    """Ventana de popularidad reciente (días) con mejor HitRate@10 en validación."""
    mejor = validacion.sort_values("hit_rate@10_val", ascending=False).iloc[0]
    return "todo" if pd.isna(mejor["recent_days"]) else int(mejor["recent_days"])


def metricas_de_modelo(resultados, modelo):
    """Dict {nombre_metrica: valor} de un modelo, para todos los K y segmentos.

    MLflow no acepta '@' en los nombres: 'hit_rate@k' con k=10 y segment=all
    queda como 'hit_rate_10_all'.
    """
    filas = resultados[resultados["model"] == modelo]
    metricas = {}
    for _, fila in filas.iterrows():
        for m in METRICAS:
            col = m if m == "coverage" else f"{m}@k"
            metricas[f"{m}_{fila['k']}_{fila['segment']}"] = float(fila[col])
    return metricas


def main():
    resultados = pd.read_csv(REPORTS_DIR / "model_comparison.csv")
    validacion = pd.read_csv(REPORTS_DIR / "validation_recent_window.csv")
    ventana = ventana_elegida(validacion)

    mlflow.set_tracking_uri(TRACKING_URI)
    mlflow.set_experiment(EXPERIMENTO)

    for modelo in resultados["model"].unique():
        with mlflow.start_run(run_name=modelo):
            mlflow.log_params({
                "modelo": modelo,
                "ventana_reciente_dias": ventana,
                "val_start": VAL_START,
                "train_end": TRAIN_END,
                "test_end": TEST_END,
                "random_state": RANDOM_STATE,
            })
            mlflow.log_metrics(metricas_de_modelo(resultados, modelo))
            mlflow.log_artifact(str(REPORTS_DIR / "model_comparison.csv"))
        print(f"  run registrado: {modelo}")

    print(f"\n{resultados['model'].nunique()} runs guardados en el experimento '{EXPERIMENTO}'")
    print("Para verlos:  mlflow ui --backend-store-uri sqlite:///mlflow.db")


if __name__ == "__main__":
    main()