"""Pipeline completo: de los CSV crudos a la comparación de modelos.

Ejecuta, en orden, los scripts que ya existen en el repo:
  1. download_data       -> verifica / descarga los 9 CSV en data/raw/
  2. make_interactions   -> limpia y genera data/processed/interactions.parquet
  3. data_quality        -> reporte de calidad en reports/dq_*.csv
  4. feature_engineering -> features de usuario y producto 
  5. train_compare       -> valida, entrena y compara modelos en reports/

Uso:  python -m src.pipeline
"""
import importlib
import time

PASOS = [
    ("1. Datos crudos", "src.data.download_data"),
    ("2. Interacciones", "src.data.make_interactions"),
    ("3. Calidad de datos", "src.data.data_quality"),
    ("4. Features", "src.data.feature_engineering"),
    ("5. Modelos", "src.models.train_compare"),
]


def correr_paso(nombre, modulo):
    """Importa el módulo y ejecuta su main(). Devuelve (estado, segundos)."""
    print(f"\n{'=' * 60}\n{nombre}  ({modulo})\n{'=' * 60}")
    inicio = time.time()
    try:
        mod = importlib.import_module(modulo)
    except ModuleNotFoundError:
        print(f"  [SALTADO] {modulo} todavía no existe en esta rama.")
        return "saltado", 0.0
    mod.main()
    return "ok", time.time() - inicio


def main():
    resumen = []
    for nombre, modulo in PASOS:
        estado, segundos = correr_paso(nombre, modulo)
        resumen.append((nombre, estado, segundos))

    print(f"\n{'=' * 60}\nRESUMEN DEL PIPELINE\n{'=' * 60}")
    for nombre, estado, segundos in resumen:
        print(f"  {nombre:<22} {estado:<8} {segundos:6.1f} s")


if __name__ == "__main__":
    main()