"""Rutas y parámetros globales del proyecto."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_RAW = ROOT / "data" / "raw"
DATA_PROCESSED = ROOT / "data" / "processed"
MODELS_DIR = ROOT / "models"
REPORTS_DIR = ROOT / "reports"

INTERACTIONS_PATH = DATA_PROCESSED / "interactions.parquet"

KAGGLE_DATASET = "olistbr/brazilian-ecommerce"

# Estados de pedido que se consideran compras válidas
VALID_ORDER_STATUS = {"delivered", "shipped", "invoiced", "processing", "approved"}

# Split temporal: se entrena con compras ANTES de TRAIN_END y se evalúa con
# compras entre TRAIN_END y TEST_END. Después de sep-2018 casi no hay datos.
TRAIN_END = "2018-06-01"
TEST_END = "2018-09-01"

K_VALUES = (5, 10, 20)
RANDOM_STATE = 42
